"""Explain the gradient-boosting model (v4): every parameter it uses, what it means, and how
predictive it is.

    python -m churn.explain        # ~2 min; writes reports/GB_FEATURES.md and gb_features.csv

Gradient boosting has no single weight per parameter (it is 400 small decision trees), so
predictive power is reported three ways:
  perm_auc_drop   how much held-out ROC-AUC falls when this parameter is shuffled
                  (5 account-grouped folds x 3 repeats). The closest thing to a "weight".
  group drop      the same, shuffling a whole family of parameters at once
  univariate_auc  ROC-AUC of the parameter on its own. 0.5 = no signal; further from 0.5
                  in either direction = stronger. Direction says which way churn goes.
Medians are in natural units (counts, %, days), not the log scale the model sees.
"""
import json
import re

import numpy as np
import openpyxl
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from . import cohorts, config, features, models, signals
from .run import groups_of, log

DICT_XLSX = config._p("FIELD_DICTIONARY_XLSX",
                      config._HOME / "Downloads/Blox_Connect/Dictionary/BloxConnect_Field_Dictionary-2.xlsx")

HAND_DEFS = {
    "days_since_last": "Days between the scoring date and the grid's most recent HEKA snapshot (telemetry silence).",
    "opted_out": "Grid's latest snapshot says data sharing is switched off (grid_opt_in = false).",
    "queries_zero_share": "Share of snapshots in the 2-month window where the daily DNS query total was 0.",
    "reporting_use_ratio": "Daily Reporting & Analytics log volume used / the grid's reporting allowance (data_reporting).",
    "n_models": "Number of distinct appliance models in the grid (data_member_models).",
    "saml_enabled": "SAML single sign-on enabled for admins.",
    "n_grid_wide_licenses": "Number of grid-wide licences (data_grid_wide_license), e.g. Reporting, Security Ecosystem.",
    "pw_min_length": "Minimum admin password length (password policy).",
    "pw_expire": "Admin passwords expire (password policy expire_enable).",
    "upgrade_failed": "Grid upgrade state contains FAILED (data_upgrade_status.grid_state).",
    "tax_score": "v0 Taxonomy composite churn score (0-66).",
    "tax_flag": "v0 Taxonomy flag (score >= 35).",
}
CORE_DEFS = {
    "member_count": "members (appliances) in the grid (data_member_count)",
    "total_queries": "DNS queries in the last 24 h, a daily counter (data_total_queries)",
    "objects": "database objects (db.objects_count)",
    "admin_count": "admin accounts (data_grid_admin_count)",
    "dashboard_count": "dashboards configured (data_grid_dashboard_count)",
}


def load_dictionary():
    d = {}
    try:
        wb = openpyxl.load_workbook(DICT_XLSX, read_only=True, data_only=True)
    except FileNotFoundError:
        return d
    for ws in wb.worksheets:
        for r in ws.iter_rows(values_only=True):
            if r and r[0] and len(r) >= 4 and (r[3] or r[2]):
                d[str(r[0]).strip()] = str(r[3] or r[2]).strip()
    return d


def definition(f, dic):
    if f in HAND_DEFS:
        return HAND_DEFS[f]
    m = re.match(r"(member_count|total_queries|objects|admin_count|dashboard_count)_(log|chg)$", f)
    if m:
        what = CORE_DEFS[m.group(1)]
        return (f"Latest value: {what}." if m.group(2) == "log"
                else f"2-month % change in {what}, first to last reading.")
    m = re.match(r"tax_(\w+)_(amber|red|na)$", f)
    if m:
        return f"v0 Taxonomy '{signals.SIGNALS[m.group(1)][0]}' verdict is {m.group(2)}."
    if f.startswith("gfu."):
        k = f[4:]
        return dic.get(k, dic.get("data_" + k, "grid_feature_usage counter (no dictionary entry)"))
    return ""


def natural(f, x):
    """Undo the model's log1p so medians read in real units."""
    if f.endswith("_log") or f.startswith("gfu."):
        return np.expm1(x)
    return x


def main():
    snaps = pd.read_pickle(config.SNAPSHOTS).sort_values("collected_at")
    ev = cohorts.build(snaps, cohorts.load_churn_list())
    ev = ev[ev.scorable].reset_index(drop=True)
    eA = cohorts.cohort(ev, "A").reset_index(drop=True)
    log("building features ...")
    usage = features.usage_columns(snaps, eA)
    X = features.build_matrix(snaps, eA, usage).reset_index(drop=True)
    y, g = eA["label"].values, groups_of(eA)
    grp = features.feature_groups(X.columns)
    fam = {c: ("profile: adoption & size" if grp[c] in ("usage", "level") else grp[c]) for c in X.columns}

    log("permutation importance (per feature and per family) ...")
    est = models.make_models()["v4_gradient_boosting"][0]
    rng = np.random.RandomState(config.RANDOM_STATE)
    per, fam_drop, base_aucs = [], [], []
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=config.RANDOM_STATE).split(X, y, g):
        m = est.fit(X.iloc[tr], y[tr])
        Xt, yt = X.iloc[te], y[te]
        base = roc_auc_score(yt, m.predict_proba(Xt)[:, 1])
        base_aucs.append(base)

        def drop(cols):
            out = []
            for _ in range(3):
                Xp = Xt.copy()
                idx = rng.permutation(len(Xp))
                Xp[cols] = Xp[cols].values[idx]
                out.append(base - roc_auc_score(yt, m.predict_proba(Xp)[:, 1]))
            return float(np.mean(out))
        per.append({c: drop([c]) for c in X.columns})
        fam_drop.append({f: drop([c for c in X.columns if fam[c] == f]) for f in sorted(set(fam.values()))})
    P = pd.DataFrame(per)
    F = pd.DataFrame(fam_drop)

    dic = load_dictionary()
    rows = []
    for c in X.columns:
        x = X[c]
        ok = x.notna()
        uni = roc_auc_score(y[ok], x[ok]) if ok.sum() > 50 and x[ok].nunique() > 1 else np.nan
        rows.append({
            "parameter": c, "family": fam[c],
            "perm_auc_drop": P[c].mean(), "perm_sd": P[c].std(),
            "univariate_auc": uni,
            "direction": "" if np.isnan(uni) else ("higher -> more churn" if uni > 0.5 else "lower -> more churn"),
            "median_churned": natural(c, x[ok & (y == 1)].median()),
            "median_retained": natural(c, x[ok & (y == 0)].median()),
            "pct_nonzero_churned": float((x[ok & (y == 1)] != 0).mean()),
            "pct_nonzero_retained": float((x[ok & (y == 0)] != 0).mean()),
            "coverage": ok.mean(),
            "definition": definition(c, dic),
        })
    T = pd.DataFrame(rows).sort_values("perm_auc_drop", ascending=False)
    T["rank"] = range(1, len(T) + 1)
    T["share_of_total_importance"] = T["perm_auc_drop"].clip(lower=0) / T["perm_auc_drop"].clip(lower=0).sum()
    T.to_csv(config.REPORTS_DIR / "gb_features.csv", index=False)

    fam_tbl = F.mean().sort_values(ascending=False)
    params = est.get_params()
    write_md(T, fam_tbl, float(np.mean(base_aucs)), params, len(X.columns), len(usage))
    (config.REPORTS_DIR / "gb_family_importance.json").write_text(json.dumps(fam_tbl.round(4).to_dict(), indent=2))
    log("wrote reports/GB_FEATURES.md and reports/gb_features.csv")


def _fmt(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    a = abs(v)
    if a >= 1e6:
        return f"{v / 1e6:.1f}M"
    if a >= 1e4:
        return f"{v / 1e3:.0f}K"
    if a >= 100:
        return f"{v:,.0f}"
    return f"{v:.2g}"


def write_md(T, fam, base_auc, params, n_feat, n_usage):
    keep = ["learning_rate", "max_iter", "max_leaf_nodes", "min_samples_leaf", "l2_regularization", "class_weight"]
    L = ["# Gradient-boosting model (v4): parameters and predictive power", "",
         "Generated by `python -m churn.explain`. Do not edit by hand.", "",
         f"Held-out ROC-AUC across the 5 folds: **{base_auc:.3f}**. {n_feat} input parameters, "
         f"{n_usage} of them grid_feature_usage counters.", "",
         "## Model settings (hyperparameters)", "", "| Setting | Value | Meaning |", "|---|---|---|"]
    meaning = {"learning_rate": "how much each new tree corrects the previous ones",
               "max_iter": "number of trees", "max_leaf_nodes": "max size of each tree",
               "min_samples_leaf": "min grids per leaf (guards against memorising single customers)",
               "l2_regularization": "penalty that shrinks extreme leaf values",
               "class_weight": "churners up-weighted so the 9.5% minority class counts equally"}
    for k in keep:
        L.append(f"| `{k}` | {params[k]} | {meaning[k]} |")
    L += ["", "## How much each family of parameters contributes", "",
          "Drop in held-out ROC-AUC when the whole family is shuffled at once.", "",
          "| Family | AUC drop |", "|---|---:|"]
    for k, v in fam.items():
        L.append(f"| {k} | {v:.3f} |")
    L += ["", "## Every parameter, ranked", "",
          "*AUC drop* is the model's reliance on that parameter. *Alone* is its univariate ROC-AUC: "
          "0.50 means no signal, and the further from 0.50 the stronger, in the direction shown. "
          "Medians are in real units.", "",
          "| # | Parameter | Family | AUC drop | Alone | Direction | Median churned | Median retained | Definition |",
          "|---:|---|---|---:|---:|---|---:|---:|---|"]
    for r in T.itertuples():
        if r.rank > 40 and r.perm_auc_drop < 0.001:
            continue
        L.append(f"| {r.rank} | `{r.parameter}` | {r.family} | {r.perm_auc_drop:.4f} | "
                 f"{'–' if np.isnan(r.univariate_auc) else f'{r.univariate_auc:.2f}'} | {r.direction} | "
                 f"{_fmt(r.median_churned)} | {_fmt(r.median_retained)} | {r.definition[:140]} |")
    L += ["", f"Parameters below rank 40 with an AUC drop under 0.001 are omitted here; all {n_feat} are in "
          "`gb_features.csv`.", ""]
    (config.REPORTS_DIR / "GB_FEATURES.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
