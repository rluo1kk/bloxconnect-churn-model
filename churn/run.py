"""End-to-end: load -> cohorts -> v0 Taxonomy backtest -> features -> models -> reports.

    python -m churn.run              # full run (~5-10 min), writes reports/
    python -m churn.run --fast       # skip permutation importance and lead-time curve

Outputs (aggregates only, no customer identifiers):
    reports/results.json         every metric, per model and cohort
    reports/RESULTS.md           the same, as tables
    reports/*.png                ROC/PR curves, improvement bars, lead-time curve, importances
"""
import argparse
import json
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

from . import cohorts, config, features, load, metrics, models, signals  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
R = config.REPORTS_DIR
COLORS = {"v0_taxonomy": "#8A8F8B", "v1_reweighted_taxonomy": "#B5541A", "v2_logistic_regression": "#2F6FB0",
          "v3_random_forest": "#3E7C24", "v4_gradient_boosting": "#8E2C4E"}


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def rnd(d):
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}


def groups_of(ev):
    return np.where(ev["account_id"].notna(), ev["account_id"].astype(str), "grid:" + ev["grid_id"].astype(str))


def evaluate_cohort(name, ev, X, sig_cols, results, curves):
    y = ev["label"].values
    g = groups_of(ev)
    res = {}
    v0 = metrics.summary(y, ev["flag"], ev["score"])
    v0["recall_at_top10pct"] = models.recall_at_top(y, ev["score"].values + 1e-6 * np.random.RandomState(0).rand(len(y)))
    res["v0_taxonomy"] = rnd(v0)
    curves["v0_taxonomy"] = ev["score"].values
    for mname, (est, subset) in models.make_models().items():
        cols = sig_cols if subset == "signals" else list(X.columns)
        p, f, thr = models.cross_validate(est, X[cols], y, g)
        m = metrics.summary(y, f, p)
        m["recall_at_top10pct"] = models.recall_at_top(y, p)
        m["fold_thresholds"] = [round(t, 4) for t in thr]
        res[mname] = rnd(m)
        curves[mname] = p
        log(f"  [{name}] {mname:26s} AUC {m['roc_auc']:.3f}  PR-AUC {m['pr_auc']:.3f}  "
            f"recall {m['recall']:.3f}  precision {m['precision']:.3f}  bal.acc {m['balanced_accuracy']:.3f}")
    results[name] = res


def learned_signal_weights(Xs, y):
    """v1 interpretation: logistic coefficients on the Taxonomy verdicts, rescaled to points."""
    est = models.make_models()["v1_reweighted_taxonomy"][0].fit(Xs, y)
    coef = est[-1].coef_[0]
    out = pd.DataFrame({"feature": Xs.columns, "log_odds": coef})
    pos = out["log_odds"].clip(lower=0)
    out["points_if_rescaled_to_100"] = (100 * pos / pos.sum()).round(1) if pos.sum() else 0
    return out.sort_values("log_odds", ascending=False)


def univariate(X, y, top=25):
    rows = []
    for c in X.columns:
        x = X[c]
        ok = x.notna()
        if ok.sum() < 50 or x[ok].nunique() < 2 or len(set(y[ok])) < 2:
            continue
        auc = roc_auc_score(y[ok], x[ok])
        rows.append({"feature": c, "auc": auc, "strength": abs(auc - 0.5),
                     "direction": "higher = more churn" if auc > 0.5 else "lower = more churn",
                     "coverage": ok.mean(),
                     "median_churn": float(x[ok & (y == 1)].median()),
                     "median_retained": float(x[ok & (y == 0)].median())})
    return pd.DataFrame(rows).sort_values("strength", ascending=False).head(top)


def perm_importance(X, y, g):
    est = models.make_models()["v4_gradient_boosting"][0]
    imps = []
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=config.RANDOM_STATE).split(X, y, g):
        m = est.fit(X.iloc[tr], y[tr])
        r = permutation_importance(m, X.iloc[te], y[te], scoring="roc_auc", n_repeats=3,
                                   random_state=config.RANDOM_STATE, n_jobs=-1)
        imps.append(r.importances_mean)
    s = pd.Series(np.mean(imps, axis=0), index=X.columns).sort_values(ascending=False)
    return s


def lead_time_curve(snaps, ev_churn, usage_cols, leads=range(0, 361, 30)):
    """Taxonomy fire rates and raw trends for churned grids at k days before churn.
    At each k, every churned grid observed for a full window before that date is included,
    so the sample shrinks further back in time (n_grids is reported per point). A fixed
    panel was tried and rejected: grids with 12+ months of history are mostly ones that
    stopped reporting early, which overstates "gone dark"."""
    by = {k: g for k, g in snaps.groupby(["source", "grid_id"])}
    need = pd.Timedelta(days=config.WINDOW_DAYS + 7)
    out = []
    for k in leads:
        T_of = {r.Index: r.churn_date - pd.Timedelta(days=k) for r in ev_churn.itertuples()}
        rows = [r for r in ev_churn.itertuples()
                if (by[(r.source, r.grid_id)]["collected_at"] <= T_of[r.Index] - need).any()]
        F = pd.DataFrame([features.grid_features(by[(r.source, r.grid_id)], T_of[r.Index], usage_cols)
                          for r in rows])
        rec = {"days_before_churn": k, "n_grids": len(F),
               "tax_flag_rate": float(F["tax_flag"].mean()),
               "silent_21d_rate": float((F["days_since_last"] > 21).mean()),
               "member_chg_median": float(F["member_count_chg"].median()),
               "objects_chg_median": float(F["objects_chg"].median()),
               "queries_chg_median": float(F["total_queries_chg"].median())}
        for s in signals.WEIGHTS:
            rec[f"{s}_fire_rate"] = float((F[f"tax_{s}_amber"] + F[f"tax_{s}_red"]).clip(upper=1).mean())
        out.append(rec)
    return pd.DataFrame(out)


def post_churn(snaps, ev_churn):
    """Do churned appliances actually go away? Compare each grid before vs after its churn date."""
    by = {k: g for k, g in snaps.groupby(["source", "grid_id"])}
    rows = []
    for r in ev_churn.itertuples():
        g = by[(r.source, r.grid_id)]
        before = g[(g.collected_at <= r.churn_date - pd.Timedelta(days=60))]
        if before.empty:
            continue
        after = g[g.collected_at >= r.churn_date]
        rows.append({
            "reports_after_churn": len(after) > 0,
            "reports_90d_after": (g.collected_at >= r.churn_date + pd.Timedelta(days=90)).any(),
            "reports_180d_after": (g.collected_at >= r.churn_date + pd.Timedelta(days=180)).any(),
            "member_chg": ((after.member_count.iloc[-1] - before.member_count.iloc[-1]) / before.member_count.iloc[-1])
            if len(after) and before.member_count.iloc[-1] else np.nan,
        })
    d = pd.DataFrame(rows)
    return {"n_grids": len(d), "share_reporting_after_churn": float(d.reports_after_churn.mean()),
            "share_reporting_90d_after": float(d.reports_90d_after.mean()),
            "share_reporting_180d_after": float(d.reports_180d_after.mean()),
            "median_member_change_after_vs_before": float(d.member_chg.median()),
            "share_with_fewer_members_after": float((d.member_chg < 0).mean())}


# ---------------------------------------------------------------- charts
def chart_roc_pr(y, curves, path, title):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    for name, p in curves.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax[0].plot(fpr, tpr, color=COLORS.get(name), lw=2 if name != "v0_taxonomy" else 2.5,
                   ls="--" if name == "v0_taxonomy" else "-", label=f"{name} (AUC {roc_auc_score(y, p):.2f})")
        pr, rc, _ = precision_recall_curve(y, p)
        ax[1].plot(rc, pr, color=COLORS.get(name), lw=2, ls="--" if name == "v0_taxonomy" else "-")
    ax[0].plot([0, 1], [0, 1], color="#ccc", lw=1)
    ax[0].set(xlabel="False-positive rate", ylabel="True-positive rate (recall)", title="ROC")
    ax[1].axhline(np.mean(y), color="#ccc", lw=1)
    ax[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall")
    ax[0].legend(fontsize=8, loc="lower right", frameon=False)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def chart_improvement(res, path):
    names = list(res)
    mets = [("roc_auc", "ROC-AUC"), ("balanced_accuracy", "Balanced accuracy"),
            ("recall", "Recall"), ("recall_at_top10pct", "Recall @ top 10%")]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    w = 0.8 / len(names)
    for i, n in enumerate(names):
        vals = [res[n][m] for m, _ in mets]
        bars = ax.bar(np.arange(len(mets)) + i * w, vals, w, color=COLORS.get(n), label=n)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
    ax.set_xticks(np.arange(len(mets)) + w * (len(names) - 1) / 2, [m for _, m in mets])
    ax.set_ylim(0, 1.05)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper left")
    ax.set_title("Churned vs retained grids: current Taxonomy model vs statistical models (out-of-fold)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def chart_lead(lt, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    x = -lt["days_before_churn"] / 30
    ax[0].plot(x, lt["silent_21d_rate"], color="#2F6FB0", lw=2, label="Silent 21+ days")
    ax[0].plot(x, lt["tax_flag_rate"], color=COLORS["v0_taxonomy"], lw=2, ls="--", label="Taxonomy model flags")
    ax[0].plot(x, lt["member_chg_median"], color="#B5541A", lw=1.5, label="Median 2-month member change")
    ax[0].plot(x, lt["objects_chg_median"], color="#3E7C24", lw=1.5, label="Median 2-month object change")
    ax[0].set(xlabel="Months relative to churn date", ylabel="Share of churned grids / % change",
              title="Churned grids before their churn date")
    for xi, n in zip(x, lt["n_grids"]):
        ax[0].annotate(f"n={n}", (xi, 0.0), textcoords="offset points", xytext=(0, -2), ha="center",
                       fontsize=6, color="#888", va="top")
    ax[0].legend(fontsize=8, frameon=False)
    for s, c in zip(signals.WEIGHTS, ["#8E2C4E", "#B5541A", "#3E7C24", "#2F6FB0"]):
        ax[1].plot(x, lt[f"{s}_fire_rate"], color=c, lw=2, label=signals.SIGNALS[s][0])
    ax[1].set(xlabel="Months relative to churn date", ylabel="Share of churned grids firing",
              title="Taxonomy Churn signals before churn")
    ax[1].legend(fontsize=8, frameon=False)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
        a.axvline(-3, color="#ccc", lw=1, ls=":")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def chart_importance(imp, path, top=20):
    s = imp.head(top)[::-1]
    grp = features.feature_groups(s.index)
    pal = {"trend": "#8E2C4E", "recency": "#2F6FB0", "level": "#3E7C24", "usage": "#B5541A", "taxonomy": "#8A8F8B"}
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(s.index, s.values, color=[pal[grp[i]] for i in s.index])
    ax.set(xlabel="Drop in ROC-AUC when shuffled (held-out folds)", title="What the gradient-boosting model relies on")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=k) for k, c in pal.items()], fontsize=8, frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    R.mkdir(parents=True, exist_ok=True)

    if not config.SNAPSHOTS.exists():
        load.main()
    snaps = pd.read_pickle(config.SNAPSHOTS).sort_values("collected_at")
    churn_list = cohorts.load_churn_list()
    ev = cohorts.build(snaps, churn_list)
    log(f"cohorts: {ev.groupby(['role', 'scorable']).size().to_dict()}")
    n_excluded = ev.attrs.get("excluded_weekly_grids")
    scorable_counts = {f"{k[0]}|scorable={k[1]}": int(v) for k, v in ev.groupby(["role", "scorable"]).size().items()}
    ev = ev[ev.scorable].reset_index(drop=True)

    by = {k: g for k, g in snaps.groupby(["source", "grid_id"])}
    v0 = pd.DataFrame([signals.score_grid(by[(r.source, r.grid_id)], r.T) for r in ev.itertuples()])
    ev = pd.concat([ev, v0], axis=1)

    log("building features ...")
    usage_cols = features.usage_columns(snaps, cohorts.cohort(ev, "A"))
    X_all = features.build_matrix(snaps, ev, usage_cols)
    sig_cols = [c for c in X_all.columns if c.startswith("tax_") and c.endswith(("_amber", "_red", "_na"))]

    results, out = {}, {"design": {}, "cohorts": {}}
    curves = {}
    for name in ("A", "B"):
        idx = cohorts.cohort(ev, name).index
        e, X = ev.loc[idx].reset_index(drop=True), X_all.loc[idx].reset_index(drop=True)
        out["cohorts"][name] = {"n": len(e), "positives": int(e.label.sum()), "negatives": int((1 - e.label).sum()),
                                "churned_accounts": int(e[e.label == 1].account_id.nunique())}
        log(f"cohort {name}: {out['cohorts'][name]}")
        c = {}
        evaluate_cohort(name, e, X, sig_cols, results, c)
        curves[name] = (e["label"].values, c)
        if name == "A":
            eA, XA = e, X

    # era check: cohort A with only churners scored in 2026 (same calendar era as the retained fleet)
    keep = (eA.label == 0) | (eA["T"] >= pd.Timestamp("2026-01-01", tz="UTC"))
    eE, XE = eA[keep].reset_index(drop=True), XA[keep].reset_index(drop=True)
    out["cohorts"]["A_2026_only"] = {"n": len(eE), "positives": int(eE.label.sum())}
    c = {}
    evaluate_cohort("A_2026_only", eE, XE, sig_cols, results, c)

    out["results"] = results
    out["design"] = {"lead_days": config.LEAD_DAYS, "window_days": config.WINDOW_DAYS,
                     "healthy_lead_days": config.HEALTHY_LEAD_DAYS, "control_anchor": config.CONTROL_ANCHOR,
                     "excluded_weekly_grids_shared_with_churn_pull": n_excluded, "scorable_counts": scorable_counts,
                     "n_features": X_all.shape[1], "n_usage_features": len(usage_cols),
                     "taxonomy_weights": signals.WEIGHTS, "taxonomy_threshold": signals.CHURN_THRESHOLD}

    # what the Taxonomy signals look like in each group
    A = eA
    out["signal_rates_A"] = {role: {k: {"fired": float(d[f"{k}_fired"].mean()),
                                        "na": float((d[f"{k}_verdict"] == "na").mean())} for k in signals.WEIGHTS}
                             for role, d in A.groupby("role")}
    out["v1_learned_weights"] = learned_signal_weights(XA[sig_cols], eA.label.values).to_dict("records")
    yA = eA.label.values
    out["univariate_A"] = univariate(XA, yA).round(4).to_dict("records")
    idxB = cohorts.cohort(ev, "B").index
    out["univariate_B"] = univariate(X_all.loc[idxB].reset_index(drop=True), ev.loc[idxB, "label"].values).round(4).to_dict("records")

    chart_roc_pr(*curves["A"], R / "roc_pr_cohortA.png", "Cohort A - churned grids (3 months before churn) vs retained fleet")
    chart_roc_pr(*curves["B"], R / "roc_pr_cohortB.png", "Cohort B - churned grids 3 months vs 12 months before churn")
    chart_improvement(results["A"], R / "improvement_cohortA.png")

    if not args.fast:
        log("permutation importance ...")
        imp = perm_importance(XA, yA, groups_of(eA))
        out["permutation_importance_A"] = imp.head(30).round(4).to_dict()
        chart_importance(imp, R / "importance_gb.png")
        log("lead-time curve ...")
        lt = lead_time_curve(snaps, eA[eA.label == 1], usage_cols)
        out["lead_time"] = lt.round(4).to_dict("records")
        chart_lead(lt, R / "lead_time.png")

    out["post_churn"] = post_churn(snaps, ev[ev.role == "churner"])
    log(f"post-churn: {out['post_churn']}")
    (R / "results.json").write_text(json.dumps(out, indent=2, default=str))
    write_markdown(out)
    log(f"wrote {R}/results.json, RESULTS.md and charts")


def write_markdown(out):
    L = ["# Churn model results", "", "Generated by `python -m churn.run`. Do not edit by hand.", ""]
    for name, desc in (("A", "churned grids 3 months before churn vs the retained live fleet (5 Jul 2026)"),
                       ("B", "churned grids 3 months before churn vs the same grids 12 months before churn"),
                       ("A_2026_only", "cohort A restricted to churners scored in 2026 (era-leakage check)")):
        if name not in out["results"]:
            continue
        c = out["cohorts"].get(name, {})
        L += [f"## Cohort {name}: {desc}", "", f"n = {c.get('n')}, churned = {c.get('positives')}", "",
              "| Model | ROC-AUC | PR-AUC | Accuracy | Balanced acc. | Precision | Recall | Specificity | F1 | Recall @ top 10% |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for m, r in out["results"][name].items():
            L.append(f"| {m} | {r.get('roc_auc', float('nan')):.3f} | {r.get('pr_auc', float('nan')):.3f} | "
                     f"{r['accuracy']:.3f} | {r['balanced_accuracy']:.3f} | {r['precision']:.3f} | {r['recall']:.3f} | "
                     f"{r['specificity']:.3f} | {r['f1']:.3f} | {r['recall_at_top10pct']:.3f} |")
        L.append("")
    L += ["## Taxonomy signal firing rates (cohort A)", "", "| Group | " + " | ".join(signals.WEIGHTS) + " |",
          "|---|" + "---:|" * len(signals.WEIGHTS)]
    for role, d in out["signal_rates_A"].items():
        L.append(f"| {role} | " + " | ".join(f"{d[k]['fired']:.1%} fired, {d[k]['na']:.0%} n/a" for k in signals.WEIGHTS) + " |")
    L += ["", "## Strongest single features (cohort A, univariate ROC-AUC)", "",
          "| Feature | AUC | Direction | Median churned | Median retained | Coverage |", "|---|---:|---|---:|---:|---:|"]
    for r in out["univariate_A"][:15]:
        L.append(f"| `{r['feature']}` | {r['auc']:.3f} | {r['direction']} | {r['median_churn']:.3g} | "
                 f"{r['median_retained']:.3g} | {r['coverage']:.0%} |")
    if "permutation_importance_A" in out:
        L += ["", "## Gradient-boosting permutation importance (top 15)", "", "| Feature | AUC drop |", "|---|---:|"]
        for k, v in list(out["permutation_importance_A"].items())[:15]:
            L.append(f"| `{k}` | {v:.4f} |")
    L += ["", "Charts: `roc_pr_cohortA.png`, `roc_pr_cohortB.png`, `improvement_cohortA.png`, "
          "`importance_gb.png`, `lead_time.png`.", ""]
    (R / "RESULTS.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
