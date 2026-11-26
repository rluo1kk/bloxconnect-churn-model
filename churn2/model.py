"""The v2 churn model: statistical pipeline, built from the data rather than from the Taxonomy.

    python -m churn2.model            # primary run + 25-split stability + robustness (~20-40 min)
    python -m churn2.model --quick    # primary run only

Every step that looks at labels runs on the DEVELOPMENT set only. The TEST set (20% of
customers, split by account) is touched exactly once, at the end.

  1 split        stratified by label, grouped by account: dev 80% / locked test 20%
  2 screen       per parameter: coverage >= 80% in both classes, not near-constant,
                 Mann-Whitney U with Benjamini-Hochberg FDR q < 0.01, effect |AUC-0.5| >= 0.05,
                 and ERA-STABLE: churners are scored 2023-2026 but retained grids in Jul 2026, so a
                 parameter whose value drifts with the scoring date among churners (Spearman
                 p < 0.01 and |rho| >= 0.15) would separate the groups by WHEN, not WHO -> excluded
  3 de-correlate Spearman |rho| >= 0.7 clusters -> keep the single strongest per cluster
  4 select       forward selection, logistic regression, 5x5 repeated grouped CV on dev;
                 a parameter is added only if the AUC gain is significant (Nadeau-Bengio corrected
                 resampled t-test, one-sided p < 0.01) and >= 0.005
  5 compare      logistic vs random forest vs gradient boosting on the selected set, and
                 gradient boosting on ALL screened parameters (does selection lose anything?)
  6 test         once: AUC with account-clustered bootstrap 95% CI, PR-AUC, confusion matrix at a
                 threshold fixed on dev, Brier score, label-permutation p-value, account level
  7 repeat       steps 1-6 on 25 random splits: test-AUC distribution, selection frequency
"""
import argparse
import json
import sys
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from churn import config
from .dataset import OUT as DATASET, OUT_UNVERIFIED

warnings.filterwarnings("ignore")
R = config.REPORTS_DIR / "v2"
META = ["grid_id", "account_id", "source", "T", "churn_date", "label", "ambiguous_churn", "ref_group", "map_source"]
FDR_Q, MIN_EFFECT, RHO_MAX, P_ENTER, MIN_GAIN = 0.01, 0.05, 0.7, 0.01, 0.005
ERA_P, ERA_RHO = 0.01, 0.15   # a parameter that drifts with calendar date among churners is excluded
K, REPEATS = 5, 5


def log(*a):
    print(*a, file=sys.stderr, flush=True)


# ------------------------------------------------------------------ helpers
def slog(X):
    """Signed log1p: tames heavy-tailed counts, keeps order and sign. Binary columns unchanged."""
    X = np.asarray(X, dtype=float)
    return np.sign(X) * np.log1p(np.abs(X))


def make_lr():
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), FunctionTransformer(slog),
                         StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))


def make_gb(lr=0.05, leaves=15):
    return HistGradientBoostingClassifier(learning_rate=lr, max_iter=300, max_leaf_nodes=leaves,
                                          min_samples_leaf=20, l2_regularization=1.0, random_state=config.RANDOM_STATE)


def make_rf():
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                         RandomForestClassifier(n_estimators=500, min_samples_leaf=5, max_features="sqrt",
                                                n_jobs=1, random_state=config.RANDOM_STATE))


def split(D, seed, test_frac=0.2):
    sgk = StratifiedGroupKFold(n_splits=int(round(1 / test_frac)), shuffle=True, random_state=seed)
    dev, test = next(sgk.split(D, D.label, D.account_id))
    return D.iloc[dev].reset_index(drop=True), D.iloc[test].reset_index(drop=True)


def cv_splits(D, seed, k=K, repeats=REPEATS):
    out = []
    for r in range(repeats):
        for tr, te in StratifiedGroupKFold(k, shuffle=True, random_state=seed * 100 + r).split(D, D.label, D.account_id):
            out.append((tr, te))
    return out


def cv_aucs(est, X, y, splits):
    a = []
    for tr, te in splits:
        m = clone(est).fit(X.iloc[tr], y[tr])
        a.append(roc_auc_score(y[te], m.predict_proba(X.iloc[te])[:, 1]))
    return np.array(a)


def cv_oof(est, X, y, splits_one_repeat):
    p = np.zeros(len(y))
    for tr, te in splits_one_repeat:
        p[te] = clone(est).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
    return p


def corrected_ttest(diff, n_train, n_test):
    """Nadeau & Bengio (2003): repeated-CV folds overlap, so the naive t-test is over-confident."""
    d = np.asarray(diff)
    J = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return (0.0 if d.mean() > 0 else 1.0)
    t = d.mean() / np.sqrt((1 / J + n_test / n_train) * var)
    return float(1 - stats.t.cdf(t, df=J - 1))


# ------------------------------------------------------------------ steps 2-4
def screen(dev, cols):
    y = dev.label.values
    t_days = (dev["T"] - pd.Timestamp("2020-01-01", tz="UTC")).dt.days.values
    rows = []
    for c in cols:
        x = pd.to_numeric(dev[c], errors="coerce")
        cov1, cov0 = x[y == 1].notna().mean(), x[y == 0].notna().mean()
        ok = x.notna()
        if min(cov1, cov0) < 0.8 or x[ok].nunique() < 2:
            continue
        if x[ok].value_counts(normalize=True).iloc[0] > 0.98:
            continue
        a, b = x[ok & (y == 1)], x[ok & (y == 0)]
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        auc = u / (len(a) * len(b))
        # era stability: among churners (scored 2023-2026), does the value drift with the date?
        tc = t_days[ok & (y == 1)]
        rho, p_era = (stats.spearmanr(a, tc) if a.nunique() > 1 else (0.0, 1.0))
        rows.append({"feature": c, "auc": auc, "effect": abs(auc - 0.5), "p": p,
                     "era_rho": float(rho), "era_p": float(p_era),
                     "median_churned": float(a.median()), "median_retained": float(b.median()),
                     "mean_churned": float(a.mean()), "mean_retained": float(b.mean()),
                     "coverage": float(ok.mean())})
    S = pd.DataFrame(rows)
    if S.empty:
        return S
    m = len(S)
    order = np.argsort(S.p.values)
    q = np.empty(m)
    q[order] = np.minimum.accumulate((S.p.values[order] * m / np.arange(1, m + 1))[::-1])[::-1]
    S["q"] = np.minimum(q, 1)
    S["era_drift"] = (S.era_p < ERA_P) & (S.era_rho.abs() >= ERA_RHO)
    S["passed"] = (S.q < FDR_Q) & (S.effect >= MIN_EFFECT) & ~S.era_drift
    return S.sort_values("effect", ascending=False).reset_index(drop=True)


def decorrelate(dev, S):
    keep = S[S.passed].feature.tolist()
    if len(keep) <= 1:
        return keep, {}
    rho = dev[keep].apply(pd.to_numeric, errors="coerce").corr(method="spearman").abs().fillna(0).to_numpy(copy=True)
    np.fill_diagonal(rho, 1)
    Z = linkage(squareform(1 - rho, checks=False), method="complete")
    cl = fcluster(Z, t=1 - RHO_MAX, criterion="distance")
    eff = S.set_index("feature").effect
    reps, members = [], {}
    for c in np.unique(cl):
        group = [f for f, k in zip(keep, cl) if k == c]
        best = max(group, key=lambda f: eff[f])
        reps.append(best)
        members[best] = [g for g in group if g != best]
    return sorted(reps, key=lambda f: -eff[f]), members


def forward_select(dev, candidates, seed, max_feats=20):
    y = dev.label.values
    X = dev[candidates].apply(pd.to_numeric, errors="coerce")
    sp = cv_splits(dev, seed)
    n_te = len(dev) / K
    n_tr = len(dev) - n_te
    chosen, base = [], np.full(len(sp), 0.5)
    path = []
    while len(chosen) < max_feats:
        best = None
        for c in candidates:
            if c in chosen:
                continue
            a = cv_aucs(make_lr(), X[chosen + [c]], y, sp)
            if best is None or a.mean() > best[1].mean():
                best = (c, a)
        if best is None:
            break
        c, a = best
        gain = a.mean() - base.mean()
        p = corrected_ttest(a - base, n_tr, n_te) if chosen else corrected_ttest(a - 0.5, n_tr, n_te)
        path.append({"step": len(chosen) + 1, "feature": c, "cv_auc": float(a.mean()),
                     "cv_auc_sd": float(a.std()), "gain": float(gain), "p": p,
                     "accepted": bool(p < P_ENTER and gain >= MIN_GAIN)})
        if not path[-1]["accepted"]:
            break
        chosen.append(c)
        base = a
    return chosen, pd.DataFrame(path)


# ------------------------------------------------------------------ step 6
def best_threshold(y, p):
    best, thr = -1, 0.5
    for t in np.unique(np.quantile(p, np.linspace(0.05, 0.99, 300))):
        f = p >= t
        tp = (f & (y == 1)).sum()
        if tp == 0:
            continue
        prec, rec = tp / f.sum(), tp / (y == 1).sum()
        f1 = 2 * prec * rec / (prec + rec)
        if f1 > best:
            best, thr = f1, t
    return float(thr)


def confusion(y, flag):
    tp = int(((flag == 1) & (y == 1)).sum()); fp = int(((flag == 1) & (y == 0)).sum())
    fn = int(((flag == 0) & (y == 1)).sum()); tn = int(((flag == 0) & (y == 0)).sum())
    rec = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    prec = tp / (tp + fp) if tp + fp else np.nan
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "accuracy": (tp + tn) / len(y), "recall": rec,
            "specificity": spec, "precision": prec, "balanced_accuracy": (rec + spec) / 2,
            "f1": 2 * prec * rec / (prec + rec) if tp else 0.0}


def ppv_at(prevalence, rec, spec):
    """Precision a flag would have at a real-world churn rate (test precision reflects the sample mix)."""
    return rec * prevalence / (rec * prevalence + (1 - spec) * (1 - prevalence))


def evaluate_test(y, p, groups, thr, n_boot=2000, n_perm=1000, seed=0):
    rng = np.random.RandomState(seed)
    auc = roc_auc_score(y, p)
    ug = np.unique(groups)
    gi = {g: np.where(groups == g)[0] for g in ug}
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([gi[g] for g in rng.choice(ug, len(ug), replace=True)])
        if len(set(y[idx])) == 2:
            boots.append(roc_auc_score(y[idx], p[idx]))
    perm = [roc_auc_score(rng.permutation(y), p) for _ in range(n_perm)]
    c = confusion(y, (p >= thr).astype(int))
    out = {"n": int(len(y)), "churned": int(y.sum()), "roc_auc": float(auc),
           "roc_auc_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
           "pr_auc": float(average_precision_score(y, p)), "brier": float(brier_score_loss(y, p)),
           "perm_p_value": float((np.sum(np.array(perm) >= auc) + 1) / (n_perm + 1)), "threshold": thr, **c}
    for prev in (0.05, 0.10, 0.20):
        out[f"precision_if_churn_rate_{int(prev * 100)}pct"] = float(ppv_at(prev, c["recall"], c["specificity"]))
    # account level: an account is flagged if any of its grids is
    acc = pd.DataFrame({"g": groups, "y": y, "p": p}).groupby("g").agg(y=("y", "max"), p=("p", "max"))
    ca = confusion(acc.y.values, (acc.p.values >= thr).astype(int))
    out["account_level"] = {"n": int(len(acc)), "roc_auc": float(roc_auc_score(acc.y, acc.p)), **ca}
    return out


# ------------------------------------------------------------------ one full run
def run(D, seed, detail=False):
    cols = [c for c in D.columns if c not in META]
    dev, test = split(D, seed)
    S = screen(dev, cols)
    reps, clusters = decorrelate(dev, S)
    chosen, path = forward_select(dev, reps, seed)
    y_dev, y_te = dev.label.values, test.label.values
    if not chosen:   # nothing significant (expected for the shuffled-label control): constant model
        res = {"seed": seed, "n_dev": len(dev), "n_test": len(test), "screened": int(len(S)),
               "passed_screen": int(S.passed.sum()) if len(S) else 0, "after_decorrelation": len(reps),
               "selected": [], "winner": "none (no significant parameter)", "model_comparison": {},
               "test": {"roc_auc": 0.5, "recall": 0.0, "precision": float("nan")}, "test_auc_all_models": {}}
        if detail:
            res.update({"_S": S, "_path": path, "_clusters": clusters, "_dev": dev, "_test": test,
                        "_p_test": np.full(len(test), y_dev.mean()), "_final": None})
        return res
    Xd = dev.apply(lambda s: pd.to_numeric(s, errors="coerce") if s.name not in META else s)
    Xt = test.apply(lambda s: pd.to_numeric(s, errors="coerce") if s.name not in META else s)
    sp = cv_splits(dev, seed + 7)
    sp1 = sp[:K]
    n_te = len(dev) / K
    models = {
        "logistic_selected": (make_lr(), chosen),
        "random_forest_selected": (make_rf(), chosen),
        "gradient_boosting_selected": (make_gb(), chosen),
        "gradient_boosting_all_screened": (make_gb(), S[S.passed].feature.tolist() or chosen),
    }
    cmp, aucs = {}, {}
    for name, (est, fs) in models.items():
        a = cv_aucs(est, Xd[fs], y_dev, sp)
        aucs[name] = a
        cmp[name] = {"n_features": len(fs), "dev_cv_auc": float(a.mean()), "dev_cv_auc_sd": float(a.std())}
    for name in models:
        if name != "logistic_selected":
            d = aucs[name] - aucs["logistic_selected"]
            cmp[name]["p_better_than_logistic"] = corrected_ttest(d, len(dev) - n_te, n_te)
    # parsimony rule: keep logistic unless another model is significantly better
    winner = "logistic_selected"
    for name in ("gradient_boosting_selected", "random_forest_selected", "gradient_boosting_all_screened"):
        if cmp[name]["p_better_than_logistic"] < 0.01 and cmp[name]["dev_cv_auc"] > cmp[winner]["dev_cv_auc"]:
            winner = name
    est, fs = models[winner]
    thr = best_threshold(y_dev, cv_oof(est, Xd[fs], y_dev, sp1))
    final = clone(est).fit(Xd[fs], y_dev)
    p_te = final.predict_proba(Xt[fs])[:, 1]
    res = {"seed": seed, "n_dev": len(dev), "n_test": len(test), "screened": int(len(S)),
           "passed_screen": int(S.passed.sum()) if len(S) else 0, "after_decorrelation": len(reps),
           "selected": chosen, "winner": winner, "model_comparison": cmp,
           "test": evaluate_test(y_te, p_te, test.account_id.values, thr,
                                 n_boot=2000 if detail else 300, n_perm=1000 if detail else 200, seed=seed)}
    # test AUC of every model, for the record (selection already made on dev)
    res["test_auc_all_models"] = {n: float(roc_auc_score(y_te, clone(e).fit(Xd[f], y_dev).predict_proba(Xt[f])[:, 1]))
                                  for n, (e, f) in models.items()}
    if detail:
        res["_S"], res["_path"], res["_clusters"] = S, path, clusters
        res["_dev"], res["_test"], res["_p_test"], res["_final"] = dev, test, p_te, final
    return res


# ------------------------------------------------------------------ reporting
def coef_table(dev, fs):
    """Final logistic regression refit with statsmodels for standard errors and p-values."""
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    X = dev[fs].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median())
    Z = pd.DataFrame(slog(X.values), columns=fs)
    Z = (Z - Z.mean()) / Z.std(ddof=0)
    fit = sm.Logit(dev.label.values, sm.add_constant(Z)).fit(disp=0)
    ci = fit.conf_int()
    vif = [variance_inflation_factor(Z.values, i) for i in range(len(fs))] if len(fs) > 1 else [1.0]
    T = pd.DataFrame({"feature": fs, "coef_per_sd": fit.params[fs].values,
                      "odds_ratio_per_sd": np.exp(fit.params[fs].values),
                      "or_ci_low": np.exp(ci.loc[fs, 0].values), "or_ci_high": np.exp(ci.loc[fs, 1].values),
                      "p_value": fit.pvalues[fs].values, "vif": vif})
    return T, float(fit.params["const"]), float(fit.prsquared)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--splits", type=int, default=25)
    args = ap.parse_args()
    R.mkdir(parents=True, exist_ok=True)
    D_all = pd.read_pickle(DATASET)
    D = D_all[~D_all.ambiguous_churn].reset_index(drop=True)
    log(f"primary dataset: {len(D)} grids, {int(D.label.sum())} churned, {D.account_id.nunique()} accounts")

    log("primary run (seed 42) ...")
    r = run(D, config.RANDOM_STATE, detail=True)
    S, path = r.pop("_S"), r.pop("_path")
    clusters, dev = r.pop("_clusters"), r.pop("_dev")
    test, p_te, final = r.pop("_test"), r.pop("_p_test"), r.pop("_final")
    S.to_csv(R / "screening.csv", index=False)
    path.to_csv(R / "forward_selection.csv", index=False)
    coefs, intercept, pseudo_r2 = coef_table(dev, r["selected"])
    coefs.to_csv(R / "coefficients.csv", index=False)
    out = {"primary": r, "coefficients": coefs.round(4).to_dict("records"), "intercept": intercept,
           "mcfadden_pseudo_r2": pseudo_r2,
           "redundant_alternatives": {k: v for k, v in clusters.items() if k in r["selected"]},
           "dataset": {"grids": len(D), "churned_grids": int(D.label.sum()), "accounts": int(D.account_id.nunique()),
                       "churned_accounts": int(D[D.label == 1].account_id.nunique()),
                       "retained_accounts": int(D[D.label == 0].account_id.nunique()),
                       "candidate_parameters": len([c for c in D.columns if c not in META])}}
    log(f"  selected {r['selected']}")
    log(f"  winner {r['winner']}  test AUC {r['test']['roc_auc']:.3f} CI {r['test']['roc_auc_ci95']}")
    pd.DataFrame({"label": test.label, "p": p_te, "T": test["T"]}).to_pickle(config.DATA_DIR / "v2_test_predictions.pkl")

    if not args.quick:
        log(f"stability: full pipeline on {args.splits} random splits ...")
        runs = Parallel(n_jobs=-1)(delayed(run)(D, s) for s in range(1000, 1000 + args.splits))
        sel = pd.Series([f for x in runs for f in x["selected"]]).value_counts() / len(runs)
        out["stability"] = {
            "n_splits": len(runs),
            "test_auc": [x["test"]["roc_auc"] for x in runs],
            "test_recall": [x["test"]["recall"] for x in runs],
            "test_precision": [x["test"]["precision"] for x in runs],
            "n_selected": [len(x["selected"]) for x in runs],
            "winners": pd.Series([x["winner"] for x in runs]).value_counts().to_dict(),
            "selection_frequency": sel.round(3).to_dict(),
            "test_auc_all_models_mean": pd.DataFrame([x["test_auc_all_models"] for x in runs]).mean().round(4).to_dict(),
            "per_split": [{"seed": x["seed"], "test_auc": x["test"]["roc_auc"], "selected": x["selected"],
                           "winner": x["winner"], "test_auc_all_models": x["test_auc_all_models"]} for x in runs],
        }
        log(f"  test AUC over splits: mean {np.mean(out['stability']['test_auc']):.3f} "
            f"[{np.percentile(out['stability']['test_auc'], 2.5):.3f}, {np.percentile(out['stability']['test_auc'], 97.5):.3f}]")

        log("robustness ...")
        rob = {}
        firmo = [c for c in D.columns if c.startswith("firmo.")]
        variants = {
            "with_ambiguous_churners": D_all.reset_index(drop=True),
            "era_matched_churners_T_from_2025_07": D[(D.label == 0) | (D["T"] >= pd.Timestamp("2025-07-01", tz="UTC"))].reset_index(drop=True),
            "telemetry_only_no_firmographics": D.drop(columns=firmo),
            "controls_from_trusted_map_only": D[(D.label == 1) | (D.map_source == "feature_usage_extract")].reset_index(drop=True),
        }
        if OUT_UNVERIFIED.exists():
            U = pd.read_pickle(OUT_UNVERIFIED)
            # the unmapped fleet has no account -> no firmographics; compare on telemetry only
            variants["reference_unverified_fleet_telemetry_only"] = pd.concat(
                [D[D.label == 1], U], ignore_index=True).drop(columns=firmo, errors="ignore")
        for name, Dv in variants.items():
            x = run(Dv, config.RANDOM_STATE)
            rob[name] = {"n": len(Dv), "churned": int(Dv.label.sum()), "selected": x["selected"],
                         "winner": x["winner"], "test_auc": x["test"]["roc_auc"], "test_recall": x["test"]["recall"],
                         "test_precision": x["test"]["precision"]}
            log(f"  {name}: AUC {x['test']['roc_auc']:.3f}  selected {x['selected']}")
        nulls = []
        for s in range(3):
            Dn = D.copy()
            Dn["label"] = np.random.RandomState(500 + s).permutation(Dn.label.values)
            x = run(Dn, 2000 + s)
            nulls.append({"test_auc": x["test"]["roc_auc"], "selected": x["selected"]})
        rob["negative_control_shuffled_labels"] = nulls
        log(f"  negative control (shuffled labels): {[round(n['test_auc'], 3) for n in nulls]}")
        out["robustness"] = rob

    (R / "results.json").write_text(json.dumps(out, indent=2, default=lambda o: o if not isinstance(o, np.generic) else o.item()))
    log(f"wrote {R}/results.json")


if __name__ == "__main__":
    main()
