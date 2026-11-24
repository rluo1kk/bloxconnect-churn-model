"""Model zoo and the cross-validation harness.

Every model is evaluated identically:
  * 5-fold StratifiedGroupKFold, grouped by ACCOUNT (a churned account's grids never sit
    in both train and test; retained grids are their own group)
  * the decision threshold is chosen INSIDE each training fold (inner 3-fold CV, maximise F1)
    and then applied unchanged to the held-out fold, so reported precision/recall are honest
  * metrics are computed on the pooled out-of-fold predictions
"""
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config

RS = config.RANDOM_STATE


def make_models():
    """name -> (estimator, feature subset key). 'signals' = the four Taxonomy verdicts only."""
    lr = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                       LogisticRegression(C=0.5, class_weight="balanced", max_iter=5000))
    return {
        "v1_reweighted_taxonomy": (
            make_pipeline(LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)), "signals"),
        "v2_logistic_regression": (lr, "all"),
        "v3_random_forest": (
            make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                          RandomForestClassifier(n_estimators=500, min_samples_leaf=5, max_features="sqrt",
                                                 class_weight="balanced_subsample", n_jobs=-1, random_state=RS)),
            "all"),
        "v4_gradient_boosting": (
            HistGradientBoostingClassifier(learning_rate=0.05, max_iter=400, max_leaf_nodes=15,
                                           min_samples_leaf=20, l2_regularization=1.0,
                                           class_weight="balanced", early_stopping=False, random_state=RS),
            "all"),
    }


def best_f1_threshold(y, p):
    order = np.unique(np.quantile(p, np.linspace(0.5, 0.995, 200)))
    best, thr = -1, 0.5
    for t in order:
        f = p >= t
        tp = (f & (y == 1)).sum()
        if tp == 0:
            continue
        prec, rec = tp / f.sum(), tp / (y == 1).sum()
        f1 = 2 * prec * rec / (prec + rec)
        if f1 > best:
            best, thr = f1, t
    return thr


def cross_validate(est, X, y, groups, n_splits=5):
    """Out-of-fold probabilities and flags, with per-fold thresholds chosen on training data."""
    y = np.asarray(y)
    proba = np.zeros(len(y))
    flag = np.zeros(len(y), dtype=int)
    thresholds = []
    outer = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RS)
    for tr, te in outer.split(X, y, groups):
        m = clone(est)
        inner = cross_val_predict(clone(est), X.iloc[tr], y[tr],
                                  cv=StratifiedKFold(3, shuffle=True, random_state=RS),
                                  method="predict_proba")[:, 1]
        thr = best_f1_threshold(y[tr], inner)
        m.fit(X.iloc[tr], y[tr])
        proba[te] = m.predict_proba(X.iloc[te])[:, 1]
        flag[te] = (proba[te] >= thr).astype(int)
        thresholds.append(float(thr))
    return proba, flag, thresholds


def recall_at_top(y, p, frac=0.10):
    """Share of churners caught if the team reviews only the top `frac` riskiest grids."""
    y = np.asarray(y)
    k = max(1, int(round(frac * len(p))))
    top = np.argsort(-p)[:k]
    return float(y[top].sum() / y.sum())
