"""Classification metrics used everywhere, so every model is reported identically."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def summary(y, flag, score=None) -> dict:
    y = np.asarray(y).astype(int)
    f = np.asarray(flag).astype(int)
    tp = int(((f == 1) & (y == 1)).sum())
    fp = int(((f == 1) & (y == 0)).sum())
    fn = int(((f == 0) & (y == 1)).sum())
    tn = int(((f == 0) & (y == 0)).sum())
    n = tp + fp + fn + tn

    def div(a, b):
        return a / b if b else float("nan")
    recall, spec = div(tp, tp + fn), div(tn, tn + fp)
    precision = div(tp, tp + fp)
    out = {
        "n": n, "positives": tp + fn, "base_rate": div(tp + fn, n),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": div(tp + tn, n),
        "balanced_accuracy": (recall + spec) / 2,
        "precision": precision, "recall": recall, "specificity": spec,
        "f1": div(2 * precision * recall, precision + recall) if tp else 0.0,
        "flag_rate": div(tp + fp, n),
    }
    if score is not None and len(set(y)) == 2:
        s = np.asarray(score, dtype=float)
        out["roc_auc"] = float(roc_auc_score(y, s))
        out["pr_auc"] = float(average_precision_score(y, s))
    return out
