"""Consensus model: the parameters selected in >= 50% of the 25 repeated pipeline runs
(DTC appears in two near-identical forms; the member-level one is used).

    python -m churn2.consensus     # -> reports/v2/consensus.json

Because this specification was read off the repeated runs, its scores here are a check of
stability, not a fresh hold-out estimate. The primary pipeline result stays the headline.
"""
import json

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from churn import config
from .dataset import OUT as DATASET
from .model import R, coef_table, make_lr, split

FEATURES = ["rec.days_silent_beyond_week", "mem.share_lic_dtc", "firmo.log_employees"]


def main():
    D = pd.read_pickle(DATASET)
    D = D[~D.ambiguous_churn].reset_index(drop=True)
    aucs = []
    for seed in range(1000, 1025):
        dev, test = split(D, seed)
        m = clone(make_lr()).fit(dev[FEATURES].astype(float), dev.label)
        aucs.append(roc_auc_score(test.label, m.predict_proba(test[FEATURES].astype(float))[:, 1]))
    coefs, intercept, r2 = coef_table(D, FEATURES)
    out = {"features": FEATURES, "test_auc_25_splits": aucs, "mean": float(np.mean(aucs)),
           "p2_5": float(np.percentile(aucs, 2.5)), "p97_5": float(np.percentile(aucs, 97.5)),
           "coefficients_full_data": coefs.round(4).to_dict("records"), "intercept": intercept,
           "mcfadden_pseudo_r2": r2}
    (R / "consensus.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "test_auc_25_splits"}, indent=1))


if __name__ == "__main__":
    main()
