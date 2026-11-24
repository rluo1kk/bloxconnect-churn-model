"""Feature matrix for the statistical models: one row per (grid, T), built only from
snapshots collected on or before T (no look-ahead).

Feature groups
  recency     days since last snapshot, opted out
  trend       for each core metric: 2-month % change (first -> last reading in the window)
  level       latest value of each core metric (log scale) and of grid-level configuration
  usage       latest grid_feature_usage counters (log scale) that are populated in both classes
  taxonomy    the four v0 signal verdicts (so models can reuse what the Taxonomy encodes)

Deliberately excluded
  * anything that depends on HOW MANY snapshots fall in the window (snapshot count,
    volatility, drawdown from peak). The weekly exports were sampled less often than
    Anand's pull (median 7 vs 9 readings per window), so those features identify the
    source file, not churn. First -> last change reads the same at either cadence.
  * NIOS version and schema version. Both track calendar time, and
churners are scored on earlier dates than the retained fleet, so they would leak the
label through "when" instead of "who".
"""
import numpy as np
import pandas as pd

from . import config, signals

CORE = ["member_count", "total_queries", "objects", "admin_count", "dashboard_count"]
STATIC = ["n_models", "saml_enabled", "n_grid_wide_licenses", "pw_min_length", "pw_expire", "upgrade_failed"]


def _change(s: pd.Series):
    s = s.dropna()
    if len(s) < 2 or not s.iloc[0]:
        return np.nan
    return float(np.clip((s.iloc[-1] - s.iloc[0]) / s.iloc[0], -1, 5))


def grid_features(g: pd.DataFrame, T: pd.Timestamp, usage_cols) -> dict:
    upto = g[g["collected_at"] <= T]
    last = upto.iloc[-1]
    w = signals.window(g, T)
    f = {
        "days_since_last": (T - last["collected_at"]).days,
        "opted_out": float(signals.opted_out(last["opt_in"])),
    }
    for m in CORE:
        f[f"{m}_chg"] = _change(w.snaps[m])
        f[f"{m}_log"] = np.log1p(last[m]) if pd.notna(last[m]) else np.nan
    q = w.snaps["total_queries"].dropna()
    f["queries_zero_share"] = (q == 0).mean() if len(q) else np.nan
    rep_cap = last["reporting_capacity_mb"]
    f["reporting_use_ratio"] = (last["reporting_usage_mb"] / rep_cap) if rep_cap else np.nan
    for c in STATIC:
        f[c] = last[c]
    for c in usage_cols:
        v = last[c]
        f[c] = np.log1p(max(v, 0)) if pd.notna(v) else np.nan
    v0 = signals.score_grid(g, T)
    for k in signals.WEIGHTS:
        f[f"tax_{k}_amber"] = float(v0[f"{k}_verdict"] == "amber")
        f[f"tax_{k}_red"] = float(v0[f"{k}_verdict"] == "red")
        f[f"tax_{k}_na"] = float(v0[f"{k}_verdict"] == "na")
    f["tax_score"] = v0["score"]
    f["tax_flag"] = v0["flag"]
    return f


def usage_columns(snaps: pd.DataFrame, ev: pd.DataFrame, min_cov=0.95):
    """grid_feature_usage counters populated for >=95% of scored rows in BOTH classes."""
    cols = [c for c in snaps.columns if c.startswith("gfu.")]
    last = []
    by = {k: g for k, g in snaps.groupby(["source", "grid_id"])}
    for r in ev.itertuples():
        g = by[(r.source, r.grid_id)]
        last.append(g[g["collected_at"] <= r.T].iloc[-1][cols])
    L = pd.DataFrame(last, index=ev.index)
    cov = L.notna().groupby(ev["label"].values).mean()
    keep = [c for c in cols if cov[c].min() >= min_cov and L[c].nunique() > 1]
    return keep


def build_matrix(snaps: pd.DataFrame, ev: pd.DataFrame, usage_cols) -> pd.DataFrame:
    by = {k: g for k, g in snaps.groupby(["source", "grid_id"])}
    rows = [grid_features(by[(r.source, r.grid_id)], r.T, usage_cols) for r in ev.itertuples()]
    return pd.DataFrame(rows, index=ev.index)


def feature_groups(columns):
    grp = {}
    for c in columns:
        if c.startswith("tax_"):
            grp[c] = "taxonomy"
        elif c.startswith("gfu."):
            grp[c] = "usage"
        elif c.endswith("_chg") or c == "queries_zero_share":
            grp[c] = "trend"
        elif c in ("days_since_last", "opted_out"):
            grp[c] = "recency"
        else:
            grp[c] = "level"
    return grp


_ = config
