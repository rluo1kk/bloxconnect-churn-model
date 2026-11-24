"""The current churn model (v0): BloxConnect Taxonomy Core (Final), Churn tier.

    S(g, T) = sum_i  w_i * 1[ signal_i(g, T) is AMBER or RED ]        i in {traffic, shrink, quiet, dark}
    churn_flag(g, T) = 1[ S(g, T) >= 35 ]

w_i comes from the sub-use-case Importance via IMP_W (HIGH 25, LOW-MEDIUM 8), exactly as the
dashboard derives it (`tierWeights('Churn')`). Each signal is evaluated on the grid's own
snapshots in the 2-month window ending at the scoring date T. A signal that cannot be
evaluated (n/a) contributes 0. n/a is not green, but it is not a trip either.

Thresholds are verbatim from 'Use cases - detailed' (Churn rows) of the Taxonomy workbook.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config

IMP_W = {"HIGH": 25, "MEDIUM-HIGH": 18, "MEDIUM": 12, "LOW-MEDIUM": 8, "LOW": 5}
CHURN_THRESHOLD = 35

# name -> (sub-use-case as named in the Taxonomy, Importance)
SIGNALS = {
    "traffic": ("Traffic falling off", "HIGH"),
    "shrink": ("Shrinking deployment", "HIGH"),
    "quiet": ("Going quiet (stagnant object count)", "LOW-MEDIUM"),
    "dark": ("Gone dark", "LOW-MEDIUM"),
}
WEIGHTS = {k: IMP_W[imp] for k, (_, imp) in SIGNALS.items()}

WINDOW_TOLERANCE_DAYS = 7   # weekly cadence: accept a baseline up to one week before T-60d
MIN_SPAN_DAYS = 28          # a "trend" needs at least ~4 weeks between first and last reading
QUERY_FLOOR = 1000          # Traffic: n/a when peak daily total < 1,000


@dataclass
class Window:
    """The grid's snapshots inside [T - 60d - tolerance, T], plus the latest snapshot <= T."""
    snaps: pd.DataFrame
    last_seen: pd.Timestamp | None
    last_opt_in: object


def window(grid_snaps: pd.DataFrame, T: pd.Timestamp, days=config.WINDOW_DAYS) -> Window:
    upto = grid_snaps[grid_snaps["collected_at"] <= T]
    start = T - pd.Timedelta(days=days + WINDOW_TOLERANCE_DAYS)
    w = upto[upto["collected_at"] >= start]
    last = upto.iloc[-1] if len(upto) else None
    return Window(w, None if last is None else last["collected_at"], None if last is None else last["opt_in"])


def pct_change(series: pd.Series):
    """First -> last relative change over the window, or None if not evaluable."""
    s = series.dropna()
    if len(s) < 2:
        return None
    a, b = float(s.iloc[0]), float(s.iloc[-1])
    if a == 0:
        return None
    return (b - a) / a


def _span_ok(w: Window):
    s = w.snaps["collected_at"]
    return len(s) >= 2 and (s.iloc[-1] - s.iloc[0]).days >= MIN_SPAN_DAYS


def sig_traffic(w: Window):
    """Traffic falling off. Daily query total OR object count, first -> last over 2 months.
    Amber: down 30-49%.  Red: down 50%+, or fallen to 0 from a non-zero baseline.
    n/a: fewer than 2 readings / span < 4 weeks; query part n/a when peak daily total < 1,000."""
    if not _span_ok(w):
        return "na", None
    worst, val = None, None
    q = w.snaps["total_queries"].dropna()
    if len(q) >= 2 and q.max() >= QUERY_FLOOR:
        if q.iloc[0] > 0 and q.iloc[-1] == 0:
            return "red", -1.0
        worst = pct_change(q)
    o = pct_change(w.snaps["objects"])
    for x in (worst, o):
        if x is not None and (val is None or x < val):
            val = x
    if val is None:
        return "na", None
    return ("red" if val <= -0.50 else "amber" if val <= -0.30 else "green"), val


def sig_shrink(w: Window):
    """Shrinking deployment. Member count over 2 months. Amber: down 10-19%. Red: down 20%+."""
    if not _span_ok(w):
        return "na", None
    v = pct_change(w.snaps["member_count"])
    if v is None:
        return "na", None
    return ("red" if v <= -0.20 else "amber" if v <= -0.10 else "green"), v


def sig_quiet(w: Window):
    """Going quiet. Object count moves < 2% (either direction) over 2 months -> red."""
    if not _span_ok(w):
        return "na", None
    v = pct_change(w.snaps["objects"])
    if v is None:
        return "na", None
    return ("red" if abs(v) < 0.02 else "green"), v


def opted_out(v) -> bool:
    """True only for an explicit opt-out. pandas hands back numpy.bool_, so `is False` is wrong."""
    return v is not None and not pd.isna(v) and not bool(v)


def sig_dark(w: Window, T: pd.Timestamp):
    """Gone dark. Days since the last snapshot at T. Amber: 14-21. Red: 21+, or opted out."""
    if w.last_seen is None:
        return "na", None
    days = (T - w.last_seen).days
    if opted_out(w.last_opt_in):
        return "red", days
    return ("red" if days > 21 else "amber" if days >= 14 else "green"), days


def score_grid(grid_snaps: pd.DataFrame, T: pd.Timestamp, fire=("amber", "red")) -> dict:
    """Evaluate the four Churn signals and the composite score for one grid at date T."""
    w = window(grid_snaps, T)
    out = {}
    for name, (verdict, val) in {
        "traffic": sig_traffic(w), "shrink": sig_shrink(w),
        "quiet": sig_quiet(w), "dark": sig_dark(w, T),
    }.items():
        out[f"{name}_verdict"] = verdict
        out[f"{name}_value"] = val
        out[f"{name}_fired"] = int(verdict in fire)
    out["score"] = sum(WEIGHTS[k] * out[f"{k}_fired"] for k in WEIGHTS)
    out["flag"] = int(out["score"] >= CHURN_THRESHOLD)
    out["n_window_snaps"] = len(w.snaps)
    return out


def weights_table() -> pd.DataFrame:
    return pd.DataFrame([
        {"signal": k, "sub_use_case": n, "importance": imp, "weight": IMP_W[imp]}
        for k, (n, imp) in SIGNALS.items()
    ])


if __name__ == "__main__":
    print(weights_table().to_string(index=False))
    print(f"threshold: {CHURN_THRESHOLD}   max score: {sum(WEIGHTS.values())}")
    # Which signal combinations reach the threshold?
    import itertools
    for r in range(1, 5):
        for combo in itertools.combinations(WEIGHTS, r):
            s = sum(WEIGHTS[c] for c in combo)
            print(f"{'+'.join(combo):28s} {s:3d}  {'FLAG' if s >= CHURN_THRESHOLD else ''}")
    _ = np  # keep numpy import for downstream users
