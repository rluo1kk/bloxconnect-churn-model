"""Labels, scoring dates and the feature matrix (one row per grid).

    python -m churn2.dataset      # -> data/v2_dataset.pkl

LABELS (decided by commercial status, never by telemetry, so no feature can define its own label)
  churned  (1)  grid in Anand's serial-matched pull; its account is on the FY25-FY26 churn list.
                 Scored at T = churn date - 90 days.
                 `ambiguous_churn` marks the 63 accounts that Customer Lifecycle still calls Active
                 (probably moved to other Infoblox products); excluded from the primary model.
  retained (0)  fleet grid whose Salesforce account is Active in Customer Lifecycle, is NOT on the
                 churn list and is NOT one of Anand's accounts. Scored at T = 5 Jul 2026 + 0-6 days
                 (Lifecycle is dated after T + 90 days, so "still active 90 days later" holds).

WHAT IS MEASURED AT T (nothing after T is ever read)
  level    every grid parameter at the latest snapshot on or before T
  trend    % change vs the value ~8 weeks earlier (as-of T-56d), for every multi-valued parameter
  member   per-appliance parameters summarised per grid, at the latest member snapshot on or before T
  recency  days silent beyond a normal week (weekday-neutral); opted out
  firmo    customer tenure at T, employee count, geo (candidates; kept only if significant)

SCORABLE: first snapshot on or before T - 63 days (a full trend window) and last snapshot on or
before T no more than 365 days old. Applied identically to both classes.
"""
import sys

import numpy as np
import pandas as pd

from churn import config
from .extract import OUT_GRID, OUT_MEM

OUT = config.DATA_DIR / "v2_dataset.pkl"
OUT_UNVERIFIED = config.DATA_DIR / "v2_unverified_fleet.pkl"
TREND_DAYS, TREND_TOL = 56, 21
F = config.CHURN_PULL_DIR


def labels():
    ch = pd.read_excel(config.CHURN_LIST_XLSX, header=1)
    ch = ch[ch["18-Digit Account ID"].notna()]
    churn_date = dict(zip(ch["18-Digit Account ID"].str.strip(),
                          pd.to_datetime(ch["Customer Until"], errors="coerce", utc=True)))
    lc = pd.read_excel(F / "Copy of Customer Lifecycle _Data_Current.xlsx", sheet_name="Complete Dataset")
    lc = lc.drop_duplicates("account_id").set_index("account_id")
    sc = pd.read_excel(F / "Copy of Customer Lifecycle _Data_Current.xlsx", sheet_name="Scores",
                       usecols=["account_id", "account_geo"]).drop_duplicates("account_id").set_index("account_id")
    return churn_date, lc, sc


def asof(df, t, tol_days=None):
    x = df[df.collected_at <= t]
    if x.empty:
        return None
    r = x.iloc[-1]
    if tol_days is not None and (t - r.collected_at).days > tol_days:
        return None
    return r


def main():
    G = pd.read_pickle(OUT_GRID)
    M = pd.read_pickle(OUT_MEM)
    churn_date, lc, sc = labels()
    rng = np.random.RandomState(config.RANDOM_STATE)


    grid_cols = [c for c in G.columns if c not in
                 ("source", "account_id", "grid_id", "collected_at", "license_uid", "opt_in", "nios_version")]
    trend_cols = [c for c in grid_cols if G[c].nunique() > 2]
    mem_cols = [c for c in M.columns if c not in ("source", "grid_id", "collected_at", "_versions")]

    Gby = {k: g for k, g in G.groupby(["source", "grid_id"])}
    Mby = {k: g for k, g in M.groupby(["source", "grid_id"])}
    pull_accounts = set(G.loc[G.source == "churn_pull", "account_id"])
    T0 = pd.Timestamp(config.CONTROL_ANCHOR, tz="UTC") + pd.Timedelta(hours=23, minutes=59)

    rows, skipped = [], {"no_label": 0, "not_scorable": 0}
    for (src, gid), g in Gby.items():
        acct = g["account_id"].dropna().iloc[-1] if g["account_id"].notna().any() else None
        if src == "churn_pull":
            cd = churn_date.get(acct)
            if cd is None or pd.isna(cd):
                skipped["no_label"] += 1
                continue
            y, T = 1, cd - pd.Timedelta(days=config.LEAD_DAYS)
            ambiguous = lc["active_customer"].get(acct) == "Active"
        else:
            if acct is None:
                ref = "unverified_fleet"      # no account mapping: status unknown, mostly retained
            elif (acct in churn_date or acct in pull_accounts
                    or lc["active_customer"].get(acct) != "Active"):
                skipped["no_label"] += 1
                continue
            else:
                ref = "verified_active"
            y, T = 0, T0 + pd.Timedelta(days=int(rng.randint(0, 7)))
            ambiguous = False
        cd = churn_date.get(acct) if y else pd.NaT
        if g.collected_at.iloc[0] > T - pd.Timedelta(days=TREND_DAYS + 7):
            skipped["not_scorable"] += 1
            continue
        last = asof(g, T)
        if last is None or (T - last.collected_at).days > 365:
            skipped["not_scorable"] += 1
            continue
        past = asof(g, T - pd.Timedelta(days=TREND_DAYS), tol_days=TREND_TOL)
        f = {"grid_id": gid, "account_id": acct if acct is not None else "grid:" + gid, "source": src, "T": T,
             "churn_date": cd, "label": y, "ambiguous_churn": bool(ambiguous),
             "ref_group": "churned" if y else ref}
        # silence BEYOND a normal weekly cycle. Raw "days since last snapshot" (0-6 for any weekly
        # reporter) encodes the weekday the scoring date happens to fall on, and churn dates
        # cluster on month starts, so the raw value leaks the label (AUC 0.53 within the week).
        f["rec.days_silent_beyond_week"] = float(max(0, (T - last.collected_at).days - 6))
        f["rec.opted_out"] = float(last.opt_in is not None and not pd.isna(last.opt_in) and not bool(last.opt_in))
        for c in grid_cols:
            f[c] = last[c]
        for c in trend_cols:
            a, b = (past[c] if past is not None else np.nan), last[c]
            f["chg." + c] = (np.clip((b - a) / abs(a), -1, 5) if pd.notna(a) and pd.notna(b) and a != 0
                             else (0.0 if a == 0 and b == 0 else np.nan))
        mg = Mby.get((src, gid))
        mrow = asof(mg, T) if mg is not None else None
        if mrow is not None:
            for c in mem_cols:
                f[c] = mrow[c]
            if pd.notna(mrow.get("mem.days_since_last_upgrade", np.nan)):
                f["mem.days_since_last_upgrade"] = mrow["mem.days_since_last_upgrade"] + (T - mrow.collected_at).days
        # NOTE: release-age features ("how old is the running NIOS release", "how many newer
        # releases existed at T") were tried and REMOVED. A release's date is only knowable from
        # when it first appears in OUR telemetry, which is sparse in 2023-25 (churners only) and
        # dense in 2026 (whole fleet). The features measured our data coverage, not the customer,
        # and a boosted model scored test AUC 0.95 on them. Upgrade activity from each member's
        # own version history (mem.upgrade_events_365d, mem.days_since_last_upgrade) is kept.
        # firmographic candidates
        start = pd.to_datetime(lc["first_order_start_dt"].get(acct), errors="coerce")
        f["firmo.tenure_years_at_T"] = (float((T.tz_localize(None) - start).days / 365.25)
                                        if pd.notna(start) else np.nan)
        emp = pd.to_numeric(lc["account_employee_count"].get(acct), errors="coerce")
        f["firmo.log_employees"] = float(np.log1p(emp)) if pd.notna(emp) else np.nan
        geo = sc["account_geo"].get(acct)
        for gname in ("AMS", "EMEA", "APJ"):
            f[f"firmo.geo_{gname}"] = float(geo == gname) if isinstance(geo, str) else np.nan
        rows.append(f)

    D = pd.DataFrame(rows)
    # primary file keeps the verified reference group; the unverified fleet is saved separately
    D[D.ref_group == "unverified_fleet"].reset_index(drop=True).to_pickle(OUT_UNVERIFIED)
    D = D[D.ref_group != "unverified_fleet"].reset_index(drop=True)
    D.to_pickle(OUT)
    print(f"skipped: {skipped}", file=sys.stderr)
    print(D.groupby(["label", "ambiguous_churn"]).agg(grids=("grid_id", "size"), accounts=("account_id", "nunique")))
    print(f"columns: {D.shape[1]}")


if __name__ == "__main__":
    main()
