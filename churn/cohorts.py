"""Build the evaluation cohorts: which grid is scored at which date, with what label.

Unit of prediction: a GRID at a scoring date T (the dashboard scores grids, and the
retained stand-in population has no account id). Account-level results are reported
as a secondary view for churners.

Cohort A  ("vs live fleet")
  positives  every churned grid in Anand's pull, scored at T = churn date - 90 days
  negatives  every grid in the weekly Apr-Jul 2026 exports, scored at T = 2026-07-05,
             EXCLUDING any grid_id or license_uid that also appears in Anand's pull
Cohort B  ("within churners")
  positives  as above
  negatives  the same churned grids scored one year earlier (T = churn date - 365 days),
             when they were not about to churn within 3 months
             Both classes come from one source, so source artefacts cannot leak.

Scoring rules, applied identically to every group (they remove artefacts of how each
group was assembled rather than anything about churn):
  * observed for the whole window: first snapshot on or before T - (60 + 7) days
  * still part of the monitored fleet: last snapshot on or before T no more than 365 days old
  * no anchor alignment: churners' T falls on an arbitrary weekday relative to their weekly
    snapshots, so retained grids get T = 5 Jul 2026 + a seeded random 0-6 days. Scoring them
    all exactly on snapshot day would make "days since last snapshot" a label giveaway.
"""
import numpy as np
import openpyxl
import pandas as pd

STALE_DAYS = 365

from . import config


def load_churn_list(path=config.CHURN_LIST_XLSX) -> pd.DataFrame:
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    rows = list(wb.worksheets[0].iter_rows(values_only=True))
    hdr_i = next(i for i, r in enumerate(rows) if r and "18-Digit Account ID" in r)
    df = pd.DataFrame(rows[hdr_i + 1:], columns=rows[hdr_i])
    df = df[df["18-Digit Account ID"].notna()]
    out = pd.DataFrame({
        "account_id": df["18-Digit Account ID"].astype(str).str.strip(),
        "account_name": df["Account Name"],
        "churn_date": pd.to_datetime(df["Customer Until"], errors="coerce", utc=True),
        "churn_fy": df["Until FY"],
        "nios_customer": df["NIOS Customer?"],
    })
    return out.drop_duplicates("account_id")


def build(snaps: pd.DataFrame, churn_list: pd.DataFrame) -> pd.DataFrame:
    """Return one row per (grid, T) to score: grid_id, account_id, T, label, cohort, role."""
    cp = snaps[snaps.source == "churn_pull"]
    wk = snaps[snaps.source == "weekly"]
    times = {k: g["collected_at"].sort_values().values
             for k, g in snaps.groupby(["source", "grid_id"])}
    need = pd.Timedelta(days=config.WINDOW_DAYS + 7)

    def scorable(key, T):
        t = times[key]
        before = t[t <= T.to_datetime64()]
        if len(before) == 0 or pd.Timestamp(t[0], tz="UTC") > T - need:
            return False
        return (T - pd.Timestamp(before[-1], tz="UTC")).days <= STALE_DAYS

    grids = (cp.groupby("grid_id")["account_id"].first().rename("account_id").reset_index()
             .merge(churn_list[["account_id", "churn_date"]], on="account_id", how="left"))
    missing = grids["churn_date"].isna().sum()
    if missing:
        print(f"warning: {missing} churned grids have no churn date; dropped")
    grids = grids.dropna(subset=["churn_date"])

    rows = []
    for _, g in grids.iterrows():
        for lead, label, role in ((config.LEAD_DAYS, 1, "churner"),
                                  (config.HEALTHY_LEAD_DAYS, 0, "churner_1y_earlier")):
            T = g.churn_date - pd.Timedelta(days=lead)
            rows.append(dict(grid_id=g.grid_id, account_id=g.account_id, source="churn_pull", T=T,
                             churn_date=g.churn_date, label=label, role=role,
                             scorable=scorable(("churn_pull", g.grid_id), T)))

    exclude_grids = set(cp.grid_id)
    exclude_lic = set(cp.license_uid.dropna())
    wk_grids = wk.groupby("grid_id")["license_uid"].last()
    T0 = pd.Timestamp(config.CONTROL_ANCHOR, tz="UTC") + pd.Timedelta(hours=23, minutes=59)
    rng = np.random.RandomState(config.RANDOM_STATE)
    n_excl = 0
    for gid, lic in wk_grids.items():
        if gid in exclude_grids or (lic and lic in exclude_lic):
            n_excl += 1
            continue
        T = T0 + pd.Timedelta(days=int(rng.randint(0, 7)))
        rows.append(dict(grid_id=gid, account_id=None, source="weekly", T=T, churn_date=pd.NaT,
                         label=0, role="retained_fleet", scorable=scorable(("weekly", gid), T)))
    ev = pd.DataFrame(rows)
    ev.attrs["excluded_weekly_grids"] = n_excl
    return ev


def cohort(ev: pd.DataFrame, name: str) -> pd.DataFrame:
    s = ev[ev.scorable]
    if name == "A":
        return s[s.role.isin(["churner", "retained_fleet"])].copy()
    if name == "B":
        return s[s.role.isin(["churner", "churner_1y_earlier"])].copy()
    raise ValueError(name)
