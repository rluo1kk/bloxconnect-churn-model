"""Load both raw sources into one tidy snapshot table: one row per grid per collection.

    python -m churn.load        # writes data/snapshots.pkl

Sources
  churn_pull   Anand's HEKA unload for churned accounts (JSON lines, nested `data`)
  weekly       Weekly HEKA grid exports, Apr-Jul 2026 (xlsx, `data_*` columns holding JSON strings)

The two arrive in different shapes. Everything downstream reads only the columns
produced here, so both sources are scored with identical logic.

HEKA rules applied (see docs/DATA_RULES.md):
  * one row per (grid, collection timestamp): duplicate runs inside a file are dropped,
    keeping the latest `collected_at` per grid per calendar day
  * never dedupe by ophid
  * `db.objects_count` comes from grid_feature_usage (flat dotted keys)
"""
import json
import re
import sys
from pathlib import Path

import openpyxl
import pandas as pd

from . import config

# grid_feature_usage keys kept as ML features (numeric or boolean, present in both sources)
_GFU_SKIP = re.compile(r"(version|_ts$|timestamp|update_style)", re.I)


def _num(v):
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _maybe_json(v):
    if isinstance(v, str) and v[:1] in "[{":
        try:
            return json.loads(v)
        except ValueError:
            return None
    return v


def _row(source, account_id, grid_id, license_uid, collected_at, d, origin_version, schema_version):
    """Common row from one grid snapshot. `d` is the HEKA `data` object (already parsed)."""
    gfu = d.get("grid_feature_usage") or {}
    if not isinstance(gfu, dict):
        gfu = {}
    rep = d.get("reporting") or {}
    pw = d.get("password_policy") or {}
    up = d.get("upgrade_status") or {}
    models = d.get("member_models") or []
    row = {
        "source": source,
        "account_id": account_id,
        "grid_id": (grid_id or "").replace("-", "").lower() or None,
        "license_uid": (license_uid or None),
        "collected_at": pd.to_datetime(collected_at, utc=True, errors="coerce"),
        "member_count": _num(d.get("member_count")),
        "total_queries": _num(d.get("total_queries")),
        "objects": _num(gfu.get("db.objects_count")),
        "admin_count": _num(d.get("grid_admin_count")),
        "dashboard_count": _num(d.get("grid_dashboard_count")),
        "opt_in": d.get("grid_opt_in"),
        "nios_version": origin_version,
        "schema_version": schema_version,
        "n_models": float(len(set(models))) if isinstance(models, list) else None,
        "saml_enabled": _num(d.get("saml_enabled")),
        "n_grid_wide_licenses": float(len(d.get("grid_wide_license") or [])),
        "reporting_usage_mb": _num(rep.get("total_daily_usage_mb")) if isinstance(rep, dict) else None,
        "reporting_capacity_mb": _num(rep.get("reporting_license_capacity_mb")) if isinstance(rep, dict) else None,
        "pw_min_length": _num(pw.get("password_min_length")) if isinstance(pw, dict) else None,
        "pw_expire": _num(pw.get("expire_enable")) if isinstance(pw, dict) else None,
        "upgrade_failed": (float("FAIL" in str(up.get("grid_state", "")).upper())
                           if isinstance(up, dict) and up.get("grid_state") else None),
    }
    for k, v in gfu.items():
        if _GFU_SKIP.search(k):
            continue
        x = _num(v)
        if x is not None:
            row["gfu." + k] = x
    return row


def load_churn_pull(paths):
    rows = []
    for p in paths:
        with config.open_text(p) as f:
            for line in f:
                r = json.loads(line)
                d = r.get("data") or {}
                rows.append(_row("churn_pull", r.get("account_id"), r.get("grid_id") or d.get("grid_uuid"),
                                 d.get("license_uid"), r.get("collected_at"), d,
                                 (r.get("origin") or {}).get("version"), (r.get("schema") or {}).get("version")))
        print(f"  {Path(p).name}: {len(rows):,} rows so far", file=sys.stderr)
    return rows


_XLSX_COLS = ["data_grid_uuid", "data_license_uid", "meta_collected_at", "origin_version", "schema_version"]


def load_weekly(paths):
    rows = []
    for p in paths:
        wb = openpyxl.load_workbook(p, read_only=True)
        ws = wb.worksheets[0]
        it = ws.iter_rows(values_only=True)
        header = list(next(it))
        idx = {h: i for i, h in enumerate(header) if h}
        n0 = len(rows)
        for vals in it:
            def g(c):
                i = idx.get(c)
                return vals[i] if i is not None and i < len(vals) else None
            if not g("data_grid_uuid") or not g("meta_collected_at"):
                continue
            d = {}
            for c in idx:
                if c.startswith("data_"):
                    d[c[5:]] = _maybe_json(g(c))
            rows.append(_row("weekly", None, g("data_grid_uuid"), g("data_license_uid"), g("meta_collected_at"),
                             d, g("origin_version"), g("schema_version")))
        wb.close()
        print(f"  {Path(p).name}: {len(rows) - n0:,} rows", file=sys.stderr)
    return rows


def dedupe(df):
    """Keep one snapshot per grid per calendar day: the latest collection that day."""
    df = df.dropna(subset=["grid_id", "collected_at"]).sort_values("collected_at")
    df["day"] = df["collected_at"].dt.floor("D")
    return df.drop_duplicates(["source", "grid_id", "day"], keep="last").drop(columns="day")


def main():
    churn_paths = sorted(config.CHURN_PULL_DIR.glob(config.CHURN_PULL_GLOB))
    weekly_paths = sorted(p for p in config.WEEKLY_GRID_DIR.glob(config.WEEKLY_GRID_GLOB)
                          if not p.name.startswith("~$"))
    if not churn_paths or not weekly_paths:
        raise SystemExit(f"missing inputs: {len(churn_paths)} churn files, {len(weekly_paths)} weekly files")
    print(f"churn pull: {len(churn_paths)} files; weekly: {len(weekly_paths)} files", file=sys.stderr)
    df = pd.DataFrame(load_churn_pull(churn_paths) + load_weekly(weekly_paths))
    raw_n = len(df)
    df = dedupe(df)
    # feature columns must exist in BOTH sources, else they just encode the source
    gfu = [c for c in df.columns if c.startswith("gfu.")]
    cov = df.groupby("source")[gfu].apply(lambda x: x.notna().mean())
    keep = [c for c in gfu if cov[c].min() >= 0.8]
    df = df.drop(columns=[c for c in gfu if c not in keep])
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_pickle(config.SNAPSHOTS)
    print(f"{raw_n:,} raw rows -> {len(df):,} after dedupe; {len(keep)} of {len(gfu)} feature-usage "
          f"keys kept (>=80% coverage in both sources) -> {config.SNAPSHOTS}", file=sys.stderr)
    print(df.groupby("source").agg(rows=("grid_id", "size"), grids=("grid_id", "nunique"),
                                   first=("collected_at", "min"), last=("collected_at", "max")))


if __name__ == "__main__":
    main()
