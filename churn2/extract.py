"""Turn raw HEKA records from BOTH sources into one wide table of parameters.

    python -m churn2.extract      # -> data/v2_grid_snapshots.pkl, data/v2_member_aggs.pkl (~3 min)

One function, `flatten_grid`, turns any HEKA grid `data` object into {dotted.path: number}.
One function, `member_aggregates`, turns any list of member records into per-grid numbers.
Both sources pass through the same two functions, so a parameter means the same thing
whichever file it came from.

  churn_pull  Anand's JSON lines (grid record with nested member_details)
  weekly      weekly grid xlsx (data_* columns holding JSON strings) + weekly member xlsx
              Only grids mapped to a Salesforce account are kept; labels are decided later.
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

from churn import config

OUT_GRID = config.DATA_DIR / "v2_grid_snapshots.pkl"
OUT_MEM = config.DATA_DIR / "v2_member_aggs.pkl"
WEEKLY_MEMBER_FILES = ["Weekly_JULY_5_HEKA_CLEANED_MEMBER_DATA.xlsx", "Weekly_JUN_28_HEKA_MEMBER_DATA.xlsx",
                       "Weekly_JUN_14_HEKA_CLEANED_MEMBER_DATA.xlsx", "Weekly_JUN_8_HEKA_MEMBER_DATA.xlsx",
                       "Weekly_JUN_1_HEKA_MEMBER_DATA.xlsx"]

# identifiers, free text and bookkeeping: never parameters
SKIP_TOP = {"member_details", "grid_name", "grid_uuid", "license_uid", "grid_opt_in_ts", "heka_action_ts",
            "member_models"}
SKIP_PATH = re.compile(r"(version|_ts$|timestamp|_time$|schedule|_date|uuid|_id$|name$|update_style|"
                       r"last_rule|serial|hostname|reporting_configuration)", re.I)


def _parse(v):
    if isinstance(v, str) and v[:1] in "[{":
        try:
            return json.loads(v)
        except ValueError:
            return None
    return v


def flatten_grid(d: dict) -> dict:
    """Every numeric / boolean leaf of a HEKA grid object, as {path: float}. Lists -> length."""
    out = {}

    def walk(v, path):
        v = _parse(v)
        if isinstance(v, bool):
            out[path] = float(v)
        elif isinstance(v, (int, float)):
            out[path] = float(v)
        elif isinstance(v, dict):
            for k, x in v.items():
                p = f"{path}.{k}" if path else k
                if not SKIP_PATH.search(k):
                    walk(x, p)
        elif isinstance(v, list):
            out[path + ".n"] = float(len(v))
    for k, v in d.items():
        if k in SKIP_TOP or SKIP_PATH.search(k):
            continue
        walk(v, k)
    # a few categorical grid fields worth keeping as flags
    up = _parse(d.get("upgrade_status")) or {}
    if isinstance(up, dict):
        gs = str(up.get("grid_state", "")).upper()
        out["upgrade_status.state_failed"] = float("FAIL" in gs)
        out["upgrade_status.test_failed"] = float("FAIL" in str(up.get("upgrade_test_status", "")).upper())
    models = _parse(d.get("member_models")) or []
    if isinstance(models, list):
        out["member_models.distinct"] = float(len(set(models)))
    lic = _parse(d.get("grid_wide_license")) or []
    if isinstance(lic, list):
        for x in lic:
            t = x.get("license_type") if isinstance(x, dict) else x
            if t:
                out[f"grid_wide_license.has_{t}"] = 1.0
    return out


_VER = re.compile(r"^(\d+)\.(\d+)\.(\d+)")
_OLD_GEN = re.compile(r"IB-V?\d+?(15|25)$")   # x15 / x25 = older appliance generation


def vnorm(v):
    m = _VER.match(str(v or ""))
    return f"{m.group(1)}.{m.group(2)}.{m.group(3)}" if m else None


def member_aggregates(members: list, T_iso: str | None = None) -> dict:
    """Per-grid numbers from member records. Dedupe by hostname (never by ophid)."""
    seen = {}
    for m in members or []:
        m = {k: _parse(v) for k, v in m.items()}
        seen[m.get("member_hostname") or id(m)] = m
    ms = list(seen.values())
    n = len(ms)
    if n == 0:
        return {}
    o = {"mem.n": float(n)}

    def share(pred):
        return float(sum(1 for m in ms if pred(m)) / n)
    o["mem.share_running"] = share(lambda m: m.get("member_operational_state") == "Running")
    o["mem.share_offline"] = share(lambda m: m.get("member_operational_state") == "Offline")
    o["mem.share_invalid"] = share(lambda m: m.get("member_operational_state") == "Invalid")
    o["mem.share_ha"] = share(lambda m: m.get("member_ha_status") in ("Active", "Passive"))
    o["mem.has_gmc"] = float(any(m.get("member_role") == "GMC" for m in ms))
    o["mem.share_standalone"] = share(lambda m: m.get("member_role") == "Standalone")
    o["mem.share_virtual"] = share(lambda m: m.get("member_type") == "VNIOS")
    o["mem.share_cloud"] = share(lambda m: m.get("member_platform") in ("AWS", "AZURE", "GCP"))
    phys = [m for m in ms if m.get("member_type") == "INFOBLOX"]
    o["mem.share_old_gen_hw"] = (sum(1 for m in phys if _OLD_GEN.search(str(m.get("member_model") or "")))
                                 / len(phys)) if phys else np.nan
    o["mem.distinct_models"] = float(len({m.get("member_model") for m in ms}))
    vers = [vnorm(m.get("member_version")) for m in ms]
    o["mem.distinct_versions"] = float(len({v for v in vers if v}))
    o["mem.share_pending_version"] = share(lambda m: str(m.get("member_version")) == "Pending")
    for lic in ("rpz", "dtc", "ms_management", "sw_tp", "tp_sub", "discovery", "threat_anl", "cloud",
                "cloud_api", "reporting", "grid", "fireeye", "vnios", "security_ecosystem"):
        o[f"mem.share_lic_{lic}"] = share(lambda m, l=lic: l in (m.get("member_license_types") or []))
    for p in ("dns", "dhcp", "ntp", "tftp", "atp", "ftp", "http"):
        o[f"mem.share_proto_{p}"] = share(lambda m, p=p: p in (m.get("member_enabled_protocols") or []))
    o["mem.n_enabled_features_mean"] = float(np.mean([len(m.get("member_enabled_features") or []) for m in ms]))
    for k in ("member_object_count", "member_disk_size", "recursive_client_limit"):
        vals = [m.get(k) for m in ms if isinstance(m.get(k), (int, float))]
        o[f"mem.{k}_sum" if k == "member_object_count" else f"mem.{k}_mean"] = (
            float(np.sum(vals) if k == "member_object_count" else np.mean(vals)) if vals else np.nan)
    for k in ("member_dns_forwarder", "smartcache_enabled", "public_ip_enabled", "log_lease_events"):
        o[f"mem.share_{k}"] = share(lambda m, k=k: bool(m.get(k)) is True and m.get(k) not in ("False", "false"))
    # member_feature_usage: booleans -> share of members, numbers -> sum and max
    agg = defaultdict(list)
    for m in ms:
        fu = m.get("member_feature_usage") or {}
        if isinstance(fu, dict):
            for k, v in fu.items():
                if SKIP_PATH.search(k):
                    continue
                if isinstance(v, bool):
                    agg[("b", k)].append(float(v))
                elif isinstance(v, (int, float)):
                    agg[("n", k)].append(float(v))
    for (t, k), vals in agg.items():
        if t == "b":
            o[f"mfu.{k}.share"] = float(np.mean(vals))
        else:
            o[f"mfu.{k}.sum"] = float(np.sum(vals))
            o[f"mfu.{k}.max"] = float(np.max(vals))
    # version history -> upgrade activity, relative to T (only events on or before T)
    if T_iso:
        T = pd.Timestamp(T_iso)
        events = []
        for m in ms:
            for e in m.get("member_version_history") or []:
                if isinstance(e, dict) and e.get("timestamp"):
                    t = pd.to_datetime(e["timestamp"], utc=True, errors="coerce")
                    if pd.notna(t) and t <= T:
                        events.append((t, str(e.get("version", ""))))
        o["mem.upgrade_events_365d"] = float(sum(1 for t, _ in events if (T - t).days <= 365) / n)
        o["mem.hotfix_events_365d"] = float(sum(1 for t, v in events if (T - t).days <= 365
                                                and v.lower().startswith("hotfix")) / n)
        o["mem.days_since_last_upgrade"] = float(min((T - t).days for t, _ in events)) if events else np.nan
    o["_versions"] = ";".join(sorted({v for v in vers if v}))
    return o


# ------------------------------------------------------------------ sources
def load_churn_pull():
    grid_rows, mem_rows = [], []
    for p in sorted(config.CHURN_PULL_DIR.glob(config.CHURN_PULL_GLOB)):
        with open(p, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                d = r.get("data") or {}
                gid = (r.get("grid_id") or d.get("grid_uuid") or "").replace("-", "").lower()
                ts = r.get("collected_at")
                base = {"source": "churn_pull", "account_id": r.get("account_id"), "grid_id": gid,
                        "collected_at": pd.to_datetime(ts, utc=True), "license_uid": d.get("license_uid"),
                        "opt_in": d.get("grid_opt_in"), "nios_version": vnorm((r.get("origin") or {}).get("version"))}
                grid_rows.append({**base, **flatten_grid(d)})
                mem_rows.append({"source": "churn_pull", "grid_id": gid, "collected_at": base["collected_at"],
                                 **member_aggregates(d.get("member_details"), ts)})
        print(f"  {p.name}: {len(grid_rows):,}", file=sys.stderr)
    return grid_rows, mem_rows


def _xlsx_rows(path, max_col=None):
    wb = openpyxl.load_workbook(path, read_only=True)
    ws = wb.worksheets[0]
    it = ws.iter_rows(values_only=True, max_col=max_col)
    hdr = list(next(it))
    for vals in it:
        yield {h: v for h, v in zip(hdr, vals) if h}
    wb.close()


def load_weekly(keep_grids: set | None, exclude: set = frozenset()):
    grid_rows = []
    for p in sorted(config.WEEKLY_GRID_DIR.glob(config.WEEKLY_GRID_GLOB)):
        if p.name.startswith("~$"):
            continue
        n0 = len(grid_rows)
        for r in _xlsx_rows(p):
            gid = str(r.get("data_grid_uuid") or "").replace("-", "").lower()
            if (keep_grids is not None and gid not in keep_grids) or gid in exclude or not r.get("meta_collected_at"):
                continue
            d = {k[5:]: v for k, v in r.items() if k.startswith("data_")}
            grid_rows.append({"source": "weekly", "account_id": None, "grid_id": gid,
                              "collected_at": pd.to_datetime(r["meta_collected_at"], utc=True, errors="coerce"),
                              "license_uid": r.get("data_license_uid"), "opt_in": _parse(r.get("data_grid_opt_in")),
                              "nios_version": vnorm(r.get("origin_version")), **flatten_grid(d)})
        print(f"  {p.name}: {len(grid_rows) - n0:,} rows kept", file=sys.stderr)
    mem_rows = []
    for name in WEEKLY_MEMBER_FILES:
        p = config.WEEKLY_GRID_DIR / name
        by = defaultdict(list)
        stamp = {}
        for r in _xlsx_rows(p, max_col=60):
            gid = str(r.get("data_grid_uuid") or "").replace("-", "").lower()
            if (keep_grids is not None and gid not in keep_grids) or gid in exclude:
                continue
            ts = pd.to_datetime(r.get("meta_collected_at"), utc=True, errors="coerce")
            if pd.isna(ts):
                continue
            # duplicate collection runs: keep only the grid's latest run in this file
            if gid in stamp and ts < stamp[gid]:
                continue
            if gid in stamp and ts > stamp[gid]:
                by[gid] = []
            stamp[gid] = ts
            by[gid].append(r)
        for gid, ms in by.items():
            mem_rows.append({"source": "weekly", "grid_id": gid, "collected_at": stamp[gid],
                             **member_aggregates(ms, stamp[gid].isoformat())})
        print(f"  {name}: {len(by):,} grids with member data", file=sys.stderr)
    return grid_rows, mem_rows


def grid_account_map():
    """grid UUID -> Salesforce account id, from the HEKA feature-usage extract (Dec 2025 - Jun 2026)."""
    p = config.CHURN_PULL_DIR / "Copy of HEKA feature usage data and member role 06.05.2026.xlsx"
    gf = pd.read_excel(p, sheet_name="Grid Features", usecols=["account id Grid Member", "Grid UUID"])
    gf.columns = ["account_id", "grid_id"]
    gf["grid_id"] = gf["grid_id"].astype(str).str.replace("-", "").str.lower()
    n_acc = gf.groupby("grid_id")["account_id"].nunique()
    ambiguous = set(n_acc[n_acc > 1].index)            # a grid claimed by 2+ accounts is not trusted
    gf = gf[~gf["grid_id"].isin(ambiguous)].drop_duplicates("grid_id")
    return gf.set_index("grid_id")["account_id"], len(ambiguous)


def main():
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    g2a, n_amb = grid_account_map()
    print(f"grid->account map: {len(g2a):,} grids ({n_amb} ambiguous grids dropped)", file=sys.stderr)
    cg, cm = load_churn_pull()
    churn_grids = {r["grid_id"] for r in cg}
    # every fleet grid except Anand's churned ones; mapped grids get a verified label later,
    # unmapped grids form the secondary (unverified) reference group
    wg, wm = load_weekly(None, exclude=churn_grids)
    G = pd.DataFrame(cg + wg)
    G.loc[G.source == "weekly", "account_id"] = G.loc[G.source == "weekly", "grid_id"].map(g2a)
    G = (G.dropna(subset=["collected_at"]).sort_values("collected_at")
         .assign(day=lambda x: x.collected_at.dt.floor("D"))
         .drop_duplicates(["source", "grid_id", "day"], keep="last").drop(columns="day"))
    M = pd.DataFrame(cm + wm).dropna(subset=["collected_at"]).sort_values("collected_at")
    M = M.assign(day=M.collected_at.dt.floor("D")).drop_duplicates(["source", "grid_id", "day"], keep="last").drop(columns="day")
    G.to_pickle(OUT_GRID)
    M.to_pickle(OUT_MEM)
    print(f"grid snapshots {G.shape}, member aggregates {M.shape}", file=sys.stderr)
    print(G.groupby("source").agg(rows=("grid_id", "size"), grids=("grid_id", "nunique"),
                                  accounts=("account_id", "nunique")))
    _ = Counter


if __name__ == "__main__":
    main()
