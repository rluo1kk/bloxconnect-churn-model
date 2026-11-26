"""Paths and constants. Override any path with an environment variable of the same name.

Easiest setup: point CHURN_HANDOFF_DIR at the Product Team OneDrive handoff folder and every
input path below follows its layout:

    export CHURN_HANDOFF_DIR="/path/to/Product_Team_Handoff"

Without it, paths default to the original analyst's laptop layout.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_HOME = Path.home()
_HANDOFF = os.environ.get("CHURN_HANDOFF_DIR")
_H = Path(_HANDOFF).expanduser() if _HANDOFF else None


def _p(env, default):
    return Path(os.environ.get(env, default)).expanduser()


def _d(handoff_rel, laptop_default):
    """Default path: inside the handoff folder if CHURN_HANDOFF_DIR is set, else the laptop path."""
    return (_H / handoff_rel) if _H else laptop_default


_FINAL = _HOME / "Downloads/Final Versions"
_CHURN = "Shared data/Churn model inputs"
_LABELS = f"{_CHURN}/Labels and account mapping"

# --- Raw inputs (never committed) -------------------------------------------------
# Anand's Redshift unload: HEKA grid snapshots for churned accounts, JSON lines (.json or .json.gz).
CHURN_PULL_DIR = _p("CHURN_PULL_DIR", _d(f"{_CHURN}/Churned customers (Anand HEKA pull)", _FINAL))
CHURN_PULL_GLOB = os.environ.get("CHURN_PULL_GLOB", "000*_part_00.json.gz" if _H else "000*_part_00.json")

# Salesforce churn list: one row per churned account with 'Customer Until'.
CHURN_LIST_XLSX = _p("CHURN_LIST_XLSX", _d(f"{_LABELS}/FY25-FY26 Churned Accounts with last products.xlsx",
                                           _FINAL / "FY25-FY26 Churned Accounts with last products.xlsx"))

# Customer Lifecycle (Active / Inactive, employees, geo) and the grid -> account extract.
LIFECYCLE_XLSX = _p("LIFECYCLE_XLSX", _d(f"{_LABELS}/Copy of Customer Lifecycle _Data_Current.xlsx",
                                         _FINAL / "Copy of Customer Lifecycle _Data_Current.xlsx"))
FEATURE_USAGE_XLSX = _p("FEATURE_USAGE_XLSX",
                        _d(f"{_LABELS}/Copy of HEKA feature usage data and member role 06.05.2026.xlsx",
                           _FINAL / "Copy of HEKA feature usage data and member role 06.05.2026.xlsx"))
# Name matching: grid -> account name, and active-account names -> Salesforce IDs.
MEMBER_DETAILS_XLSX = _p("MEMBER_DETAILS_XLSX", _d(f"{_LABELS}/Member Details.xlsx",
                                                   _HOME / "Downloads/Blox_Connect/HEKA customer data/Member Details.xlsx"))
ACTIVE_ACCOUNTS_XLSX = _p("ACTIVE_ACCOUNTS_XLSX", _d(f"{_LABELS}/SFDC NIOS Active Accounts.xlsx",
                                                     _HOME / "Downloads/SFDC NIOS Active Accounts.xlsx"))

# Weekly HEKA grid + member exports (Apr-Jul 2026): the fleet the retained group comes from.
WEEKLY_GRID_DIR = _p("WEEKLY_GRID_DIR", _d("Shared data/HEKA weekly exports (Apr-Jul 2026)",
                                           _HOME / "Downloads/Blox_Connect/BloxConnect Raw Data (All)/Data I Actually Use"))
WEEKLY_GRID_GLOB = os.environ.get("WEEKLY_GRID_GLOB", "*[Gg][Rr][Ii][Dd]*.xlsx")
FIELD_DICTIONARY_XLSX = _p("FIELD_DICTIONARY_XLSX", _d("Shared data/Reference/BloxConnect_Field_Dictionary.xlsx",
                                                       _HOME / "Downloads/Blox_Connect/Dictionary/BloxConnect_Field_Dictionary-2.xlsx"))

# --- Intermediate + outputs --------------------------------------------------------
DATA_DIR = _p("CHURN_DATA_DIR", ROOT / "data")            # git-ignored
REPORTS_DIR = _p("CHURN_REPORTS_DIR", ROOT / "reports")   # committed (aggregates only)
SNAPSHOTS = DATA_DIR / "snapshots.pkl"

# --- Evaluation design -------------------------------------------------------------
LEAD_DAYS = 90            # score each churner 3 months before its churn date (Gainsight convention)
WINDOW_DAYS = 60          # Taxonomy: trends are read "over the course of 2 months"
HEALTHY_LEAD_DAYS = 365   # within-churner baseline: the same grid one year before churn
CONTROL_ANCHOR = "2026-07-05"   # last weekly snapshot; retained grids are scored as of this date
RANDOM_STATE = 42


def open_text(path):
    """Open a JSON-lines file whether it is plain or gzip-compressed."""
    import gzip
    path = Path(path)
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else open(path, encoding="utf-8")
