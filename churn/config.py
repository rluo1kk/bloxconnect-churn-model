"""Paths and constants. Override any path with an environment variable of the same name."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_HOME = Path.home()


def _p(env, default):
    return Path(os.environ.get(env, default)).expanduser()


# --- Raw inputs (never committed) -------------------------------------------------
# Anand's Redshift unload: HEKA grid snapshots for churned accounts, JSON lines.
CHURN_PULL_DIR = _p("CHURN_PULL_DIR", _HOME / "Downloads/Final Versions")
CHURN_PULL_GLOB = os.environ.get("CHURN_PULL_GLOB", "000*_part_00.json")

# Salesforce churn list: one row per churned account with 'Customer Until'.
CHURN_LIST_XLSX = _p("CHURN_LIST_XLSX",
                     _HOME / "Downloads/Final Versions/FY25-FY26 Churned Accounts with last products.xlsx")

# Weekly HEKA grid exports (Apr-Jul 2026), used as the stand-in "retained" population.
WEEKLY_GRID_DIR = _p("WEEKLY_GRID_DIR",
                     _HOME / "Downloads/Blox_Connect/BloxConnect Raw Data (All)/Data I Actually Use")
WEEKLY_GRID_GLOB = os.environ.get("WEEKLY_GRID_GLOB", "*[Gg][Rr][Ii][Dd]*.xlsx")

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
