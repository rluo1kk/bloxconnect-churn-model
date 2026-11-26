"""Extend the grid -> Salesforce account map by matching account NAMES.

    python -m churn2.namematch      # -> data/v2_name_map.pkl and a validation summary

Source: `HEKA customer data/Member Details.xlsx`, a Tableau crosstab with Account Name and
Grid UUID per member. Account Name and Grid UUID appear only on the first row of each group,
and UUIDs are truncated to 30 characters.

Name -> ID uses exact matching after normalisation (case, punctuation, legal suffixes) against
`SFDC NIOS Active Accounts.xlsx` and the churn list. A name is used only if it resolves to
exactly ONE account. The method is validated on grids that already have an ID from the
feature-usage extract: agreement with that ID is reported, and if it is low the name map is
not to be trusted.
"""
import re
import sys
import unicodedata
from pathlib import Path

import openpyxl
import pandas as pd

from churn import config

OUT = config.DATA_DIR / "v2_name_map.pkl"
MEMBER_DETAILS = Path.home() / "Downloads/Blox_Connect/HEKA customer data/Member Details.xlsx"
ACTIVE_XLSX = Path.home() / "Downloads/SFDC NIOS Active Accounts.xlsx"

SUFFIX = re.compile(r"\b(incorporated|inc|llc|l l c|ltd|limited|corp|corporation|co|company|plc|gmbh|ag|sa|s a|"
                    r"bv|b v|nv|n v|spa|s p a|srl|sas|pty|pte|oy|ab|as|a s|kk|k k|lp|llp|the)\b")


def norm(name):
    if not isinstance(name, str):
        return None
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"&", " and ", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = SUFFIX.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s or None


def grid_names():
    """grid UUID prefix (30 chars) -> account name, forward-filling the crosstab."""
    wb = openpyxl.load_workbook(MEMBER_DETAILS, read_only=True)
    it = wb.worksheets[0].iter_rows(values_only=True)
    hdr = list(next(it))
    ia, ig = hdr.index("Account Name"), hdr.index("Grid UUID")
    name = grid = None
    pairs = set()
    for r in it:
        a, g = r[ia], r[ig]
        if a not in (None, ""):
            name = str(a).strip()
        if g not in (None, ""):
            grid = str(g).strip().replace("-", "").lower()
        if name and grid:
            pairs.add((grid[:30], name))
    df = pd.DataFrame(sorted(pairs), columns=["grid30", "account_name"])
    n = df.groupby("grid30")["account_name"].nunique()
    return df[df.grid30.isin(n[n == 1].index)].drop_duplicates("grid30"), int((n > 1).sum())


def name_index():
    """normalised name -> account id, from the active-accounts list and the churn list."""
    act = pd.read_excel(ACTIVE_XLSX)
    act = act[["Account ID", "Account Name"]].dropna()
    act.columns = ["account_id", "account_name"]
    ch = pd.read_excel(config.CHURN_LIST_XLSX, header=1)[["18-Digit Account ID", "Account Name"]].dropna()
    ch.columns = ["account_id", "account_name"]
    allr = pd.concat([act.assign(src="active"), ch.assign(src="churn_list")])
    allr["key"] = allr.account_name.map(norm)
    allr = allr.dropna(subset=["key"])
    ids = allr.groupby("key")["account_id"].nunique()
    unique = allr[allr.key.isin(ids[ids == 1].index)].drop_duplicates("key")
    return unique.set_index("key")["account_id"], int((ids > 1).sum())


def main():
    from .extract import grid_account_map
    gn, n_multi = grid_names()
    idx, n_ambig_names = name_index()
    gn["key"] = gn.account_name.map(norm)
    gn["account_id"] = gn.key.map(idx)
    matched = gn.dropna(subset=["account_id"])
    print(f"grids with an account name: {len(gn):,} ({n_multi} grids with conflicting names dropped)", file=sys.stderr)
    print(f"names resolving to exactly one Salesforce account: {len(matched):,} grids "
          f"({n_ambig_names} normalised names were ambiguous and skipped)", file=sys.stderr)

    # validation against the trusted feature-usage map
    g2a, _ = grid_account_map()
    trusted = pd.DataFrame({"grid30": g2a.index.str[:30], "trusted_id": g2a.values})
    v = matched.merge(trusted, on="grid30")
    agree = (v.account_id == v.trusted_id).mean() if len(v) else float("nan")
    print(f"VALIDATION: {len(v):,} grids have both a trusted ID and a name-matched ID; agreement {agree:.1%}",
          file=sys.stderr)
    out = matched[~matched.grid30.isin(set(trusted.grid30))][["grid30", "account_id", "account_name"]]
    out.attrs.update({"validation_n": int(len(v)), "validation_agreement": float(agree)})
    out.to_pickle(OUT)
    print(f"new grids gaining an account ID from names: {len(out):,} -> {OUT}", file=sys.stderr)
    if len(v):
        print(v[v.account_id != v.trusted_id].head(8)[["account_name", "account_id", "trusted_id"]].to_string(),
              file=sys.stderr)


if __name__ == "__main__":
    main()
