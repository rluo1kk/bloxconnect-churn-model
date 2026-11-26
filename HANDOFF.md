# BloxConnect Churn Model: Handoff

**From:** Richard Luo (Product Management Intern), last day 25 Nov 2026
**Manager:** Manash Kirtania
**Repository (private):** https://github.com/rluo1kk/bloxconnect-churn-model

---

## 1. What it is

A churn model for NIOS customers built from BloxConnect (HEKA) telemetry, plus a test of the
current **BloxConnect Taxonomy churn score**.

**Use v2** ([docs/v2/MODEL_V2.md](docs/v2/MODEL_V2.md)). It was built from the data, with a locked
test set and significance-gated parameters, comparing **316 churned grids** with **2,973
verified-active grids**. It uses **3 independent parameters**:
- silence;
- DNS Traffic Control adoption;
- company size.

| | Taxonomy churn score | **v2 model** |
|---|---:|---:|
| ROC-AUC (0.5 = guessing) | 0.52 | **0.82–0.88** |
| Churners caught, 3 months ahead | 1 of 378 | 50% at a 46% precision (locked test set) |

The rest of this file describes **v1**, the first study (Taxonomy backtest and a 171-parameter
gradient boosting model). Its 0.84 AUC was later found to be inflated by a weekday artifact. It's
kept for reference.

## Why "3 months ahead"

Every churned customer is scored using only telemetry up to **90 days before its churn date** (the
"Customer Until" date on the Salesforce churn list). Nothing from the final 90 days is used. Retained
customers are still Active more than 90 days after their scoring date. The 90-day lead was chosen for
three reasons:

1. **It leaves time to act.** A flag arrives about a quarter before the contract ends: enough for a
   CSM conversation, an enablement push or a retention offer. A 1-month warning is usually too late
   to change a renewal decision.
2. **It matches the Gainsight ML model,** which also scores accounts 3 months before renewal. Using
   the same horizon keeps the two models comparable, and possible to combine later.
3. **It keeps most of the churned sample.** Anand's pull covers each customer's last reporting year,
   so longer leads leave more churners with no data before their scoring date:

   | Lead time | Churned grids that can be scored |
   |---|---:|
   | 1 month | 330 |
   | 2 months | 329 |
   | **3 months** | **316** |
   | 6 months | 280 |
   | 9 months | 242 |
   | 12 months | 98 |

   Three months costs only 14 churners compared with one month. Six months or more starts losing
   them quickly.

**Limits of the choice:**
- The model was trained and tested **only at the 90-day mark.** Its accuracy at other lead times
  hasn't been measured.
- It doesn't predict the exact churn date. Predicting *when* would need survival analysis (see
  next steps).
- The lead time is one setting, `LEAD_DAYS` in `churn/config.py`. To try 60 or 180 days, change it
  and re-run the pipeline.

## 2. Read these first

| File | What it is |
|---|---|
| [SUMMARY.md](SUMMARY.md) | one page, plain language |
| [EXPLAINED.md](EXPLAINED.md) | step-by-step method, plus gradient descent and boosting explained |
| [docs/MODEL.md](docs/MODEL.md) | the current model as a formula, with weights and thresholds |
| [docs/FINDINGS.md](docs/FINDINGS.md) | full results, data leaks found, trends, proposed methods, limitations |
| [reports/GB_FEATURES.md](reports/GB_FEATURES.md) | every parameter the best model uses: definition, importance, churned vs retained medians |
| [reports/RESULTS.md](reports/RESULTS.md) | all metrics for every model (generated) |

## 3. Running it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m churn.load       # raw files -> data/snapshots.pkl   (~30 s)
.venv/bin/python -m churn.run        # backtest + models -> reports/    (~4 min)
.venv/bin/python -m churn.explain    # parameter table -> reports/      (~2 min)
.venv/bin/python -m unittest discover tests
```

**Data is not in the repo** because it's customer data. Point these environment variables at
the files:

| Variable | File | Where it came from |
|---|---|---|
| `CHURN_PULL_DIR` | `0000–0003_part_00.json`, HEKA snapshots of churned accounts | **Anand** (Redshift unload) |
| `CHURN_LIST_XLSX` | `FY25-FY26 Churned Accounts with last products.xlsx` | Salesforce churn list |
| `WEEKLY_GRID_DIR` | weekly `*GRID*.xlsx` HEKA exports, Apr–Jul 2026 | BloxConnect raw data folder |

On Richard's laptop these sit in `~/Downloads/Final Versions/` and
`~/Downloads/Blox_Connect/BloxConnect Raw Data (All)/Data I Actually Use/`.

## 4. How it works (short version)

1. **Load** both sources into one table: one row per grid per day, deduplicated.
2. **Score churners 3 months before their churn date.** "Retained" grids from the live fleet are
   scored at 5 Jul 2026. Only data from before the scoring date is used.
3. **v0:** the Taxonomy rules exactly as written. 25 points each for Traffic falling off and
   Shrinking deployment, 8 each for Going quiet and Gone dark; flagged at 35.
4. **v1–v4:** re-weighted Taxonomy, logistic regression, random forest, gradient boosting.
   Each uses 171 parameters and is tested with **5-fold cross-validation grouped by account**,
   so a model is never tested on a customer it trained on.
5. **Leak checks:** two artifacts that made the models look too good (0.98 AUC) were found and
   removed. Details are in FINDINGS § 1.

Code map: `churn/load.py` → `cohorts.py` → `signals.py` (v0) → `features.py` → `models.py` →
`run.py` / `explain.py`.

## 5. What we learned

1. **Customers don't wind down before churning.** Members, objects and traffic barely change in
   the year before churn, and about half are still reporting after the contract ends. Churn is a
   commercial decision.
2. **Silence is the one working trend.** Churned grids go 21+ days without reporting much more
   often than retained ones.
3. **Who the customer is matters most.** Churners are thin deployments: no DNS Traffic Control,
   low reporting use, few queries, admins and dashboards.
4. **Two Taxonomy churn signals point the wrong way.** "Going quiet" and "Traffic falling off"
   fire more on customers who stayed.

## 6. What we were working on

Proving whether BloxConnect telemetry can predict churn, and replacing the dashboard's churn
tier with something that works. For comparison, the **Gainsight ML team's** model (sent by
Deepak) reports AUC 0.67, 81% recall and 22% precision. It was measured on different data, so
it's a reference point, not a head-to-head comparison.

## 7. Next steps, in priority order

1. **Get a proper comparison group from Anand.** Ask for the same HEKA pull for a random set of
   **renewed** accounts, covering the same years and in the same format, **with account IDs**.
   Today's "retained" group is a stand-in and may include unflagged churners, which makes the
   models look *worse*. With the new data, rerun `churn.run`; no code changes are needed beyond
   the input paths.
2. **Test new parameters that already exist in HEKA:**
   - signs of migration away from Infoblox: DNS zones in AWS Route 53 or Azure (`data_multi_cloud`), more forwarding or secondary zones, Microsoft DNS taking over (`ms_sync`);
   - paid features used vs paid for;
   - licence events;
   - old hardware or no upgrades;
   - intermittent reporting.
3. **Check whether some "churners" moved to Infoblox cloud** (BloxOne / Universal DDI), using the
   churn list's "Product Groups — Final Month" column. Those aren't real losses and blur the
   labels.
4. **Add Salesforce / Gainsight data:** days to renewal, falling ARR, support cases, NPS. Churn is
   commercial, so telemetry alone hits a ceiling.
5. **Make it production-ready:**
   - probability calibration (a real "x% chance");
   - survival analysis (*when* a customer will churn);
   - SHAP explanations per customer (*why*);
   - a train-on-FY25, test-on-FY26 check.
6. **Feed it back into the dashboard:**
   - raise the weight of "Gone dark";
   - drop or rethink "Going quiet";
   - add an adoption / thin-deployment signal.

## 8. Caveats

- Grid-level scoring: accounts with several grids count once per grid.
- Churners are scored on dates from 2024 to 2026, retained grids in July 2026. A 2026-only check
  (AUC 0.82) shows this isn't driving the result, but it has only 147 churners.
- `reports/` contains aggregates only. Never commit `data/` or the raw files.
