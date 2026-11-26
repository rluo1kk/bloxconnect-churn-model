# BloxConnect Churn Model

> **Start here: v2** ([docs/v2/MODEL_V2.md](docs/v2/MODEL_V2.md)), a model built from the data with a
> locked test set and significance-gated parameters. 316 churned vs 2,973 verified-active grids;
> 3 independent parameters (silence, DNS Traffic Control adoption, company size), test AUC 0.82–0.88.
> Data: set `CHURN_HANDOFF_DIR` to the OneDrive handoff folder (see HANDOFF.md).
> Run: `python -m churn2.namematch && python -m churn2.extract && python -m churn2.dataset &&
> python -m churn2.model && python -m churn2.report && python -m churn2.consensus`.
> The material below is the earlier v1 study (Taxonomy backtest), kept for reference.

Backtests the BloxConnect Taxonomy churn score against real churned customers and compares it
with statistical models trained on the same HEKA telemetry.

| | Taxonomy model today (v0) | Gradient boosting (v4) |
|---|---:|---:|
| ROC-AUC | 0.52 | **0.84** |
| Churners caught, 3 months ahead | 1 of 378 | **163 of 378** |
| Precision of a flag | 4% | **42%** |
| Churners found when reviewing the top 10% riskiest grids | 7% | **43%** |

Start with [docs/MODEL.md](docs/MODEL.md) for the current model's formula and weights, then
[docs/FINDINGS.md](docs/FINDINGS.md) for the backtest, the improvements, the trends found and
the limitations. Generated tables are in [reports/RESULTS.md](reports/RESULTS.md).

> **Confidential.** Inputs are Infoblox customer telemetry and Salesforce churn records. They
> are never committed (`data/` and `*.xlsx`/`*.json` inputs are git-ignored). `reports/` holds
> aggregates only, with no account names or IDs.

## Run it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m churn.load        # raw files -> data/snapshots.pkl      (~30 s)
.venv/bin/python -m churn.run         # backtest + models -> reports/       (~4 min)
.venv/bin/python -m unittest discover tests
```

Input locations default to the original analyst's machine. Override them with environment
variables (see `churn/config.py`):

| Variable | What |
|---|---|
| `CHURN_PULL_DIR` | folder with Anand's `000*_part_00.json` (HEKA unload of churned accounts) |
| `CHURN_LIST_XLSX` | `FY25-FY26 Churned Accounts with last products.xlsx` |
| `WEEKLY_GRID_DIR` | folder with the weekly `*GRID*.xlsx` HEKA exports |

## Layout

```
churn/
  config.py     paths, scoring design (3-month lead, 2-month window)
  load.py       both raw sources -> one snapshot table, with HEKA de-duplication rules
  cohorts.py    who is scored when, with what label (and the anti-artifact rules)
  signals.py    v0: the Taxonomy Churn tier, rule by rule
  features.py   171 features for the statistical models (no look-ahead)
  models.py     v1-v4 and the grouped cross-validation harness
  metrics.py    one metrics function for every model
  run.py        end to end; writes reports/
tests/          unit tests for every Taxonomy rule (synthetic data, no customer data needed)
docs/           MODEL.md (formula + weights), FINDINGS.md (results, trends, next steps)
reports/        results.json, RESULTS.md, charts (generated)
```

## Changing the model

- **Taxonomy thresholds or weights:** edit `churn/signals.py`, update the tests, rerun.
- **New feature:** add it in `features.grid_features`. It must use only snapshots at or before `T`.
- **New model:** add it to `models.make_models()`. It's cross-validated and reported automatically.
- **Matched retained accounts from Anand:** point `WEEKLY_GRID_DIR`/`load.py` at them and rerun.
  This is the planned next step.
