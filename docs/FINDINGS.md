# Backtest, improvements and what the data says

All numbers come from `reports/results.json`, produced by `python -m churn.run`. Every model
score is **out-of-fold** (5-fold cross-validation grouped by account), and each decision
threshold was chosen on training folds only.

## 1. How the model was tested

| | Churned (positives) | Retained (negatives) |
|---|---|---|
| Source | Anand's HEKA pull: 488 churned accounts, 573 grids, Oct 2022 – Oct 2026 | Weekly HEKA grid exports, 29 Apr – 5 Jul 2026, excluding 167 grids whose grid or licence ID is in Anand's pull |
| Scored on | **3 months before** each account's churn date (`Customer Until`, FY25–FY26 churn list) | 5 Jul 2026 + a random 0–6 days |
| Scorable | 378 grids / 332 accounts | 3,612 grids |

**Cohort A** (the main test) compares churned grids against the live fleet. The base rate is
9.5% churners.

**Cohort B** compares the same churned grids 3 months vs 12 months before churn (378 vs 102).
Both groups come from one source, so no source quirk can leak into the result.

**Grids not scored:**
- **Churned grids (195 of 573):** most had no telemetry covering the full 2-month window before the scoring date.
- **Retained grids (977):** either first seen after 29 Apr, or silent for over a year.

### Artifacts found and removed before trusting any result

| Artifact | Symptom | Fix |
|---|---|---|
| Retained grids all scored exactly on snapshot day | "days since last snapshot" = 0 vs 6 for churners; AUC 0.99 | random 0–6 day offset, like churners' arbitrary dates |
| Weekly exports sampled less often (7 vs 9 readings per window) | snapshot count, volatility and drawdown identified the *file*, not churn | dropped every count-dependent feature; first → last change only |
| Partial history | n/a rates driven by when Anand's pull started | both groups must be observed for the full window |
| Opt-out check (`is False` on a NumPy bool) | opt-out never fired | fixed and covered by a unit test |

Removing the first two artifacts took gradient boosting from a too-good AUC of 0.98 down to 0.84.
For the same grid on the same day, both sources agree on every field (713 overlapping
grid-days, 100% match), so the remaining differences are real differences between customers.

## 2. Results: current model vs improved models (Cohort A)

| Model | ROC-AUC | Accuracy | Balanced acc. | Precision | Recall | Recall @ top 10% |
|---|---:|---:|---:|---:|---:|---:|
| **v0 Taxonomy (current)** | **0.52** | 0.90 | **0.50** | 0.04 | **0.003** (1 of 378) | 0.07 |
| v1 Taxonomy signals, re-weighted (logistic) | 0.64 | 0.85 | 0.61 | 0.26 | 0.32 | 0.28 |
| v2 Logistic regression, all features | 0.80 | 0.89 | 0.67 | 0.39 | 0.40 | 0.40 |
| v3 Random forest | 0.83 | 0.90 | 0.65 | 0.49 | 0.33 | 0.40 |
| **v4 Gradient boosting** | **0.84** | 0.89 | **0.68** | **0.42** | **0.43** (163 of 378) | **0.43** |

**Read accuracy with care.** Only 9.5% of grids churn, so a model that never flags anyone scores
90.5%. That's why v0 shows 0.90 accuracy while catching 1 churner. Balanced accuracy, recall and
AUC are the honest measures.

**Improvement, v0 → v4:**
- **ROC-AUC:** 0.52 → 0.84.
- **Churners caught:** 1 → 163 of 378.
- **Precision of a flag:** 4% → 42%.
- **Ranking:** if a team reviews only the top 10% riskiest grids, it now finds 43% of churners, up from 7%.

**Robustness checks:**

| Check | v0 | v4 | What it shows |
|---|---:|---:|---|
| Churners scored in 2026 only (same period as the retained fleet) | AUC 0.54 | AUC 0.82; recall @ top 10%: 58% | the gain is not an artifact of comparing different years |
| Cohort B, same grids 3 vs 12 months before churn | AUC 0.47 | AUC 0.69 (random forest 0.74) | separating *when* a grid is close to churn is harder than *who* will churn |

For context, the Gainsight ML model reports AUC 0.67, 81% recall and 22% precision. That was
measured on different data with a different design, so it's a reference point, not a
head-to-head comparison.

## 3. Why the current model fails

Firing rates in Cohort A:

| Signal | Churned grids | Retained grids | Verdict |
|---|---:|---:|---|
| Traffic falling off (25) | 6.9% | 11.2% | fires **more** on retained grids |
| Shrinking deployment (25) | 0.5% | 1.9% | almost never fires; churners don't shed members first |
| Going quiet (8) | 48.9% | 58.3% | fires on most of the fleet; **inversely** related to churn |
| Gone dark (8) | **34.7%** | **11.8%** | the only signal that separates the groups, and it has the lowest weight |

Learned weights (v1, logistic regression on the same four verdicts), largest first:

| Verdict | Log-odds | Reading |
|---|---:|---|
| Gone dark, amber (silent 14–21 days) | +1.42 | strongest Taxonomy evidence |
| No trend data in the window | +0.95 | missing telemetry is itself a churn sign |
| Going quiet, red | +0.31 | weak |
| Gone dark, red | +0.28 | weak (overlaps with "no trend data") |
| Shrinking amber / red, Traffic amber | **negative** | the opposite of what the Taxonomy assumes |

## 4. Trends in the data we were missing

1. **Churn here is a commercial event, not a technical wind-down.** The median 2-month change is 0%
   for members and under 0.5% for objects at every point in the year before churn (`reports/lead_time.png`).
   51% of churned grids are still reporting on or after their churn date. Among those, member
   counts are unchanged (median 0%; only 1.3% have fewer members). Customers don't dismantle the
   grid before the contract ends, so trend-of-usage signals can't see churn coming.
2. **Silence is the leading indicator.** Churned grids are silent for 21+ days about 31–34% of the
   time 3–9 months before churn, rising to 41% at the churn date. "Gone dark" fires on only 12% of
   retained grids. "Days since last snapshot" is the model's most important feature.
3. **Who churns differs from who stays.** Churned grids are smaller and use less of the product:
   - fewer DTC members (DNS Traffic Control);
   - lower reporting usage relative to their allowance;
   - fewer daily queries (median about 3,000 vs 1.5 million);
   - fewer admins, zones and forwarding zones.

   These *profile* features carry most of the gain from v1 to v4. The Taxonomy currently has no
   churn signal for low adoption.
4. **Going quiet is mislabelled.** Flat object counts are normal for a stable, healthy deployment
   (58% of the retained fleet). As a churn signal it adds noise.

## 5. Statistical methods: proposals and status

| Method | Status | Why |
|---|---|---|
| Re-weight Taxonomy signals with **logistic regression** | done (v1) | keeps the rule-based model and its explainability; turns weights into evidence |
| **L2 logistic regression** on engineered features | done (v2) | transparent coefficients, good baseline |
| **Random forest** | done (v3) | non-linear interactions, robust to scale and outliers |
| **Gradient boosting** (histogram GBM) | done (v4, best) | strongest on tabular data; handles missing values natively |
| **Probability calibration** (Platt / isotonic) | next | turn scores into "x% chance of churn", as the Gainsight model does |
| **Survival analysis** (Cox / discrete-time hazard) | next | predicts *when*, uses every snapshot instead of one date, handles accounts still active |
| **SHAP explanations** per account | next | the dashboard can show *why* a customer is at risk, like the Taxonomy chips do today |
| Join with **Salesforce / Gainsight** features (ARR trend, renewal date, support cases, NPS) | next | churn is commercial (finding 1); telemetry alone hits a ceiling |
| **Temporal holdout** (train on FY25, test on FY26) | next | closest test to production use |

## 6. Limitations (read before quoting a number)

- **Stand-in retained group.** The weekly exports have no Salesforce account ID. Some
  "retained" grids may belong to churners that aren't in Anand's pull (about 527 NIOS churners
  on the list are missing from it). That contamination would *understate* performance.
- **Different time periods.** Retained grids are scored in July 2026, churners from 2024 to 2026.
  The 2026-only check (AUC 0.82) shows this isn't driving the result, but it has only 147
  churners.
- **Grid-level scoring.** Accounts with several grids are counted once per grid.
- **Next step:** with **Anand's matched pull of retained accounts** (same format, same years,
  with account IDs), rerun `python -m churn.run`. That gives the definitive accuracy, and no code
  changes are needed beyond pointing `config.py` at the new files.
