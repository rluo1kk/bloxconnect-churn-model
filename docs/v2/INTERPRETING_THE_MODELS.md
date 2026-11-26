# How to interpret the gradient boosting model and the new model

**The short version:**
- **Gradient boosting** has no weights to read. You interpret it from the outside, by testing what
  it relies on.
- **The new model** is a logistic regression. You read it from the inside: each parameter has one
  number that says how strongly, and in which direction, it moves churn risk.

---

## 1. Gradient boosting: interpret it by its behaviour

### How it decides

- It's about **400 small decision trees**, each one a short chain of questions such as "silent for
  more than 21 days? and no DNS Traffic Control? then raise the risk".
- Each tree corrects the mistakes left by the trees before it.
- A customer's score is **the sum of all 400 trees' outputs**, converted to a probability.
- No single tree, and no single number, explains the result.

### Three ways to read it

| Question | Tool | What it tells you |
|---|---|---|
| What does it rely on? | **Permutation importance**: scramble one parameter and see how much accuracy drops | which parameters matter most (`reports/GB_FEATURES.md`) |
| Which direction? | **Partial dependence**, or comparing churned and retained medians | e.g. lower DTC adoption goes with higher risk |
| Why *this* customer? | **SHAP values** (not built yet) | per-account reasons, e.g. "+12 points from silence, −5 from DTC" |

### Cautions

- **Importance isn't impact.** It says the model *uses* a parameter, not how much risk changes per
  unit.
- **Correlated parameters split the credit.** DTC and RPZ share credit, so either one can look less
  important than it really is.
- **It finds shortcuts.** In this project, gradient boosting reached a fake AUC of 0.98, and later
  0.95, by exploiting data artifacts. It's powerful, so it has to be audited.
- **In v2 it wasn't significantly better** than the simple model: average AUC 0.859 vs 0.842,
  p = 0.28. That's why it wasn't chosen.

---

## 2. The new model (logistic regression): read it directly

### The formula

<div class="formula">churn score = −2.787 + 0.628 × silence − 0.872 × DTC − 0.479 × employees<br>
churn probability = 1 ÷ (1 + e<sup>−score</sup>)</div>

**Its parts:**
- **silence, DTC and employees** are each customer's values put on a common scale: how unusual
  the customer is compared with the average, in standard deviations.
- Each raw value is **log-transformed first**, so going from 0 to 30 days of silence matters more
  than going from 300 to 330.
- **The signs give the direction:**
  - silence is **+**, so more silence means more risk;
  - DTC and employees are **−**, so more of either means less risk.
- **The size of each number gives the strength.**

### The odds ratio: the easiest number to quote

| Parameter | Odds ratio per standard deviation | In words |
|---|---:|---|
| Silence | **1.87** | each step up **raises** the odds of churn by **87%** |
| DTC adoption | **0.42** | each step up **cuts** the odds by **58%** |
| Company size | **0.62** | each step up **cuts** the odds by **38%** |

### What the numbers mean for real customers

These are calculated with the fitted model.

| Customer | Churn probability (next 3 months) |
|---|---:|
| Reporting normally, no DTC, 1,000 employees | **11%**, about average |
| Same, but **half its members on DTC** | **2%** |
| Same as the first, but **silent 30 extra days** | **37%** |
| Silent 30 days, no DTC, **200 employees** | **45%** |
| Reporting normally, full DTC, 10,000 employees | **1%** |

This is the main advantage over gradient boosting: **any customer's score can be explained with
three facts.**

### How to use the probabilities

- The model was fitted on a sample where about **10% of grids churn**, close to reality. So "37%"
  roughly means *about 37 in 100 customers like this would churn within 3 months*.
- **Rank customers by probability and work down the list.**
- The flag cut-off (about 18% in the primary run) is **a dial**:
  - lower it to catch more churners, with more false alarms;
  - raise it for fewer, more certain flags.

### What it does *not* say

- **It isn't causal.** DTC users stay, but that may be because engaged customers buy DTC, not
  because DTC keeps them. Offering DTC trials to thin deployments would test this.
- **It only sees customers who send telemetry.** The 563 churned accounts with no HEKA data are
  invisible to it.

---

## 3. Rule of thumb

| Use | Model |
|---|---|
| Decisions, dashboard scores, explaining a customer's risk | **Logistic model (the new model)** |
| Benchmark: is there a pattern we're missing? | **Gradient boosting** |

If gradient boosting ever becomes **significantly** better than the logistic model, a new pattern
exists in the data. Find it, and add it to the simple model.
