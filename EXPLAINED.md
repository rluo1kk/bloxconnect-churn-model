# How the churn model was built, step by step

This walks through what was done, in order, and why. Part 2 explains gradient descent and
gradient boosting from scratch.

---

# Part 1: The process

## Step 1. Gather the data

| Source | What it is | Size |
|---|---|---|
| **Anand's HEKA pull** (4 JSON files) | weekly telemetry snapshots of grids belonging to customers who **churned** | 19,033 snapshots · 488 accounts · 573 grids · Oct 2022 – Oct 2026 |
| **Churn list** (FY25–FY26 spreadsheet) | each churned account's **churn date** ("Customer Until") | 1,108 accounts; all 488 of Anand's are on it |
| **Weekly HEKA exports** (8 spreadsheets) | the whole live fleet, 29 Apr – 5 Jul 2026 | 4,756 grids |

**Problem found:** Anand's data contains *only* churners. A model can't be judged without
customers who stayed: a model that flags everyone would look perfect. So the weekly fleet
exports became the stand-in "retained" group.

## Step 2. Clean and combine

- Both sources were converted into **one table**: one row per grid per snapshot date, with the
  same columns (members, daily DNS queries, database objects, admins, dashboards, opt-in status,
  and 137 feature-usage counters).
- **Duplicates removed:** some grids were collected more than once a day, so only the last
  collection per day is kept. That left 49,825 snapshots.
- **Sanity check:** 148 grids appear in both sources. On the same day, every field matched
  100%, so the two sources record data identically.

## Step 3. Decide who gets scored, and when

A prediction is only useful *before* churn happens, so:

- **Churners** are scored **3 months before their churn date**, the same lead time the Gainsight
  model uses.
- **Retained grids** are scored at **5 July 2026**, the latest data we have.
- **Only data from on or before the scoring date is used.** The model can never see the future.
- A grid is only scored if it had been reporting for the full 2 months before that date, and
  hadn't been silent for over a year.

Result: **378 churned grids** (332 accounts) vs **3,612 retained grids**. About **9.5%**
churners, roughly the real-world mix.

## Step 4. Rebuild the current Taxonomy model and test it (v0)

Each of the four Churn rules was coded exactly as the Taxonomy spreadsheet states:

| Signal | Points | Fires when (over 2 months) |
|---|---:|---|
| Traffic falling off | 25 | DNS queries or objects down 30%+ |
| Shrinking deployment | 25 | members down 10%+ |
| Going quiet | 8 | object count moves less than 2% |
| Gone dark | 8 | no report for 14+ days, or opted out |

The score adds the points of every signal that fires, and **35 or more = flagged**.

**Result: 1 of 378 churners flagged.** The model fails because two of its signals almost never
fire before churn, and the other two fire *more* on healthy customers.

## Step 5. Build a much bigger set of clues ("features")

Instead of 4 yes/no signals, each grid got **171 measurements** at its scoring date:

| Family | Examples | Count |
|---|---|---:|
| **Silence** | days since last report, opted out | 2 |
| **Trends** | 2-month % change in members, queries, objects, admins, dashboards | 6 |
| **Size and setup** | daily queries, objects, admins, dashboards, licences, password policy | about 14 |
| **Feature usage** | DNS Traffic Control %, A records, DHCP leases, zones, … | 137 |
| **Taxonomy verdicts** | each of the 4 signals: green / amber / red / n/a, plus the score | 14 |

## Step 6. Check for "cheating" (data leaks)

A model can score well by spotting *which file a row came from* instead of who churned. This
happened twice, and both were fixed before any result was trusted:

1. **Scoring dates lined up with snapshot days.** Every retained grid was scored on the exact day
   of a snapshot, so its "days since last report" was always 0. The model learned "0 = retained".
   **Fix:** retained grids were moved to a random 0–6 days after 5 July, like churners.
2. **Different snapshot frequency.** The fleet files had 7 snapshots per 2 months; Anand's had 9.
   Anything depending on the *number* of snapshots identified the file.
   **Fix:** those measurements were removed.

Before the fixes the best model scored a too-good-to-be-true **0.98**. After them it scored an
honest **0.84**. The unit tests also caught a bug that stopped "opted out" from ever firing,
and that's fixed.

## Step 7. Train the models, testing on customers they haven't seen

Four models were tried, from simple to powerful:

| Model | In plain words |
|---|---|
| **v1 Re-weighted Taxonomy** | same 4 signals, but the data picks the points instead of the spreadsheet |
| **v2 Logistic regression** | one weight per clue, added up and turned into a probability |
| **v3 Random forest** | 500 decision trees, each built on a random sample; they vote |
| **v4 Gradient boosting** | 400 small trees built *one after another*, each fixing the last one's mistakes (see Part 2) |

**How they were tested: 5-fold cross-validation.**
1. Split the customers into 5 groups. All grids of one account stay in the same group, so the
   model can't recognise a customer it has already seen.
2. Train on 4 groups, then predict the 5th group, which the model never saw.
3. Repeat 5 times so every customer gets predicted exactly once by a model that never saw it.

The cut-off for "flag" vs "don't flag" was also chosen using only the training groups.

## Step 8. Measure

| Measure | Question it answers | v0 Taxonomy | v4 Gradient boosting |
|---|---|---:|---:|
| **Recall** | Of the real churners, how many did we catch? | 0.3% (1 of 378) | **43%** (163) |
| **Precision** | Of the customers we flagged, how many really churned? | 4% | **42%** |
| **Recall @ top 10%** | If the team reviews only the riskiest 10%, what share of churners do they find? | 7% | **43%** |
| **ROC-AUC** | How well does the score rank churners above non-churners? (0.5 = coin flip) | 0.52 | **0.84** |
| **Accuracy** | What share of all predictions were right? | 90% | 89% |

**Why accuracy misleads:** about 90% of customers don't churn, so a model that flags *nobody*
scores 90%. That's exactly what v0 almost does.

## Step 9. Explain what the model learned

**Permutation importance:** take one clue, shuffle its values randomly across customers, and see
how much the score drops. A big drop means the model relied on it.

| Family shuffled | ROC-AUC drop |
|---|---:|
| Customer profile (size and feature usage) | **0.18** |
| Silence | **0.08** |
| Trends | 0.02 |
| Taxonomy verdicts | 0.00 |

**Conclusion:** churners don't visibly wind down. They are **thin deployments** (few queries,
no DNS Traffic Control, low reporting use) that **go silent** more often.

---

# Part 2: Gradient descent and gradient boosting

## Gradient descent: walking downhill in fog

Imagine you're on a hillside in thick fog and want to reach the lowest point. You can't see the
valley, but you can feel which way the ground slopes under your feet. So you:
1. feel the slope;
2. take a small step **downhill**;
3. repeat until the ground is flat.

That is gradient descent:
- the **hill** is the *loss*, a number measuring how wrong the model is;
- your **position** is the model's settings (its *weights*);
- the **slope** is the *gradient*, which says in which direction the error grows;
- the **step size** is the *learning rate*.

**Update rule:** new weight = old weight − learning rate × slope.

### Tiny worked example

Say the error is `(w − 3)²`, which is smallest at w = 3. Its slope is `2 × (w − 3)`. Start at
w = 0 with a learning rate of 0.1:

| Step | w | Slope 2(w−3) | New w = w − 0.1 × slope |
|---:|---:|---:|---:|
| 1 | 0 | −6.0 | 0.60 |
| 2 | 0.60 | −4.8 | 1.08 |
| 3 | 1.08 | −3.84 | 1.46 |
| … | … | … | … approaches **3** |

The steps shrink as the slope flattens, which is how it "lands" at the bottom.
- **Learning rate too big:** you overshoot and bounce around.
- **Learning rate too small:** it takes forever.

### For churn, the "hill" is log loss

The model outputs a churn probability *p* for each customer. **Log loss** punishes confident
wrong answers heavily:
- a real churner given p = 0.9 costs little;
- a real churner given p = 0.01 costs a lot.

Gradient descent (or a smarter cousin of it) adjusts the model to make the total log loss as
small as possible.

## Gradient boosting: gradient descent, but each step is a small decision tree

A **decision tree** is a flowchart of questions:

```
days since last report > 21?
 ├─ yes → DTC members % = 0?
 │         ├─ yes → raise risk a lot
 │         └─ no  → raise risk a little
 └─ no  → daily queries < 10,000? → …
```

One small tree is a weak predictor. **Boosting** builds many of them **in sequence**, each one
correcting the mistakes left over by all the trees before it:

1. **Start with a guess.** Give every grid the same risk, the average churn rate (about 9.5%).
2. **Measure the mistakes.** For each grid, mistake = actual − predicted. A churner (1) predicted
   at 0.095 has a mistake of **+0.905**. A retained grid (0) has a mistake of **−0.095**.
   *Mathematically, this "actual minus predicted" is exactly the downhill slope of the log loss.
   That's where "gradient" comes from.*
3. **Fit a small tree to the mistakes,** not to the original answers. The tree finds questions
   that separate "under-predicted" grids from "over-predicted" ones.
4. **Take a small step.** Add that tree's output × the learning rate (0.05) to every grid's score.
   Small steps stop any single tree from dominating.
5. **Repeat** 400 times. Each new tree targets what's still wrong.

**Gradient descent vs gradient boosting:**
- **Gradient descent** moves the model's *weights* downhill.
- **Gradient boosting** moves the model's *predictions* downhill, using a tree as each step.

### Toy example: three customers

| Grid | Actually churned? | Start | After tree 1 | After many trees |
|---|:-:|---:|---:|---:|
| A: silent 30 days, no DTC | yes | 0.095 | 0.13 | **0.85** |
| B: reporting, heavy DTC use | no | 0.095 | 0.09 | **0.03** |
| C: reporting, tiny deployment | yes | 0.095 | 0.10 | **0.55** |

*The numbers are illustrative, to show the direction of movement.*

### The settings used, and why

| Setting | Value | What it does |
|---|---|---|
| Number of trees | 400 | how many correction steps |
| Learning rate | 0.05 | step size; small means careful, less overfitting |
| Max leaves per tree | 15 | keeps each tree small and "weak" |
| Min grids per leaf | 20 | stops a tree memorising one customer |
| L2 regularisation | 1.0 | shrinks extreme corrections |
| Class weight | balanced | churners (9.5%) count as much as the 90.5% |

### Why it beat the other models

- **Combinations:** it captures combinations, such as "silent *and* no DTC *and* small", that a
  points-based score can't express.
- **Missing values:** it handles them natively, so a missing reporting allowance doesn't break it.
- **Overfitting control:** many small careful steps generalise better than one big model.

**Trade-off:** it's harder to explain than a points table. That's why Step 9 (importance) matters,
and why SHAP explanations per customer are a recommended next step.

---

## Glossary

| Term | Meaning |
|---|---|
| **Feature / parameter** | one measurable clue about a grid, e.g. days since last report |
| **Label** | the right answer: churned (1) or retained (0) |
| **Training / testing** | learning from some customers, then checking on unseen ones |
| **Cross-validation** | rotating which customers are held out, so every one is tested once |
| **Overfitting** | memorising the training customers instead of learning general patterns |
| **Data leak** | a clue that gives away the answer for the wrong reason |
| **Threshold** | the probability above which a grid is flagged |
