# How 264 parameters became 5: the selection steps explained

The pipeline started with 264 parameters that had enough data to test. Each step below is a filter.
A parameter has to pass all of them to reach the final model. Examples use the real numbers from
this project.

| Step | What it asks | Parameters left |
|---|---|---:|
| 1. Significant and stable over time | Do churned and retained customers really differ on this, and not just because of the date they were measured? | 64 |
| 2. Independent | Is this measuring something different from the parameters already kept? | 27 |
| 3. Adds significant accuracy | Does adding it make predictions *meaningfully* better? | 5 |

---

## Step 1: Significant and stable over time (264 → 64)

Four checks happen at once here.

### a) Mann-Whitney U test: "do the two groups differ?"

- It **ranks every customer** on a parameter, from lowest to highest, and checks whether churned
  customers sit noticeably lower or higher in the ranking than retained ones.
- It works on **ranks, not raw values**, so a few giant customers (e.g. 50 million queries a day)
  can't distort it. It also makes no assumption that the data follows a bell curve, which HEKA
  counts rarely do.
- Its result can be written as an **AUC**: the chance that a randomly picked churner has a *higher*
  value than a randomly picked retained customer.
  - **0.50 means no difference.**
  - DNS Traffic Control adoption scored **0.31**: churners have *more* DTC only 31% of the time, so
    they clearly tend to have *less*.
  - Silence scored **0.63**: churners tend to be silent longer.

### b) Benjamini-Hochberg FDR, q < 0.01: "is it a fluke?"

- Test 264 parameters and some will look significant **by pure chance**. At the usual p < 0.01,
  about 2–3 of 264 would pass even if nothing were real.
- The **false discovery rate (FDR)** correction adjusts for testing so many at once. q < 0.01 means
  *at most about 1% of the parameters we keep are expected to be flukes*.
- Effect here: 114 parameters passed a plain p < 0.01; **98** passed after the correction.

### c) Effect size, |AUC − 0.5| ≥ 0.05: "is the difference big enough to matter?"

- With thousands of customers, even tiny differences become "statistically significant". This
  check requires a **real-world difference**: AUC at or below 0.45, or at or above 0.55.
- Example: DHCP templates had q = 0.008 (significant) but AUC 0.45, a barely-there difference.
  **Dropped.**
- Effect here: 98 → **71**.

### d) Era drift, p < 0.01 and |ρ| ≥ 0.15: "is it really about the date?"

- Churners were measured between 2023 and 2026; retained customers only in 2026. If a parameter
  **changes with the calendar** (for example, newer NIOS versions report more of something), it
  would separate the groups by **when** they were measured, not by **who** churned.
- The test: among churners only, check whether the parameter rises or falls with the measurement
  date (Spearman correlation **ρ**, explained below).
- Example: *enabled features per member* rose steadily with the date (ρ = +0.30). It looked
  predictive, but partly because churners were measured earlier. **Dropped.** So were tenure,
  HTTP certificates, share of virtual appliances, two geography flags and one zone count.
- Effect here: 71 → **64**.

**An analogy.** Compare phones of people who left a gym in 2023–2025 with phones of current members
checked in 2026. The members' phones have more storage. That isn't because members are different
people; phones just got bigger over those years. A model would wrongly learn "big phone storage →
stays at the gym". Era drift is the same trap.

**The cases caught in this project:**

| Parameter | Why it changes with the date | Drift (ρ) | Churned vs retained: all churners → churners measured in 2026 only |
|---|---|---:|---|
| Enabled features per member | newer NIOS versions report more feature entries, and churners were measured on older software | +0.30 | 0.26 → 0.39: the gap shrinks a lot |
| HTTP certificates | how certificates are recorded changed over time | −0.24 | 0.63 → 0.55 |
| Customer tenure at scoring date | grows automatically: the same customer has more years in 2026 than in 2024 | +0.29 | 0.38 → 0.46 |

How to read the table:
- **ρ** is how strongly the parameter rises (+) or falls (−) with the measurement date, among
  churners only. 0 means no drift.
- **The last column** uses AUC, where 0.5 means no difference. Comparing only churners measured in
  2026 (the same year as retained customers) shrinks each gap, which shows part of the difference
  was the date rather than churn.

**None of the final model's parameters drift:** silence, DTC adoption and employee count all have
|ρ| ≤ 0.04.

---

## Step 2: Independent, Spearman |ρ| ≥ 0.7 (64 → 27)

- **Spearman correlation (ρ)** measures how strongly two parameters move together, from −1 to +1:
  - **0** means unrelated;
  - **±1** means they always move together.

  Like Mann-Whitney, it uses ranks, so outliers can't distort it.
- Many HEKA fields **measure the same thing twice**. Keeping both adds no information and makes the
  model unstable, because it can't tell which of the twins deserves the credit.
- The pipeline groups parameters with |ρ| ≥ 0.7 into clusters and **keeps only the strongest one
  in each**.

**Real examples:**

| Kept | Dropped as near-duplicates |
|---|---|
| % of members on DTC (grid view) | % of members on DTC (member view), ρ = 0.98 with the kept one; share licensed for RPZ |
| Daily Reporting volume | Reporting licence capacity, and five per-category Reporting volumes (DNS, DHCP, IPAM, …) |

---

## Step 3: Adds significant accuracy (27 → 5)

### Forward selection: build the model one parameter at a time

1. Start with **no parameters**.
2. Try each remaining parameter, and add the one that improves accuracy the most.
3. **Only keep it if the improvement passes a significance test** (below).
4. Repeat until the next-best parameter fails the test.

| Step | Parameter added | Accuracy (AUC) | Gain | p-value | Result |
|---:|---|---:|---:|---:|---|
| 1 | % of members on DTC | 0.694 | +0.194 | < 0.000001 | **added** |
| 2 | Daily Reporting volume | 0.785 | +0.091 | < 0.000001 | **added** |
| 3 | Employee count | 0.831 | +0.046 | 0.000007 | **added** |
| 4 | Virtual-node management IPs | 0.842 | +0.011 | 0.001 | **added** |
| 5 | Days silent beyond a week | 0.865 | +0.023 | 0.002 | **added** |
| 6 | Daily DNS queries | 0.869 | +0.004 | 0.09 | **rejected, stop** |

Daily DNS queries is strongly related to churn on its own. But once DTC, Reporting, employees and
silence are known, it adds almost nothing new, so it's left out.

### Logistic regression

The model used to score each candidate set. It's the same kind of model as the final one: each
parameter gets one weight, the weights are summed, and the sum is turned into a probability.

### 5×5 repeated cross-validation: "how accurate is it on customers it hasn't seen?"

1. Split the development customers into **5 groups**.
2. Train on 4 groups, then test on the 5th. Rotate so each group is tested once. That gives
   **5 accuracy scores**.
3. Reshuffle the customers into new groups and do it again, **5 times in all**.
4. Result: **25 accuracy scores** per candidate, instead of one lucky or unlucky split.

### Nadeau-Bengio corrected t-test: "is the gain real?"

- Each candidate's 25 scores are compared with the current model's 25 scores: is the new
  parameter's gain bigger than random variation?
- A normal t-test would be **too generous**, because the 25 scores aren't independent: the
  training groups overlap heavily, so the scores move together.
- The **Nadeau-Bengio correction** accounts for that overlap and widens the uncertainty, so a gain
  has to be genuinely large to count.
- **p < 0.01** means there's less than a 1% chance of seeing a gain this big if the parameter
  really added nothing.

---

## What happened after these steps

The 5 parameters above came from **one** split of the data. The whole pipeline was then repeated on
**25 different random splits**, to see which parameters keep being chosen:

| Parameter | Chosen in |
|---|---:|
| DNS Traffic Control adoption | 96% of runs |
| Silence | 80% |
| Employee count | 76% |
| Reporting volume | 36% |
| Virtual-node management IPs | rarely |

The three chosen in most runs form the **recommended model**. The other two were real on this one
split, but not reliable across splits.
