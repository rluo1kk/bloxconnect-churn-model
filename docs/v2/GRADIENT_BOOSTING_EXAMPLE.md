# Gradient boosting, a worked example with 8 customers

**The idea in one line:** start with a rough guess, measure how wrong it is for each customer, add a
small decision tree that corrects those mistakes, and repeat.

*All numbers below are calculated, not illustrative. The example is simplified: the real model
works on the log-odds scale, uses larger trees and runs 300–400 rounds with smaller steps.*

---

## The customers

| Customer | Days silent (beyond a normal week) | % of members on DTC | Actually churned? |
|---|---:|---:|:---:|
| A | 30 | 0 | **yes** |
| B | 25 | 0 | **yes** |
| C | 0 | 0 | **yes** |
| D | 0 | 0 | no |
| E | 0 | 60 | no |
| F | 21 | 70 | no |
| G | 0 | 50 | no |
| H | 0 | 0 | no |

3 of the 8 customers churned.

---

## Step 0: start with the simplest guess

Give everyone the average churn rate, **3 ÷ 8 = 0.375**, so a 37.5% predicted risk for every
customer.

## Step 1: measure the mistakes (the "residuals")

**Mistake = what actually happened − what we predicted.** Churned counts as 1, retained as 0.

| Customer | Actual | Predicted | Mistake |
|---|---:|---:|---:|
| A, B, C (churned) | 1 | 0.375 | **+0.625**, too low |
| D, E, F, G, H (stayed) | 0 | 0.375 | **−0.375**, too high |

## Step 2: tree 1, find the question that best sorts the mistakes

The model tries possible yes/no questions and keeps the one that best separates the "too low"
mistakes from the "too high" ones:

<div class="tree"><b>Tree 1: "Is any member on DTC?"</b><br>
YES (E, F, G): average mistake <b>−0.375</b>, so these predictions are too high<br>
NO (A, B, C, D, H): average mistake <b>+0.225</b>, so these predictions are too low</div>

## Step 3: take a small step in that direction

**New prediction = old prediction + learning rate × the tree's correction.** The learning rate here
is 0.5, which means the model only moves halfway, so no single tree can overreact.

| Group | Calculation | New prediction |
|---|---|---:|
| DTC users (E, F, G) | 0.375 + 0.5 × (−0.375) | **0.188** |
| Non-DTC (A, B, C, D, H) | 0.375 + 0.5 × (+0.225) | **0.488** |

## Step 4: repeat, and tree 2 picks up what tree 1 missed

New mistakes: A, B and C are still too low (+0.512), while D and H are now too high (−0.488). Tree 1
can't separate these, because none of them use DTC. So tree 2 asks a different question:

<div class="tree"><b>Tree 2: "Silent for more than 14 days?"</b><br>
YES (A, B, F): average mistake <b>+0.279</b>, push up<br>
NO (C, D, E, G, H): average mistake <b>−0.168</b>, push down</div>

| Customer | After tree 1 | Tree 2's step | After tree 2 |
|---|---:|---:|---:|
| A, B (silent, no DTC) | 0.488 | +0.140 | **0.627** |
| C, D, H (reporting, no DTC) | 0.488 | −0.084 | **0.404** |
| F (silent, but uses DTC) | 0.188 | +0.140 | **0.327** |
| E, G (reporting, use DTC) | 0.188 | −0.084 | **0.104** |

## After 3 rounds

Tree 3 asks about DTC again, refining what's left over.

| Customer | Churned? | Start | Round 1 | Round 2 | **Round 3** |
|---|:---:|---:|---:|---:|---:|
| A | yes | 0.375 | 0.488 | 0.627 | **0.681** |
| B | yes | 0.375 | 0.488 | 0.627 | **0.681** |
| C | yes | 0.375 | 0.488 | 0.404 | **0.457** |
| D | no | 0.375 | 0.488 | 0.404 | **0.457** |
| H | no | 0.375 | 0.488 | 0.404 | **0.457** |
| F | no | 0.375 | 0.188 | 0.327 | **0.238** |
| E | no | 0.375 | 0.188 | 0.104 | **0.015** |
| G | no | 0.375 | 0.188 | 0.104 | **0.015** |

**What happened:**
- **Silent customers without DTC** (A, B) rose to the highest risk.
- **DTC users who keep reporting** (E, G) fell to almost zero.
- **F** (silent, but uses DTC) landed in between. The trees *combined* the two signals, which is
  the "interaction" a points-based score can't capture.
- **C, D and H look identical** in this data (reporting, no DTC), so the model can't tell them
  apart. It gives all three the same middle risk. No model can beat the information it's given.

---

## Why it's called "gradient" boosting

Each tree is fitted to the **mistakes** (actual − predicted). Those mistakes are exactly the
**downhill direction of the error**, so every round is one step of **gradient descent**: walking
downhill on the error, one small tree at a time.

## Mapping the example to the real model

| In this example | In the real churn model |
|---|---|
| 8 customers | about 2,600 grids in the development set |
| 2 parameters | up to 64 screened parameters |
| 1 question per tree = 2 leaves | up to 14 questions per tree = up to 15 leaves (at least 20 grids in each) |
| 3 rounds | 300 rounds |
| learning rate 0.5 | learning rate 0.05 (smaller, more careful steps) |
| predicts probability directly | predicts log-odds, then converts to probability |

### What "15 leaves" means

A **leaf** is an end bucket of a tree: where a customer lands after answering its questions. Each
leaf holds one correction ("push risk up or down by this much").

- In the example above, each tree asked **one question**, so it had **2 leaves** (yes / no).
- In the real model, a tree can ask **follow-up questions**, up to 14 of them, so customers can be
  split into **as many as 15 buckets**, each with its own correction.

<div class="tree"><b>A 4-leaf tree</b><br>
Silent more than 14 days?<br>
&nbsp;&nbsp;├─ yes → Any members on DTC?<br>
&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ yes → <b>leaf 1</b>: push risk up a little<br>
&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└─ no&nbsp;&nbsp;→ <b>leaf 2</b>: push risk up a lot<br>
&nbsp;&nbsp;└─ no&nbsp;&nbsp;→ Fewer than 500 employees?<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ yes → <b>leaf 3</b>: push risk up slightly<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└─ no&nbsp;&nbsp;→ <b>leaf 4</b>: push risk down</div>

**Why more leaves help:** one tree can capture **combinations**, such as "silent *and* no DTC",
instead of one factor at a time.

**Why cap it at 15:**
- With unlimited leaves, a tree could make a bucket for almost every customer and **memorise**
  them instead of learning patterns. That's overfitting.
- A second rule backs this up: every leaf must contain **at least 20 grids**.

**Why we didn't pick it in the end:** with this much data, all those trees weren't significantly
more accurate than the simple 3-parameter logistic model (average AUC 0.859 vs 0.842). The logistic
model can be explained in one sentence per customer.
