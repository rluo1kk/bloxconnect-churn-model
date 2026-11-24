# The current churn model (v0): BloxConnect Taxonomy, Churn tier

Source: `BloxConnect_Taxonomy_Core_Final.xlsx`, tab *Use cases – detailed*, Churn rows. The
weights come from the Importance → weight map the dashboard uses (`IMP_W`). The implementation
is `churn/signals.py`, and every rule below has a unit test in `tests/test_signals.py`.

## Mathematical expression

For a grid *g* scored on date *T*:

$$
S(g,T) \;=\; \sum_{i \in \{\text{traffic},\,\text{shrink},\,\text{quiet},\,\text{dark}\}} w_i \cdot \mathbb{1}\big[\, v_i(g,T) \in \{\text{amber},\text{red}\}\,\big]
$$

$$
\text{Churn flag}(g,T) \;=\; \mathbb{1}\big[\, S(g,T) \ge 35 \,\big], \qquad 0 \le S \le 66
$$

where $v_i(g,T)$ is signal *i*'s verdict (green / amber / red / n/a). It is computed from the
grid's own snapshots in the window $W = [T-60\text{d},\, T]$, and n/a contributes 0.

The trend signals use the first-to-last relative change inside the window:

$$
\Delta x \;=\; \frac{x_{\text{last}} - x_{\text{first}}}{x_{\text{first}}}
$$

## Weights and parameters

| # | Signal (Taxonomy sub-use-case) | Importance | Weight $w_i$ | HEKA fields | Amber | Red | n/a when |
|---|---|---|---:|---|---|---|---|
| 1 | **Traffic falling off** | HIGH | **25** | `total_queries` (24 h daily total), `db.objects_count` | Δ queries *or* Δ objects ≤ −30% (to −49%) | Δ ≤ −50%, or queries fall to 0 from a non-zero baseline | < 2 readings or < 4 weeks of span; the query part is n/a if peak daily total < 1,000 |
| 2 | **Shrinking deployment** | HIGH | **25** | `member_count` | Δ members −10% to −19% | Δ members ≤ −20% | < 2 readings / < 4 weeks |
| 3 | **Going quiet** (stagnant object count) | LOW-MEDIUM | **8** | `db.objects_count` | – | \|Δ objects\| < 2% | < 2 readings / < 4 weeks |
| 4 | **Gone dark** | LOW-MEDIUM | **8** | `collected_at`, `grid_opt_in` | silent 14–21 days | silent > 21 days, or opted out | grid never seen before T |

| Parameter | Value | Source |
|---|---|---|
| Importance → weight | HIGH 25 · MEDIUM-HIGH 18 · MEDIUM 12 · LOW-MEDIUM 8 · LOW 5 | dashboard `IMP_W` |
| Flag threshold | **35** | dashboard `CHURN_THRESHOLD` |
| Trend window | 2 months (60 days, plus 7 days of tolerance to catch the weekly baseline reading) | Taxonomy: "over the course of 2 months" |
| A signal "fires" | amber **or** red. Red-only is a one-line change (`fire=("red",)`) | "whenever a signal's threshold is crossed" |

## What the threshold actually means

The two LOW-MEDIUM signals can't both fire alongside the object-count rule:
- "Going quiet" needs objects flat, while the object part of "Traffic" needs objects down 30% or more.
- "Gone dark" means there is no recent data, so "Going quiet" can't be evaluated.

So in practice:

| Signals that fire | Score | Flagged? |
|---|---:|---|
| any single signal | 8 or 25 | no |
| one HIGH + one LOW-MEDIUM | 33 | **no** |
| Traffic + Shrinking | 50 | yes |
| one HIGH + Going quiet + Gone dark | 41 | yes, but almost impossible (see above) |

**The current model flags a grid essentially only when traffic is falling *and* members are being
removed at the same time.**

## Not included

- The dashboard also gives up to 10 points of partial credit for the share of members that are
  offline. That isn't in the Taxonomy, and member status isn't in the weekly grid export, so
  it's left out here.
- The "Shrinking deployment" escalation for a missing GMC only turns amber into red, so it
  can't change a fired / not-fired verdict.
- DHCP LPS is still a placeholder in the Taxonomy, with no data behind it.
