# Churn Model: Summary

## What we did
We tested the current Taxonomy churn score on real churned customers from Anand's HEKA data.
We scored each one 3 months before it churned and compared it with the live fleet as
"retained" customers. Then we built better models on the same data.

## The current model
**Score = 25·Traffic falling off + 25·Shrinking deployment + 8·Going quiet + 8·Gone dark.**
A customer is flagged at 35 or more.

In practice, a customer is only flagged when traffic is falling **and** members are being
removed at the same time.

## Results

| | Current model | Best new model (gradient boosting) |
|---|---:|---:|
| Churners caught | **1 of 378** | **163 of 378** |
| Flags that were real churners | 4% | **42%** |
| Churners found in the top 10% riskiest grids | 7% | **43%** |
| ROC-AUC (0.5 = guessing, 1.0 = perfect) | 0.52 | **0.84** |

The current model shows **90% "accuracy"**, but that number is misleading. Only about 10% of
customers churn, so flagging nobody also scores about 90%.

## What we learned
1. **Customers don't wind down before they leave.** Members, objects and traffic barely change in
   the year before churn, and about half are still reporting after the contract ends. Churn is a
   business decision, not a technical shutdown.
2. **Going silent is the best warning sign.** About a third of churners stop reporting for 21+
   days in the months before churn, rising to 41% by the churn date. Among retained customers,
   12% trip the "Gone dark" signal.
3. **Who the customer is matters most.** Churners are thin deployments:
   - no DNS Traffic Control;
   - low reporting use;
   - about 3,000 daily queries vs 1.5 million for retained customers;
   - fewer records, admins and dashboards.

   Infoblox isn't deeply embedded for them, so it's easy to replace.
4. **Two current signals work backwards.** "Traffic falling off" and "Going quiet" fire *more*
   often on customers who stayed.

## What the best model relies on

| Family | Importance (drop in ROC-AUC when removed) |
|---|---:|
| Customer profile: how much of the product they use | **0.18**, the biggest |
| Silence: days since last report | **0.08** |
| Trends: changes over 2 months | 0.02, almost nothing |
| Current Taxonomy signals | 0.00 |

Top 5 parameters:

| # | Parameter | Churners have… |
|---:|---|---|
| 1 | Days since last report | longer silences |
| 2 | DNS Traffic Control members % | 0% (retained: 10%) |
| 3 | Reporting use vs allowance | 4% (retained: 10%) |
| 4 | Change in daily DNS queries | no meaningful difference |
| 5 | Daily DNS queries | about 3,000 (retained: about 1.5 million) |

## What to look at next
**Using HEKA data we already have:**
1. **Signs of moving away from Infoblox:** DNS zones showing up in AWS or Azure, more
   forwarding or secondary zones, or Microsoft DNS taking over.
2. **Paid features used vs paid for:** customers paying for features they don't use.
3. **Licence events:** temporary licences, licences about to expire, licences being dropped.
4. **Old hardware or no upgrades for a long time:** the hardware-refresh decision is a natural
   moment to leave.
5. **Reporting patterns:** intermittent reporting or opt-outs.

**Needing other data:**
6. **Salesforce / Gainsight:** days to renewal, falling ARR, support cases, NPS.
7. **Moves to Infoblox cloud:** some "churners" may have moved to BloxOne / Universal DDI, which
   isn't a real loss.
8. **Admin logins and API use:** true engagement, not collected today.

**Test first:** #1 and #7, which are possible today with data we already have.

## Caveats
- The "retained" group is a stand-in. It may include some churners, which would make the new
  model look *worse* than it really is.
- **Ask Anand** for the same data on customers who renewed, with account IDs. Rerunning
  `python -m churn.run` then gives the final accuracy.

Full detail: `docs/FINDINGS.md` · Model formula: `docs/MODEL.md` · All parameters:
`reports/GB_FEATURES.md`
