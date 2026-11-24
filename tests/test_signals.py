"""Unit tests for the Taxonomy churn model on synthetic grids (no customer data needed).

    .venv/bin/python -m pytest -q        or        .venv/bin/python -m unittest discover tests
"""
import unittest

import numpy as np
import pandas as pd

from churn import metrics, signals

T = pd.Timestamp("2026-07-05", tz="UTC")


def grid(members, objects, queries, days_apart=7, end=T, opt_in=True):
    n = len(members)
    times = [end - pd.Timedelta(days=days_apart * (n - 1 - i)) for i in range(n)]
    return pd.DataFrame({"collected_at": times, "member_count": members, "objects": objects,
                         "total_queries": queries, "opt_in": [opt_in] * n})


class TaxonomySignals(unittest.TestCase):
    def test_weights_and_threshold_match_taxonomy(self):
        self.assertEqual(signals.WEIGHTS, {"traffic": 25, "shrink": 25, "quiet": 8, "dark": 8})
        self.assertEqual(signals.CHURN_THRESHOLD, 35)

    def test_healthy_grid_scores_zero(self):
        g = grid([10] * 10, np.linspace(1000, 1200, 10), [50_000] * 10)
        r = signals.score_grid(g, T)
        self.assertEqual(r["score"], 0)
        self.assertEqual(r["flag"], 0)

    def test_traffic_red_and_shrink_red_flags(self):
        g = grid(list(np.linspace(10, 6, 10)), list(np.linspace(1000, 400, 10)), [50_000] * 10)
        r = signals.score_grid(g, T)
        self.assertEqual(r["traffic_verdict"], "red")   # objects -60%
        self.assertEqual(r["shrink_verdict"], "red")    # members -40%
        self.assertEqual(r["score"], 50)
        self.assertEqual(r["flag"], 1)

    def test_queries_fall_to_zero_is_red(self):
        g = grid([5] * 10, np.linspace(1000, 1100, 10), [80_000] * 9 + [0])
        self.assertEqual(signals.score_grid(g, T)["traffic_verdict"], "red")

    def test_query_floor_makes_query_part_na(self):
        g = grid([5] * 10, [1000] * 10, [900] * 9 + [0])          # peak < 1,000
        r = signals.score_grid(g, T)
        self.assertNotEqual(r["traffic_verdict"], "red")

    def test_shrink_amber_band(self):
        g = grid([10] * 9 + [8.5], np.linspace(1000, 1100, 10), [50_000] * 10)  # -15%
        self.assertEqual(signals.score_grid(g, T)["shrink_verdict"], "amber")

    def test_going_quiet(self):
        g = grid([5] * 10, [1000] * 9 + [1010], [50_000] * 10)   # +1% objects
        self.assertEqual(signals.score_grid(g, T)["quiet_verdict"], "red")

    def test_gone_dark_bands(self):
        for silent, want in ((3, "green"), (15, "amber"), (30, "red")):
            g = grid([5] * 10, np.linspace(1000, 1100, 10), [50_000] * 10, end=T - pd.Timedelta(days=silent))
            self.assertEqual(signals.score_grid(g, T)["dark_verdict"], want, silent)

    def test_opted_out_is_dark_red(self):
        g = grid([5] * 10, np.linspace(1000, 1100, 10), [50_000] * 10, opt_in=False)
        self.assertEqual(signals.score_grid(g, T)["dark_verdict"], "red")

    def test_single_snapshot_is_na_not_green(self):
        g = grid([5], [1000], [50_000])
        r = signals.score_grid(g, T)
        for s in ("traffic", "shrink", "quiet"):
            self.assertEqual(r[f"{s}_verdict"], "na")
        self.assertEqual(r["score"], 0)

    def test_no_lookahead(self):
        g = grid([10] * 10, np.linspace(1000, 1100, 10), [50_000] * 10)
        future = grid([1], [1], [0], end=T + pd.Timedelta(days=7))
        self.assertEqual(signals.score_grid(pd.concat([g, future]), T), signals.score_grid(g, T))

    def test_one_high_plus_one_low_does_not_flag(self):
        self.assertLess(signals.WEIGHTS["shrink"] + signals.WEIGHTS["dark"], signals.CHURN_THRESHOLD)


class Metrics(unittest.TestCase):
    def test_summary(self):
        m = metrics.summary([1, 1, 0, 0], [1, 0, 0, 1], [0.9, 0.2, 0.1, 0.8])
        self.assertEqual((m["tp"], m["fn"], m["tn"], m["fp"]), (1, 1, 1, 1))
        self.assertAlmostEqual(m["accuracy"], 0.5)
        self.assertAlmostEqual(m["recall"], 0.5)


if __name__ == "__main__":
    unittest.main()
