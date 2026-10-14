import unittest
import json
import tempfile
from pathlib import Path

import numpy as np

from src.data import build_industry_prior, load_data, normalize_industry
from src.dualpath import make_dual_path_estimator
from src.evaluation import eval_ic, eval_modularity
from src.network import topk_adj, weighted_topk_adj
from src.portfolio import eval_portfolio_metrics, min_var_weights
from src.utils import cov_to_corr, ensure_psd


class CoreValidationTests(unittest.TestCase):
    def test_unknown_industry_is_not_connected(self):
        stocks = ["000001", "000002", "000003"]
        prior = build_industry_prior(
            stocks,
            {"000001": "Bank", "000002": "Bank", "000003": None},
        )
        np.testing.assert_array_equal(
            prior,
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        )
        self.assertIsNone(normalize_industry("Unknown"))
        self.assertEqual(normalize_industry(" Bank "), "Bank")

        adj = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        # Unknown labels are excluded from IC and remain singleton communities
        # for modularity instead of being grouped into a fake industry.
        self.assertEqual(eval_ic(adj, stocks, {
            "000001": "Bank", "000002": "Bank", "000003": None,
        }), 1.0)
        self.assertTrue(np.isfinite(eval_modularity(adj, stocks, {
            "000001": "Bank", "000002": "Bank", "000003": None,
        })))

    def test_topk_zero_and_invalid_inputs(self):
        corr = np.eye(3)
        np.testing.assert_array_equal(topk_adj(corr, 0), np.zeros((3, 3)))
        np.testing.assert_array_equal(
            weighted_topk_adj(corr, 0), np.zeros((3, 3))
        )
        with self.assertRaises(ValueError):
            topk_adj(corr, 3)
        with self.assertRaises(ValueError):
            weighted_topk_adj(np.array([[1.0, np.nan], [np.nan, 1.0]]), 1)

    def test_covariance_helpers_validate_and_repair(self):
        cov = np.array([[1.0, 2.0], [0.0, 1.0]])
        repaired = ensure_psd(cov)
        self.assertGreaterEqual(np.linalg.eigvalsh(repaired).min(), 1e-6 - 1e-10)
        corr = cov_to_corr(np.eye(2))
        np.testing.assert_allclose(corr, np.eye(2))

    def test_optimizer_rejects_infeasible_cap(self):
        with self.assertRaises(ValueError):
            min_var_weights(np.eye(3), max_weight=0.05)

    def test_portfolio_metrics_reject_bad_returns(self):
        with self.assertRaises(ValueError):
            eval_portfolio_metrics([0.01, np.nan])
        with self.assertRaises(ValueError):
            eval_portfolio_metrics([-1.0, 0.0])
        with self.assertRaises(ValueError):
            eval_portfolio_metrics([0.01, 0.0])
        metrics = eval_portfolio_metrics(np.full(20, 0.001))
        self.assertTrue(np.isfinite(metrics["sharpe"]))

    def test_dualpath_marks_prior_assisted_scores(self):
        prior = build_industry_prior(
            ["000001", "000002"],
            {"000001": "Bank", "000002": "Bank"},
        )

        def base(history):
            return np.eye(2), np.eye(2)

        assisted = make_dual_path_estimator(base, prior, cp=0.5)
        independent = make_dual_path_estimator(base, prior, cp=0.0)
        self.assertTrue(assisted.uses_industry_prior)
        self.assertFalse(independent.uses_industry_prior)

    def test_historical_loader_rejects_current_universe_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "returns_clean.csv").write_text("date,000001\n2020-01-01,0\n")
            (root / "industry_info.csv").write_text("code,industry\n000001,Bank\n")
            (root / "data_manifest.json").write_text(json.dumps({
                "universe_source": "provider_current_snapshot"
            }))
            with self.assertRaises(ValueError):
                load_data(root, require_point_in_time=True)


if __name__ == "__main__":
    unittest.main()
