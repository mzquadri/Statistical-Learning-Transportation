"""Tests for the committed artifacts and the invariants they have to satisfy.

These are not tests of the coursework's conclusions, which were graded on their
own terms. They check the things that would silently break if a file were
edited by hand or an artifact regenerated in isolation: that the result tables
are internally consistent, that a probability is a probability, that the counts
in one table agree with the counts in another, and that the verification script
can still fail.
"""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
A1 = ROOT / "Assignment_1"
A2 = ROOT / "Assignment_2"
A3 = ROOT / "Assignment_3" / "solution"


class Layout(unittest.TestCase):
    """The files each README promises are the files that are here."""

    def test_every_assignment_has_a_readme_and_requirements(self):
        for folder in (A1, A2, A3):
            with self.subTest(folder=folder.name):
                self.assertTrue((folder / "README.md").is_file())
                self.assertTrue((folder / "requirements.txt").is_file())

    def test_every_assignment_has_an_executed_notebook(self):
        for folder in (A1, A2, A3):
            found = list(folder.glob("*.ipynb"))
            with self.subTest(folder=folder.name):
                self.assertEqual(len(found), 1)

    def test_the_supplied_mlp_skeleton_is_kept_apart_from_the_completed_one(self):
        """Starter code and the work done on it must stay distinguishable."""
        supplied = (ROOT / "Assignment_3" / "mlp.py").read_text(encoding="utf-8")
        completed = (A3 / "mlp.py").read_text(encoding="utf-8")
        self.assertNotEqual(supplied, completed)
        # The skeleton ships five gaps. The completed file keeps the marker
        # comments around them, so which lines were written and which were
        # supplied stays visible in the file rather than only in the README.
        self.assertEqual(supplied.count("### YOUR CODE STARTS HERE ###"), 5)
        self.assertEqual(completed.count("### YOUR CODE STARTS HERE ###"), 5)
        # No gap is left unfilled: the skeleton's placeholders are all gone.
        self.assertIn("return ...", supplied)
        self.assertNotIn("return ...", completed)
        self.assertNotIn("self.W2 = ...", completed)

    def test_the_completed_mlp_says_what_was_changed(self):
        completed = (A3 / "mlp.py").read_text(encoding="utf-8")
        self.assertIn("supplied", completed.lower())
        self.assertIn("alpha", completed)

    def test_the_dataset_is_not_committed_twice(self):
        copies = list(ROOT.rglob("nyc-yellow-may2oct.csv"))
        self.assertEqual(len(copies), 1, f"duplicated: {copies}")


class Notebooks(unittest.TestCase):
    """Every notebook is stored executed, in order, with no error output."""

    def notebooks(self):
        return sorted(p for p in ROOT.rglob("*.ipynb")
                      if ".ipynb_checkpoints" not in p.parts)

    def test_three_notebooks_are_present(self):
        self.assertEqual(len(self.notebooks()), 3)

    def test_execution_counts_run_from_one_without_gaps(self):
        """A gap means the stored outputs came from more than one session."""
        for path in self.notebooks():
            notebook = json.loads(path.read_text(encoding="utf-8"))
            code = [c for c in notebook["cells"] if c["cell_type"] == "code"]
            counts = [c.get("execution_count") for c in code]
            with self.subTest(notebook=path.name):
                self.assertEqual(counts, list(range(1, len(code) + 1)))

    def test_no_cell_stored_an_error(self):
        for path in self.notebooks():
            notebook = json.loads(path.read_text(encoding="utf-8"))
            errors = [i for i, c in enumerate(notebook["cells"])
                      if any(o.get("output_type") == "error"
                             for o in c.get("outputs", []))]
            with self.subTest(notebook=path.name):
                self.assertEqual(errors, [])

    def test_no_notebook_carries_an_absolute_path(self):
        import re
        pattern = re.compile(r"(?i)[A-Z]:[\\/]Users[\\/]|/home/[a-z_]+/|/Users/[a-z]")
        for path in self.notebooks():
            text = path.read_text(encoding="utf-8")
            with self.subTest(notebook=path.name):
                self.assertIsNone(pattern.search(text))


class Assignment2Tables(unittest.TestCase):
    """Internal consistency of the Problem Set 2 result tables."""

    @classmethod
    def setUpClass(cls):
        cls.results = A2 / "results"

    def table(self, name, **kwargs):
        return pd.read_csv(self.results / name, **kwargs)

    def test_all_twenty_two_tables_are_present(self):
        self.assertEqual(len(list(self.results.glob("*.csv"))), 22)

    def test_the_pca_eigenvalues_sum_to_the_number_of_variables(self):
        """Standardising with ddof=1 is what makes this exact rather than 5.001."""
        variance = self.table("p1_3_variance_explained.csv", index_col=0)
        self.assertAlmostEqual(variance["eigenvalue"].sum(), 5.0, places=9)

    def test_the_variance_proportions_are_a_distribution(self):
        variance = self.table("p1_3_variance_explained.csv", index_col=0)
        proportions = variance["proportion of variance"]
        self.assertTrue((proportions > 0).all())
        # The table stores six decimal places, so the column sums to 0.999999
        # rather than exactly 1. The eigenvalue column is the exact one.
        self.assertAlmostEqual(proportions.sum(), 1.0, places=5)

    def test_the_eigenvalues_are_in_descending_order(self):
        variance = self.table("p1_3_variance_explained.csv", index_col=0)
        values = list(variance["eigenvalue"])
        self.assertEqual(values, sorted(values, reverse=True))

    def test_the_cumulative_column_is_the_running_total(self):
        variance = self.table("p1_3_variance_explained.csv", index_col=0)
        running = variance["proportion of variance"].cumsum()
        self.assertTrue(np.allclose(running, variance["cumulative proportion"],
                                    atol=1e-6))

    def test_the_correlation_matrix_is_symmetric_with_a_unit_diagonal(self):
        matrix = self.table("p1_3_correlation_matrix.csv", index_col=0)
        values = matrix.to_numpy(float)
        self.assertTrue(np.allclose(values, values.T))
        self.assertTrue(np.allclose(np.diag(values), 1.0))
        self.assertTrue((np.abs(values) <= 1.0 + 1e-12).all())

    def test_the_predicted_probabilities_lie_in_the_unit_interval(self):
        probabilities = self.table("p1_2_predicted_probabilities.csv")
        numeric = probabilities.select_dtypes("number")
        for column in numeric.columns:
            if not column.lower().startswith(("p", "prob")):
                continue
            with self.subTest(column=column):
                self.assertTrue((numeric[column] >= 0).all())
                self.assertTrue((numeric[column] <= 1).all())

    def test_the_class_counts_add_up_to_the_sample(self):
        balance = self.table("p3_4_class_balance.csv")
        self.assertEqual(balance["count"].sum(), 1200)
        self.assertAlmostEqual(balance["share"].sum(), 1.0, places=3)

    def test_each_split_beats_a_zero_information_gain(self):
        """A split that gains nothing would not be worth reporting."""
        splits = self.table("p3_1_split_criteria.csv")
        self.assertTrue((splits["information gain"] > 0).all())
        self.assertTrue((splits["Gini reduction"] > 0).all())

    def test_the_weighted_impurity_is_the_child_average(self):
        splits = self.table("p3_1_split_criteria.csv")
        for _, row in splits.iterrows():
            total = row["n_left"] + row["n_right"]
            with self.subTest(split=row["split"]):
                self.assertAlmostEqual(
                    row["weighted H"],
                    (row["n_left"] * row["H(left)"]
                     + row["n_right"] * row["H(right)"]) / total, places=6)
                self.assertAlmostEqual(
                    row["weighted G"],
                    (row["n_left"] * row["G(left)"]
                     + row["n_right"] * row["G(right)"]) / total, places=6)

    def test_every_test_metric_is_a_proportion(self):
        metrics = self.table("p3_4_test_metrics.csv", index_col=0)
        for column in metrics.columns:
            with self.subTest(metric=column):
                self.assertTrue((metrics[column] >= 0).all())
                self.assertTrue((metrics[column] <= 1).all())

    def test_the_f1_column_is_the_harmonic_mean_of_the_other_two(self):
        metrics = self.table("p3_4_test_metrics.csv", index_col=0)
        for name, row in metrics.iterrows():
            if row["precision"] + row["recall"] == 0:
                continue
            with self.subTest(model=name):
                self.assertAlmostEqual(
                    row["F1-score"],
                    2 * row["precision"] * row["recall"]
                    / (row["precision"] + row["recall"]), places=3)

    def test_every_fitted_model_beats_the_majority_baseline_on_auc(self):
        metrics = self.table("p3_4_test_metrics.csv", index_col=0)
        baseline = metrics.loc["Baseline (majority class)", "ROC-AUC"]
        self.assertEqual(baseline, 0.5)
        fitted = metrics.drop(index="Baseline (majority class)")
        self.assertTrue((fitted["ROC-AUC"] > baseline).all())

    def test_lowering_the_threshold_trades_precision_for_recall(self):
        """The direction is a property of the trade-off, not of this data."""
        table = self.table("p3_4_threshold_final_test.csv")
        default = table.iloc[0]
        selected = table.iloc[1]
        self.assertGreater(selected["recall"], default["recall"])
        self.assertLess(selected["precision"], default["precision"])
        self.assertGreater(selected["false positives"], default["false positives"])
        self.assertLess(selected["false negatives"], default["false negatives"])

    def test_the_kernel_matrices_are_symmetric(self):
        for name in ("p2_6_rbf_kernel_matrix.csv",
                     "p2_6_feature_map_kernel_matrix.csv"):
            matrix = self.table(name, index_col=0).to_numpy(float)
            with self.subTest(matrix=name):
                self.assertTrue(np.allclose(matrix, matrix.T))

    def test_the_rbf_kernel_matrix_is_positive_semidefinite(self):
        """A valid kernel: the report claims its smallest eigenvalue is positive."""
        matrix = self.table("p2_6_rbf_kernel_matrix.csv", index_col=0).to_numpy(float)
        eigenvalues = np.linalg.eigvalsh(matrix)
        self.assertGreater(eigenvalues.min(), 0)

    def test_the_rbf_kernel_has_a_unit_diagonal(self):
        matrix = self.table("p2_6_rbf_kernel_matrix.csv", index_col=0).to_numpy(float)
        self.assertTrue(np.allclose(np.diag(matrix), 1.0))


class Assignment3Registry(unittest.TestCase):
    """The registry that binds the Problem Set 3 report to its notebook."""

    @classmethod
    def setUpClass(cls):
        cls.registry = pd.read_csv(A3 / "results" / "reported_values.csv",
                                   dtype={"problem": str})

    def test_the_registry_holds_126_values(self):
        self.assertEqual(len(self.registry), 126)

    def test_every_metric_name_is_unique(self):
        self.assertEqual(self.registry["metric"].nunique(), len(self.registry))

    def test_every_value_is_finite(self):
        self.assertTrue(np.isfinite(self.registry["value"]).all())

    def test_every_problem_of_the_four_is_represented(self):
        top_level = {p.split(".")[0] for p in self.registry["problem"]}
        self.assertEqual(top_level, {"1", "2", "3", "4"})

    def test_the_cleaning_audit_accounts_for_every_raw_row(self):
        values = self.registry.set_index("metric")["value"]
        self.assertEqual(values["n_retained"],
                         values["n_raw"] - values["n_excluded_outside_period"])

    def test_the_gradient_checks_are_at_machine_precision(self):
        """Problem 1 rests on the hand-derived gradients being right."""
        values = self.registry.set_index("metric")["value"]
        self.assertLess(values["gradient_check_max_rel_error"], 1e-6)
        self.assertLess(values["l2_gradient_check_max_rel_error"], 1e-6)
        self.assertLess(values["autograd_max_abs_diff"], 1e-12)

    def test_true_convolution_is_the_negated_cross_correlation(self):
        """For the antisymmetric kernel of Problem 4.1 the flip is a sign change."""
        values = self.registry.set_index("metric")["value"]
        self.assertAlmostEqual(values["true_convolution_value"],
                               -values["cross_correlation_value"], places=12)

    def test_the_from_scratch_convolution_matches_the_reference(self):
        values = self.registry.set_index("metric")["value"]
        self.assertEqual(values["n_verification_cases"], 81)
        self.assertLess(values["max_diff_vs_reference"], 1e-12)

    def test_the_discounted_return_matches_its_step_table(self):
        steps = pd.read_csv(A3 / "results" / "p3_1_return.csv")
        values = self.registry.set_index("metric")["value"]
        rebuilt = float((steps["reward"] * 0.9 ** steps["step k"]).sum())
        self.assertAlmostEqual(values["discounted_return"], rebuilt, places=9)

    def test_the_policy_evaluation_table_is_exact_arithmetic(self):
        """Values are stored as fractions such as -4/3, not as rounded decimals."""
        table = pd.read_csv(A3 / "results" / "p3_3_policy_evaluation.csv")
        self.assertIn("v_1", table.columns)
        self.assertIn("v_2", table.columns)
        self.assertTrue(any("/" in str(v) for v in table["v_2"]))


class Verification(unittest.TestCase):
    """The verification script has to be able to fail, or it proves nothing."""

    def test_the_recorded_verification_passed_every_check(self):
        recorded = ROOT / "verification" / "verification.json"
        if not recorded.is_file():
            self.skipTest("run scripts/verify_results.py first")
        data = json.loads(recorded.read_text(encoding="utf-8"))
        self.assertEqual(data["failed"], [])
        self.assertEqual(data["totals"]["passed"], data["totals"]["checks"])
        self.assertGreater(data["totals"]["checks"], 40)

    def test_a_check_fails_when_the_reported_value_is_wrong(self):
        import sys
        sys.path.insert(0, str(ROOT / "scripts"))
        from verify_results import check
        good = check("x", "1", "claim", 1.0, 1.0, 1e-9, "test")
        bad = check("x", "1", "claim", 1.0, 2.0, 1e-9, "test")
        self.assertTrue(good.passed)
        self.assertFalse(bad.passed)

    def test_the_tolerance_is_relative_for_large_values(self):
        import sys
        sys.path.insert(0, str(ROOT / "scripts"))
        from verify_results import check
        self.assertTrue(check("x", "1", "c", 1e6, 1e6 + 1, 1e-5, "t").passed)
        self.assertFalse(check("x", "1", "c", 1e6, 1e6 + 1e4, 1e-5, "t").passed)

    def test_every_assignment_is_covered(self):
        recorded = ROOT / "verification" / "verification.json"
        if not recorded.is_file():
            self.skipTest("run scripts/verify_results.py first")
        data = json.loads(recorded.read_text(encoding="utf-8"))
        self.assertEqual(set(data["by_assignment"]),
                         {"Problem Set 1", "Problem Set 2", "Problem Set 3"})


class Assignment1Data(unittest.TestCase):
    """Problem Set 1 writes no tables, so the data is what there is to check."""

    def test_both_supplied_files_are_present_and_the_right_shape(self):
        three = pd.read_csv(A1 / "data_problem_3.csv")
        four = pd.read_csv(A1 / "data_problem_4.csv")
        self.assertEqual(len(three), 21)
        self.assertEqual(four.shape, (1816, 5))

    def test_the_regression_columns_are_present(self):
        three = pd.read_csv(A1 / "data_problem_3.csv")
        for column in ("cid", "rhp", "etw", "cmp", "axle", "n_v", "mpg"):
            with self.subTest(column=column):
                self.assertIn(column, three.columns)

    def test_the_freeway_columns_are_present_and_non_negative(self):
        four = pd.read_csv(A1 / "data_problem_4.csv")
        for column in ("flow", "occupancy", "speed", "speed_down", "flow_ratio"):
            with self.subTest(column=column):
                self.assertIn(column, four.columns)
                self.assertGreaterEqual(four[column].min(), 0)

    def test_the_two_speed_columns_are_strongly_correlated(self):
        """The claim Section 4.1 makes before using speed_down as a predictor."""
        four = pd.read_csv(A1 / "data_problem_4.csv")
        self.assertGreater(four["speed"].corr(four["speed_down"]), 0.8)

    def test_flow_and_occupancy_are_nearly_collinear(self):
        """Which is why Section 4.3 keeps one of them and drops the other."""
        four = pd.read_csv(A1 / "data_problem_4.csv")
        self.assertGreater(four["flow"].corr(four["occupancy"]), 0.99)


class Bootstrap(unittest.TestCase):
    """The closed forms Problem Set 2 quotes for bagging."""

    def test_the_out_of_bag_share_approaches_one_over_e(self):
        for n in (100, 1000, 5000, 100_000):
            with self.subTest(n=n):
                share = (1 - 1 / n) ** n
                self.assertLess(abs(share - 1 / math.e), 0.01)

    def test_the_reported_expectations_hold_at_n_equals_5000(self):
        n = 5000
        out_of_bag = n * (1 - 1 / n) ** n
        self.assertAlmostEqual(out_of_bag, 1839.21, places=1)
        self.assertAlmostEqual(n - out_of_bag, 3160.79, places=1)


if __name__ == "__main__":
    unittest.main()
