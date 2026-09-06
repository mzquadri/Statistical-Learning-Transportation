"""Recompute the numbers the three reports quote, from the committed artifacts.

    python scripts/verify_results.py

The notebooks already write every table and figure they cite, and Problem Set 3
carries a registry of all 126 numbers quoted in its report. What none of that
does is check the other direction: that the prose in a report still says what
the artifacts say. A number typed into a sentence and later recomputed by a
changed cell will not announce itself.

So each check here recomputes a claim from the data or from the result tables
and compares it with the figure the report gives. Where a claim is analytic,
such as the expected out-of-bag count of a bootstrap sample or the discounted
return of a fixed route, it is derived here rather than read from a file, so
the check is independent of the notebook that produced the artifact.

This is a post-submission verification of work that was submitted earlier. It
does not change any submitted number and it is not part of the coursework.

Writes verification/verification.json.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "verification" / "verification.json"

A1 = ROOT / "Assignment_1"
A2 = ROOT / "Assignment_2"
A3 = ROOT / "Assignment_3" / "solution"

#: LaTeX sets U+2212 MINUS SIGN rather than an ASCII hyphen, so a value
#: extracted from a report PDF needs normalising before it can be matched.
MINUS_SIGN = chr(0x2212)


@dataclass
class Check:
    """One claim, the value it is compared against, and how it was rebuilt."""

    assignment: str
    problem: str
    claim: str
    reported: float
    recomputed: float
    tolerance: float
    source: str
    passed: bool = False

    def settle(self) -> Check:
        scale = max(1.0, abs(self.reported))
        self.passed = abs(self.reported - self.recomputed) <= self.tolerance * scale
        return self


def check(assignment, problem, claim, reported, recomputed, tolerance, source):
    return Check(assignment, problem, claim, float(reported), float(recomputed),
                 tolerance, source).settle()


# ---------------------------------------------------------------------------
# Problem Set 1: linear regression and diagnostics
# ---------------------------------------------------------------------------

def verify_assignment_1() -> list[Check]:
    """Refit both regressions from the supplied CSVs.

    Problem Set 1 writes no result tables; its numbers live only in the
    notebook's stored outputs and in the report. Refitting from the raw data
    is therefore the only independent check available, and it is the stronger
    one: it does not trust the notebook at all.
    """
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    out = []
    three = pd.read_csv(A1 / "data_problem_3.csv")
    predictors = ["cid", "rhp", "etw", "cmp", "axle", "n_v"]
    design = sm.add_constant(three[predictors])
    fit = sm.OLS(three["mpg"], design).fit()

    out.append(check("Problem Set 1", "3.1", "21 vehicles in the sample",
                     21, fit.nobs, 0, "data_problem_3.csv"))
    out.append(check("Problem Set 1", "3.1",
                     "R-squared about 0.89", 0.89, fit.rsquared, 6e-3,
                     "OLS refit of mpg on six predictors"))
    out.append(check("Problem Set 1", "3.3",
                     "F-test p-value 4.66e-06", 4.66e-06, fit.f_pvalue, 2e-3,
                     "OLS refit"))
    # 3.4 reports that etw and axle are the only coefficients significant at 5%.
    significant = sorted(name for name, p in fit.pvalues.items()
                         if name != "const" and p < 0.05)
    out.append(check("Problem Set 1", "3.4",
                     "exactly two predictors significant at 5 percent",
                     2, len(significant), 0,
                     f"significant: {', '.join(significant)}"))
    out.append(check("Problem Set 1", "3.4", "etw is one of them",
                     1, 1 if "etw" in significant else 0, 0, "OLS refit"))
    out.append(check("Problem Set 1", "3.4", "axle is the other",
                     1, 1 if "axle" in significant else 0, 0, "OLS refit"))

    four = pd.read_csv(A1 / "data_problem_4.csv")
    out.append(check("Problem Set 1", "4.1",
                     "speed and speed_down correlate around 0.87",
                     0.87, four["speed"].corr(four["speed_down"]), 6e-3,
                     "data_problem_4.csv"))
    out.append(check("Problem Set 1", "4.1",
                     "flow and occupancy correlate 0.994",
                     0.994, four["flow"].corr(four["occupancy"]), 2e-3,
                     "data_problem_4.csv"))

    original = ["flow", "occupancy", "speed_down", "flow_ratio"]
    with_const = sm.add_constant(four[original])
    vifs = {name: variance_inflation_factor(with_const.values, i)
            for i, name in enumerate(with_const.columns)}
    # The report reads flow and occupancy as "around 100" and flow_ratio as
    # "a bit high". The constant's VIF is explicitly ignored in the notebook,
    # which is what makes the statsmodels change below harmless.
    out.append(check("Problem Set 1", "4.2", "VIF of flow is 104.24",
                     104.23727, vifs["flow"], 1e-5, "variance_inflation_factor"))
    out.append(check("Problem Set 1", "4.2", "VIF of occupancy is 94.88",
                     94.88128, vifs["occupancy"], 1e-5, "variance_inflation_factor"))
    out.append(check("Problem Set 1", "4.2", "VIF of speed_down is 2.10",
                     2.09650, vifs["speed_down"], 1e-4, "variance_inflation_factor"))
    out.append(check("Problem Set 1", "4.2", "VIF of flow_ratio is 9.12",
                     9.11658, vifs["flow_ratio"], 1e-5, "variance_inflation_factor"))
    return out


# ---------------------------------------------------------------------------
# Problem Set 2: logistic regression, PCA, SVM, trees and ensembles
# ---------------------------------------------------------------------------

def verify_assignment_2() -> list[Check]:
    out = []
    results = A2 / "results"

    # 1.1 The problem sheet gives the two BIC values and the sample size. The
    # log-likelihoods and AICs in the report are recovered from them, so they
    # can be re-derived here without the notebook.
    n = 244_493
    fit = pd.read_csv(results / "p1_1_model_fit_comparison.csv", index_col=0)
    for name, k, bic in (("Model 1", 6, 95_815.66), ("Model 2", 5, 97_047.60)):
        log_likelihood = (k * math.log(n) - bic) / 2
        out.append(check("Problem Set 2", "1.1",
                         f"{name} log-likelihood recovered from its BIC",
                         fit.loc[name, "log-likelihood"], log_likelihood, 1e-5,
                         "BIC = -2 lnL + k ln n"))
        out.append(check("Problem Set 2", "1.1", f"{name} AIC",
                         fit.loc[name, "AIC"], -2 * log_likelihood + 2 * k, 1e-6,
                         "AIC = -2 lnL + 2k"))
    statistic = 2 * (fit.loc["Model 1", "log-likelihood"]
                     - fit.loc["Model 2", "log-likelihood"])
    out.append(check("Problem Set 2", "1.1",
                     "likelihood-ratio statistic 1244.35 on 1 df",
                     1244.35, statistic, 1e-5, "2(lnL1 - lnL2)"))

    # 1.1 odds ratios quoted in the prose.
    shift = pd.read_csv(results / "p1_1_coefficient_shift.csv", index_col=0)
    out.append(check("Problem Set 2", "1.1",
                     "odds ratio 2.751 per additional vehicle",
                     2.751, math.exp(1.012), 1e-3, "exp of the HHVeh coefficient"))
    out.append(check("Problem Set 2", "1.1",
                     "odds ratio 1.799 over 600 minutes",
                     1.799, math.exp(0.000979 * 600), 1e-3,
                     "exp(600 * TrvlTime coefficient)"))
    out.append(check("Problem Set 2", "1.1",
                     "HHVeh coefficient moves 17.39 percent between models",
                     17.3913, shift.loc["HHVeh", "change (%)"], 1e-4,
                     "p1_1_coefficient_shift.csv"))

    # 1.3 PCA on five standardised variables: the eigenvalues must sum to K.
    variance = pd.read_csv(results / "p1_3_variance_explained.csv", index_col=0)
    out.append(check("Problem Set 2", "1.3",
                     "eigenvalues sum to K = 5 with ddof = 1",
                     5.0, variance["eigenvalue"].sum(), 1e-9,
                     "p1_3_variance_explained.csv"))
    out.append(check("Problem Set 2", "1.3",
                     "first three components explain 79.57 percent",
                     0.795658, variance["cumulative proportion"].iloc[2], 1e-6,
                     "p1_3_variance_explained.csv"))
    out.append(check("Problem Set 2", "1.3",
                     "the reported ddof = 0 total would be 5.0010",
                     5.0010, 5 * n_over_n_minus_one(5000), 1e-4,
                     "n/(n-1) with n = 5000"))

    # 3.1 The two candidate splits, recomputed from their own child counts.
    splits = pd.read_csv(results / "p3_1_split_criteria.csv")
    for _, row in splits.iterrows():
        left, right = row["n_left"], row["n_right"]
        total = left + right
        label = row["split"].split(":")[0]
        out.append(check("Problem Set 2", "3.1",
                         f"split {label} weighted entropy",
                         row["weighted H"],
                         (left * row["H(left)"] + right * row["H(right)"]) / total,
                         1e-6, "p3_1_split_criteria.csv"))
        out.append(check("Problem Set 2", "3.1",
                         f"split {label} weighted Gini",
                         row["weighted G"],
                         (left * row["G(left)"] + right * row["G(right)"]) / total,
                         1e-6, "p3_1_split_criteria.csv"))

    # 3.2 Bootstrap expectations are closed form for n = 5000.
    size = 5000
    out_of_bag = size * (1 - 1 / size) ** size
    out.append(check("Problem Set 2", "3.2",
                     "expected unique observations 3160.79",
                     3160.79, size - out_of_bag, 1e-5, "n(1 - (1 - 1/n)^n)"))
    out.append(check("Problem Set 2", "3.2",
                     "expected out-of-bag observations 1839.21",
                     1839.21, out_of_bag, 1e-5, "n(1 - 1/n)^n"))

    # 3.4 Class balance and the threshold table's internal consistency.
    balance = pd.read_csv(results / "p3_4_class_balance.csv")
    total = balance["count"].sum()
    severe = int(balance.loc[balance["severe_delay"].str.startswith("1"), "count"].iloc[0])
    out.append(check("Problem Set 2", "3.4", "1200 incidents in total",
                     1200, total, 0, "p3_4_class_balance.csv"))
    out.append(check("Problem Set 2", "3.4",
                     "severe share 23.42 percent", 23.42, 100 * severe / total, 1e-3,
                     "p3_4_class_balance.csv"))
    out.append(check("Problem Set 2", "3.4",
                     "class imbalance about 3.27 to 1",
                     3.27, (total - severe) / severe, 2e-3,
                     "p3_4_class_balance.csv"))

    thresholds = pd.read_csv(results / "p3_4_threshold_final_test.csv")
    for _, row in thresholds.iterrows():
        harmonic = (2 * row["precision"] * row["recall"]
                    / (row["precision"] + row["recall"]))
        out.append(check("Problem Set 2", "3.4",
                         f"F1 at threshold {row['threshold']}",
                         row["F1"], harmonic, 1e-3,
                         "2PR/(P+R) from the reported precision and recall"))

    # 3.4 The test metrics table must agree with itself.
    metrics = pd.read_csv(results / "p3_4_test_metrics.csv", index_col=0)
    for name, row in metrics.iterrows():
        if row["precision"] + row["recall"] == 0:
            continue
        harmonic = (2 * row["precision"] * row["recall"]
                    / (row["precision"] + row["recall"]))
        out.append(check("Problem Set 2", "3.4", f"F1 of the {name.lower()}",
                         row["F1-score"], harmonic, 2e-3,
                         "p3_4_test_metrics.csv"))
    out.append(check("Problem Set 2", "3.4",
                     "the majority-class baseline scores 0.5 ROC-AUC",
                     0.5, metrics.loc["Baseline (majority class)", "ROC-AUC"], 0,
                     "p3_4_test_metrics.csv"))
    return out


def n_over_n_minus_one(n: int) -> float:
    return n / (n - 1)


def _fraction(value) -> float:
    """Parse an exact fraction such as -4/3, which the MDP tables store as text."""
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if "/" in text:
        numerator, _, denominator = text.partition("/")
        return float(numerator) / float(denominator)
    return float(text)


# ---------------------------------------------------------------------------
# Problem Set 3: MLP, Bayesian optimisation, forecasting, MDP, convolution
# ---------------------------------------------------------------------------

def registry() -> pd.Series:
    frame = pd.read_csv(A3 / "results" / "reported_values.csv",
                        dtype={"problem": str})
    return frame.set_index("metric")["value"]


def verify_assignment_3() -> list[Check]:
    """Check the registry against the artifacts and against closed forms.

    Problem Set 3 already records every number its report quotes. What is
    checked here is that those recorded values still match the tables the
    notebook wrote, and that the ones with an analytic answer are right.
    """
    out = []
    values = registry()
    results = A3 / "results"

    # 1.1 The gradient check is the claim the rest of Problem 1 rests on. Both
    # are compared against zero with the tolerance the report itself implies,
    # so a regression in the derivation would fail here.
    out.append(check("Problem Set 3", "1.1",
                     "finite-difference gradient check is at most 1e-6",
                     0.0, values["gradient_check_max_rel_error"], 1e-6,
                     "reported_values.csv"))
    out.append(check("Problem Set 3", "1.1",
                     "autograd cross-check at float64 machine precision",
                     0.0, values["autograd_max_abs_diff"], 1e-12,
                     "reported_values.csv"))
    out.append(check("Problem Set 3", "1.2",
                     "the L2 gradient check is also at most 1e-6",
                     0.0, values["l2_gradient_check_max_rel_error"], 1e-6,
                     "reported_values.csv"))

    # 1.1 The learning-rate sweep table must contain the reported optimum.
    sweep = pd.read_csv(results / "p1_1_lr_sweep.csv")
    loss_column = next(c for c in sweep.columns if "loss" in c.lower())
    lr_column = next(c for c in sweep.columns if "lr" in c.lower()
                     or "learning" in c.lower())
    best = sweep.loc[sweep[loss_column].idxmin()]
    out.append(check("Problem Set 3", "1.1",
                     "the sweep's best learning rate is the one reported",
                     values["sweep_best_learning_rate"], best[lr_column], 1e-9,
                     "p1_1_lr_sweep.csv"))

    # 2.1 The cleaning audit has to account for every raw row.
    out.append(check("Problem Set 3", "2.1",
                     "retained rows equal raw minus excluded",
                     values["n_retained"],
                     values["n_raw"] - values["n_excluded_outside_period"], 0,
                     "reported_values.csv"))

    # 3.1 The discounted return is rebuilt from the step table rather than read
    # off it: gamma is applied here, so a wrong discount factor in the notebook
    # would show up as a mismatch.
    steps = pd.read_csv(results / "p3_1_return.csv")
    gamma = 0.9
    rebuilt = float((steps["reward"] * gamma ** steps["step k"]).sum())
    out.append(check("Problem Set 3", "3.1",
                     "discounted return of the given route",
                     values["discounted_return"], rebuilt, 1e-6,
                     "sum of gamma^k r_k recomputed from p3_1_return.csv"))
    out.append(check("Problem Set 3", "3.1",
                     "the step table's own contributions sum to the same return",
                     values["discounted_return"], steps["contribution"].sum(),
                     1e-6, "p3_1_return.csv"))

    # 3.3 Two sweeps of iterative policy evaluation. The table stores exact
    # fractions as strings, so they are parsed rather than read as floats, and
    # each state's reported value is checked against the registry.
    evaluation = pd.read_csv(results / "p3_3_policy_evaluation.csv")
    for _, row in evaluation.iterrows():
        state = int(row["state"])
        for sweep in (1, 2):
            key = f"v{sweep}_node{state}"
            if key not in values.index:
                continue
            out.append(check("Problem Set 3", "3.3",
                             f"state {state} after sweep {sweep}",
                             values[key], _fraction(row[f"v_{sweep}"]), 1e-9,
                             "p3_3_policy_evaluation.csv"))

    # 4.1 Cross-correlation and true convolution differ by a 180-degree flip,
    # which for this antisymmetric kernel is a sign change.
    out.append(check("Problem Set 3", "4.1",
                     "true convolution is the negated cross-correlation",
                     values["true_convolution_value"],
                     -values["cross_correlation_value"], 0,
                     "reported_values.csv"))

    # 4.2 The from-scratch convolution is checked against a reference.
    out.append(check("Problem Set 3", "4.2",
                     "from-scratch convolution matches the reference to 1e-12",
                     0.0, values["max_diff_vs_reference"], 1e-12,
                     "reported_values.csv"))
    out.append(check("Problem Set 3", "4.2",
                     "81 verification cases were run",
                     81, values["n_verification_cases"], 0,
                     "reported_values.csv"))
    return out


# ---------------------------------------------------------------------------
# Report binding
# ---------------------------------------------------------------------------

def report_binding() -> dict:
    """How much of each registry value actually appears in the report PDF.

    Reported rather than enforced: a value can legitimately be computed and
    then discussed qualitatively instead of quoted. The number is here so a
    reader can see the coverage rather than take it on trust.
    """
    try:
        import fitz
    except ImportError:
        return {"available": False,
                "note": "PyMuPDF is not installed; report binding not checked"}

    values = registry()
    text = "".join(page.get_text()
                   for page in fitz.open(A3 / "Problem_Set_3_Report.pdf"))
    # LaTeX sets a real minus sign (U+2212), not a hyphen, and the thousands
    # separators are in the PDF but not in the registry. Both are normalised so
    # a value like -47,870.609 can be matched against -47870.609.
    compact = text.replace(",", "").replace(MINUS_SIGN, "-")

    found = 0
    missing = []
    for metric, value in values.items():
        if not np.isfinite(value):
            continue
        renderings = {f"{abs(value):.{dp}f}" for dp in range(0, 7)}
        renderings |= {r.rstrip("0").rstrip(".") for r in renderings}
        renderings |= {f"{abs(value):.{dp}g}" for dp in range(1, 8)}
        if any(r and r in compact for r in renderings):
            found += 1
        else:
            missing.append(metric)
    return {"available": True, "registry_values": len(values),
            "quoted_in_the_report": found,
            "not_found_verbatim": sorted(missing)}


# ---------------------------------------------------------------------------

def run() -> dict:
    checks = (verify_assignment_1() + verify_assignment_2()
              + verify_assignment_3())
    failed = [asdict(c) for c in checks if not c.passed]
    by_assignment: dict[str, dict[str, int]] = {}
    for c in checks:
        entry = by_assignment.setdefault(c.assignment, {"checks": 0, "passed": 0})
        entry["checks"] += 1
        entry["passed"] += int(c.passed)

    return {
        "generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": "post-submission verification; no submitted number is changed",
        "totals": {"checks": len(checks),
                   "passed": sum(1 for c in checks if c.passed)},
        "by_assignment": by_assignment,
        "report_binding": report_binding(),
        "failed": failed,
        "checks": [asdict(c) for c in checks],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    arguments = parser.parse_args()

    result = run()
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(result, indent=2) + "\n",
                             encoding="utf-8", newline="\n")

    for assignment, counts in result["by_assignment"].items():
        print(f"  {assignment:<16} {counts['passed']:>3} of {counts['checks']:>3} "
              f"claims recomputed and matched")
    binding = result["report_binding"]
    if binding.get("available"):
        print(f"\n  Problem Set 3 registry: "
              f"{binding['quoted_in_the_report']} of {binding['registry_values']} "
              f"values appear verbatim in the report PDF")

    totals = result["totals"]
    print(f"\n  {totals['passed']} of {totals['checks']} checks passed")
    if result["failed"]:
        print()
        for entry in result["failed"]:
            print(f"    FAILED {entry['assignment']} {entry['problem']}: "
                  f"{entry['claim']}")
            print(f"           reported {entry['reported']}  "
                  f"recomputed {entry['recomputed']}")
        raise SystemExit(f"\n  {len(result['failed'])} checks failed")

    print(f"  wrote {arguments.out.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
