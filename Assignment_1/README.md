# Problem Set 1: Linear regression

**Course:** Statistical Learning and Data Analytics for Transportation Systems
**Programme:** M.Sc. Mathematics in Science and Engineering, Technical University of Munich
**Author:** Mohd Zamin Quadri

The problem sheet has four problems. Problems 1 and 2 are worked by hand in the
report; only Problems 3 and 4 require code, and those are what this notebook
contains.

## What is here

| File | What it is |
|---|---|
| `Problem_Set_1_Notebook.ipynb` | The solution to Problems 3 and 4, executed, outputs stored |
| `Stats_assessment_1_15052026.pdf` | The problem sheet, the chair's own material |
| `data_problem_3.csv` | Supplied with the assignment: 21 vehicles, 17 columns |
| `data_problem_4.csv` | Supplied with the assignment: 1,816 five-minute intervals, 5 columns |

**The report is not published here.** It carries the matriculation number on its
title page and, unlike the reports for Problem Sets 2 and 3, has no LaTeX source
in the repository to rebuild a redacted copy from. It is excluded by
`.gitignore` rather than redacted, so the notebook is the only solution artifact
in this folder.

## Problems

**Problem 3, gasoline mileage (40 points).** Highway fuel economy of 21 vehicles
against six engine and vehicle variables. Fit the model by ordinary least
squares, estimate the error variance, test the regression as a whole, test each
coefficient, give 99 percent confidence intervals, read the residual plots, and
put confidence and prediction bands on the mean response.

**Problem 4, freeway traffic speed (30 points).** Segment speed on a two-lane
stretch of the SR241-N freeway in California, recorded every five minutes,
against flow, occupancy, downstream speed and lane flow ratio. Examine the pairs,
diagnose the multicollinearity, choose variables and transformations, and compare
ordinary least squares with ridge and lasso.

## Data

Both files were supplied with the assignment. The problem sheet does not name a
public source for either, so none is claimed here. `data_problem_3.csv` is
described in the sheet as 2005 DaimlerChrysler highway mileage test results;
`data_problem_4.csv` as loop-detector data from SR241-N.

Neither file is synthetic as far as the assignment states, but the provenance
beyond the sheet is not something this repository can establish.

## Running it

```bash
python -m pip install -r requirements.txt
jupyter notebook Problem_Set_1_Notebook.ipynb     # then Cell > Run All
```

or headless:

```bash
python -m nbconvert --to notebook --execute --inplace Problem_Set_1_Notebook.ipynb
```

Paths are relative to this folder. The notebook is stored with its outputs, so
it does not need to be re-run in order to be read.

## One value that no longer reproduces

Re-running the notebook on a current `statsmodels` reproduces every number
except one. The variance inflation factor of the constant column in Section 4.2
was `1961.68748` when the notebook was executed and is `1.00000` now:
`statsmodels` changed how `variance_inflation_factor` treats an intercept.

The four real predictors are unaffected and still give `flow` 104.23727,
`occupancy` 94.88128, `speed_down` 2.09650 and `flow_ratio` 9.11658. No
conclusion depends on the constant's figure; Section 4.2 of the notebook says
outright that the constant's VIF has no meaning and ignores it.

The stored output is left as it was submitted. `scripts/verify_results.py` in
the repository root checks the four predictor VIFs against a fresh fit.
