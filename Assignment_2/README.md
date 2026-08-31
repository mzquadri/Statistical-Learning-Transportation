# Problem Set 2 — Statistical Learning and Data Analytics for Transportation Systems

**Name:** Mohd Zamin Quadri
**Course:** Statistical Learning and Data Analytics for Transportation Systems
**Programme:** M.Sc. Mathematics in Science and Engineering
**University:** Technical University of Munich

Topics covered: dimensionality reduction (PCA), binomial logistic regression, support
vector machines, decision trees, bagging, random forests and gradient boosting.

---

## Directory structure

```
Solution/
├── Problem_Set_2_Notebook.ipynb    Executed notebook — all code, outputs and figures
├── Problem_Set_2_Report.pdf        Written report (26 pages) — the main deliverable
├── Problem_Set_2_Report.tex        LaTeX source of the report
├── README.md                       This file
├── requirements.txt                Python packages required to run the notebook
├── data/
│   └── mode_choice_pcasample.csv   Survey sample for Problem 1.3 (5000 × 6)
├── figures/                        18 PNG figures written by the notebook
└── results/                        22 CSV result tables written by the notebook
```

The `data/` copy of `mode_choice_pcasample.csv` is identical to the file supplied with the
assignment; the original in the parent folder is untouched.

---

## Requirements

**Python 3.9 or newer.** Developed and verified on Python 3.13.5 (Anaconda distribution).

| Package | Version used | Purpose |
|---|---|---|
| numpy | 2.3.3 | numerical arrays, linear algebra |
| pandas | 2.3.3 | data frames, CSV input and output |
| matplotlib | 3.10.8 | all figures |
| scikit-learn | 1.7.2 | PCA, SVM, decision tree, random forest, gradient boosting, metrics |
| scipy | 1.16.2 | chi-squared and binomial distributions |
| jupyter / nbconvert | 7.17.1 | running the notebook |

The notebook deliberately avoids `seaborn` and `statsmodels`; every plot is drawn with
matplotlib and every model comes from scikit-learn, which keeps the dependency list short.

### Installation

```bash
python -m pip install -r requirements.txt
```

Or, if you prefer conda:

```bash
conda create -n ps2 python=3.13
conda activate ps2
python -m pip install -r requirements.txt
```

---

## How to run

From inside the `Solution/` folder:

```bash
# Option 1 — run headless and write the outputs back into the notebook
python -m nbconvert --to notebook --execute --inplace Problem_Set_2_Notebook.ipynb

# Option 2 — run interactively
jupyter notebook Problem_Set_2_Notebook.ipynb    # then Cell > Run All
```

Total runtime is roughly one minute on a standard laptop. The notebook is submitted with
its outputs already stored, so it does not need to be re-run in order to be read.

**File paths.** No absolute paths appear anywhere in the notebook. The dataset is located by
a small helper, `find_data()`, which searches `data/`, the notebook's own folder, and the
parent folder in that order. The notebook therefore also runs correctly if it is placed
directly in the `Assignment 2/` folder next to `mode_choice_pcasample.csv` — this was
tested explicitly. Figures and result tables are written to `figures/` and `results/`,
which are created automatically if they do not exist.

**Reproducibility.** `random_state=42` (and `np.random.default_rng(42)` for the synthetic
dataset of Problem 3.4) is used wherever a random choice is made, so repeated runs give
identical numbers.

---

## Rebuilding the report PDF

The report is written in LaTeX and needs a TeX distribution (MiKTeX or TeX Live) with the
standard packages `geometry`, `amsmath`, `graphicx`, `booktabs`, `caption`, `float`,
`hyperref`, `microtype` and `enumitem`. Run the notebook first so that `figures/` is
populated, then:

```bash
pdflatex Problem_Set_2_Report.tex
pdflatex Problem_Set_2_Report.tex     # second pass resolves cross-references
```

---

## Generated files

### Figures (`figures/`)

| File | Shows |
|---|---|
| `p1_2_probability_curves.png` | Predicted car probability against travel time for both respondents |
| `p1_3_correlation_matrix.png` | Correlation matrix of the five independent variables |
| `p1_3_scree_and_variance.png` | Scree plot with the Kaiser line, and explained variance |
| `p1_3_loadings.png` | Component loadings as a heatmap |
| `p1_3_biplot.png` | Biplot of PC1 against PC2 with loading vectors |
| `p2_1_data.png` | The eight Problem 2 points by class |
| `p2_3_linear_svm.png` | Soft-margin linear SVM: boundary, margins, support vectors |
| `p2_4_feature_map.png` | Original versus mapped space under the modulo-2 map |
| `p2_5_new_predictions.png` | Predictions for the four new samples |
| `p2_6_kernel_matrices.png` | RBF kernel matrix and the kernel induced by the smooth map |
| `p3_1_split_comparison.png` | Impurity reduction of the two candidate splits |
| `p3_2_bootstrap_probabilities.png` | Bootstrap inclusion and out-of-bag probabilities against n |
| `p3_3_majority_vote.png` | Majority-vote accuracy against individual accuracy |
| `p3_4_class_balance.png` | Class balance of `severe_delay` |
| `p3_4_decision_tree.png` | The fitted depth-3 decision tree |
| `p3_4_confusion_and_roc.png` | Confusion matrices for the three models and their ROC curves |
| `p3_4_feature_importances.png` | Impurity-based and permutation importances |
| `p3_4_threshold_tradeoff.png` | Threshold selection on training out-of-fold predictions |

### Result tables (`results/`)

Problem 1: `p1_1_model_fit_comparison.csv`, `p1_1_coefficient_shift.csv`,
`p1_2_predicted_probabilities.csv`, `p1_3_descriptives.csv`,
`p1_3_correlation_matrix.csv`, `p1_3_variance_explained.csv`, `p1_3_loadings.csv`,
`p1_3_eigenvectors.csv`, `p1_3_sensitivity_log_traveltime.csv`

Problem 2: `p2_3_svm_C_sweep.csv`, `p2_3_svm_points.csv`, `p2_5_new_predictions.csv`,
`p2_6_rbf_kernel_matrix.csv`, `p2_6_feature_map_kernel_matrix.csv`

Problem 3: `p3_1_split_criteria.csv`, `p3_3_boosting_update.csv`,
`p3_4_class_balance.csv`, `p3_4_test_metrics.csv`, `p3_4_feature_importances.csv`,
`p3_4_permutation_importances.csv`, `p3_4_threshold_selection_oof.csv`,
`p3_4_threshold_final_test.csv`

---

## Notes on the analysis

- Every number quoted in the report comes from an executed cell of the notebook. Nothing
  was computed by hand and transcribed.
- In Problem 1.3 the PCA is run on the **five independent variables only**;
  `travel_mode` is the dependent variable and is excluded. The variables are standardised
  with the sample standard deviation (`ddof=1`) so the eigenvalues sum to exactly `K = 5`;
  `StandardScaler` uses `ddof=0` and would give `5.0010`. The notebook also shows what
  happens without standardisation, as a justification for standardising at all.
- In Problem 2.3 non-separability is established rigorously through the convex-hull
  criterion, not by inspection: the class hulls `[1.5,3.5]x[1,3]` and `[0.5,2.5]x[0,2]`
  intersect, and the notebook exhibits a point of the intersection as a convex combination
  of each class.
- In Problem 2.6 the comparison kernel is induced by the smooth map
  `(sin(pi*x1), cos(pi*x2))`, whose class images have equal norm, so the Gram matrix can be
  read directly as a similarity structure (`+2` within class, `-2` across).
- In Problem 3.4 the test set is used **only** for final evaluation. Both the
  cross-validated model comparison and the classification-threshold selection were carried
  out on the training partition, the latter via five-fold out-of-fold predictions, so no
  decision was informed by the test data. ROC-AUC is computed from `predict_proba`, not
  from hard labels.
