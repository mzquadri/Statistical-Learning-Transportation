# Statistical Learning and Data Analytics for Transportation Systems

Coursework for *Statistical Learning and Data Analytics for Transportation Systems*,
M.Sc. Mathematics in Science and Engineering, Technical University of Munich.

Three problem sets, each with the executed notebook, the written report and its
LaTeX source, plus every intermediate table and figure the report cites.

## Contents

| | Topics | Deliverables |
|---|---|---|
| **Assignment 1** | Multiple linear regression, model diagnostics, fuel-economy prediction | Notebook, report |
| **Assignment 2** | PCA, binomial logistic regression, SVMs, decision trees, bagging, random forests, gradient boosting | Notebook, 26-page report, LaTeX source, figures, results |
| **Assignment 3** | Neural networks and backpropagation, Bayesian optimisation, sequence forecasting, Markov decision processes, 2-D convolution | Notebook, 17-page report, LaTeX source, figures, results |

## Layout

```
Assignment_1/   Problem Set 1 - notebook, report, data
Assignment_2/   Problem Set 2 - notebook, report (+ .tex), figures/, results/, data/
Assignment_3/   Problem Set 3 - problem sheet, data, mlp.py
  solution/     notebook, report (+ .tex), figures/, results/, build scripts
```

Every figure and table under `figures/` and `results/` is written by the notebook —
nothing is transcribed by hand. In Assignment 3, `results/reported_values.csv`
lists each number quoted in the report next to the cell that produced it.

## Reproducing

Each assignment folder carries its own `requirements.txt`:

```bash
pip install -r Assignment_3/solution/requirements.txt
jupyter lab Assignment_3/solution/Problem_Set_3_Notebook.ipynb
```

Paths inside the notebooks are relative to the notebook's own folder.

## Notes

Reports are rebuilt from their `.tex` sources with `pdflatex`. The matriculation
number has been removed from this public copy; `build_zip.py` reads it from a
`MATRIC` environment variable if you rebuild the submission archive.

Problem sheets included in these folders are the course's own material and remain
the property of the chair; they are here for context only.

## Licence

Coursework, published for reference. Please don't submit it as your own.
