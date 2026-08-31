# Problem Set 3: Statistical Learning and Data Analytics for Transportation Systems

**Name:** Mohd Zamin Quadri

Topics: neural networks and backpropagation, Bayesian optimization, sequence
forecasting, Markov decision processes, and 2-D convolution.

## Contents

| File | Description |
|---|---|
| `Problem_Set_3_Report.pdf` | The report, covering Problems 1–4 |
| `Problem_Set_3_Notebook.ipynb` | All code, executed, with outputs stored |
| `mlp.py` | The supplied MLP skeleton with the missing parts completed (Problem 1) |
| `data/nyc-yellow-may2oct.csv` | Demand data for Problem 2, as supplied with the assignment |
| `requirements.txt` | Packages needed to run the notebook |
| `results/` | Every table the notebook writes, including `reported_values.csv` |

`results/reported_values.csv` lists all 126 numbers quoted in the report next to the
problem they belong to, so any figure in the PDF can be checked against the notebook
one value at a time.

The notebook also writes its plots to `figures/`. That folder is not included in the
archive, because all seven figures are already embedded in the report PDF and both
folders are recreated automatically when the notebook runs.

## Running the notebook

```bash
python -m pip install -r requirements.txt
jupyter notebook Problem_Set_3_Notebook.ipynb     # then Cell > Run All
```

or headless:

```bash
python -m nbconvert --to notebook --execute --inplace Problem_Set_3_Notebook.ipynb
```

The notebook is submitted with its outputs already stored, so it does not need to
be re-run in order to be read.

**Runtime.** Roughly one hour end to end on a machine with a CUDA GPU. Most of that
is two deliberately expensive experiments: the Bayesian optimization in Problem 1.2
evaluates a 3-fold cross-validated objective 50 times, and Problem 2.2 compares three
architectures, sweeps the lookback of the selected one, re-compares at the chosen
lookback and repeats the top two over three seeds. The code runs on CPU as well, more
slowly.

**Paths.** No absolute paths appear anywhere. The dataset is located by a small
helper that looks in `data/`, the notebook's own folder, and the parent folder, so
the notebook also runs if placed next to the original CSV.

**Reproducibility.** Random seeds are fixed throughout (`RANDOM_STATE = 42`). Model
weights are seeded at construction rather than inside the training loop, so results
do not depend on how many cells ran beforehand. One caveat is documented in the
notebook: `Conv1d` backward on CUDA accumulates with atomics, so the 1-D CNN
reproduces only to about 0.03 in validation MAE. The selected model is an LSTM,
which reproduces exactly, and no reported test figure depends on the CNN.

## Note on Problem 1.2

The supplied `mlp.py` defines no training hyperparameter called `alpha`. The only
`alpha` in the file is `plt.pcolormesh(..., alpha=0.5)` in the test snippet, which
sets plot transparency and is unrelated to training. Problem 1.2 nevertheless asks
for the best `learning_rate` and `alpha`, so an interpretation is required. **I
assume `alpha` is the coefficient of an L2 penalty on the network weights**, the
usual meaning of the name in `sklearn`'s `MLPClassifier` and in ridge regression.
The assumption is stated explicitly in Section 1.2 of the report before it is used.
The extension is backward-compatible: `alpha` defaults to `0.0`, which reproduces
the supplied unregularised skeleton exactly, so Problem 1.1 is unaffected. Biases
are left unpenalised.
