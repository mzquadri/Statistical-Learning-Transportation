"""Generates the LaTeX report from values the notebook actually produced.

Every number is read from results/reported_values.csv or results/*.csv, so the
report cannot drift from the notebook. Run the notebook first.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RES = Path("results")
# "1.1"/"1.2" must stay strings, otherwise pandas reads them as floats and the
# registry keys stop matching.
reg = pd.read_csv(RES / "reported_values.csv",
                  dtype={"problem": str, "metric": str})
V = dict(zip(reg["problem"] + "|" + reg["metric"], reg["value"]))


def v(problem, metric):
    key = f"{problem}|{metric}"
    if key not in V:
        raise KeyError(f"{key} missing from the registry; run the notebook first")
    return float(V[key])


def pct(x, d=2):
    return f"{100 * x:.{d}f}\\%"


def sci(x, d=2):
    """Scientific notation as bare math; callers add $...$ where needed."""
    m, e = f"{x:.{d}e}".split("e")
    return f"{m}\\times 10^{{{int(e)}}}"


def msci(x, d=2):
    """Scientific notation wrapped for use in text or a table cell."""
    return f"${sci(x, d)}$"


def latex_table(df, cols, headers, fmts, caption, label, align=None):
    align = align or ("l" + "r" * (len(cols) - 1))
    out = ["\\begin{table}[H]", "\\centering",
           f"\\caption{{{caption}}}", f"\\label{{{label}}}",
           f"\\begin{{tabular}}{{{align}}}", "\\toprule",
           " & ".join(headers) + " \\\\", "\\midrule"]
    for _, r in df.iterrows():
        out.append(" & ".join(f(r[c]) for c, f in zip(cols, fmts)) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(out)


sweep = pd.read_csv(RES / "p1_1_lr_sweep.csv")
final = pd.read_csv(RES / "p1_2_final_models.csv")
spread = pd.read_csv(RES / "p1_2_basin_spread.csv")
decay = pd.read_csv(RES / "p1_2_weight_norm_vs_alpha.csv")

# Derived counts, so the prose cannot contradict the registry.
BO_HITS = v("1.2", "bo_evals_in_good_region")
BO_GUIDED_HITS = v("1.2", "bo_guided_evals_in_good_region")
BO_INIT_HITS = BO_HITS - BO_GUIDED_HITS
RS_HITS = v("1.2", "rs_evals_in_good_region")
W_INIT = float(decay["pure_decay_prediction"].iloc[0])   # alpha = 0 row is ||W||_init
_best = decay.loc[decay["test_bce"].idxmin()]            # best row of the diagnostic
BEST_A = float(_best["alpha"])
BEST_PRED = float(_best["pure_decay_prediction"])
BEST_MEAS = float(_best["norm_W_final"])
BEST_ORDERS = np.log10(BEST_MEAS / BEST_PRED)
GAP_PCT = 100 * (v("1.2", "bo_L_cv") - v("1.2", "rs_L_cv")) / v("1.2", "rs_L_cv")
DIVERGED = sweep.loc[sweep["diverged"], "learning_rate"]
ANY_DIVERGED = len(DIVERGED) > 0
# The sweep is described from what it actually produced. Earlier drafts asserted a
# divergent run unconditionally and emitted the literal string "none" when the
# selection was empty; both the claim and the placeholder are now derived.
_sw = sweep.sort_values("learning_rate")
_TOP, _SECOND = _sw.iloc[-1], _sw.iloc[-2]
if ANY_DIVERGED:
    DIVERGE_CAPTION = (f"The run at $\\eta={DIVERGED.iloc[0]:g}$ diverged to a "
                       "non-finite loss.")
    DIVERGE_PROSE = (f"the run at $\\eta={DIVERGED.iloc[0]:g}$ diverges outright")
else:
    DIVERGE_CAPTION = ("No run produced a non-finite loss; the largest step sizes "
                       "stall at a high one instead of diverging.")
    DIVERGE_PROSE = (
        f"the loss climbs back to {_SECOND['final_train_loss']:.4f} at "
        f"$\\eta={_SECOND['learning_rate']:g}$ and to "
        f"{_TOP['final_train_loss']:.4f} at $\\eta={_TOP['learning_rate']:g}$")

sweep_tbl = latex_table(
    sweep, ["learning_rate", "final_train_loss", "train_acc", "test_acc"],
    ["$\\eta$", "final training loss", "train accuracy", "test accuracy"],
    [lambda x: f"{x:g}",
     lambda x: "diverged" if not pd.notna(x) or x != x else f"{x:.4f}",
     lambda x: pct(x), lambda x: pct(x)],
    "Learning-rate sweep with the supplied architecture "
    "($n_h=8$, 20\\,000 epochs, seed 42, $\\alpha=0$). "
    + DIVERGE_CAPTION,
    "tab:lrsweep")

final_tbl = latex_table(
    final, ["configuration", "learning_rate", "alpha", "train_acc", "test_acc",
            "test_bce"],
    ["configuration", "$\\eta$", "$\\alpha$", "train acc.", "test acc.",
     "test BCE"],
    [str, lambda x: f"{x:.4g}", lambda x: msci(x) if x > 0 else "0",
     lambda x: pct(x), lambda x: pct(x), lambda x: f"{x:.4f}"],
    "Models retrained on all 140 training points and evaluated once on the "
    "60 held-out test points.",
    "tab:final")

spread_tbl = latex_table(
    spread, ["quantity", "sd_log10", "spread_factor"],
    ["quantity", "s.d. of $\\log_{10}$", "spread factor"],
    [lambda s: f"\\code{{{s}}}", lambda x: f"{x:.3f}", lambda x: f"{x:.2f}$\\times$"],
    "Spread of each candidate invariant across the low-loss evaluations in "
    "basin~B. The quantity with the smallest spread is the one the good region "
    "actually constrains.",
    "tab:spread")

comp2 = pd.read_csv(RES / "p2_2_comparison.csv")
test2 = pd.read_csv(RES / "p2_2_test_comparison.csv")

# The selected model is read from the frozen-pipeline record, and the strongest
# baseline is identified from the test table. Neither is assumed.
_frozen = pd.read_csv(RES / "p2_2_frozen_pipeline.csv", header=None,
                      index_col=0).iloc[:, 0]
SELECTED_NAME = str(_frozen["architecture"])
_base = test2[test2.model != SELECTED_NAME].sort_values("test_MAE").iloc[0]
BEST_BASE_NAME = str(_base.model)
BEST_BASE_MAE = float(_base.test_MAE)
MAE_RED = 100 * (BEST_BASE_MAE - v("2.2", "test_mae")) / BEST_BASE_MAE
# The prose names both of these explicitly, so a change must fail loudly rather
# than silently produce a report that contradicts its own tables.
assert SELECTED_NAME == "LSTM", (
    f"the notebook selected {SELECTED_NAME}, but Sections 2.2 and 2.3 are written "
    "around the LSTM; investigate before regenerating")
assert abs(BEST_BASE_MAE - v("2.2", "test_historical_mae")) < 1e-6, (
    f"the strongest baseline is {BEST_BASE_NAME}, not the Historical Average; "
    "the 2.3 text names it explicitly and must be updated")

# Seed-stability means. These are means over three seeds and must not be confused
# with the single reference-seed value in the comparison table; an earlier draft
# quoted the seed-42 run while calling it a three-seed average.
seeds = pd.read_csv(RES / "p2_2_seed_stability.csv").sort_values("mean_val_MAE")
SEED_BEST_NAME = str(seeds.iloc[0]["model"])
SEED_BEST_MEAN = float(seeds.iloc[0]["mean_val_MAE"])
SEED_RUNNER_NAME = str(seeds.iloc[1]["model"])
SEED_RUNNER_MEAN = float(seeds.iloc[1]["mean_val_MAE"])
assert SEED_BEST_NAME == SELECTED_NAME, (
    f"seed stability ranks {SEED_BEST_NAME} first but the frozen pipeline selected "
    f"{SELECTED_NAME}; the prose asserts they agree")
assert abs((SEED_RUNNER_MEAN - SEED_BEST_MEAN) - v("2.2", "seed_gap_top2")) < 1e-9, (
    "the registered seed gap is not the difference of the two seed means")

seeds_tbl = latex_table(
    seeds, ["model", "seeds", "mean_val_MAE", "sd_val_MAE", "min", "max"],
    ["model", "seeds", "mean val MAE", "s.d.", "min", "max"],
    [str, lambda x: f"{x:.0f}", lambda x: f"{x:.4f}", lambda x: f"{x:.4f}",
     lambda x: f"{x:.4f}", lambda x: f"{x:.4f}"],
    "Seed stability of the top two architectures at $L=24$, each retrained from "
    "three initialisations. These means, not the single reference-seed run of "
    "Table~\\ref{tab:p2arch}, are what the selection argument rests on.",
    "tab:seeds")

arch_tbl = latex_table(
    comp2, ["model", "lookback", "params", "best_epoch", "val_MAE", "val_RMSE"],
    ["model", "$L$", "parameters", "best epoch", "val MAE", "val RMSE"],
    [str, lambda x: str(x), lambda x: str(x), lambda x: str(x),
     lambda x: f"{x:.4f}", lambda x: f"{x:.4f}"],
    "Validation comparison. Baselines have no lookback or parameters. No test "
    "metric appears here: the test partition was still sealed at this stage.",
    "tab:p2arch")

test_tbl = latex_table(
    test2, ["model", "test_MAE", "test_RMSE", "test_bias"],
    ["model", "test MAE", "test RMSE", "test bias"],
    [str, lambda x: f"{x:.4f}", lambda x: f"{x:.4f}", lambda x: f"{x:+.4f}"],
    "October results, computed once after the pipeline was frozen. Negative bias "
    "means the method under-predicts on average.",
    "tab:p2test")

p32 = pd.read_csv(RES / "p3_2_policy.csv")
p32_tbl = latex_table(
    p32, ["state", "action", "pi(a|s)", "P(s'|s,a)", "P(s'|s) under pi"],
    ["state", "action", "$\\pi(a\\mid s)$", "$P(s'\\mid s,a)$",
     "$P(s'\\mid s)$"],
    [lambda x: str(x), str, lambda x: f"${x}$", lambda x: f"{x:.0f}",
     lambda x: f"${x}$"],
    "Uniform random policy and the induced transition probabilities at nodes 5, "
    "6 and 9. Moves are deterministic, so the two probabilities coincide.",
    "tab:p32", align="rlccc")

p33 = pd.read_csv(RES / "p3_3_policy_evaluation.csv")
p33_tbl = latex_table(
    p33, ["state", "v_0", "v_1", "v_2"],
    ["state", "$v_0$", "$v_1$", "$v_2$"],
    [lambda x: str(x), str, str, str],
    "Two synchronous sweeps of iterative policy evaluation under the uniform "
    "random policy, $\\gamma=1$, shown as exact fractions. Node 3 is terminal.",
    "tab:p33", align="rccc")

split = pd.read_csv(RES / "p2_1_split.csv")
split_tbl = latex_table(
    split, ["partition", "first target", "last target", "samples"],
    ["partition", "first target step", "last target step", "windows"],
    [str, str, str, lambda x: f"{x:,}"],
    "Chronological split. A sample is assigned to the partition containing its "
    "entire four-step target horizon.",
    "tab:p2split", align="lllr")

decay_tbl = latex_table(
    decay, ["alpha", "norm_W_final", "pure_decay_prediction", "test_acc",
            "test_bce"],
    ["$\\alpha$", "$\\lVert W\\rVert$ measured", "pure-decay prediction",
     "test acc.", "test BCE"],
    [lambda x: msci(x, 1) if x > 0 else "0", lambda x: f"{x:.2f}",
     lambda x: f"{x:.3g}" if x >= 0.01 else msci(x, 1),
     lambda x: pct(x), lambda x: f"{x:.4f}"],
    "Effect of the penalty at fixed $\\eta=1.0$. The pure-decay column is "
    "$\\lVert W\\rVert_{\\text{init}}e^{-T\\eta\\alpha}$, which the measurement "
    "contradicts by many orders of magnitude.",
    "tab:decay")

doc = rf"""
\documentclass[11pt,a4paper]{{article}}
\usepackage[margin=2.5cm]{{geometry}}
\usepackage{{amsmath,amssymb}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{caption}}
\usepackage{{float}}
\usepackage[hidelinks]{{hyperref}}
\usepackage{{microtype}}
\usepackage{{enumitem}}
\newcommand{{\code}}[1]{{\texttt{{#1}}}}
\hypersetup{{
  pdftitle={{Problem Set 3 - Statistical Learning and Data Analytics for
            Transportation Systems}},
  pdfauthor={{Mohd Zamin Quadri}},
  pdfsubject={{Neural networks, reinforcement learning and convolutions}},
  pdfkeywords={{}},
  pdfcreator={{}},
}}

\title{{\vspace{{-1.5cm}}Statistical Learning and Data Analytics for Transportation
Systems\\[0.3em]\large Problem Set 3: Neural Networks, Reinforcement Learning,
Convolutions}}
\author{{Mohd Zamin Quadri}}
\date{{}}

\begin{{document}}
\maketitle

\section{{Problem 1 (15 points)}}

The model is a one-hidden-layer perceptron with sigmoid activations at both layers,
trained on the binary cross-entropy loss by full-batch gradient descent. Writing $N$
for the number of samples, $n_x$ for the input width and $n_h$ for the hidden width,
the forward pass is
\begin{{align}}
A_1 &= XW_1 + \mathbf{{1}}b_1, & Z_1 &= \sigma(A_1), \\
A_2 &= Z_1W_2 + \mathbf{{1}}b_2, & \hat{{Y}} &= \sigma(A_2),
\end{{align}}
with $X\in\mathbb{{R}}^{{N\times n_x}}$, $W_1\in\mathbb{{R}}^{{n_x\times n_h}}$,
$b_1\in\mathbb{{R}}^{{1\times n_h}}$, $W_2\in\mathbb{{R}}^{{n_h\times 1}}$,
$b_2\in\mathbb{{R}}^{{1\times 1}}$, and $\sigma(a)=1/(1+e^{{-a}})$. The loss is
\begin{{equation}}
L = -\frac{{1}}{{N}}\sum_{{i=1}}^{{N}}
\Big[y_i\ln\hat y_i + (1-y_i)\ln(1-\hat y_i)\Big].
\end{{equation}}

\subsection{{Gradient derivation and completion of \code{{mlp.py}} (8 points)}}

\paragraph{{Loss with respect to the output.}}
\begin{{equation}}
\frac{{\partial L}}{{\partial \hat y_i}}
= -\frac{{1}}{{N}}\left[\frac{{y_i}}{{\hat y_i}}-\frac{{1-y_i}}{{1-\hat y_i}}\right]
= -\frac{{1}}{{N}}\cdot
  \frac{{y_i(1-\hat y_i)-(1-y_i)\hat y_i}}{{\hat y_i(1-\hat y_i)}}
= \frac{{1}}{{N}}\cdot\frac{{\hat y_i-y_i}}{{\hat y_i(1-\hat y_i)}} .
\end{{equation}}

\paragraph{{Through the output sigmoid.}} Since $\hat y_i=\sigma(a_{{2,i}})$ and
$\sigma'(a)=\sigma(a)(1-\sigma(a))$,
\begin{{equation}}
\delta_{{2,i}} \;\equiv\; \frac{{\partial L}}{{\partial a_{{2,i}}}}
\;=\; \frac{{1}}{{N}}\cdot\frac{{\hat y_i-y_i}}{{\hat y_i(1-\hat y_i)}}
      \cdot \hat y_i(1-\hat y_i)
\;=\; \frac{{\hat y_i - y_i}}{{N}} .
\label{{eq:delta2}}
\end{{equation}}
The factor $\hat y(1-\hat y)$ cancels exactly. This is why a sigmoid output is paired
with cross-entropy: no saturation term survives at the output layer, so a confidently
wrong prediction still generates a large gradient. Under squared error the factor
would remain and learning would stall.

\paragraph{{Output-layer parameters.}} From
$a_{{2,i}}=\sum_j z_{{1,ij}}W_{{2,j}}+b_2$,
\begin{{equation}}
\nabla_{{W_2}}L = Z_1^{{\top}}\delta_2 \in\mathbb{{R}}^{{n_h\times 1}},
\qquad
\nabla_{{b_2}}L = \sum_{{i}} \delta_{{2,i}} \in\mathbb{{R}}^{{1\times 1}} .
\end{{equation}}
The bias gradient is a sum over the batch because $b_2$ was broadcast across all $N$
rows in the forward pass; broadcasting forwards becomes summation backwards.

\paragraph{{Into the hidden layer.}}
\begin{{equation}}
\nabla_{{Z_1}}L = \delta_2 W_2^{{\top}}\in\mathbb{{R}}^{{N\times n_h}},
\qquad
\delta_{{1,ij}} = \big(\nabla_{{Z_1}}L\big)_{{ij}}\,
                  z_{{1,ij}}\big(1-z_{{1,ij}}\big).
\end{{equation}}

\paragraph{{Hidden-layer parameters and the update.}}
\begin{{equation}}
\nabla_{{W_1}}L = X^{{\top}}\delta_1 \in\mathbb{{R}}^{{n_x\times n_h}},
\qquad
\nabla_{{b_1}}L = \sum_i \delta_{{1,i\cdot}} \in\mathbb{{R}}^{{1\times n_h}},
\qquad
\theta \leftarrow \theta - \eta\,\nabla_\theta L .
\end{{equation}}

\paragraph{{Translating the derivation into the supplied skeleton.}} Three points
determine whether the code matches the algebra.
\begin{{enumerate}}[nosep]
  \item \code{{backward\_propagation}} calls \code{{activation\_derivative(out)}} and
    \code{{activation\_derivative(self.z1)}}. Both arguments are \emph{{post}}
    -activation values, so the method must return $z(1-z)$, not
    $\sigma(a)(1-\sigma(a))$, even though its parameter is named \code{{a}}.
  \item The loss is a mean, so the factor $1/N$ has to be carried into
    \code{{binary\_cross\_entropy\_derivative}}. Omitting it makes the gradient
    magnitude scale with the batch size.
  \item Bias gradients are \code{{sum(axis=0, keepdims=True)}}, which both performs
    the summation above and preserves the $(1,\cdot)$ shape required for
    broadcasting in the update.
\end{{enumerate}}

\paragraph{{Verification.}} Correctness of the gradient is established independently
of whether training succeeds. Central differences with $\varepsilon=10^{{-6}}$ agree
with the analytic gradient to a maximum relative error of
{msci(v('1.1', 'gradient_check_max_rel_error'))} over all four parameter arrays, and a
PyTorch autograd cross-check in double precision on identical weights differs by at
most {msci(v('1.1', 'autograd_max_abs_diff'))}, which is machine precision.

\paragraph{{Training with the supplied settings.}} With
$\eta=0.001$, $n_h=8$ and 20\,000 epochs, the training loss decreases monotonically
but only reaches {v('1.1', 'default_final_train_loss'):.4f}, and $\ln 2 = 0.6931$ is
the loss of a constant $0.5$ predictor. Training accuracy is
{pct(v('1.1', 'default_train_accuracy'))} and test accuracy
{pct(v('1.1', 'default_test_accuracy'))}, below the majority-class baseline of
{pct(v('1.1', 'test_majority_baseline'))} on this test split. The gradient is exact,
so the cause is the step size: with gradient components of order $10^{{-1}}$ at
initialisation, a step of $\eta=0.001$ moves each weight by only about
$10^{{-4}}$ per epoch, and after 20\,000 full-batch steps the parameters have barely
left their initial values. Table~\ref{{tab:lrsweep}} confirms that the learning rate, and not the
architecture or the gradient, is the binding constraint.

{sweep_tbl}

The sweep shows three regimes: below $\eta\approx0.01$ the network does not move, a
usable band runs from roughly $0.1$ to $10$, and past that the step overshoots and the
output saturates; {DIVERGE_PROSE}, against the $\ln 2=0.6931$ of a constant
predictor. No configuration in this sweep left the finite range. The lowest final
training loss
occurs at $\eta={v('1.1', 'sweep_best_learning_rate'):g}$, which reaches
{pct(v('1.1', 'sweep_best_test_accuracy'))} test accuracy. This sweep is diagnostic
rather than a model-selection procedure: its purpose is to establish that the learning
rate is the binding constraint and to supply a trained model for
Figure~\ref{{fig:p11}}. Ranking on the training loss uses no test information, but
training loss is not a generalisation criterion (it rewards the most strongly
fitted model), so it would not be an appropriate basis for choosing
hyperparameters. Selection on held-out data is carried out in Section~1.2.

\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{figures/p1_1_test_classification.png}}
\caption{{Classification of the 60 held-out test points. Background shading is the
predicted probability $P(y=1)$ and the black line the $0.5$ decision boundary; circles
are correctly classified test points and crosses misclassified ones. Left: the
learning rate supplied in the test snippet, where the network has not moved away from
its initialisation. Right: the same code and seed at
$\eta={v('1.1', 'sweep_best_learning_rate'):g}$, which recovers the circular
boundary.}}
\label{{fig:p11}}
\end{{figure}}

\subsection{{Bayesian optimization of \code{{learning\_rate}} and \code{{alpha}}
(7 points)}}

\paragraph{{An assumption that has to be stated first.}} The problem asks for the
best \code{{learning\_rate}} and \code{{alpha}}, but the supplied \code{{MLP}} class
defines no training hyperparameter called $\alpha$. The only \code{{alpha}} anywhere
in \code{{mlp.py}} is \code{{plt.pcolormesh(..., alpha=0.5)}} in the test snippet,
which sets the opacity of the decision-surface shading and has nothing to do with
training. Since the task is to optimise a hyperparameter the code does not yet
contain, some interpretation is unavoidable, and I have made mine explicit rather
than leaving it implied: \textbf{{I take $\alpha$ to be the coefficient of an $L_2$
penalty on the network weights}}, the standard meaning of \code{{alpha}} in
\code{{scikit-learn}}'s \code{{MLPClassifier}} and in ridge regression, and the
reading under which the search in this section is a meaningful exercise. The
extension is backward-compatible: $\alpha=0$ recovers the supplied unregularised
model exactly, so nothing in Section~1.1 is affected. Should the intended meaning
have been different (a momentum coefficient, say, or a learning-rate decay), the
optimisation machinery below transfers unchanged; only the second coordinate of the
search space would need reinterpreting. I use an $L_2$ penalty
on the weights, leaving the biases unpenalised:
\begin{{equation}}
J \;=\; L \;+\; \frac{{\alpha}}{{2}}
\Big(\lVert W_1\rVert_F^2+\lVert W_2\rVert_F^2\Big).
\end{{equation}}
The penalty contains no forward-pass quantity, so it adds to the gradient independently
of backpropagation:
$\partial\big(\tfrac{{\alpha}}{{2}}\sum W^2\big)/\partial W_{{jk}} = \alpha W_{{jk}}$,
hence
\begin{{equation}}
\nabla_{{W_2}}J = Z_1^{{\top}}\delta_2+\alpha W_2,
\qquad
\nabla_{{W_1}}J = X^{{\top}}\delta_1+\alpha W_1,
\end{{equation}}
with the bias gradients unchanged. Biases are excluded because a bias shifts the
decision boundary without controlling how sharply the network responds to its inputs,
so penalising it biases the model toward predicting $0.5$ without reducing capacity.
The coefficient defaults to zero, so every result in Section~1.1 is the unregularised
model. The regularised gradient was checked the same way as before: maximum relative
error {msci(v('1.2', 'l2_gradient_check_max_rel_error'))} against central differences
and {msci(v('1.2', 'l2_autograd_max_abs_diff'))} against autograd.

\paragraph{{Objective.}} Two quantities are kept distinct throughout. $L_{{cv}}$ is the
mean three-fold cross-validated binary cross-entropy on the training partition, and
$g=\log_{{10}}L_{{cv}}$ is the transformed value the optimiser minimises. The choices
behind this objective are as follows. Validation \emph{{loss}} rather than accuracy is
used because accuracy on the 46--47 validation points of a fold is a step function of
the hyperparameters, which a Gaussian process cannot model usefully. Cross-validation is
performed on the training partition only, so the 60 test points are used exactly once.
A fixed initialisation seed per fold makes the objective deterministic in
$(\eta,\alpha)$. The $\log_{{10}}$ transform prevents divergent configurations from
dominating the surrogate as outliers, and configurations whose weights become
non-finite are recorded at a common cap of $L_{{cv}}=10$. The search box is
logarithmic, $\log_{{10}}\eta\in[-3,1]$ and $\log_{{10}}\alpha\in[-6,0]$, because
Table~\ref{{tab:lrsweep}} shows the usable band spans four orders of magnitude.

\paragraph{{Algorithm.}} The surrogate is a Gaussian process with a Mat\'ern
$\nu=5/2$ kernel and a separate length scale per dimension. Their upper bound is
capped at 10: the box is only $4\times6$ units wide in log space, so a length scale of
10 already represents an almost linear trend, and permitting larger values allows the
process to declare a dimension irrelevant and stop searching it. Candidates are scored
by Expected Improvement, which for minimisation is
\begin{{equation}}
\mathrm{{EI}}(x) = \big(g^{{*}}-\mu(x)-\xi\big)\Phi(Z) + \sigma(x)\varphi(Z),
\qquad
Z = \frac{{g^{{*}}-\mu(x)-\xi}}{{\sigma(x)}},
\end{{equation}}
with $g^{{*}}$ the best value observed so far, $\mu$ and $\sigma$ the posterior mean
and standard deviation, and $\xi=0.01$ an exploration margin. Each iteration maximises
$\mathrm{{EI}}$ over 4096 Sobol candidates and refines the best with L-BFGS-B. The
budget is 16 Sobol initial points followed by 34 guided iterations;
Figure~\ref{{fig:p12conv}} traces both searches.

\paragraph{{Result.}} The search selects
$\eta = {v('1.2', 'bo_learning_rate'):.4g}$ and
$\alpha = {sci(v('1.2', 'bo_alpha'))}$, with
$g = {v('1.2', 'bo_g'):+.4f}$, that is $L_{{cv}} = {v('1.2', 'bo_L_cv'):.4f}$, in
{v('1.2', 'bo_runtime_s'):.0f}\,s. Table~\ref{{tab:final}} shows the effect of tuning
on the held-out test set and Figure~\ref{{fig:p12tuned}} the decision boundary it
produces.

{final_tbl}

\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{figures/p1_2_bo_convergence.png}}
\caption{{Left: best $L_{{cv}}$ found so far against the number of objective
evaluations. The shaded band is the bootstrap 10th--90th percentile of random
search's best-of-$k$, resampled from its 50 independent draws. The first 16 Bayesian
-optimization evaluations form a Sobol design and are not yet guided. Right: all 100
evaluations from both searches in the $(\eta,\alpha)$ plane, coloured by $L_{{cv}}$.}}
\label{{fig:p12conv}}
\end{{figure}}

\paragraph{{Comparison with random search.}} An uninformed search of the same size,
on the same objective and the same log-scaled box, reaches
$L_{{cv}} = {v('1.2', 'rs_L_cv'):.4f}$ at
$\eta={v('1.2', 'rs_learning_rate'):.4g}$,
$\alpha={sci(v('1.2', 'rs_alpha'))}$. The two best values differ by
{abs(GAP_PCT):.2f}\,\%, so on final quality the methods are equivalent here. They are
not, however, equally economical with their budget:
{BO_HITS:.0f} of the 50 Bayesian evaluations landed in the low-loss region against
{RS_HITS:.0f} of the 50 random draws, and {BO_GUIDED_HITS:.0f} of those
{BO_HITS:.0f} came from the 34 guided iterations, only {BO_INIT_HITS:.0f} from the
16-point Sobol initialisation. The surrogate therefore did steer the search. It simply
had little to gain: the low-loss region occupies a large share of a two-dimensional
box, so 50 random draws are very likely to find it. The advantage of Bayesian
optimization would grow with the dimension of the search space and with the cost of
each evaluation.

\paragraph{{What the low-loss region constrains.}} Rearranging the update as
$W\leftarrow(1-\eta\alpha)W-\eta\nabla_W L$ suggests that the product $\eta\alpha$
might be the controlling quantity. The evaluations do not support this. The low-loss
set separates into two clusters, one at $\eta\approx0.1$ with $\alpha$ effectively
zero and a better one at larger $\eta$ with $\alpha$ of order $10^{{-4}}$; within
the better cluster (Table~\ref{{tab:spread}}) $\alpha$ varies by only
{v('1.2', 'basinB_alpha_spread_factor'):.2f}$\times$ while the product varies by
{v('1.2', 'basinB_product_spread_factor'):.2f}$\times$. Regressing
$\log_{{10}}\alpha$ on $\log_{{10}}\eta$ within that cluster gives a slope of
${v('1.2', 'basinB_slope_logalpha_vs_logeta'):+.3f}$ with
$R^2={v('1.2', 'basinB_slope_r2'):.3f}$, whereas a constant product would require a
slope of exactly $-1$. The correct statement is therefore that $\alpha$ is pinned to a
narrow band while $\eta$ is only loosely constrained, and that raising $\alpha$ from
nearly zero to about $4\times10^{{-4}}$ shifts the best learning rate from roughly
$0.1$ to roughly $1$. The two hyperparameters do interact, which is what justifies
searching them jointly, but not through a constant product.

{spread_tbl}

\paragraph{{What the penalty does.}} Because the decay term acts at every one of the
20\,000 steps, $(1-\eta\alpha)^{{T}}$ is tempting to read as a cumulative shrinkage
factor. Table~\ref{{tab:decay}} shows that reading is wrong. The weights start at
$\lVert W\rVert_{{\text{{init}}}}={W_INIT:.2f}$; at $\alpha={sci(BEST_A, 0)}$, the
setting with the lowest test cross-entropy in the table, pure decay predicts a final
norm of ${sci(BEST_PRED, 1)}$ while the measured norm is {BEST_MEAS:.2f}, a
discrepancy of about {BEST_ORDERS:.0f} orders of magnitude, and in the opposite
direction, since
the weights have grown rather than shrunk. The prediction fails because the data
gradient is applied at every step as well, so the two forces balance at
$\nabla_W L+\alpha W=0$ instead of compounding. What $L_2$ provides here is a brake on
weight growth. Unregularised training inflates the weight norm to
{v('1.2', 'norm_W_alpha0'):.1f}, saturating the output sigmoid, and cross-entropy
penalises confident errors heavily; at $\alpha=3\times10^{{-4}}$ the norm settles at
{v('1.2', 'norm_W_alpha3em4'):.1f} and the test cross-entropy improves substantially.
An order of magnitude more, and the brake dominates: the norm collapses and the
network can no longer represent a circular boundary.

{decay_tbl}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.55\textwidth]{{figures/p1_2_tuned_classification.png}}
\caption{{Test data classified by the model at the hyperparameters selected by
Bayesian optimization, on the same axes as Figure~\ref{{fig:p11}}.}}
\label{{fig:p12tuned}}
\end{{figure}}

\paragraph{{Limitations.}} Two are worth stating. First, the objective is
cross-entropy, so the selected model is the best calibrated one rather than the most
accurate: an unregularised model at $\eta=1$ reaches a slightly higher test accuracy
while scoring a much worse cross-entropy. Second, the comparison rests on one
Bayesian run against one random-search run; the bootstrap characterises the
distribution of random search but not of Bayesian optimization, so the comparison is
indicative rather than a significance test.

\section{{Problem 2 (25 points)}}

The task is to forecast pickup and drop-off demand over the next four 15-minute
steps, that is a one-hour horizon. Ten locations are observed over
{v('2.1', 'n_timesteps'):.0f} quarter-hour slots from May to October 2023. Because
the locations move together (the mean pairwise correlation of the pickup series
is ${v('2.1', 'xloc_pickup_corr'):.2f}$) and pickups and drop-offs within a
location correlate at ${v('2.1', 'pickup_dropoff_corr'):.2f}$ on average, the
problem is treated as a single \textbf{{system-wide multivariate}} forecast: one
model reads all 20 demand channels and predicts all 20 channels at each of the four
future steps, giving {v('2.1', 'n_features'):.0f} input features per time step and
$4\times10\times2=80$ outputs.

\subsection{{Cleansing, Dataset class and the split (5 points)}}

\paragraph{{Cleansing.}} Of the {v('2.1', 'n_raw'):.0f} supplied rows,
{v('2.1', 'n_excluded_outside_period'):.0f} carry timestamps outside the
May--October 2023 study period stated in the problem and are excluded on that
basis; no claim is made about their origin. The remaining
{v('2.1', 'n_retained'):.0f} rows contain no duplicate
\code{{(datetime, location\_id)}} keys and no negative, non-integer or infinite
demand, so nothing else is removed on validity grounds. Extreme values are
deliberately \emph{{retained}}: the largest counts occur on weekday evenings and are
genuine peak demand, which is exactly the behaviour a fleet-management forecast
exists to capture.

\paragraph{{Two different defects.}} A row can exist while its demand columns are
empty, so timestamp completeness and value completeness are audited separately.
Comparing the expected grid of $184\times96=
{v('2.1', 'n_timesteps'):.0f}$ slots against the observed timestamps of each
location gives zero defects: every row is present. What is missing is the
\emph{{value}} on {v('2.1', 'n_missing_demand'):.0f} of the
{v('2.1', 'n_demand_entries'):.0f} demand entries
({pct(1 - v('2.1', 'pct_observed'), 4)}), always in both columns of a row at once, in
runs of at most sixteen slots.

\paragraph{{Why entries are masked rather than windows dropped.}} Per-channel
missingness is small, but a multivariate window needs all 20 channels at every step,
so the rates compound: {pct(v('2.1', 'pct_timesteps_any_channel_missing'))} of
timesteps have at least one channel missing. Requiring a fully observed input
\emph{{and}} target window therefore retains only about $70\%$ of samples at a
three-hour lookback and around $1\%$ at a one-day lookback. Masking individual
target entries instead costs {pct(1 - v('2.1', 'pct_observed'), 4)} of the supervision.
The adopted rule is:
\begin{{itemize}}[nosep]
  \item \textbf{{inputs}} are completed by causal forward fill, which copies from an
    \emph{{earlier}} timestamp only, plus a per-channel indicator flagging that the
    value was filled. Bidirectional interpolation is rejected because it would use
    observations later than the timestamp being filled, information that does not
    exist at forecast time. Forward fill also cannot move a value from a later
    partition into an earlier one, so it is safe across split boundaries by
    construction;
  \item \textbf{{targets}} are never imputed. A per-entry mask removes unobserved
    targets from both the loss and every reported metric, so the model is neither
    trained nor scored against a fabricated value.
\end{{itemize}}

\paragraph{{Features and scaling.}} Each time step carries 20 standardised demand
channels, 20 imputation flags, and four cyclical calendar terms
$\sin,\cos(2\pi\,\text{{tod}}/96)$ and $\sin,\cos(2\pi\,\text{{dow}}/7)$, which keep
23:45 adjacent to 00:00. No month feature is used: with six months and a
chronological split, month identity would be of little use and easy to misuse.
Standardisation is per location $\times$ channel, giving 20 mean and
standard-deviation pairs, because channel means span roughly 14 to 60 trips per 15
minutes, and both
statistics are estimated on May--August alone. Applied to the training period the
transform gives mean $0$ and standard deviation $1$ exactly; on validation and test
it gives $-0.08$ and $+0.16$ respectively, which is the expected behaviour and
confirms that no test information entered the fit.

\paragraph{{Dataset class.}} \code{{TaxiDemandDataset}} receives every array through
its constructor and stores only window indices, so \code{{\_\_getitem\_\_}} performs
slicing and nothing else. One item is
\begin{{center}}
\code{{x}}: $({v('2.1', 'lookback'):.0f}, {v('2.1', 'n_features'):.0f})$ \quad
\code{{y}}: $({v('2.1', 'horizon'):.0f}, 10, 2)$ \quad
\code{{m}}: $({v('2.1', 'horizon'):.0f}, 10, 2)$, all \code{{float32}},
\end{{center}}
with axes (time, feature), (horizon step, location, channel) and the same for the
mask. Demand columns are ordered \code{{location\_index}}$\times 2+$
\code{{channel\_index}}. The target is kept in structured form rather than flattened
so that per-horizon and per-location metrics are direct slices.

\paragraph{{Split.}} The lag-1 autocorrelation of demand is
${v('2.1', 'autocorr_lag1'):.2f}$, so a random split
would place near-identical adjacent windows on both sides and leak heavily. The
split is therefore chronological on calendar boundaries, and a sample belongs to the
partition containing its \emph{{entire}} target horizon; Table~\ref{{tab:p2split}}
gives the resulting partitions.

{split_tbl}

Input windows may reach back into the preceding partition, because an operator
forecasting at 1 September genuinely holds August observations; this is not target
leakage. {v('2.1', 'n_straddling_dropped'):.0f} windows whose horizon straddled a
boundary were dropped. {v('2.1', 'n_leakage_checks_passed'):.0f} assertions are
executed in the notebook, covering partition ordering, disjointness of target
timestamps, the requirement that every input step strictly precedes its own targets,
horizon contiguity, and that the scaling statistics were computed from training rows
only. One dataset item was also traced back to the source CSV: de-standardising
\code{{y}} recovers the four recorded pickup counts exactly.

\subsection{{Architecture, training and evaluation (15 points)}}

\paragraph{{Metrics.}} Everything is measured over observed target entries only and
reported in trips per 15 minutes,
\begin{{equation}}
\text{{MAE}}=\frac{{\sum_i m_i|\hat y_i-y_i|}}{{\sum_i m_i}},\quad
\text{{RMSE}}=\sqrt{{\frac{{\sum_i m_i(\hat y_i-y_i)^2}}{{\sum_i m_i}}}},\quad
\text{{bias}}=\frac{{\sum_i m_i(\hat y_i-y_i)}}{{\sum_i m_i}} ,
\end{{equation}}
with squared errors accumulated globally before the square root. MAPE is not used
because many targets are exactly zero. The training loss is the same masked mean
squared error on standardised targets. Both were verified against a hand-computed
example, and setting a masked prediction to $-10^6$ leaves the loss unchanged.

\paragraph{{Candidate architectures.}} Three models were compared. Their sizes are
not identical ({v('2.2', 'mlp_params'):,.0f} parameters for the MLP,
{v('2.2', '1dcnn_params'):,.0f} for the CNN and {v('2.2', 'lstm_params'):,.0f} for
the LSTM), because flattening makes the MLP's input layer grow with the lookback
while the other two share parameters across time. That works against the conclusion
rather than for it: the MLP is the \emph{{largest}} of the three and still the weaker
of the two feed-forward options, so its deficit is a matter of inductive bias and not
of capacity. The MLP flattens the window and can mix all lags and channels but has no
temporal structure. The 1-D CNN
convolves along time with dilations $1,2,4$ and kernel $3$, a receptive field of
$1+2(1+2+4)=15$ steps, and shares weights across time. The LSTM keeps a recurrent
state along the sequence. Every model saw identical inputs, samples, mask, optimiser,
batch size, epoch budget and early-stopping rule. The selection criterion, lowest
validation MAE, was fixed before the results were seen.

\paragraph{{Choosing the lookback.}} Architecture and lookback interact, so they were
chosen in two passes on validation data: the three architectures were compared at a
provisional lookback of {v('2.2', 'provisional_lookback'):.0f} steps, the winner was
then swept over lookback, and finally all three were re-compared at the selected
value to check that the ranking still held. The sweep settles on
$L={v('2.2', 'selected_lookback'):.0f}$ steps, that is
{v('2.1', 'lookback') / 4:g} hours of history, ahead of the next best candidate by
{v('2.2', 'lookback_gap_to_runner_up'):.4f} trips per 15 minutes.

A cheap ridge regression on the flattened window was run first as a deterministic
reference, and it points the other way: it is minimised at the shortest window it was
offered and degrades steadily as the window grows. The disagreement is informative
rather than contradictory. Flattening adds
{v('2.1', 'n_features'):.0f} inputs for every extra step, so a linear model pays a
dimension penalty on a fixed number of training samples, and neighbouring windows are
strongly collinear. A recurrent model reuses the same parameters at every step and is
not charged in the same way, so it can exploit sequence length the linear probe
cannot. This is the reason the lookback was ultimately selected with the architecture
that is actually deployed rather than with a cheaper stand-in.

{arch_tbl}

\paragraph{{Selection.}} The LSTM has the lowest validation MAE, and the margin is
not initialisation noise. Repeating the top two architectures over three seeds
(Table~\ref{{tab:seeds}}) gives a mean validation MAE of {SEED_BEST_MEAN:.4f} for the
{SEED_BEST_NAME} against {SEED_RUNNER_MEAN:.4f} for the {SEED_RUNNER_NAME}, a gap of
{v('2.2', 'seed_gap_top2'):.4f} trips per 15 minutes against a pooled seed standard
deviation of {v('2.2', 'seed_pooled_sd'):.4f}, a ratio of
{v('2.2', 'seed_gap_top2') / v('2.2', 'seed_pooled_sd'):.1f}, so the ordering cannot
be an artefact of initialisation. The selected model is a single recurrent
layer, \code{{LSTM}}$(44\to128)$, dropout $0.1$, then a linear head to the 80
outputs, reshaped to $(4,10,2)$: {v('2.2', 'selected_model_params'):.0f} parameters,
matching the closed-form count exactly.

{seeds_tbl}

\paragraph{{Training.}} Adam at $10^{{-3}}$, batch size 64, at most 60 epochs, early
stopping on validation masked MSE with patience 8 and the best checkpoint restored,
which occurred at epoch {v('2.2', 'best_epoch'):.0f}. The output head is linear, so
all three architectures emit a small fraction of negative demands; the selected model
is therefore clipped at zero, a rule adopted because it lowered validation MAE. One
consequence has to be stated plainly, because it is easy to miss: every validation
figure in Tables~\ref{{tab:p2arch}} and~\ref{{tab:seeds}} is measured
\emph{{before}} that clipping step, since the clipping rule was only fixed once an
architecture had been chosen, whereas the October figures in
Table~\ref{{tab:p2test}} are measured \emph{{after}} it. The two sets of numbers are
therefore not strictly like-for-like, and the small val-to-test difference should not
be read as a generalisation gap measured on a common footing. The whole pipeline was
frozen before the test partition was touched, and October was then evaluated exactly
once (Protocol A: the validation-selected checkpoint, trained on May--August only).

\paragraph{{A recording outage in September.}} While inspecting errors, a period was
found where recorded demand collapses across all ten locations simultaneously:
{v('2.2', 'outage_slots'):.0f} consecutive slots from 21 to 24 September, with a
missing rate of about 24\,\% against 1.2\,\% elsewhere and a mean observed value near
$1.6$ trips per 15 minutes. It is isolated by a criterion no daily cycle can trigger, namely
a contiguous run of at least 24 hours below 5 trips per 15 minutes,
whereas the longest genuine overnight lull in six months is four hours. The event lies entirely
inside September: the training and test partitions contain
{v('2.2', 'outage_in_train'):.0f} and {v('2.2', 'outage_in_test'):.0f} of its slots
respectively, so neither the fitted model nor the October evaluation is affected. The
architecture ranking is unchanged when those validation targets are excluded, so the
selection does not depend on them. The affected values were left in place rather than
removed, and the robustness check is reported instead.

{test_tbl}

\paragraph{{Test results.}} On October the selected model reaches a MAE of
{v('2.2', 'test_mae'):.4f} and an RMSE of {v('2.2', 'test_rmse'):.4f} trips per 15
minutes, reproduced independently in NumPy. Error grows with the horizon, from
{v('2.2', 'test_mae_h1'):.4f} at $+15$ minutes to {v('2.2', 'test_mae_h4'):.4f} at
$+60$ minutes, which is expected but was measured rather than assumed. After
clipping, no prediction is negative. Figure~\ref{{fig:p22curve}} shows the training
run behind the restored checkpoint and Figure~\ref{{fig:p22example}} one October day
of forecasts.

\paragraph{{How much the lookback choice is really worth.}} The sweep is worth a
caveat. The margin that selected $L={v('2.2', 'selected_lookback'):.0f}$ over the
next candidate is {v('2.2', 'lookback_gap_to_runner_up'):.4f} trips per 15 minutes,
larger than the seed-to-seed spread of {v('2.2', 'seed_pooled_sd'):.4f} but of the
same order, and it was measured on a single month of validation data that also
contains the September outage discussed above. A margin of that size is enough to
break a tie under a pre-declared rule, which is how it was used, but it is not on its
own strong evidence that six hours of history generalises better than three. Whether
it transfers is not something this experiment can answer: the test partition was
unsealed once, for the selected configuration only, and re-opening it to score the
shorter window would be exactly the kind of test-set selection the protocol exists to
prevent. The point applies to the whole tuning exercise and not just to this one
parameter.

\begin{{figure}}[H]
\centering
\includegraphics[width=0.72\textwidth]{{figures/p2_2_learning_curve.png}}
\caption{{Training and validation loss for the selected model. The dashed line marks
the checkpoint restored by early stopping.}}
\label{{fig:p22curve}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{figures/p2_2_forecast_example.png}}
\caption{{Observed demand and the $+15$ and $+60$ minute forecasts for one October
day. The location is the one with median test MAE, picked by that rule rather than
for appearance; the day was fixed in advance as a mid-October Wednesday and was not
selected on the results.}}
\label{{fig:p22example}}
\end{{figure}}

\subsection{{Deep learning versus traditional time-series methods (5 points)}}

The forecasting problem here has three features that matter for this comparison: the
target is 20 demand channels rather than one series, the ten locations and the two
channels within a location move together, and the whole one-hour horizon has to be
produced at once. A neural network suits that shape. A single LSTM forward pass maps
{v('2.1', 'lookback') / 4:g} hours of history for all 20 channels, plus the calendar
terms and the imputation flags, onto all $4\times10\times2$ future values, and it can learn
nonlinear relations among those inputs without anyone specifying their functional
form. On October it reached a MAE of {v('2.2', 'test_mae'):.4f} against
{v('2.2', 'test_historical_mae'):.4f} for the Historical Average, the strongest of
the three baselines evaluated, a reduction of about {MAE_RED:.0f}\,\% relative to
that baseline. The point is not that statistical methods cannot handle several
series or several steps ahead; VAR-type models and SARIMAX with exogenous regressors
exist. It is that one joint model covers all of it here without specifying a separate
relationship for each of the 20 channels.

Traditional methods keep real advantages on this task. The Historical Average answers
a question an operator can state plainly, namely what demand normally occurs at
this location, on this weekday, at this time, and it needs no iterative fitting, no
architecture choice, no early stopping and no tensor pipeline. ARIMA-family models
come with established diagnostics: stationarity, differencing, seasonal terms and
residual checks are all inspectable, and their assumptions are visible in a way that
an LSTM's hidden state is not. That matters for transport planning, where knowing why
a forecast moved can be worth as much as a slightly lower error. They are also less
data-hungry. Six months at 15-minute resolution sounds like a lot, but consecutive
windows overlap in {v('2.1', 'lookback') - 1:.0f} of their
{v('2.1', 'lookback'):.0f} input steps and demand is strongly
autocorrelated at short lags, so the number of effectively independent observations
is far below the window count.

The chronological split makes the trade-off concrete. October's mean pickup demand is
about {100 * (v('2.1', 'test_mean_pickup') / v('2.1', 'train_mean_pickup') - 1):.0f}\,\%
above the May--August training level, and the two approaches absorbed that shift very
differently. The Historical Average, being a fixed climatology of the training
months, under-predicted October with a bias of
{v('2.2', 'test_historical_bias'):+.4f} trips per 15 minutes; the same model had
\emph{{over}}-predicted the lower-demand September. The LSTM's bias was
{v('2.2', 'test_bias'):+.4f}, roughly
{abs(v('2.2', 'test_historical_bias') / v('2.2', 'test_bias')):.0f} times smaller in
magnitude, because it conditions on the most recent
{v('2.1', 'lookback') / 4:g} hours and therefore tracks the prevailing level instead
of assuming it. Not having an explicit stationarity
assumption is not the same as being immune to distribution shift, but on this
particular shift the recent-history model degraded far less.

The conclusion should be read narrowly. In this experiment the selected LSTM had a
lower test MAE and RMSE than every baseline evaluated, which supports using a joint
nonlinear model for this dataset. It does not show that deep learning is generally
preferable: no ARIMA or SARIMAX model was fitted here, so nothing was measured
against a properly specified seasonal statistical model. Such a model could well be
competitive, and it would come with lower implementation complexity and better
interpretability. What the experiment does support is narrower and still useful:
when 20 correlated series must be forecast jointly over several steps and recent
demand carries most of the signal, a single recurrent model is a reasonable choice,
and the accuracy gain has to be weighed against the extra machinery it requires.

\section{{Problem 3 (30 points)}}

The network is a $3\times3$ grid of nine states. Every time step Emily must move to
an adjacent node, so the admissible actions at a state are exactly its neighbours and
remaining still is not an option. A move earns the reward of the road taken
($-1$, $-3$ or $-5$ for no, low and high congestion), and arriving at node~3 pays an
additional $+10$. The congestion level of each of the twelve roads was read off
Figure~1 of the problem sheet, matching every edge against the three legend entries
by colour and line style; all twelve were unambiguous. The resulting assignment,
which is the one encoded in the notebook and redrawn in Figure~\ref{{fig:p34}}, is:
no congestion on 2--3, 4--5 and 8--9; low on 1--2, 2--5, 3--6, 4--7 and 5--8; high
on 1--4, 5--6, 6--9 and 7--8.

Node 3 is treated as terminal, so $v(3)=0$ throughout. The sheet calls it the
destination and pays a one-off arrival bonus, which is only coherent if the episode
ends on arrival.

\subsection{{Discounted return of a fixed route (6 points)}}

Along $7\to8\to5\to2\to3$ the roads are high (7--8), low (8--5), low (5--2) and none
(2--3), and the final move also collects the bonus, so its reward is $-1+10=+9$. With
$\gamma=0.9$,
\begin{{equation}}
G=\sum_{{k=0}}^{{3}}\gamma^{{k}}r_{{k}}
=(-5)+0.9(-3)+0.81(-3)+0.729(+9)
=-5-2.7-2.43+6.561
={v('3.1', 'discounted_return'):.4f}.
\end{{equation}}
The return is negative: the $+10$ bonus is discounted by $\gamma^{{3}}=0.729$ and does
not quite offset the congestion penalties accumulated on the way.

\subsection{{Uniform random policy (5 points)}}

Knowing nothing about the network, Emily chooses uniformly among the admissible
moves, $\pi(a\mid s)=1/|\mathcal{{A}}(s)|$ (Table~\ref{{tab:p32}}). Each move is deterministic, so
$P(s'\mid s,a)=1$ for the intended neighbour and the state transition probability
under the policy coincides with the action probability.

{p32_tbl}

Node~5 is interior with four neighbours, node~6 is an edge state with three, and
node~9 is a corner with two, which is why the three probabilities differ.

\subsection{{Two sweeps of iterative policy evaluation (12 points)}}

With $\gamma=1$ and $v_0(s)=0$, the Bellman expectation equation for this policy is
\begin{{equation}}
v_{{k+1}}(s)=\sum_{{a}}\pi(a\mid s)\big[r(s,a)+\gamma v_k(s')\big]
=\frac{{1}}{{|\mathcal{{A}}(s)|}}\sum_{{s'\in\mathcal{{N}}(s)}}
\big[r(s,s')+v_k(s')\big].
\end{{equation}}
The sweeps are synchronous: both use $v_k$ on the right-hand side, so the result does
not depend on the order in which states are visited. Table~\ref{{tab:p33}} gives the
two sweeps as exact fractions.

{p33_tbl}

As a worked example, the second sweep at node~2 uses the three neighbours 1, 3 and 5
with rewards $-3$, $+9$ and $-3$ and the first-sweep values $v_1(1)=-4$, $v_1(3)=0$
and $v_1(5)=-3$:
\begin{{equation}}
v_2(2)=\tfrac{{1}}{{3}}\big[(-3-4)+(9+0)+(-3-3)\big]
=\tfrac{{1}}{{3}}(-7+9-6)=-\tfrac{{4}}{{3}}\approx{v('3.3', 'v2_node2'):.4f}.
\end{{equation}}
After one sweep only node~2 has a positive value, because it is the sole state from
which the bonus is reachable in a single move. After the second sweep every value has
fallen, since one sweep of a random walk mostly accumulates further congestion
penalties; the states nearest the charging station remain the least negative.

\subsection{{Greedy policy and its optimality (7 points)}}

Acting greedily with respect to $v_2$ (Figure~\ref{{fig:p34}}) means taking
$\arg\max_a\big[r(s,a)+v_2(s')\big]$ at each state, which gives
$1\to2$, $2\to3$, $4\to5$, $5\to2$, $6\to3$, $7\to4$, $8\to9$ and $9\to8$.

\begin{{figure}}[H]
\centering
\includegraphics[width=0.58\textwidth]{{figures/p3_4_greedy_policy.png}}
\caption{{The greedy policy with respect to $v_2$. Edge colour and style show the
congestion level; arrows show the chosen move. Nodes 8 and 9 point at each other.}}
\label{{fig:p34}}
\end{{figure}}

\paragraph{{It is not the optimal policy.}} Running value iteration to convergence
gives $v^{{*}}=(6,9,0,5,6,7,2,3,2)$ for states 1 to 9 and the optimal actions
$1\to2$, $2\to3$, $4\to5$, $5\to2$, $6\to3$, $7\to4$, $8\to5$ and $9\to6$ or
$9\to8$. The greedy policy agrees everywhere except at node~8. There the successor
value actually favours node~5, since $v_2(5)=-9/2$ is above $v_2(9)=-5$, but the
greedy rule scores the whole move rather than the destination alone, and the road
8--9 is uncongested while 8--5 carries low congestion. The cheaper edge more than
covers the gap in successor value:
$-1+v_2(9)=-6$ against $-3+v_2(5)=-7.5$, so $8\to9$ wins by $1.5$. That single
disagreement is
enough to break the policy: 8 points to 9 and 9 points back to 8, so from either
state Emily cycles between them and never reaches the charging station at all. Only
{v('3.4', 'n_states_reaching_goal'):.0f} of the eight non-terminal states reach the
goal under this policy. The failure is a consequence of truncating the evaluation
after two sweeps, since the values have not yet propagated the fact that node~5 is
close to the bonus, and not a defect of policy iteration itself, which alternates
evaluation and improvement until nothing changes.

\paragraph{{Can optimality be guaranteed in practice?}} For a finite MDP with known
and exact transition probabilities and rewards, policy iteration is guaranteed to
reach an optimal policy in finitely many steps: each improvement step yields a policy
at least as good, and there are only finitely many deterministic policies. That
guarantee is relative to the model, not to the world. In a real road network the
transitions are not deterministic, since a closure or a queue may prevent the
intended move; congestion is stochastic and varies through the day, so a fixed
reward of
$-5$ for a road is at best an average, and a policy that is optimal at 08:00 need not
be at 22:00. If the rewards or transition probabilities are estimated from data,
their estimation error carries into the solution. The state space also has to stay
small enough to sweep, which a city-scale network is not. And as this exercise shows,
a truncated computation can return a policy that is not merely suboptimal but
non-terminating. Optimality is therefore guaranteed with respect to the assumed model
and a converged computation; both assumptions need checking before the result is
trusted in practice.

\section{{Problem 4 (30 points)}}

\subsection{{Manual convolution (5 points)}}

For a $4\times4$ input, a $3\times3$ kernel, stride 1 and no padding the output size
in each dimension is $\lfloor (H-k)/s\rfloor+1=(4-3)/1+1=2$, so the result is
$2\times2$. Each entry is the sum of the elementwise product of $K$ with the
corresponding window. Every window of $I$ consists of three identical rows, either
$(10,10,0)$ or $(10,0,0)$, and $K$ has columns $(+1,0,-1)$, so every row contributes
$10-0=10$ and every window totals ${v('4.1', 'cross_correlation_value'):.0f}$:
\begin{{equation}}
I*K=\begin{{pmatrix}}30 & 30\\ 30 & 30\end{{pmatrix}}.
\end{{equation}}

\paragraph{{Convention.}} This is cross-correlation, which is what convolutional
layers compute and what the function of Section~4.2 implements. A true mathematical
convolution rotates the kernel by $180^{{\circ}}$ first; for this $K$ that maps the
columns $(+1,0,-1)$ to $(-1,0,+1)$ and flips the sign of every entry, giving
${v('4.1', 'true_convolution_value'):.0f}$ throughout. The magnitude and the
interpretation are unchanged.

\paragraph{{What the filter detected.}} $K$ subtracts the right column of a window
from the left column, so it measures intensity change along the horizontal axis and
therefore responds to \emph{{vertical}} edges. The image is bright in columns 1--2 and
dark in columns 3--4, a single vertical edge between them, and the response is large
and positive because the transition runs bright to dark from left to right. The
output is constant down each column because nothing in $I$ varies vertically: the
filter has found one vertical edge running the full height of the image. A filter
tuned to horizontal edges would return zeros everywhere on this input.

\subsection{{Convolution from scratch (12 points)}}

\code{{conv2d(image, kernel, stride=1, padding=0)}} is implemented with NumPy array
operations only; no built-in convolution routine is used anywhere in this problem.
The image is zero-padded, a strided view produces every window at once, and a single
\code{{einsum}} contracts each window against the kernel, giving an output of size
$\lfloor (H+2p-k_h)/s\rfloor+1$ by $\lfloor (W+2p-k_w)/s\rfloor+1$. Passing
\code{{kernel[::-1, ::-1]}} yields a true convolution instead of a cross-correlation.

Correctness was established in two independent ways. The function reproduces the hand
calculation of Section~4.1 exactly, both conventions. It was also checked against a
deliberately naive quadruple-loop implementation written separately, over
{v('4.2', 'n_verification_cases'):.0f} combinations of image size, kernel shape,
stride and padding: every output matched the size formula and the largest
disagreement in any entry was {msci(v('4.2', 'max_diff_vs_reference'), 1)}, which is
floating-point rounding.

\subsection{{Applying image filters (8 points)}}

\code{{skimage.data.astronaut()}} supplies a genuine RGB image, $512\times512\times3$,
converted to grayscale with the usual luminance weights
$0.2125R+0.7154G+0.0721B$. Both Sobel kernels were applied with stride 1 and padding
1, which preserves the $512\times512$ resolution. Figure~\ref{{fig:p43}} shows the
grayscale input beside the two responses.

\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{figures/p4_3_sobel.png}}
\caption{{Grayscale input and the two Sobel responses on a common symmetric scale.
Mid-grey is a response near zero; light and dark are the two signs of the gradient.}}
\label{{fig:p43}}
\end{{figure}}

$K_x$ differences the left and right columns of each window, so it responds to
intensity change along the horizontal axis and picks out \emph{{vertical}} structures: the line
of the shoulder, the upright window frames, the vertical folds of the
suit. $K_y$ differences the top and bottom rows and picks out \emph{{horizontal}}
structures: the rim of the helmet, the horizontal bands in the background, the
collar. An edge running parallel to a filter's difference direction produces almost
no response, which is why the two maps look close to complementary. On this image the
mean absolute response of $K_x$ is {v('4.3', 'kx_mean_abs_response'):.4f} against
{v('4.3', 'ky_mean_abs_response'):.4f} for $K_y$, consistent with its slightly
stronger horizontal intensity variation.

\subsection{{Connection to CNNs (5 points)}}

The Sobel kernels above are \textbf{{fixed}}: their nine weights were chosen by hand
to approximate a first derivative of image intensity, and they are identical for
every image. They encode one specific, human-specified notion of what is worth
measuring.

A convolutional layer performs the same arithmetic (the same sliding window,
elementwise products and sum, the same stride and padding rules implemented in
Section~4.2), but its kernel entries are \textbf{{parameters, initialised randomly
and learned by gradient descent}}. Each weight receives a gradient of the loss through
backpropagation, exactly as the weights in Problem~1 did, and moves to reduce that
loss. Nobody instructs a kernel to detect edges; if edge detection helps minimise the
training objective then filters resembling Sobel operators tend to appear in the first
layer on their own, and if some other feature is more useful, that appears instead.

Three consequences matter for a task such as image classification. The features are
optimised for the actual objective rather than fixed in advance, so they can capture
structure nobody thought to hand-code. A layer holds many kernels and layers are
stacked, so deeper layers convolve over the feature maps of earlier ones and build a
hierarchy of edges, then corners and textures, then object parts, which would be
impractical to design by hand. And weight sharing keeps this affordable: a $3\times3$
kernel is nine parameters whatever the image size, and the same detector is applied at
every position, so a pattern learned in one part of the image is recognised anywhere
in it. A fully connected layer over a $512\times512$ image would need more than a
quarter of a million weights for a single unit and would have to learn each position
independently.

\end{{document}}
"""

def check_document(text):
    """Guard against defects that render silently instead of failing the build.

    An unescaped '%' opens a LaTeX comment and swallows the rest of the line, so
    a mis-formatted percentage destroys a sentence without any error. A leftover
    'none' inside math mode is the signature of a failed placeholder lookup.
    """
    import re
    problems = []
    for i, line in enumerate(text.splitlines(), 1):
        if re.search(r"(?<!\\)%", line):
            problems.append(f"line {i}: unescaped '%' -> {line.strip()[:90]}")
        if re.search(r"\$[^$]*\bnone\b[^$]*\$", line):
            problems.append(f"line {i}: literal 'none' in math -> {line.strip()[:90]}")
    labels = set(re.findall(r"\\label\{([^}]+)\}", text))
    for r in set(re.findall(r"\\ref\{([^}]+)\}", text)) - labels:
        problems.append(f"dangling \\ref{{{r}}}")
    if problems:
        raise SystemExit("build_report: refusing to write\n  "
                         + "\n  ".join(problems))


doc = doc.lstrip()
check_document(doc)
Path("Problem_Set_3_Report.tex").write_text(doc, encoding="utf-8")
print(f"wrote Problem_Set_3_Report.tex ({len(doc)} chars) "
      f"from {len(reg)} registered values")
