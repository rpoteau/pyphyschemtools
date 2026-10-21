# Chemical Kinetics

**`kinetics.py` module**

This module contains three classes and one standalone function:

* **`kbasic`** — a minimal, self-contained, pedagogical tool: fit a single rate
  constant $k$ for one chosen reaction order. See [3. `kbasic`](kbasic).
* **`KORD`** (Kinetic Order of Reaction Determination) — the full tool: fit
  $k$, $Y_0$, $Y_\infty$ (and $b_\infty$) for reaction orders 0, 1 and 2, and
  decide which order the data actually support. See
  [4. `KORD`](kord).
* **`ReactionNetwork`** — the general tool: build an arbitrary network of
  (ir)reversible elementary reactions, simulate every species' concentration
  by numerically integrating the mass-action ODE system, and fit the rate
  constant(s) against experimental data. Not limited to a single A -> B
  transformation like `kbasic`/`KORD` — covers parallel reactions, consecutive
  reactions, reversible reactions, and any network built out of those. See
  [5. `ReactionNetwork`](reaction-network).
* **`superpose_plots()`** — the module's only standalone function (not a
  method of any class): overlays the fitted curves of several `KORD`
  studies on one plot, e.g. to compare the same data fitted freely vs. with
  `fix_Y0`/`fix_Yinf`. See [4.7.1 Comparing several fits](kord-advanced-compare).

## Table of Contents

* [1. Finding a Mathematical Model That Fits the Data](rmsd)
  * [1.1 RMSE](rmse)
  * [1.2 R2](r2)
* [2. Integrated rate laws: $a(t)$, $b(t)$ and half-lives for orders 0, 1, 2](rate-laws)
  * [2.1 Reactant Expression $a(t)$](rate-laws-a)
  * [2.2 Product Expression $b(t)$](rate-laws-b)
  * [2.3 Half-life $t_{1/2}$](rate-laws-halflife)
* [3. `kbasic`: a minimal, pedagogical rate-constant fit](kbasic)
  * [3.1 Purpose and limitations](kbasic-purpose)
  * [3.2 Data format](kbasic-format)
  * [3.3 Quick Start](kbasic-quickstart)
  * [3.4 Theory](kbasic-theory)
* [4. `KORD`: Kinetic Order of Reaction Determination](kord)
  * [4.1 Basic principle](principle)
    * [4.1.1 Experimental Measurements](experimental-measurements)
    * [4.1.2 Theoretical Model](theo-model-def)
  * [4.2 Quick Start](kord-quickstart)
    * [4.2.1 Basic use](kord-quickstart-basic)
    * [4.2.2 Specific fit](kord-quickstart-specific)
    * [4.2.3 With your own file](kord-quickstart-own)
  * [4.3 Fixing parameters: `fix_Y0`, `fix_Yinf`, and why $b_\infty$ never is](kord-fixing-parameters)
  * [4.4 Input](input)
  * [4.5 Model Selection: `get_best_order()` and $\Delta$BIC](model-selection)
  * [4.6 Derivation of the theoretical model](theoretical-model)
  * [4.7 Advanced use](kord-advanced)
    * [4.7.1 Comparing several fits](kord-advanced-compare)
    * [4.7.2 `from_spreadsheet` vs `load_from_spreadsheet`](kord-advanced-loading)
    * [4.7.3 Reading a different sheet / experiment](kord-advanced-sheet)
* [5. `ReactionNetwork`: general reaction networks](reaction-network)
  * [5.1 Purpose and scope](reaction-network-purpose)
  * [5.2 Quick Start](reaction-network-quickstart)
  * [5.3 Reaction syntax](reaction-network-syntax)
  * [5.4 Units of rate constants](reaction-network-units)
  * [5.5 Parallel, consecutive and reversible reactions](reaction-network-examples)
  * [5.6 Fitting rate constants](reaction-network-fit)
  * [5.7 Checking the overall equation (bilan)](reaction-network-overall)
  * [5.8 Quasi-steady-state check (QSSA)](reaction-network-qssa)
  * [5.9 Worked example 1: stratospheric ozone decomposition (QSSA)](reaction-network-ozone)
  * [5.10 Worked example 2: consecutive reactions with a fit](reaction-network-fit-example)
  * [5.11 Worked example 3: generating a synthetic kinetics dataset](reaction-network-synth-data)

---

(rmsd)=
## 1. Finding a Mathematical Model That Fits the Data

Curve fitting, in general, is the search for the free parameters — generically
$a, b, c, \ldots$ (not to be confused with this module's own $a_0$, $\alpha$,
$\beta$) — of a mathematical model $Y_\mathrm{THEO}(t;\, a, b, c, \ldots)$ that
make it track the measured data $Y_\mathrm{EXP}(t)$ as closely as possible.
"As closely as possible" needs a single number to judge by. This module uses
two, for two different jobs: the **RMSE**, which is what `scipy.optimize`
actually minimizes during the fit itself, and the **R2**, computed afterward
as a normalized, easier-to-interpret check on the result.

(rmse)=
### 1.1 RMSE

The number both `kbasic.fit()` (via `scipy.optimize.minimize`) and
`KORD.fit()` (via `scipy.optimize.curve_fit`) actually search to minimize is
the **Root-Mean-Square Deviation** between the measured values and the model
evaluated at its best-fit parameters, $\hat{Y}_\mathrm{THEO}(t)$ (the hat marks
a *prediction*, exactly as in the usual machine-learning notation), over the
experimental time points $t_k$:

$$RMSE = \sqrt{\frac{1}{n} \sum_{k=1}^{n} \{Y_\mathrm{EXP}(t_{k}) - \hat{Y}_\mathrm{THEO}(t_{k})\}^{2}}$$

*(You may see this quantity called RMSD — Root Mean Square* ***D****eviation —
elsewhere, e.g. in crystallography or molecular dynamics; RMSE — Root Mean
Square* ***E****rror, the name used throughout this module and by
scikit-learn's `root_mean_squared_error` — is the same formula, just the name
preferred in statistics/ML.)*

The RMSE should tend toward 0 as the fit improves, and it will, **provided
the chosen mathematical model is the right one for the data**: a correct
model, once its parameters are optimized, reproduces the measurements almost
exactly. A wrong model (the wrong reaction order, say) cannot be pushed to a
near-zero RMSE no matter how its parameters are tuned, simply because its
shape doesn't match the data's. `KORD` builds on this further — see
[4.5 Model Selection](model-selection) — to compare reaction orders
against each other, not just to fit one order's parameters.

Being an *absolute* deviation, expressed in $Y_\mathrm{EXP}$'s own units, the
RMSE alone doesn't say whether a given value is a good or a bad fit — that
depends on the scale of the data itself (an RMSE of 0.02 is excellent against
values around 1.4, much less so against values around 0.05). $R^2$, below,
fixes that by being unitless.

(r2)=
### 1.2 R2

$R^2$ (the **coefficient of determination**) rescales the fit's error into a
dimensionless number that doesn't depend on the units or the scale of
$Y_\mathrm{EXP}$, by comparing the model's squared error against the squared
error of the simplest possible "model" — just predicting the mean
$\bar{Y}_\mathrm{EXP}$ for every point:

$$R^{2} = 1 - \frac{\sum_{k=1}^{n}\{Y_\mathrm{EXP}(t_k) - \hat{Y}_\mathrm{THEO}(t_k)\}^{2}}{\sum_{k=1}^{n}\{Y_\mathrm{EXP}(t_k) - \bar{Y}_\mathrm{EXP}\}^{2}}$$

Both `kbasic.fit()` (`kb.r2`) and `KORD.fit()`
(`study.results[order]['R2']`) compute this with scikit-learn's `r2_score`,
alongside the RMSE they already minimize:

* **$R^2 = 1$**: a perfect fit — the model reproduces every data point exactly.
* **$R^2$ close to 1** (e.g. 0.9997 in the `kbasic` example in
  [3.3 Quick Start](kbasic-quickstart)): the model explains almost all of the
  variation in the data — the usual sign of a good fit.
* **$R^2 = 0$**: the model does no better than just predicting the mean of
  the data for every point — it has explained none of the variation.
* **$R^2 < 0$**: the model does *worse* than that flat, mean-value line.

  For a fit obtained by actually minimizing the squared error against *this
  same* data (a plain, unconstrained `fit()`/`kb.fit()` call), this should not
  happen: the mean-value line is itself reachable within `KORD`'s model
  family (set $Y_0 = Y_\infty = \bar{Y}_\mathrm{EXP}$, any $k$), so the true
  least-squares optimum can only do at least as well. A negative $R^2$
  straight out of such a fit is therefore a red flag that the optimizer did
  **not** converge to the actual optimum — check the starting guesses ($k$,
  $Y_0$, $Y_\infty$) and the bounds — rather than a sign that the chosen
  reaction order is merely a poor one (a poor-but-converged fit shows up as
  $R^2$ positive but low, not negative).

  A legitimate negative $R^2$ *can* occur in two other situations: when
  `fix_Y0`/`fix_Yinf` pins a parameter away from $\bar{Y}_\mathrm{EXP}$,
  shrinking the model family so that the mean-value line is no longer
  reachable at all; or when $Y_\mathrm{THEO}$ is evaluated on data it was
  **not** fitted on (e.g. a held-out test set, or the raw guess curve from
  `simulate_plot()` before any `fit()` call) — there, nothing guarantees the
  model beats the mean.

---

(rate-laws)=
## 2. Integrated rate laws: $a(t)$, $b(t)$ and half-lives for orders 0, 1, 2

Consider a single-reactant, single-product transformation $\alpha A
\rightarrow \beta B$, under the usual simplifying assumptions used throughout
this module:

- **Single-component reaction**: a **unique reactant** $A$ transforms into a
  **unique product** $B$.
- **Total reaction**: the reaction goes to completion (the extent of
  reaction is 100%).
- **Closed system**: no exchange of matter occurs between the system and its
  environment; only energy exchanges are possible.
- **Homogeneous system**: the concentration of any compound $C_i$ is uniform
  throughout the entire system.
- **Isochoric system**: the volume of the system remains constant throughout
  the reaction.

The rate law is defined as:

$$v = -\frac{1}{\alpha} \frac{d[A]}{dt} = k [A]^{n}$$

This section gives the integrated form $a(t)=[A](t)$, the corresponding
product expression $b(t)=[B](t)$, and the resulting half-life, for
$n \in \{0,1,2\}$. `kbasic` (section 3) and `KORD` (section 4) both build on
these — `kbasic` as the special case $\alpha=1$, $Y_\infty=0$ (see
[3.4 Theory](kbasic-theory)), `KORD` directly (see
[4.6 Derivation of the theoretical model](theoretical-model)).

(rate-laws-a)=
### 2.1 Reactant Expression $a(t)$

* **Order 0**:

$$a(t)=\begin{cases}
a_{0}-\alpha kt & t\leq\frac{a_{0}}{\alpha k}\\
0 & t>\frac{a_{0}}{\alpha k}
\end{cases}$$

* **Order 1**:

$$a(t) = a_{0} \exp(-\alpha kt)$$

* **Order 2**:

$$a(t) = \frac{1}{\frac{1}{a_{0}} + \alpha kt}$$

(rate-laws-b)=
### 2.2 Product Expression $b(t)$

Derived from mass balance ($M_{A} a(t) + M_{B} b(t) = M_{A} a_{0} = M_{B} b_{\infty}$):

* **Order 0**:

$$b(t)=\begin{cases}
\beta kt & t\leq\frac{b_{\infty}}{\beta k}\\
b_{\infty} & t>\frac{b_{\infty}}{\beta k}
\end{cases}$$

* **Order 1**:

$$b(t) = b_{\infty} \{1 - \exp(-\alpha kt)\}$$

* **Order 2**:

$$b(t)=\frac{b_{\infty}}{1+\frac{\beta}{b_{\infty}\alpha^{2}kt}}$$

(rate-laws-halflife)=
### 2.3 Half-life $t_{1/2}$

The half-life is the time at which $a(t)$ has fallen to half its initial
value, $a(t_{1/2}) = a_{0}/2$. Inverting the $a(t)$ expressions above gives,
for each order:

* **Order 0**: $t_{1/2} = \dfrac{a_{0}}{2\alpha k}$ — depends on $a_0$.
* **Order 1**: $t_{1/2} = \dfrac{\ln 2}{\alpha k}$ — independent of $a_0$.
* **Order 2**: $t_{1/2} = \dfrac{1}{\alpha k a_{0}}$ — inversely
  proportional to $a_0$.

`KORD.halflife(order, binf, k)` implements exactly these three formulas
(recovering $a_0$ from $b_\infty$, $\alpha$, $\beta$ internally), and
`fit()`/`get_best_order()` store the result in `study.results[order]['t_half']`.

---

(kbasic)=
## 3. `kbasic`: a minimal, pedagogical rate-constant fit

(kbasic-purpose)=
### 3.1 Purpose and limitations

`kbasic` is a deliberately minimal teaching tool. Given a single time series
$(t, Y)$, it finds the one rate constant $k$ that minimizes the RMSE (see
[1. Finding a Mathematical Model That Fits the Data](rmsd) above), for a
reaction order you choose yourself (0, 1 or 2). Its point is to make the
fitting procedure itself — build a model, define an RMSE, call
`scipy.optimize.minimize` — easy to read and follow end to end in a first
course, before moving on to `KORD`'s richer (and more opaque) model.

To stay that transparent, `kbasic` makes simplifying assumptions `KORD` does
not, and these are real **limitations**, not just simplifications:

* **No stoichiometry**: it assumes $\alpha = 1$ and reports an *apparent* $k$;
  it has no $\beta$, $a_0$ or $b_\infty$ at all, and cannot separate a true
  rate constant from the stoichiometric coefficient.
* **No baseline/plateau**: $Y_\infty$ is implicitly 0 — the three models in
  `_modele` decay from $Y_0$ toward 0 (the order-0 case is floored at exactly
  0 by `np.maximum`). There is no adjustable $Y_\infty$. **Do not use
  `kbasic` on data whose measured quantity levels off at a non-zero value** —
  use `KORD` instead.
* **A single free parameter**: only $k$ is optimized by `fit()`; $Y_0$ is
  always taken from the data (first point) or supplied by the user, never
  fitted — `kbasic` already does, permanently, what `KORD.fit(fix_Y0=True)`
  does on request.
* **One order at a time, no automatic model comparison**: `fit()` does report
  RMSE and R2 (see [3.3 Quick Start](kbasic-quickstart)), but unlike
  `KORD.get_best_order()` there is no AIC/BIC-based verdict across orders —
  you pick the order yourself, and `kbasic` only reports those two numbers
  for that one choice.
* **Order 2 gives an apparent $k$, not the true rate constant**: see
  [3.4 Theory](kbasic-theory) below for why, and how to recover the true
  value if you need it.

**In short**: reach for `kbasic` for a quick, from-scratch look at a single
order on data that genuinely decays to 0 at large $t$. Reach for `KORD` as
soon as $Y_\infty \ne 0$, stoichiometry matters, or you want the reaction
order itself determined for you rather than assumed.

(kbasic-format)=
### 3.2 Data format

`kbasic.read()` expects a plain two-column spreadsheet: time in the first
column, the measured quantity $Y$ in the second — nothing else. An optional
header row (text) sets the axis labels for `plot()`; without one, the
default labels "Temps"/"Y" are used.

| Temps / s | Abs |
|---|---|
| 0 | 1.40 |
| 7.59 | 1.33 |
| 15.18 | 1.27 |
| ... | ... |

There are no parameter rows here (unlike `KORD`'s format, see
[4.4 Input](input) below) — `kbasic` has no $\alpha$, $\beta$ or $a_0$ to
read from the file.

(kbasic-quickstart)=
### 3.3 Quick Start

```python
from pyphyschemtools import kbasic   # lazy-loaded from the package root

kb = kbasic()
kb.read("measurements.xlsx")   # 2-column file: time, Y
kb.fit(ordre=1)                # try order 1 (exponential decay to 0)
kb.plot()                      # data vs. model, and the RMSE landscape
```

`fit()` estimates its own starting value for $k$ directly from the data: it
inverts the chosen order's $Y(t)$ equation (see [3.4 Theory](kbasic-theory)
below) at one reference point, rather than always starting from an arbitrary
constant. You can still impose your own starting value with `k_depart=...`
if needed.

Alongside the RMSE, `fit()` also prints and stores the fit's R2 (`kb.r2`),
a quick check of goodness of fit beyond the RMSE alone:

```
Ordre 1 : k optimal = 2.9937e-01  (t1/2 = 2.3154e+00)
  ✅ RMSE: 2.36e-02
       R2: 0.9997
```

**With your own file.** Where your file lives decides how to read it, and the
difference between "Google Colab" and "my own computer" matters here, even
if both look like a notebook in a browser: on Colab, the notebook runs on a
Google machine, which cannot see the files of your PC.

```python
# On Google Colab: an upload button appears, pick your Excel file
kb = kbasic()
kb.read(colab=True)

# On your own PC: give the name (or the full path) of the file
kb = kbasic()
kb.read("measurements.xlsx")           # file in the same folder as the notebook
kb.read("C:/Users/me/data/kin.xlsx")   # or anywhere else, with its path
```

`kb.read(colab=True)` is a convenience built into `kbasic`. The same upload
can be done in two explicit steps with the generic `colab_upload()` function
(it returns the path of the uploaded file, which you can also pass to
`pandas`):

```python
from pyphyschemtools import colab_upload

path = colab_upload()    # upload button, returns the file path
kb.read(path)            # same as kb.read(colab=True)
```

(kbasic-theory)=
### 3.4 Theory

`kbasic`'s three models are the special case $\alpha=1$, $Y_\infty=0$ of the
general $a(t)$ expressions in [2.1](rate-laws-a) above, with $a_0$ renamed
$Y_0$:

* **Order 0**: $Y(t)=\max(Y_{0}-kt,\ 0)$
* **Order 1**: $Y(t)=Y_{0}\exp(-kt)$
* **Order 2**: $Y(t)=\dfrac{Y_{0}}{1+kt}$

and correspondingly, the half-life (special case $\alpha=1$, $a_0=Y_0$ of
[2.3](rate-laws-halflife)):

* **Order 0**: $t_{1/2} = \dfrac{Y_{0}}{2k}$
* **Order 1**: $t_{1/2} = \dfrac{\ln 2}{k}$
* **Order 2**: $t_{1/2} = \dfrac{1}{k}$

`fit()` computes and prints this $t_{1/2}$ alongside $k$ and the RMSE, and
stores it in `self.t_half`.

<div class="rqE">

**Order 2 caveat.** The true integrated second-order rate law is
$a(t) = a_{0}/(1+a_{0}kt)$ — note the extra $a_0$ multiplying $kt$ in the
denominator (consistent with [2.1](rate-laws-a) for $\alpha=1$). `kbasic`'s
order-2 model, $Y(t)=Y_{0}/(1+kt)$, omits that factor. As a result, the $k$
`kbasic` reports for order 2 is an **apparent** constant,
$k_{\mathrm{app}} = k_{\mathrm{true}} \times Y_{0}$, not the true molecular
rate constant — and it is *not* comparable between two experiments run at
different $Y_0$. If you need the true $k$ for order 2, divide `kbasic`'s
result by $Y_0$, or use `KORD` instead, which tracks $a_0$ explicitly.

</div>

---

(kord)=
## 4. `KORD`: Kinetic Order of Reaction Determination

(principle)=
### 4.1 Basic principle

<div class="rqE">

**Scope and limitations.** KORD only models the transformation of a single
followed species, $\alpha A \rightarrow \beta B$, with rate law
$v=k[A]^n$ ($n \in \{0,1,2\}$). This covers:

* a genuine unimolecular transformation of A (most order-1 cases);
* a genuine bimolecular reaction **of A with itself**, $2A \rightarrow \text{products}$
  ($n=2$), since there is still only one concentration, $[A]$, varying;
* a mixed bimolecular reaction between two *different* species, $A+B\rightarrow\ldots$,
  only in the **pseudo-order approximation**: one reactant held in large
  excess so its concentration stays essentially constant (the "isolation
  method"/Ostwald method, "flooding", or *dégénérescence de l'ordre* in
  French). The apparent order and $k$ obtained then describe the minority
  reactant only.

KORD does **not** correctly describe a true mixed second-order reaction where
both $[A]_0$ and $[B]_0$ are comparable and both vary independently — that
needs the (different, more complex) two-species integrated rate law, which
this module does not implement.

$\alpha$ and $\beta$ are plain stoichiometric coefficients of $\alpha A
\rightarrow \beta B$, not "association" vs. "dissociation": $\alpha=2$ means
2 molecules of A are consumed per reaction event (e.g. $2A\rightarrow P$),
$\beta=2$ means each event produces 2 molecules of B (e.g. $A\rightarrow 2B$,
a fragmentation).

</div>

(experimental-measurements)=
#### 4.1.1 Experimental Measurements ($Y_\mathrm{EXP}$)

In chemical kinetics, we track the evolution of molar concentrations over time: $C_{i}(t)$. Experimentally, we measure a physical quantity proportional to these concentrations: $Y_\mathrm{EXP}(t)$.

* **Spectrophotometry**: $Y_\mathrm{EXP} = A = \sum_{i} \epsilon_{i} \cdot l \cdot C_{i}$, where $A$ is the absorbance
* **Polarimetry**: $Y_\mathrm{EXP} = \alpha = \sum_{i} [\alpha]_{i} \cdot l \cdot C_{i}^{w}$
* **Conductivity**: $Y_\mathrm{EXP} = \sigma = \sum_{i} \lambda_{i} \cdot C_{i}$

**General Form:** $Y_\mathrm{EXP}(t) = \sum_{i} \eta_{i} \cdot C_{i}(t)$

(theo-model-def)=
#### 4.1.2 Theoretical Model ($Y_\mathrm{THEO}$)

The model has three different expressions (see
[4.6 Derivation of the theoretical model](theoretical-model)), depending on
the order of the reaction:

**Standard Formula for Order 0**:

$$\hat{Y}^{[0]}(t)=\begin{cases}
Y_{0}+\beta\dfrac{kt}{b_{\infty}}\left(Y_{\infty}-Y_{0}\right) & t\leq\dfrac{a_{0}}{\alpha k}=\dfrac{b_{\infty}}{\beta k}\\
Y_{\infty} & t>\dfrac{a_{0}}{\alpha k}=\dfrac{b_{\infty}}{\beta k}
\end{cases}$$

**Standard Formula for Order 1**:

$$\hat{Y}^{[1]}(t)=Y_{\infty}+\exp\left(-\alpha kt\right)\left(Y_{0}-Y_{\infty}\right)$$

**Standard Formula for Order 2**:

$$\hat{Y}^{[2]}(t)=
Y_{0}+\dfrac{\left(Y_{\infty}-Y_{0}\right)}{1+\dfrac{\beta}{b_{\infty}\alpha^{2}kt}}=Y_{0}+\frac{\left(Y_{\infty}-Y_{0}\right)b_{\infty}\alpha^{2}kt}{b_{\infty}\alpha^{2}kt+\beta}$$

$\hat{Y}_\mathrm{THEO}$ is defined by two types of values: **fixed parameters**
(input by the user) and **adjustable variables** (optimized by the algorithm,
unless held fixed — see [4.3](kord-fixing-parameters) below). $b_\infty$ is
neither: it is always *derived*, never a free fit parameter.

| Type | Symbol | Name | Default | Notes |
|---|---|---|---|---|
| Fixed | $n$ | Reaction order | — (chosen via `fit(order=...)`) | 0, 1 or 2; one order at a time |
| Fixed | $\alpha$, $\beta$ | Stoichiometric coefficients | — (user-provided) | smallest possible positive integers |
| Fixed | $a_{0}$ | Initial concentration of A | — (user-provided) | $\hat{Y}_\mathrm{THEO}$ is independent of $a_0$ for Order 1 |
| *Derived* | $b_{\infty}$ | Final concentration of B | $b_{\infty}=a_{0}\,\beta/\alpha$ | **never free** — see [4.3](kord-fixing-parameters) for why |
| Adjustable | $k$ | Rate constant | smart guess from the initial slope (`k_guess`) | always free |
| Adjustable | $Y_{0}$ | Initial value of $Y$ | `Y_exp[0]` (`Y0_guess`) | free unless `fix_Y0=True` |
| Adjustable | $Y_{\infty}$ | Final value of $Y$ | `Y_exp[-1]` (`Yinf_guess`) | free unless `fix_Yinf=True` |

So, by default — no switch passed — exactly three parameters are optimized:
$k$, $Y_0$ and $Y_\infty$ ($b_\infty$ is never among them, for any order).
`fix_Y0`/`fix_Yinf` remove $Y_0$/$Y_\infty$ from that list, leaving as few as
just $k$ when both are set.

The optimization is performed for a specific reaction order at a time to determine which model best describes the experimental data.

<div class="rqE">

By default, KORD chooses the first and last $Y_\mathrm{EXP}$ values as $Y_{0}$ and $Y_{\infty}$. And a default $k$ value is also setup by KORD. If you need to change that because of a convergence issue, ensure your starting values for $k$, $Y_{0}$ and $Y_{\infty}$ are realistic to help the algorithm converge. 

</div>

---

(kord-quickstart)=
### 4.2 Quick Start

(kord-quickstart-basic)=
#### 4.2.1 Basic use

```python
from pyphyschemtools import KORD, get_ppct_data

# Load a bundled example file (experiment 1) and build the KORD instance in one call:
file = get_ppct_data("Kinetics/Test_Examples_VP.ods")
study = KORD.from_spreadsheet(file, exp_number=1)

study.get_best_order()      # fits orders 0, 1, 2 and reports the best one
study.plot_all_fits()       # visual comparison of the three fits
```

`plot_all_fits()` fits (or reuses, from `study.results`) all three orders and
plots them together. An order already cached in `study.results` from an
earlier `fit()` call is **not** re-fitted here, even if the `fix_Y0`/`fix_Yinf`
passed to `plot_all_fits()` differ from the ones that produced that cached
fit — your earlier, explicit choice is kept rather than silently overwritten.
If that creates a mismatch (some orders fitted under different
`fix_Y0`/`fix_Yinf` than the others), a warning table is printed, since
comparing orders fitted under different constraints can be misleading. The
same rule, and the same warning, apply to
[`superpose_plots()`](kord-advanced-compare) below.

(kord-quickstart-specific)=
#### 4.2.2 Specific fit

You can also fix $Y_0$ (or $Y_\infty$) at a chosen value instead of letting
the optimizer adjust it freely — fully explained in
[4.3](kord-fixing-parameters) below:

```python
# A better starting point than the raw first data point (e.g. because it's
# visibly noisy), used here as a hard constraint rather than a free guess:
study = KORD.from_spreadsheet(file, exp_number=1, Y0_guess=1.40)
study.simulate_plot()              # the three guesses (order 0, 1, 2) vs. the data
study.fit(order=1, fix_Y0=True)    # Y0 is held fixed at 1.40

# Changing the guess afterwards and fitting order 1 again works the same way,
# without re-creating study:
study.Y0_guess = 1.35
study.fit(order=1, fix_Y0=True)             # now fixed at 1.35
study.simulate_plot(order=1)                # only the order-1 guess vs. the data
```

`simulate_plot(order=None, show_experimental=True, save_img=None)` plots the
current *guess* curve(s) — before or without fitting — against the measured
data (shown by default; pass `show_experimental=False` to hide it). Use
`order=None` (the default) to compare all three orders' guesses at once, or
`order=0`/`1`/`2` to see just one.

Note that `fit()` itself never plots anything — it only computes and stores
numbers in `study.results[order]`; `plot_all_fits()` (above) or
`simulate_plot()` (just for the initial guesses, before any fit) are what
draw a figure.

(kord-quickstart-own)=
#### 4.2.3 With your own file

As for `kbasic` (see [3.3 Quick Start](kbasic-quickstart)), it depends on
where the notebook runs. The file must follow the format described in
[4.4 Input](input).

```python
from pyphyschemtools import KORD

# On Google Colab: an upload button appears, pick your spreadsheet
study = KORD.from_spreadsheet(exp_number=1, colab=True)

# On your own PC: give the name (or the full path) of the file
study = KORD.from_spreadsheet("my_kinetics.xlsx", exp_number=1)

study.get_best_order()
study.plot_all_fits()
```

The two-step version also works, with `colab_upload()`:

```python
from pyphyschemtools import KORD, colab_upload

path = colab_upload()    # Colab only: upload button, returns the file path
study = KORD.from_spreadsheet(path, exp_number=1)
```

And a line that works in both environments:

```python
import sys
study = KORD.from_spreadsheet("my_kinetics.xlsx", exp_number=1,
                              colab="google.colab" in sys.modules)
```

---

(kord-fixing-parameters)=
### 4.3 Fixing parameters: `fix_Y0`, `fix_Yinf`, and why $b_\infty$ never is

Letting the optimizer adjust $Y_0$ freely can let it drift away from the actually
measured starting point just to minimize the RMSE — which defeats the purpose of
measuring it in the first place. Conversely, some experiments have a $Y_\infty$ that
is known on physical/chemical grounds (e.g. a non-absorbing product, $Y_\infty = 0$),
and leaving it free can make the optimizer report a spurious non-zero plateau.
`fit()` therefore accepts two independent boolean switches, `fix_Y0` and
`fix_Yinf`, to hold either one at its guess value instead of optimizing it
(only $k$ remains free when both are `True` — see the note on $b_\infty$
below, which is never a free fit parameter for any order). BIC is computed
using the actual number of free parameters, so
model comparison across orders stays meaningful whichever switches are used
(see [4.5 Model Selection](model-selection) below — `fit()` also stores an
AIC for reference in `study.results[order]['AIC']`, but the printed verdict
is based on BIC only).

Unless you pass `Y0_guess`/`Yinf_guess` yourself when creating the `KORD`
instance, $Y_0$'s guess defaults to `Y_exp[0]` (the first measured point) and
$Y_\infty$'s to `Y_exp[-1]` (the last). **The guess and the fix switch are
independent**: giving your own `Y0_guess` does not force `fix_Y0` one way or
the other — it only changes *which* value is used, whether it ends up fixed
or merely as the optimizer's starting point.

`fit(order=1, fix_Y0=False, fix_Yinf=False)` never takes `k_guess`/`Y0_guess`/
`Yinf_guess`/`binf_guess` as arguments: it always reads the current
`self.k_guess`, `self.Y0_guess`, `self.Yinf_guess`, `self.binf_guess` —
set once at `KORD(...)`/`from_spreadsheet(...)` creation time, or changed
later by assigning the attribute directly. See [4.2.2](kord-quickstart-specific)
above for an example.

**There is no `fix_binf` switch, because $b_\infty$ is never a free fit
parameter in the first place.** It is always derived from mass balance,
$b_\infty = a_0\,\beta/\alpha$, for every order — the same relation Order 1
already relied on. This is not a simplification: for Orders 0 and 2, if $k$
and $b_\infty$ were *both* left free, the model would depend on them only
through a single combined ratio ($\beta k/b_\infty$ for order 0,
$\beta/(b_\infty\alpha^2 k)$ for order 2). Infinitely many $(k, b_\infty)$
pairs then trace the exact same curve and give the exact same RMSE, so
`curve_fit` would converge to an arbitrary point on that family — whichever
one happens to sit closest to `binf_guess` — rather than to a unique,
physically meaningful value. Deriving $b_\infty$ removes that degeneracy:
`binf_guess` still exists (used only to draw the guess curve in
[`simulate_plot()`](kord-quickstart-specific) before any fit is run), but
`fit()` itself ignores it and always uses the mass-balance value, so `k` is
the only quantity the data actually has to say about $b_\infty$'s combined
effect — and it reports a `binf` value you can trust.

This buys physical consistency, but it also removes the fit's ability to
absorb noise or systematic error on that one point (e.g. mixing dead-time on
the first point, or a $Y_\infty$ assumed from "known" chemistry that is in
fact slightly off): a wrong fixed value does not just affect that one
parameter, it biases $k$ (and whichever of $Y_0$/$Y_\infty$/$b_\infty$ is
still free) as the optimizer tries to compensate for it elsewhere.

One case is fixed by necessity rather than by choice: for order 1,
$\hat{Y}^{[1]}(t)=Y_{\infty}+\exp(-\alpha kt)(Y_{0}-Y_{\infty})$ does not
depend on $b_\infty$ at all, so there is nothing in the order-1 curve shape
that could identify it — `fit()` always derives $b_\infty$ from $a_0$,
$\alpha$, $\beta$ instead of optimizing it, for order 1 only.

(input)=
### 4.4 Input

Data input is performed through a structured spreadsheet file (`.xlsx`, `.ods`,
...), read by `KORD.load_from_spreadsheet(file_path, exp_number, sheet_name=0)`
— or, more conveniently, by `KORD.from_spreadsheet(...)`, which calls it for
you (see [4.2 Quick Start](kord-quickstart)). Within the chosen sheet, each
experiment occupies a pair of columns (t, Y), laid out like this:

| Row | Column $2k{-}1$ (t) | Column $2k$ (Y) |
|---|---|---|
| 1 | header for $t$ | header for $Y$ |
| 2 | [A]0 | $[A]_0$ value |
| 3 | alpha | $\alpha$ value |
| 4 | beta | $\beta$ value |
| 5+ | $t$ data | $Y_\mathrm{EXP}$ data |

where $k$ is the experiment number (`exp_number`, 1-indexed). In the
Column $2k{-}1$ (t), `[A]0`, `alpha` and `beta` are unused labels, written in
the table purely for clarity — only the Column $2k$ (Y) cells in rows 2-4 are
actually read. Several experiments can sit side by side in the same sheet
this way, each in its own column pair; `load_from_spreadsheet` returns, for
one experiment: `t`, `Y`, `(label_t, label_Y)`, `(a0, alpha, beta)`.

As a sanity check, `load_from_spreadsheet` inspects rows 2-4 of the **t**
column (which should be blank or a text label, never real data at this
stage): if it finds a genuine number there instead, it prints a warning
reminding you of the expected format, since this usually means the data
doesn't follow the layout above (e.g. actual $(t,Y)$ data starting at row 2
instead of row 5).

(model-selection)=
### 4.5 Model Selection: `get_best_order()` and $\Delta$BIC

`get_best_order()` fits orders 0, 1 and 2 (any order not already in
`study.results`) and prints a conclusion, in two separate steps:

1. **Best model**: the order with the lowest **RMSE** (plain goodness of fit).
2. **Final Statistical Verdict**: a confidence label on how clearly the data
   favor one order over the others, based on the **BIC** (Bayesian Information
   Criterion) rather than the RMSE. BIC trades off goodness of fit against the
   number of free parameters $p$ actually optimized (fewer free parameters is
   rewarded — this matters more once you can fix $Y_0$/$Y_\infty$ with
   `fix_Y0`/`fix_Yinf`, since that changes $p$):

   $$BIC = n\ln(RMSE^{2}) + p\ln(n)$$

   where $n$ is the number of experimental points (the same for every order,
   since it's the same dataset) and $p$ is the number of parameters actually
   optimized **for that order** — normally 3 for order 1 ($k$, $Y_0$,
   $Y_\infty$; $b_\infty$ is fixed) and 4 for orders 0 and 2 ($k$, $Y_0$,
   $Y_\infty$, $b_\infty$), one less for each of $Y_0$/$Y_\infty$ held fixed
   via `fix_Y0`/`fix_Yinf`.

   `fit()`'s printed GUESS/OPTIM block also shows an **AIC** (Akaike
   Information Criterion) value alongside BIC, for reference:

   $$AIC = n\ln(RMSE^{2}) + 2p$$

   Same $n$ and $p$ as BIC above; the only difference is the penalty for
   extra parameters ($2p$ instead of $p\ln(n)$), which is milder — AIC
   penalizes model complexity less harshly than BIC does once $n$ is more
   than a handful of points. `study.results[order]['AIC']` stores it, but
   **the printed verdict below is based on BIC only** — AIC is kept for
   reference, not used in `get_best_order()`'s conclusion.

   The three orders (0, 1, 2) are ranked by their BIC (lowest = best), and

   $$\Delta BIC = BIC(\text{2}^{\text{nd}}\text{ best}) - BIC(\text{best})$$

   is the strength of evidence for the best-BIC order over its closest rival.
   The verdict label follows the standard scale from Kass, R. E., & Raftery,
   A. E. (1995), "Bayes Factors", *J. Am. Stat. Assoc.*, 90(430), 773–795 —
   see also the [Bayesian information criterion](https://en.wikipedia.org/wiki/Bayesian_information_criterion)
   and [Bayes factor](https://en.wikipedia.org/wiki/Bayes_factor) Wikipedia
   pages (ΔBIC approximates twice the log Bayes factor between the two
   models):

   | $\Delta$BIC | Verdict |
   |---|---|
   | < 2  | Weak / Inconclusive |
   | 2–6  | Positive |
   | 6–10 | Strong |
   | > 10 | Decisive |

   Example: `Model Selection Confidence: Decisive`, `ΔBIC (Best [order 1] vs
   2nd Best [order 2]): 359.15` means order 1 beats order 2 by 359 BIC points —
   an overwhelming margin, far past the "decisive" threshold of 10: the data
   leave essentially no doubt that order 1 is the right kinetic order.

   Note: the "Best model" (step 1, by RMSE) and the two orders compared in the
   verdict (step 2, by BIC) are usually the same, but are determined
   independently and need not coincide exactly, especially when parameters are
   fixed asymmetrically across orders.

---

(theoretical-model)=
### 4.6 Derivation of the theoretical model

$\hat{Y}_\mathrm{THEO}(t)$ is built directly from the integrated rate laws of
[section 2](rate-laws): it is a linear combination of the reactant
expression [$a(t)$](rate-laws-a) and the product expression
[$b(t)$](rate-laws-b) introduced there, under the same assumptions
(single-component, total, closed, homogeneous, isochoric reaction):

$$\hat{Y}_\mathrm{THEO}(t) = \frac{Y_{0}}{a_{0}} \cdot a(t) + \frac{Y_{\infty}}{b_{\infty}} \cdot b(t)$$

Given the expressions for $a(t)$ and $b(t)$ at orders 0, 1 and 2 (see
[2.1](rate-laws-a) and [2.2](rate-laws-b)), this gives:

**Standard Formula for Order 0**:

$$\hat{Y}^{[0]}(t)=\begin{cases}
Y_{0}+\beta\dfrac{kt}{b_{\infty}}\left(Y_{\infty}-Y_{0}\right) & t\leq\dfrac{a_{0}}{\alpha k}=\dfrac{b_{\infty}}{\beta k}\\
Y_{\infty} & t>\dfrac{a_{0}}{\alpha k}=\dfrac{b_{\infty}}{\beta k}
\end{cases}$$

<div class="rqE">
    
Warning: This mathematical model for Order 0 is a linear equation. Unlike Order 1 or 2, this linear model does not naturally plateau. Depending on the values of $k$ and $t$, the model may predict non-physical values (e.g., negative absorbance or negative concentration) if the time $t$ exceeds the theoretical completion time $t_{\mathrm{end}} = \frac{a_{0}}{\alpha k}=\frac{b_{\infty}}{\beta k}$. These values are mathematical artifacts and should be ignored.
    
</div>
<br>

**Standard Formula for Order 1**:

$$\hat{Y}^{[1]}(t)=Y_{\infty}+\exp\left(-\alpha kt\right)\left(Y_{0}-Y_{\infty}\right)$$

**Standard Formula for Order 2**:

$$\hat{Y}^{[2]}(t)=
Y_{0}+\dfrac{\left(Y_{\infty}-Y_{0}\right)}{1+\dfrac{\beta}{b_{\infty}\alpha^{2}kt}}=Y_{0}+\frac{\left(Y_{\infty}-Y_{0}\right)b_{\infty}\alpha^{2}kt}{b_{\infty}\alpha^{2}kt+\beta}$$

(kord-advanced)=
### 4.7 Advanced use

(kord-advanced-compare)=
#### 4.7.1 Comparing several fits

`superpose_plots(studies, order=1, show_experimental=True, labels=None,
fix_Y0=False, fix_Yinf=False, save_img=None)` is a standalone function (not
a `KORD` method, since it takes a *list* of instances) that overlays the
**fitted** curve of one chosen order — and, by default, each study's data —
across several `KORD` studies on the same graph. Typical use: the same data
fitted under different `fix_Y0`/`fix_Yinf` settings, to see directly how
that choice moves the fitted curve.

Being standalone, it is called as `superpose_plots(...)`, imported
separately (`from pyphyschemtools.kinetics import superpose_plots`) —
**not** as `KORD.superpose_plots(...)`, which raises an `AttributeError`
since `KORD` has no such method.

`study.copy()` returns an independent copy — data, guesses and cached
results are duplicated, not shared, so fitting one does not affect the
other — which is the natural way to get two variants of the same study to
compare:

```python
from pyphyschemtools.kinetics import superpose_plots

study_free = KORD.from_spreadsheet(file, exp_number=1)
study_fixed = study_free.copy()

study_free.fit(order=1, fix_Y0=False)   # Y0 optimized
study_fixed.fit(order=1, fix_Y0=True)   # Y0 held fixed at its guess value

superpose_plots([study_free, study_fixed], order=1)
```

Each study's legend entry is tagged automatically (`free fit`, `Y0 fixed`,
`Yinf fixed`, or both), based on the actual fit found in `study.results`.
If an order isn't fitted yet for a given study, it is fitted now with
`fix_Y0`/`fix_Yinf`; if it's already cached under *different* settings,
the cache is kept as-is (same rule as `plot_all_fits()`, see
[4.2.1](kord-quickstart-basic) above) and a warning is printed for that
study.

(kord-advanced-loading)=
#### 4.7.2 `from_spreadsheet` vs `load_from_spreadsheet`

`from_spreadsheet(file_path, exp_number, sheet_name=0, show_data=True, colab=False, **kwargs)`
is a convenience constructor: it reads the experiment with
`load_from_spreadsheet` (below) and forwards `headers`, `a0`, `alpha`, `beta`
to `KORD()` for you — any extra keyword (`k_guess`, `Y0_guess`, `verbose`,
...) is passed straight through to `KORD.__init__`. The returned `study` is a
perfectly ordinary `KORD` instance: all its attributes (`study.k_guess`,
`study.Y0_guess`, `study.a0`, ...) are set exactly as they would be from a
direct `KORD(...)` call. `file_path` can be any local path (`"data/exp.ods"`,
`get_ppct_data(...)` is just one way to point at a *bundled* example file —
your own files work the same way), or, on Google Colab, pass `colab=True`
and omit `file_path`: an upload button appears and the chosen file is read
instead (same mechanism as `kbasic.read(colab=True)`). `show_data=True`
(the default) also prints the usual summary table.

`from_spreadsheet` is what you should reach for by default — it's what every
example in this doc uses. The lower-level `load_from_spreadsheet` (which it
calls internally) stays available separately, and is worth using directly
when you need to inspect, filter or correct the loaded data or parameters
**before** creating the `KORD` instance — e.g. overriding a stoichiometry
value read from the file, dropping an outlier point, or feeding the same
raw $(t, Y)$ into `kbasic` instead:

```python
t, Y, headers, parameters = KORD.load_from_spreadsheet(file, exp_number=1)
# ... inspect / fix t, Y, parameters here if needed ...
study = KORD(t, Y, headers, a0=parameters[0], alpha=parameters[1], beta=parameters[2])
```

(kord-advanced-sheet)=
#### 4.7.3 Reading a different sheet / experiment

`load_from_spreadsheet(file_path, exp_number, sheet_name=0, show_data=True)`
(formerly named `load_from_excel` — same method; it reads with
`pandas.read_excel`, so it already worked on `.ods` files as well as `.xlsx`,
and the old, Excel-specific name was misleading. `load_from_excel` still
works, as a deprecated alias.) reads **one** sheet (`sheet_name`, by index or
by name — the first sheet by default) and, within that sheet, picks the
column pair identified by `exp_number` (1 for the first pair of columns, 2
for the next pair, etc. — see the data format in
[4.4 Input](input) above). `from_spreadsheet` takes the exact same
`sheet_name`/`exp_number` arguments and forwards them:

* If every experiment lives in its own pair of columns **within the same
  sheet**, just change `exp_number` (`exp_number=2` for the second experiment).
* If each experiment instead lives on its **own sheet/tab** of the workbook,
  pass `sheet_name` explicitly, by index or by name, e.g.:

  ```python
  study = KORD.from_spreadsheet(file, exp_number=1, sheet_name=1)
  # or, by sheet name:
  study = KORD.from_spreadsheet(file, exp_number=1, sheet_name="Feuille2")
  ```

---

(reaction-network)=
## 5. `ReactionNetwork`: general reaction networks

(reaction-network-purpose)=
### 5.1 Purpose and scope

`kbasic` and `KORD` both rely on **closed-form integrated rate laws**, valid
only for a single reactant transforming into a single product
($\alpha A \rightarrow \beta B$), of order 0, 1 or 2 (see
[2. Integrated rate laws](rate-laws)). That covers a lot of real experiments,
but not every mechanism a student or a researcher might need to fit:

* **Parallel (competitive) reactions**: $A \rightarrow B$ and $A \rightarrow C$
  at the same time, with their own rate constants $k_1$, $k_2$.
* **Consecutive reactions**: $A \rightarrow B \rightarrow C$, where $B$ is an
  intermediate that builds up, then decays — its concentration goes through a
  maximum (the classic setting for introducing the quasi-steady-state
  approximation, QSSA).
* **Reversible reactions**: $A \rightleftharpoons B$, which relax toward an
  equilibrium set by the ratio of the forward and backward rate constants,
  not all the way to $A=0$.
* Any **network** combining several of the above.

None of these have a simple closed-form solution in general.
`ReactionNetwork` drops the closed-form restriction entirely: it builds the
system of ODEs for an arbitrary set of elementary reactions straight from
**mass-action kinetics** (a reaction's rate is $k$ times the product of its
reactants' concentrations, each raised to its own stoichiometric
coefficient), integrates it numerically with
[`scipy.integrate.solve_ivp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html),
and fits the rate constant(s) against experimental data the same way `KORD`
does — a measured signal as a linear combination of concentrations,
$Y_\mathrm{EXP}(t) = \sum_i \eta_i\, C_i(t)$ (see
[4.1.1 Experimental Measurements](experimental-measurements)).

This is a genuine generalization: `kbasic`/`KORD`'s orders 0/1/2 are just the
special case of a single $A \rightarrow B$ reaction, and the quasi-steady-state
approximation can now be checked **numerically** against the exact solution
of a consecutive network, instead of only illustrated on a closed-form
special case.

Mass action is the standard assumption for an *elementary* step; it is not
automatically valid for an overall, multi-step reaction written as a single
line (the same caveat `KORD`'s 4.2 Principle raises for pseudo-order
kinetics) — build the actual elementary steps of the mechanism, not a
shortcut lumped equation, if you want physically meaningful rate constants.

(reaction-network-quickstart)=
### 5.2 Quick Start

Four steps, in order: read the experimental data, declare the reactions, set
the initial concentrations, then either `fit()` against the data or
`simulate()`/`plot()` directly.

**Step 1: read the data.** `fit()` just needs two plain arrays, `t_exp` (time)
and `Y_exp` (the measured quantity), so any spreadsheet can be read with
`pandas`. Start with the bundled example file, which works everywhere (Colab
or your own computer):

```python
import pandas as pd
from pyphyschemtools import get_ppct_data

file = get_ppct_data("Kinetics/consecutive_fit_data.ods")
df = pd.read_excel(file, engine="odf")   # .ods needs the odfpy package
display(df)
t_exp = df["Time / min"].values
Y_exp = df["Y"].values           # the intermediate, as actually observed
```

To use **your own file** (two columns: time, then the measured quantity),
only the first lines change. The rest of the example is identical:

```python
# On Google Colab: an upload button appears, pick your file
from pyphyschemtools import colab_upload
file = colab_upload()

# On your own PC: give the name (or the full path) of the file
file = "my_kinetics.xlsx"

df = pd.read_excel(file)          # for an .ods file: engine="odf"
display(df)
t_exp = df["Time / min"].values
Y_exp = df["Y"].values           # the intermediate, as actually observed
```

**Steps 2 to 4: build the network and fit.**

```python
from pyphyschemtools import ReactionNetwork

net = ReactionNetwork()
net.add_reaction("A -> B", k_guess=0.5)
net.add_reaction("B -> C", k_guess=0.1)
net.set_initial_conditions({"A": 1.0, "B": 0.0, "C": 0.0})

# If only B is measured (e.g. it's the one colored/absorbing species):
net.fit(t_exp, Y_exp, observed_species="B")
net.plot(t_exp, Y_exp, time_unit="min")   # all three species + the experimental overlay
net.summary()                             # fitted k1, k2 for each reaction
```

The bundled file tracks the intermediate of an $A\rightarrow I\rightarrow P$
mechanism, which is why it is declared here as `B` (the middle species); the
time is in minutes, hence `time_unit="min"`. [5.10 Worked example 2](reaction-network-fit-example)
runs the same fit with the species named $A$, $I$, $P$.

`plot()` draws every species by default; pass `species=[...]` to zoom in on a
subset — handy for a reactive intermediate whose concentration is orders of
magnitude below the reactant/product, especially combined with
`log_scale=True`:

```python
net.plot(species=["A", "B"], log_scale=True, t_max=20, time_unit="s")
```

Each name is checked against the network's actual species first, so a typo
(or a species that was never declared via `add_reaction()`) raises a clear
error listing the known ones, instead of silently plotting nothing for it.

**No fit needed at all** if you already know the rate constants and just want
to see how the concentrations evolve: `simulate(t)` integrates the ODE system
with the guesses given to `add_reaction()` — or the fitted values, once
`fit()` has been called — and returns a `{species: concentration array}`
dict:

```python
import numpy as np
t = np.linspace(0, 20, 200)
C = net.simulate(t)            # {"A": array(...), "B": array(...), "C": array(...)}
```

`copy()` returns an independent deep copy of the whole network (reactions,
initial conditions, any fitted result), e.g. to fit the same network twice
under different conditions — different `weights`, a different subset of the
data — without having to declare every reaction a second time, and without
one `fit()` call overwriting the other's `results`.

`plot()`/`plot_qssa()` save the figure to disk, in addition to showing it,
with `save_img="path/to/figure.png"` (any extension matplotlib's `savefig`
understands — png, pdf, svg...); both ask before overwriting a file that
already exists. Pass `time_unit="s"` (or `"min"`, `"h"`, `"yr"`...) to label
the x-axis "Time / s" — see [5.4 Units](reaction-network-units) for why this
is just a label and not an actual unit conversion.

(reaction-network-syntax)=
### 5.3 Reaction syntax

`add_reaction(equation, k_guess=..., kf_guess=..., kr_guess=..., order=...,
order_back=...)` parses a plain chemical-equation string:

* Species are **single-word** names — letters/digits/underscore, Unicode
  accents included (`A`, `B2`, `H2O`, `ethylene`, `éthylène`...). A species
  name cannot contain a space, so a multi-word compound needs a short alias
  first, exactly as on paper: `A = ethylene, B = cyclobutane`, then
  `net.add_reaction("2 A -> B", ...)`.
* Every name is **normalized** on parsing, so the same compound is always
  recognized as the same species no matter how it's typed across different
  `add_reaction()` calls, case-insensitively — `"ethylene"`, `"Ethylene"`,
  `"ETHYLENE"` and `"éthylène"` all resolve to the single species
  "Ethylene" (accents are stripped, and the very first spelling seen
  becomes the one used for the rest of the network). This matching is
  purely about *recognizing* the same name typed differently — it never
  lowercases a letter you actually typed in uppercase, precisely so that
  a short formula with no digit in it, like `"NO"` (nitric oxide),
  `"CO"`, `"OH"`, `"HCl"` or `"HCN"`, is never mistaken for a spelled-out
  word and mangled into something like `"No"` (indistinguishable from the
  element symbol for Nobelium). So `"2 ethylene -> cyclobutane"` and
  `"Cyclobutane -> 2 ETHYLENE"` are recognized as the forward/backward of
  the very same reaction between the very same two species, while
  `"N2O5 -> NO2 + NO3"` keeps `"NO2"`/`"NO3"` exactly as written, however
  later reactions spell them.
* Terms are separated by `+`, with an optional leading integer stoichiometric
  coefficient: `"A + 2 B -> C"`, or `"2 ethylene -> cyclobutane"` directly if
  you'd rather not alias.
* `->` for an **irreversible** reaction (give `k_guess`); `<=>` for a
  **reversible** one (give `kf_guess`/`kr_guess` — each defaults to 1.0 if
  omitted, and `kf_guess` also accepts the value passed as `k_guess`, for
  convenience). Reversible reactions are fully supported, with any kinetic
  order (see below) on either side.
* Any species not seen in an earlier call is added to the network
  automatically, in order of first appearance — no separate species
  declaration needed.
* All rate constants are constrained to be $\geq 0$ (physical consistency,
  exactly like $k$/$b_\infty$ in `KORD.fit()`).
* **Kinetic order vs. stoichiometry**: by default, the rate law is
  elementary mass action — the exponent on each reactant (product, for the
  backward direction) equals its stoichiometric coefficient. Pass
  `order={"A": 1.5}` (forward) and/or `order_back={"B": 2}` (backward,
  reversible only) to override this species by species, when the actual
  kinetic order differs from the written stoichiometry (the same caveat
  [4.1 Basic principle](principle) raises for `KORD`'s pseudo-order case) — e.g. a
  reversible reaction that is first order forward but second order
  backward: `net.add_reaction("A <=> 2 B", kf_guess=0.5, kr_guess=0.2,
  order_back={"B": 2})` (this one actually matches the stoichiometry, so
  `order_back` isn't even needed here — it's for when it *doesn't* match).

(reaction-network-units)=
### 5.4 Units of rate constants

`ReactionNetwork` is entirely **unit-agnostic**: `t`, `C0` and every rate
constant are just numbers — there is no internal unit system, no
conversion, and nothing to configure. You choose the units once (e.g.
mol·L⁻¹ for concentration and s for time) and keep every input in that same
system; the output comes out consistently in it too.

Given that, the *unit* a rate constant carries is fixed by the reaction's
overall kinetic order $n$ (the sum of the forward exponents — the
stoichiometric coefficients by default, or the `order` dict when it's been
overridden), from the requirement that the rate itself always has units of
concentration/time:

$$[k] = \text{(concentration)}^{1-n}\cdot\text{(time)}^{-1}$$

* order 1 (e.g. `"A -> B"`): $[k]=\text{time}^{-1}$ (e.g. s⁻¹) — same as
  `kbasic`'s order-1 case.
* order 2 (e.g. `"A + B -> C"` or `"2 A -> B"`): $[k]=\text{concentration}^{-1}\cdot\text{time}^{-1}$
  (e.g. L·mol⁻¹·s⁻¹).
* order 0: $[k]=\text{concentration}\cdot\text{time}^{-1}$.

A reversible reaction's $k_f$ and $k_r$ don't have to share the same unit —
each follows this rule from its own side's order, e.g. `"A <=> 2 B"` has
$k_f$ in time⁻¹ (order 1 forward) but $k_r$ in concentration⁻¹·time⁻¹ (order
2 backward, unless overridden via `order_back`).

Since the module itself carries no unit label, `plot()`/`plot_qssa()` accept
`time_unit="s"` (or `"min"`, `"h"`, `"yr"`...) purely to annotate the x-axis
as "Time / s" — this labels the figure only, it does not check or convert
anything, so it's on you to make sure `time_unit` actually matches the time
unit `t`/`t_exp` and the rate constants were expressed in.

(reaction-network-examples)=
### 5.5 Parallel, consecutive and reversible reactions

**Parallel reactions** ($A\rightarrow B$, $A\rightarrow C$, with $k_1$, $k_2$):
the classic result is that the **branching ratio** $[B]/[C]$ equals $k_1/k_2$
at every instant (not just at completion), since both channels drain $A$ at
rates proportional to their own $k$:

```python
net = ReactionNetwork()
net.add_reaction("A -> B", k_guess=0.5)
net.add_reaction("A -> C", k_guess=0.2)
net.set_initial_conditions({"A": 2.0, "B": 0.0, "C": 0.0})
```

**Consecutive reactions** ($A\rightarrow B\rightarrow C$): $[B]$ rises then
falls, peaking at $t_\mathrm{max}=\dfrac{\ln(k_2/k_1)}{k_2-k_1}$ (the Bateman
equations give the exact closed form for this particular two-step case —
`ReactionNetwork`'s numerical solution matches it to $\sim 10^{-9}$). This is
the standard setting to introduce the **quasi-steady-state approximation**
(QSSA): when $k_2\gg k_1$, $B$ is consumed almost as fast as it forms, so
$d[B]/dt\approx 0$ is a good approximation after a short induction period —
checkable directly here by comparing `net.simulate(t)['B']` for $k_2\gg k_1$
against the QSSA prediction $[B]_\mathrm{QSSA}\approx \frac{k_1}{k_2}[A](t)$.

```python
net = ReactionNetwork()
net.add_reaction("A -> B", k_guess=0.6)
net.add_reaction("B -> C", k_guess=0.15)
net.set_initial_conditions({"A": 1.5, "B": 0.0, "C": 0.0})
```

**Reversible reactions** ($A\rightleftharpoons B$): the system relaxes toward
an equilibrium with $[B]_\mathrm{eq}/[A]_\mathrm{eq}=k_f/k_r$ (not to
$[A]=0$) — `ReactionNetwork` includes the backward rate in $A$'s and $B$'s
ODEs automatically as soon as a reaction is declared with `<=>`:

```python
net = ReactionNetwork()
net.add_reaction("A <=> B", kf_guess=0.8, kr_guess=0.3)
net.set_initial_conditions({"A": 1.0, "B": 0.0})
```

(reaction-network-fit)=
### 5.6 Fitting rate constants

`fit(t_exp, Y_exp, weights=...)` or the `observed_species="X"` shortcut
(equivalent to `weights={"X": 1.0}`) fits every reaction's rate constant(s)
by nonlinear least squares, wrapping the ODE integration inside
`scipy.optimize.curve_fit` exactly as `KORD.fit()` wraps its closed-form
model. `weights={...}` is for a signal mixing several species, e.g. a UV-Vis
absorbance both $A$ and $B$ contribute to: `weights={"A": eta_A, "B": eta_B}`.

The returned dict (also stored in `net.results`) has the same shape as
`KORD.fit()`'s: `{'k': {param_name: value, ...}, 'RMSE', 'R2', 'AIC', 'BIC',
'weights'}` — `'k'` has one entry per irreversible reaction (`k1`, `k2`, ...,
numbered by the order reactions were added), two per reversible reaction:
`k{i}` for the forward constant and `k-{i}` for the reverse one — the
classic $k_1$/$k_{-1}$ notation for the two directions of the *same* step
`i` (e.g. a reversible reaction added first gives `k1`/`k-1`; the next
reaction added, even if irreversible, continues at `k2`). The
fitted values are also stored in `net.k_fit`, so a later `simulate()` or
`plot()` call with no `k_values` argument uses them automatically, and
`summary()` prints each reaction with its fitted (or, before any `fit()`
call, guessed) rate constant(s).

(reaction-network-overall)=
### 5.7 Checking the overall equation (bilan)

For a multi-step mechanism, `overall_equation()` sums the stoichiometry of
every declared reaction (forward direction as written, each counted once)
and cancels out whatever appears on both sides — typically the
intermediate(s) — to get the net equation the whole sequence implies:

```python
net = ReactionNetwork()
net.add_reaction("A -> I", k_guess=0.5)   # I: reactive intermediate
net.add_reaction("I -> P", k_guess=0.1)
print(net.overall_equation())             # "A -> P"  (I cancels out)
```

This is an important sanity check on a mechanism: the overall equation it
implies should match the balanced equation you started from. If it doesn't,
either a step is missing/wrong, or a stoichiometric coefficient is off —
exactly the kind of mistake this check catches before any fitting happens.
Note that a reversible reaction contributes its forward stoichiometry only
(this is about the mechanism's bookkeeping across the whole sequence, not
about net flux at a given time, which depends on the actual rate constants
and concentrations — see [5.8 Quasi-steady-state check](reaction-network-qssa)
for that).

(reaction-network-qssa)=
### 5.8 Quasi-steady-state check (QSSA)

For a reactive intermediate (e.g. $B$ in $A\rightarrow B\rightarrow C$), the
quasi-steady-state approximation (QSSA) assumes $d[B]/dt\approx 0$ once $B$
is being consumed about as fast as it forms — valid when $k_2\gg k_1$, but
not obviously so otherwise. `qssa_check(species, t)` answers this
numerically, without needing a closed-form solution: at each time in `t`, it
compares the *net* rate of change of `species` to its *gross* rate — the sum
of the absolute value of every reaction's individual contribution to it
(production and consumption both counted positive, so they can't cancel each
other the way they do in the net rate):

$$\text{QSSA ratio}(t) = \frac{|d[\text{species}]/dt|}{\sum_j |\text{contribution}_j(t)|}$$

The numerator **is** the first derivative of the concentration (in absolute
value) — exactly $d[\text{species}]/dt$, the same net rate `simulate()`
integrates. What makes it a *ratio* rather than just the derivative is the
denominator: the gross rate, i.e. how much is flowing through `species` in
total (production and consumption summed as positive numbers, so a large
forward flux and a large backward flux don't cancel there the way they do
in the numerator). So the ratio asks "how small is the net change *compared
to the traffic actually going through this species*?" — a small $|d[B]/dt|$
on its own doesn't mean much if nothing is happening at all, but a small
$|d[B]/dt|$ next to a large gross flux is the real signature of a
species sitting at quasi-steady state.

A ratio near 0 means production and consumption are nearly balanced — the
QSSA holds; a ratio near 1 means `species` is still mostly accumulating or
depleting on its own.

```python
net = ReactionNetwork()
net.add_reaction("A -> B", k_guess=0.05)
net.add_reaction("B -> C", k_guess=5.0)     # k2 >> k1
net.set_initial_conditions({"A": 1.0, "B": 0.0, "C": 0.0})

t = np.linspace(0.01, 20, 40)
ratio = net.qssa_check("B", t)   # << 1 here (after a short induction period):
                                 # the QSSA is a good approximation for B
```

With `k_guess=0.5` for both steps instead (so $k_2\sim k_1$), the same check
gives a ratio climbing well above 0.5 — the QSSA clearly does **not** hold,
exactly as expected since $B$ never gets consumed fast enough to stay near
steady state.

(reaction-network-ozone)=
### 5.9 Worked example 1: stratospheric ozone decomposition (QSSA)

A textbook mechanism that ties every feature above together: the classical
two-step scheme for ozone decomposition,

$$\mathrm{O_3}\rightleftharpoons \mathrm{O_2}+\mathrm{O}\qquad(k_{f1},\,k_{r1},\text{ fast})$$
$$\mathrm{O}+\mathrm{O_3}\rightarrow 2\,\mathrm{O_2}\qquad(k_2,\text{ slow})$$

with atomic oxygen $\mathrm{O}$ as a short-lived intermediate. $\mathrm{O_2}$
is given a sizeable initial concentration here (as it would have in a real
stratospheric mixture) — it has to, for the reverse of the first step to
have anything to act on from the very start; starting it at exactly 0 instead
leaves the backward rate $k_{r1}[\mathrm{O_2}][\mathrm{O}]$ pinned near 0
until $\mathrm{O_2}$ builds up on its own, so the pre-equilibrium needs an
induction period far longer than the one shown below:

```python
net = ReactionNetwork()
net.add_reaction("O3 <=> O2 + O", kf_guess=1.0, kr_guess=100.0)
net.add_reaction("O + O3 -> 2 O2", k_guess=50.0)
net.set_initial_conditions({"O3": 1.0, "O2": 1.0, "O": 0.0})

print(net.overall_equation())   # "2 O3 -> 3 O2"  (O cancels: it's the
                                 #  intermediate, consistent with the
                                 #  textbook balanced equation)
```

Checking that $\mathrm{O}$ is indeed a good QSSA candidate here — the whole
point of calling the first step "fast" and the second "slow" — is exactly
what `qssa_check()`/`plot_qssa()` are for:

```python
import numpy as np
t = np.linspace(1e-5, 8, 300)

ratio = net.qssa_check("O", t)
# ratio ~ 1 only for a brief induction period (t << 0.03), then drops
# below 1% and stays there for the rest of the decomposition: the QSSA
# for atomic oxygen is excellent once that short transient is over.

net.plot_qssa("O", t, threshold=0.05, time_unit="s")   # two-panel diagnostic plot
net.plot(species=["O3", "O"], log_scale=True, t_max=8, time_unit="s")  # zoom on
                                                                         # the trace intermediate
```

This is also a good illustration of why the induction period matters when
choosing initial conditions for a QSSA demonstration: with $k_{f1}$, $k_{r1}$
both very large and $\mathrm{O_2}(0)=0$ instead, `qssa_check()` would
correctly report the approximation failing for a much longer stretch, not
because the mechanism is wrong but because the reverse step has nothing to
react with yet — a reminder that the QSSA is a statement about *rates*
staying balanced, not simply about $k_{r1}\gg k_{f1}$.

(reaction-network-fit-example)=
### 5.10 Worked example 2: consecutive reactions with a fit

A second worked example, this time exercising `fit()` rather than
`qssa_check()`: the classic two-step consecutive mechanism
$A\rightarrow I\rightarrow P$, where only the intermediate $I$ is
experimentally observed (e.g. a UV-Vis absorbing species, as in
[4.1.1 Experimental Measurements](experimental-measurements) for `KORD`) —
the exact setting `observed_species=` is for.

`ReactionNetwork` has no built-in `read()`/`load_from_excel()` of its own
(unlike `kbasic`/`KORD`) — not because it can't be done, but because
`fit()`/`simulate()` just need two plain arrays `t_exp`, `Y_exp`, so reading
whatever file they come from is a two-line `pandas` call, with no dedicated
method needed: `pandas.read_excel()` reads `.xlsx` as well as `.ods`
(LibreOffice Calc's native format — needs the `odfpy` package) the same
way, so here the data (one $A\rightarrow I\rightarrow P$ "experiment",
with $[I]$ tracked over time) is loaded straight from a two-column
spreadsheet, `consecutive_fit_data.ods`, exactly as a real lab measurement
would be:

```python
import pandas as pd

df = pd.read_excel("consecutive_fit_data.ods", engine="odf")
t_exp = df["Time / min"].values
Y_exp = df["Y"].values           # the intermediate [I], as actually observed

net = ReactionNetwork()
net.add_reaction("A -> I", k_guess=1.0)   # deliberately uninformed guesses
net.add_reaction("I -> P", k_guess=1.0)
net.set_initial_conditions({"A": 1.0, "I": 0.0, "P": 0.0})

results = net.fit(t_exp, Y_exp, observed_species="I")
net.plot(t_exp, Y_exp, time_unit="min", save_img="./fig_examples/Kinetics/consecutive_fit.png")
net.summary()
```

The rate constants used to actually generate `consecutive_fit_data.ods`
are not needed to run this fit — that's the point: `fit()` recovers them
from the data alone, starting from guesses that give it no head start.
[5.11 Worked example 3](reaction-network-synth-data) shows exactly how
that file was produced, for anyone who wants to check the fit against the
true values or generate a similar file for a different mechanism.

Unlike the ozone example, there's no reversible step and no QSSA claim
here — it's a plain nonlinear least-squares fit of two rate constants
against one observed species, the same workflow as `net.fit(t_exp, Y_exp,
observed_species="B")` in [5.2 Quick Start](reaction-network-quickstart).

**A genuine identifiability trap, not a code bug**: fitting the exact same
data with `observed_species="P"` instead of `"I"` converges just as well
(comparable RMSE) but can recover $k_1\approx 0.15$, $k_2\approx 0.54$ —
the two rate constants **swapped**. This isn't `curve_fit` failing: the
Bateman closed form for the final product,

$$[P](t) = a_0\left(1+\frac{k_2e^{-k_1t}-k_1e^{-k_2t}}{k_1-k_2}\right),$$

is exactly symmetric under exchanging $k_1\leftrightarrow k_2$ (swap them
and $[P](t)$ is unchanged, for every $t$) — the final product's time
profile carries no information about *which* of the two steps is the fast
one, only about the unordered pair $\{k_1, k_2\}$. `curve_fit` is free to
land on either equally-good solution depending on the starting guesses.
It's the intermediate $[I](t)$ — which rises then falls, asymmetrically in
$k_1$ and $k_2$ — that breaks this symmetry and pins down the correct
assignment; whenever only the final product can realistically be followed,
this is a hard identifiability limit of the data itself, not something a
different `k_guess` or a tighter fit can get around.

(reaction-network-synth-data)=
### 5.11 Worked example 3: generating a synthetic kinetics dataset

This is how `consecutive_fit_data.ods`, read blind in
[5.10 Worked example 2](reaction-network-fit-example), was actually
produced: a network with known, "true" rate constants simulates
noise-free concentrations, Gaussian noise is added to $[I]$ only (mimicking
real measurement noise on the one species actually followed), and the
result is written out to a two-column spreadsheet — the same logic used
elsewhere in this module to validate `KORD.fit()` against synthetic data,
just saved to a file instead of being fit immediately in the same script.

```python
import numpy as np
import pandas as pd

np.random.seed(0)   # reproducible noise

true_net = ReactionNetwork(verbose=False)
true_net.add_reaction("A -> I", k_guess=0.5)     # true k1 = 0.5 min^-1
true_net.add_reaction("I -> P", k_guess=0.15)    # true k2 = 0.15 min^-1
true_net.set_initial_conditions({"A": 1.0, "I": 0.0, "P": 0.0})

t_exp = np.linspace(0.5, 25, 18)
Y_exp = true_net.simulate(t_exp)["I"] + np.random.normal(0, 0.01, size=t_exp.shape)

df = pd.DataFrame({"Time / min": t_exp, "Y": Y_exp})
df.to_excel("consecutive_fit_data.ods", index=False, engine="odf")
```

**A note on naming, for a reversible reaction**: `add_reaction()` already
names a reversible step's rate constants the standard textbook way, $k_1$
forward and $k_{-1}$ backward (not $k_{f1}$/$k_{r1}$) — e.g.
`net.add_reaction("A <=> B", kf_guess=0.8, kr_guess=0.3)` reports as
`k1=0.8, k-1=0.3` in `summary()` and in `results['k']`, and a second
reversible reaction added afterwards would be `k2`/`k-2`, and so on,
numbered in the order the reactions were declared (irreversible and
reversible steps share the same counter, so `"A -> B"` then `"B <=> C"`
gives `k1`, then `k2`/`k-2`).


## API reference

```{eval-rst}
.. automodule:: pyphyschemtools.kinetics
   :members:
   :undoc-members:
   :show-inheritance:
```
