# Chemical Kinetics

**`kinetics.py` module**

This module contains two classes and one standalone function:

* **`kbasic`** — a minimal, self-contained, pedagogical tool: fit a single rate
  constant $k$ for one chosen reaction order. See [3. `kbasic`](kbasic).
* **`KORD`** (Kinetic Order of Reaction Determination) — the full tool: fit
  $k$, $Y_0$, $Y_\infty$ (and $b_\infty$) for reaction orders 0, 1 and 2, and
  decide which order the data actually support. See
  [4. `KORD`](kord).
* **`superpose_plots()`** — the module's only standalone function (not a
  method of either class): overlays the fitted curves of several `KORD`
  studies on one plot, e.g. to compare the same data fitted freely vs. with
  `fix_Y0`/`fix_Yinf`. See [4.4.1 Comparing several fits](kord-advanced-compare).

## Table of Contents

* [1. Finding a Mathematical Model That Fits the Data](rmsd)
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
  * [4.1 Quick Start](kord-quickstart)
    * [4.1.1 Basic use](kord-quickstart-basic)
    * [4.1.2 Specific fit](kord-quickstart-specific)
  * [4.2 Principle](principle)
  * [4.3 Theoretical Model](theoretical-model)
  * [4.4 Advanced use](kord-advanced)
    * [4.4.1 Comparing several fits](kord-advanced-compare)
    * [4.4.2 `from_spreadsheet` vs `load_from_spreadsheet`](kord-advanced-loading)
    * [4.4.3 Reading a different sheet / experiment](kord-advanced-sheet)

---

(rmsd)=
## 1. Finding a Mathematical Model That Fits the Data

Curve fitting, in general, is the search for the free parameters — generically
$a, b, c, \ldots$ (not to be confused with this module's own $a_0$, $\alpha$,
$\beta$) — of a mathematical model $Y_\mathrm{THEO}(t;\, a, b, c, \ldots)$ that
make it track the measured data $Y_\mathrm{EXP}(t)$ as closely as possible.
"As closely as possible" needs a single number to minimize, and that number is
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

Both `kbasic.fit()` and `KORD.fit()` work exactly this way: they search — via
`scipy.optimize.minimize` (`kbasic`) or `scipy.optimize.curve_fit` (`KORD`) —
for the parameter values that make the RMSE as small as possible.

The RMSE should tend toward 0 as the fit improves, and it will, **provided
the chosen mathematical model is the right one for the data**: a correct
model, once its parameters are optimized, reproduces the measurements almost
exactly. A wrong model (the wrong reaction order, say) cannot be pushed to a
near-zero RMSE no matter how its parameters are tuned, simply because its
shape doesn't match the data's. `KORD` builds on this further — see
[4.2.4 Model Selection](model-selection) — to compare reaction orders
against each other, not just to fit one order's parameters.

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
[4.3 Theoretical Model](theoretical-model)).

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
* **One order at a time, no model comparison**: unlike `KORD.get_best_order()`,
  there is no AIC/BIC-based verdict. You pick the order yourself, and `kbasic`
  only reports the RMSE for that one choice.
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
[4.2.3 Input](input) below) — `kbasic` has no $\alpha$, $\beta$ or $a_0$ to
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

(kord-quickstart)=
### 4.1 Quick Start

(kord-quickstart-basic)=
#### 4.1.1 Basic use

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
#### 4.1.2 Specific fit

You can also fix $Y_0$ (or $Y_\infty$) at a chosen value instead of letting
the optimizer adjust it freely — fully explained in
[4.2 Principle](principle) below:

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

---

(principle)=
### 4.2 Principle

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

#### 4.2.1 Experimental Measurements ($Y_\mathrm{EXP}$)

In chemical kinetics, we track the evolution of molar concentrations over time: $C_{i}(t)$. Experimentally, we measure a physical quantity proportional to these concentrations: $Y_\mathrm{EXP}(t)$.

* **Spectrophotometry**: $Y_\mathrm{EXP} = A = \sum_{i} \epsilon_{i} \cdot l \cdot C_{i}$, where $A$ is the absorbance
* **Polarimetry**: $Y_\mathrm{EXP} = \alpha = \sum_{i} [\alpha]_{i} \cdot l \cdot C_{i}^{w}$
* **Conductivity**: $Y_\mathrm{EXP} = \sigma = \sum_{i} \lambda_{i} \cdot C_{i}$

**General Form:** $Y_\mathrm{EXP}(t) = \sum_{i} \eta_{i} \cdot C_{i}(t)$

(theo-model-def)=
#### 4.2.2 Theoretical Model ($Y_\mathrm{THEO}$)

The model has three different expressions (see <a href="#theoretical-model">Theoretical model section</a>), depending on the order of the reaction:

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


$Y_\mathrm{THEO}$ is defined by two types of values: **fixed parameters** (input by the user) and **adjustable variables** (optimized by the algorithm).

**<u>Fixed Parameters</u>:**
* **Reaction Order**: $n \in \{0, 1, 2\}$ (The user selects the order to test).
* **Stoichiometry**: $\alpha$ and $\beta$ are known constants, provided by the user.
      
  $\alpha$ and $\beta$ must be the smallest possible positive integers

* **Initial Concentration**: $a_{0}$ (Note: For Order 1, $\hat{Y}_\mathrm{THEO}$ is independent of $a_{0}$). The concentration must also be provided by the user

  The final concentration of B, $b_{\infty}$ is related to $a_0$ by the relation:

  $$\frac{a_0}{\alpha} = \frac{b_{\infty}}{\beta}$$

**<u>Adjustable Variables</u>:**
By default, the model fits the experimental data by adjusting the following:
* **Rate Constant**: $k$
* **Final Value**: $Y_{\infty}$
* **Initial Value**: $Y_{0}$ (While $Y_{0}$ is measured, the algorithm also adjusts it by default to ensure the best fit starting point).

The optimization is performed for a specific reaction order at a time to determine which model best describes the experimental data.

<div class="rqE">

By default, KORD chooses the first and last $Y_\mathrm{EXP}$ values as $Y_{0}$ and $Y_{\infty}$. And a default $k$ value is also setup by KORD. If you need to change that because of a convergence issue, ensure your starting values for $k$, $Y_{0}$ and $Y_{\infty}$ are realistic to help the algorithm converge. 

</div>

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
(see [4.2.4 Model Selection](model-selection) below — `fit()` also stores an
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
later by assigning the attribute directly. See [4.1.2](kord-quickstart-specific)
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
#### 4.2.3 Input

Data input is performed through a structured spreadsheet file (`.xlsx`, `.ods`,
...), read by `KORD.load_from_spreadsheet(file_path, exp_number, sheet_name=0)`
— or, more conveniently, by `KORD.from_spreadsheet(...)`, which calls it for
you (see [4.1 Quick Start](kord-quickstart)). Within the chosen sheet, each
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
#### 4.2.4 Model Selection: `get_best_order()` and $\Delta$BIC

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
### 4.3 Theoretical Model

$Y_\mathrm{THEO}(t)$ is built directly from the integrated rate laws of
[section 2](rate-laws): it is a linear combination of the reactant
expression [$a(t)$](rate-laws-a) and the product expression
[$b(t)$](rate-laws-b) introduced there, under the same assumptions
(single-component, total, closed, homogeneous, isochoric reaction):

$$\hat{Y}_\mathrm{THEO}(t) = \frac{Y_{0}}{a_{0}} \cdot a(t) + \frac{Y_{\infty}}{b_{\infty}} \cdot b(t)$$

Compte tenu des équations de $a(t)$ et $b(t)$ pour les ordres 0, 1 et 2, il
vient :

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
### 4.4 Advanced use

(kord-advanced-compare)=
#### 4.4.1 Comparing several fits

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
[4.1.1](kord-quickstart-basic) above) and a warning is printed for that
study.

(kord-advanced-loading)=
#### 4.4.2 `from_spreadsheet` vs `load_from_spreadsheet`

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
#### 4.4.3 Reading a different sheet / experiment

`load_from_spreadsheet(file_path, exp_number, sheet_name=0, show_data=True)`
(formerly named `load_from_excel` — same method; it reads with
`pandas.read_excel`, so it already worked on `.ods` files as well as `.xlsx`,
and the old, Excel-specific name was misleading. `load_from_excel` still
works, as a deprecated alias.) reads **one** sheet (`sheet_name`, by index or
by name — the first sheet by default) and, within that sheet, picks the
column pair identified by `exp_number` (1 for the first pair of columns, 2
for the next pair, etc. — see the data format in
[4.2.3 Input](input) above). `from_spreadsheet` takes the exact same
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


## API reference

```{eval-rst}
.. automodule:: pyphyschemtools.kinetics
   :members:
   :undoc-members:
   :show-inheritance:
```
