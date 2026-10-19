# Hydrogen and hydrogen-like atoms

**`Hydrogen.py` module**

Bohr model / analytical solution of the Schrödinger equation for a
one-electron system (H, He⁺, Li²⁺, ...), with an optional finite
nuclear-mass (reduced-mass) correction. Covers energy levels, ionization
energies, spectral series, and their visualization as energy-level
diagrams and line spectra.

> **Note:** this module covers energy levels, ionization energies, and
> spectra (lines and diagrams). The atomic orbitals themselves — radial
> parts, angular parts, and their 2D/3D visualization — are in the
> separate `AO.py` module.

## Quick Start

```python
from pyphyschemtools import (
    rydberg_Z, En_hydrogenic, Ei_hydrogenic,
    spectral_series, plot_energy_levels, plot_line_spectrum
)

# Ground-state energy of hydrogen, in eV
En_hydrogenic(1)

# Ionization energy of He+ (Z=2)
Ei_hydrogenic(Z=2)

# Print the Lyman to Humphreys series, transitions up to n2=12
spectral_series(n2_max=12)

# Energy-level diagram, absorption transitions
plot_energy_levels(n2_max=6, transition_type="absorption")

# Line spectrum on a log wavelength axis
plot_line_spectrum(n2_max=12)
```

<u>ou bien</u>

```python
import pyphyschemtools.hydrogen as H
H.Ei_hydrogenic(Z=1)
H.plot_line_spectrum(n2_max=12)
```


## Reduced-mass correction

All energy calculations go through a Rydberg constant that can be
corrected for the finite mass of the nucleus:

$$R_M = R_\infty \frac{\mu}{m_e}, \quad \mu = \frac{m_e M_{\text{nucleus}}}{m_e + M_{\text{nucleus}}}$$

- `rydberg_Z(Z=1, A=None, M_atom=None)` returns $R_M$ in m⁻¹. The nuclear
  mass is looked up via `mendeleev` for isotope `A` of element `Z` (the
  most abundant isotope by default), or can be given directly as
  `M_atom` (kg), bypassing the lookup entirely.
- Every other function accepts `reduced_mass=True` (default, uses
  $R_M$) or `reduced_mass=False` (uses $R_\infty$, the infinite-nucleus
  limit — `A` and `M_atom` are then ignored).
- The correction is small (~10⁻⁴ relative) and shrinks as the nucleus
  gets heavier; it is included mainly for pedagogical completeness
  rather than to close the gap with experiment, which — for Z > 1 — is
  dominated by relativistic and QED effects that this module does not
  include.

## Energy levels

$$E_n = -h c R \frac{Z^2}{n^2}$$

`En_hydrogenic(n, Z=1, A=None, M_atom=None, reduced_mass=True, unit='eV')`
returns the (negative, bound-state) energy of level $n$ of a hydrogenic
atom/ion of nuclear charge $Z$, in joules or eV.

## Ionization energy

$$E_i = h c R Z^2$$

`Ei_hydrogenic(Z, A=None, M_atom=None, reduced_mass=True)` returns the
energy (eV) needed to remove the single electron from the ground state
to the continuum. The $Z$-th ionization energy of an element matches the
hydrogen-like ion of nuclear charge $Z$, which makes this a convenient
reference point against tabulated ionization energies.

## Spectral series

Named series are keyed by their lower level $n_1$:

| $n_1$ | Name      | Short |
|-------|-----------|-------|
| 1     | Lyman     | Ly    |
| 2     | Balmer    | Ba    |
| 3     | Paschen   | Pa    |
| 4     | Brackett  | Br    |
| 5     | Pfund     | Pf    |
| 6     | Humphreys | Hu    |

For $Z > 1$ these names are historically defined for hydrogen only and
are used "by analogy" — printed and labeled with a "-like" suffix
(e.g. the Lyman-like series of He⁺).

`spectral_series(Z=1, A=None, M_atom=None, reduced_mass=True, n2_max=12,
series_names=None)` prints, for each series in `series_names` (default:
`SERIES_NAMES`, Lyman to Humphreys), its series limit ($n_2 \to \infty$)
and every transition $n_2 \to n_1$ with its energy (eV), wavelength (nm),
and frequency (Hz).

## Energy-level diagrams

`plot_energy_levels(Z=1, A=None, M_atom=None, reduced_mass=True,
n2_max=7, series_names=None, to_scale=True, transition_type="absorption",
size=(9, 11), show_level_labels=True, show_transition_labels=True,
save_as=None)` draws:

- a horizontal line for each level $n = 1 \ldots n_2^{\max}$, labeled with
  its energy;
- the ionization limit at 0 eV, with the continuum of free-electron states
  shaded above it;
- the transitions of each series as colored arrows labeled with Greek
  letters ($\alpha, \beta, \ldots$, up to $\epsilon$ — there is no
  standard name for higher members).

Two display modes, controlled by `to_scale`:

- **`to_scale=True`** (default): levels sit at their true eV energy. Since
  $E_n \propto 1/n^2$, levels $n \geq 2$ crowd together near the
  ionization limit while $n=1$ sits far below — itself a useful
  illustration of why the spectrum bunches up with increasing $n$.
- **`to_scale=False`**: levels are evenly spaced by rank — the usual
  simplified textbook diagram, more readable for $n \geq 2$ but no
  longer a true energy axis.

`transition_type` controls both arrow direction and layout:

- **`"absorption"`** (default): arrows point up ($n_1 \to n_2$), ordered
  left to right by increasing transition size (α first).
- **`"emission"`**: arrows point down ($n_2 \to n_1$, the electron
  falling back to $n_1$), ordered left to right by decreasing transition
  size, all converging on the same $n_1$.

Figures are saved through `pyphyschemtools.core.save_fig` (missing
directories created automatically).

## Line spectra

`plot_line_spectrum(Z=1, A=None, M_atom=None, reduced_mass=True,
n2_max=12, series_names=None, show_visible=True, log_scale=True,
ax=None)` draws a wavelength axis (nm) with one colored vertical line per
transition $n_2 \to n_1$, colored by series, the α-line of each series
labeled (e.g. "Ly-α"), and — by default — the visible range (380–780 nm)
shaded and a log wavelength scale (the series span several decades, from
~100 nm to several microns).

## API reference

```{eval-rst}
.. automodule:: pyphyschemtools.hydrogen
   :members:
   :undoc-members:
   :show-inheritance:
```