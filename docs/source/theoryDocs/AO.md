```{eval-rst}
.. currentmodule:: pyphyschemtools.AO
```

# Atomic Orbitals

**`AO.py` module**

The `AO` module provides the exact, analytical hydrogenic atomic orbitals
(radial parts and real spherical harmonics, as implemented in SymPy's
`sympy.physics.hydrogen` and `sympy.Znm`) together with a set of plotting
functions to visualize them: radial curves, 2D angular polar diagrams,
density-weighted electron clouds, and interactive 3D nodal surfaces.

## SymPy and `sympy.physics.hydrogen`

[SymPy](https://www.sympy.org) is a Python library for symbolic
mathematics: it manipulates algebraic expressions exactly, as formulas
(derivatives, integrals, simplifications, ...), rather than as floating-point
numbers. Exact analytical answers are available for the nonrelativistic
hydrogen atom, and SymPy's `physics` submodule provides them directly, so
there is no need to re-derive or hard-code them.

This module relies on:
- `sympy.physics.hydrogen.Psi_nlm` and `R_nl` — the full wavefunction
  $\psi_{n\ell m}$ and the radial part $R_{n\ell}(r)$ (see the
  [hydrogen module documentation](https://docs.sympy.org/latest/modules/physics/hydrogen.html));
- `sympy.Znm` and `sympy.Ynm` — the real and complex spherical harmonics
  (see the [spherical harmonics documentation](https://docs.sympy.org/latest/modules/functions/special.html#sympy.functions.special.spherical_harmonics.Ynm)).

More generally, the full catalogue of physics-oriented SymPy submodules
(mechanics, quantum, optics, continuum mechanics, ...) is listed in the
[SymPy physics reference](https://docs.sympy.org/latest/reference/public/physics/index.html).

## Physical background

The Schrödinger equation allows one to calculate the stationary states of
quantum systems. Exact analytical answers are available for the
nonrelativistic hydrogen atom. Its time-independent form is conveniently
written and solved in spherical coordinates:

$$\hat{H}(r,\theta,\varphi)\,\psi(r,\theta,\varphi) = E\,\psi(r,\theta,\varphi)$$

with $\hat{H}=\hat{T}+\hat{V}$, where

$$\hat{V} = -\frac{e^{2}}{4\pi\varepsilon_{0}}\frac{1}{r}$$

is the potential energy of the electron under the influence of the
proton, and

$$\hat{T}=-\frac{\hbar^{2}}{2\mu_\mathrm{H}}\nabla^{2}, \qquad
\nabla^{2}=\frac{1}{r^{2}}\left[\frac{\partial}{\partial r}\left(r^{2}\frac{\partial}{\partial r}\right)+\Lambda^{2}\right]$$

$\mu_\mathrm{H}=m_e m_p/(m_e+m_p)$ is the reduced mass of the electron,
and $\Lambda^2$ is the Legendre (angular momentum) operator:

$$\Lambda^{2}=\frac{1}{\sin\theta}\frac{\partial}{\partial\theta}\left(\sin\theta\frac{\partial}{\partial\theta}\right)+\frac{1}{\sin^{2}\theta}\frac{\partial^{2}}{\partial\varphi^{2}}$$

Since $\Lambda^2$ does not involve the radial variable $r$, the equation
is separable: $\psi(r,\theta,\varphi)$ is a **product wavefunction**,

$$\psi_{n\ell m}(r,\theta,\varphi) = R_{n\ell}(r)\, Y_\ell^m(\theta,\varphi)$$

called an **atomic orbital**. Schrödinger's approach requires three
quantum numbers ($n$, $\ell$, $m$) to specify one. Here:

- $Y_\ell^m(\theta,\varphi)$ is the **angular part**: a normalized
  eigenvector of $\Lambda^2$, labeled by the azimuthal quantum number
  $\ell$ and the magnetic quantum number $m$ ($-\ell \le m \le \ell$):

  $$\Lambda^{2}Y_\ell^m=-\ell(\ell+1)Y_\ell^m$$

  - This module uses the *real* combinations $Z_\ell^m$ (SymPy's `Znm`)
    rather than the complex $Y_\ell^m$ (SymPy's `Ynm`) — the familiar
    real-valued $p_x$, $p_y$, $p_z$, $d_{z^2}$, $d_{x^2-y^2}$, ...
    orbitals used in chemistry.
  - They are obtained from $Y_\ell^m$ and $Y_\ell^{-m}$ via Euler's
    formula:

    $$\cos x = \frac{e^{ix}+e^{-ix}}{2}, \qquad \sin x = \frac{e^{ix}-e^{-ix}}{2i}$$

  - $Y_\ell^m$ (and $Z_\ell^m$) has **angular nodes**: cones
    ($\theta$ = const) and/or planes ($\varphi$ = const) where the
    function changes sign.
- $R_{n\ell}(r)$ is the **radial part**, labeled by the principal quantum
  number $n$ and $\ell$ ($0 \le \ell \le n-1$), solution of the radial
  equation

  $$\left(-\frac{\hbar^{2}}{2\mu_\mathrm{H}}\left[\frac{1}{r^{2}}\frac{\partial}{\partial r}\left(r^{2}\frac{\partial}{\partial r}\right)-\frac{\ell(\ell+1)}{r^{2}}\right]-\frac{e^{2}}{4\pi\varepsilon_{0}}\frac{1}{r}\right)R_{n\ell}(r)=E_{n}R_{n\ell}(r)$$

  whose eigenvalue $E_n$ is the hydrogen energy level. $R_{n\ell}(r)$ has
  $n-\ell-1$ **radial nodes** (spherical surfaces where $R_{n\ell}=0$).

By Born's statistical interpretation, $|\psi_{n\ell m}|^2$ is the electron
probability density, normalized over all space:

$$\langle\psi|\psi\rangle = \iiint \psi^*\psi\, dv = 1, \qquad
dv = r^{2}\,dr\,\sin\theta\,d\theta\,d\varphi$$

Integrating over all angles gives the **radial probability density** —
the probability of finding the electron in a thin spherical shell at
distance $r$:

$$P_{n\ell}(r) = r^2 R_{n\ell}(r)^2$$

This is the quantity plotted by {func}`plot_radial` when `density=True`.

`plot_radial` can also overlay, for each orbital, the mean radius and its
dispersion, computed numerically from the already-normalized probability
density $P_{n\ell}(r) = r^2 R_{n\ell}(r)^2$:

$$\langle r\rangle = \int_0^\infty r\,P_{n\ell}(r)\,dr, \qquad
\sigma_r = \sqrt{\langle r^2\rangle - \langle r\rangle^2}$$

Pass `density=True, show_mean_r=True` to `plot_radial` to draw, for each
curve, a dashed vertical line at $\langle r\rangle$ and an error bar
spanning $\langle r\rangle \pm \sigma_r$.

All lengths are expressed in units of the Bohr radius $a_0$ (scaled by
$1/Z$ for hydrogenic ions with nuclear charge $Z \ne 1$), consistent with
SymPy's convention for `R_nl` and `Psi_nlm`.

## Quick start

```python
from pyphyschemtools import AO

# Radial parts of the 1s, 2s and 2p orbitals
AO.plot_radial([(1, 0), (2, 0), (2, 1)])

# Radial probability density r^2 R_nl(r)^2 instead of R_nl(r)
AO.plot_radial([(1, 0), (2, 0), (2, 1)], density=True)

# 2D angular polar diagram of a 3d_(x^2-y^2)-like orbital (l=2, m=2), xy plane
AO.plot_polar(l=2, m=2, plane="xy")

# Density-weighted electron cloud of the 2p_z orbital, xz projection
AO.plot_electron_cloud(n=2, l=1, m=0, plane="xz")

# Interactive 3D nodal surfaces of the 3d_z2 orbital, with the electron
# cloud overlaid
AO.plot_orbital_3D(n=3, l=2, m=0, show_cloud=True)

# Radial probability density of 1s, 2s, 2p, with <r> ± sigma_r overlaid
AO.plot_radial([(1, 0), (2, 0), (2, 1)], density=True, show_mean_r=True)
```

Every plotting function accepts `show_formula=True` to display (via
IPython) the symbolic SymPy expression of the radial and/or angular part
being plotted — useful for checking a result against a textbook formula,
or for teaching.

Set `save_as=...` on any function to save the figure to disk (a static
image for `plot_radial`, `plot_polar` and `plot_electron_cloud`; a static
image **or** an interactive `.html` for `plot_orbital_3D`, selected by the
file extension).

## Orbital naming convention

Subshells of AOs and AOs are identified throughout the module by `(n, l)` or
`(n, l, m)` tuples, and labeled on plots with the usual spectroscopic
notation (`1s`, `2s`, `2p`, `3d`, ...) via the
`l = 0, 1, 2, 3, 4, 5 -> s, p, d, f, g, h` correspondence.

## API reference

```{eval-rst}
.. automodule:: pyphyschemtools.AO
   :members:
   :undoc-members:
   :show-inheritance:
```
