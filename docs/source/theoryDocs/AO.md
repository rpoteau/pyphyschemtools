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

## Physical background

For the hydrogen atom (or any hydrogenic ion of nuclear charge $Z$), the
time-independent Schrödinger equation

$$\hat{H}(r,\theta,\varphi)\,\psi(r,\theta,\varphi) = E\,\psi(r,\theta,\varphi)$$

is separable in spherical coordinates. Each stationary state is a product
wavefunction

$$\psi_{n\ell m}(r,\theta,\varphi) = R_{n\ell}(r)\, Y_\ell^m(\theta,\varphi)$$

where:

- $R_{n\ell}(r)$ is the **radial part**, labeled by the principal quantum
  number $n$ and the azimuthal quantum number $\ell$ ($0 \le \ell \le n-1$).
  It has $n-\ell-1$ **radial nodes** (spherical surfaces where $R_{n\ell}=0$).
- $Y_\ell^m(\theta,\varphi)$ is the **angular part**, a spherical harmonic
  labeled by $\ell$ and the magnetic quantum number $m$
  ($-\ell \le m \le \ell$). The module uses the *real* combinations
  $Z_\ell^m$ (SymPy's `Znm`), which are the familiar real-valued $p_x$,
  $p_y$, $p_z$, $d_{z^2}$, $d_{x^2-y^2}$, ... orbitals used in chemistry.
  $Y_\ell^m$ has **angular nodes**: cones ($\theta$ = const) and/or planes
  ($\varphi$ = const) where the function changes sign.

The electron probability density is $|\psi_{n\ell m}|^2$, and the radial
probability density — the probability of finding the electron in a thin
spherical shell at distance $r$, integrated over all angles — is

$$P_{n\ell}(r) = r^2 R_{n\ell}(r)^2$$

This is the quantity plotted by {func}`plot_radial` when `density=True`.

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

Orbitals are identified throughout the module by `(n, l)` or `(n, l, m)`
tuples, and labeled on plots with the usual spectroscopic notation
(`1s`, `2s`, `2p`, `3d`, ...) via the `l = 0, 1, 2, 3, 4, 5 -> s, p, d, f, g, h`
correspondence.

## API reference

```{eval-rst}
.. automodule:: pyphyschemtools.AO
   :members:
   :undoc-members:
   :show-inheritance:
```
