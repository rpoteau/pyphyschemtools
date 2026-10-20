'''
Atomic orbitals of the hydrogen atom and hydrogenic ions: radial parts,
real spherical harmonics, and their visualization (radial plots, 2D polar
cross-sections, electron cloud projections, and interactive 3D nodal
surfaces).

Public functions:
- plot_radial         : radial part R_nl(r) or radial probability density
- plot_polar           : 2D polar diagram of the angular part, in a chosen plane
- plot_electron_cloud  : electron cloud of an orbital (density-weighted point sampling), projected onto a chosen plane
- plot_orbital_3D      : interactive 3D nodal surfaces of an orbital (radial nodes as spheres, angular nodes as cones and planes), with the electron cloud as an optional overlay
'''
import numpy as np
import sympy as sp

from scipy.optimize import brentq

import matplotlib.pyplot as plt
import plotly.graph_objects as go

from sympy.physics.hydrogen import Psi_nlm, R_nl
from sympy import Ynm, Znm

from .core import save_fig
from .visualID_Eng import fg, hl, bg, color

ORBITAL_LETTERS = {0: "s", 1: "p", 2: "d", 3: "f", 4: "g", 5: "h"}


def plot_radial(orbitals, Z=1, density=False, show_formula=False, show_mean_r=False,
                 r_max=None, n_points=500, size=(9, 5), save_as=None):
    '''
    Plot the radial part R_nl(r) (default) or the radial probability
    density r^2 R_nl(r)^2 of one or several hydrogenic atomic orbitals, as
    a function of r (in units of the Bohr radius a0, scaled by 1/Z).

    input:

    - orbitals = list of (n, l) tuples, one curve per orbital, e.g. [(1, 0), (2, 0), (2, 1)] for 1s, 2s, 2p
    - Z        = nuclear charge (default: 1 for hydrogen)
    - density  = if False (default), plot R_nl(r); if True, plot the radial probability density r^2 R_nl(r)^2 (probability of finding the electron in a thin spherical shell at r)
    - show_formula = if True, display (via IPython) the symbolic expression of the quantity plotted (and its squared/density form if density=True); default: False
    - show_mean_r = if True, overlay the mean radius <r> (dashed vertical line), the ±sigma_r range shaded in grey under the curve, and an error bar for each orbital, computed numerically from the plotted density as <r> = integral of r*P_nl(r) dr and sigma_r = sqrt(<r^2> - <r>^2); error bars are nudged upward if they would overlap another orbital's; requires density=True; default: False
    - r_max    = largest r shown, in units of a0/Z; if None (default), set automatically from the largest n among orbitals
    - n_points = number of points used to sample r
    - size     = (width, height) in inches for the figure
    - save_as  = file path to save the figure to (passed to pyphyschemtools.core's save_fig); not saved if None (default)

    output: None (the figure is drawn and shown, and saved if save_as is
    given)
    '''
    if show_mean_r and not density:
        raise ValueError("show_mean_r=True requires density=True")

    r = sp.symbols('r', positive=True)
    if r_max is None:
        n_max = max(n for n, l in orbitals)
        r_max = 3 * n_max**2        # rough outer extent of the radial density
    r_vals = np.linspace(1e-6, r_max, n_points)

    fig, ax = plt.subplots(figsize=size)
    curves = []   # (y_vals, color), in the same order as orbitals
    for n, l in orbitals:
        Rnl_expr = R_nl(n, l, r, Z)
        label = f"{n}{ORBITAL_LETTERS.get(l, f'(l={l})')}"
        if show_formula:
            display(sp.Eq(sp.Symbol(f"R_{{{label}}}"), Rnl_expr))
            if density:
                display(sp.Eq(sp.Symbol(f"r^2 R_{{{label}}}^2"),
                               sp.simplify(r**2 * Rnl_expr**2)))
        Rnl_func = sp.lambdify(r, Rnl_expr, "numpy")
        Rnl_vals = Rnl_func(r_vals)
        y_vals = r_vals**2 * Rnl_vals**2 if density else Rnl_vals
        line, = ax.plot(r_vals, y_vals, label=label)
        curves.append((y_vals, line.get_color()))

    if show_mean_r:
        global_max = max(np.max(y_vals) for y_vals, _ in curves)
        step = 0.08 * global_max
        placed = []     # (x_low, x_high, y) of error bars already placed
        for y_vals, color in curves:
            mean_r = np.trapezoid(r_vals * y_vals, r_vals)
            mean_r2 = np.trapezoid(r_vals**2 * y_vals, r_vals)
            sigma_r = np.sqrt(max(mean_r2 - mean_r**2, 0.0))
            x_lo, x_hi = mean_r - sigma_r, mean_r + sigma_r

            # grey out the +/- sigma_r region under the curve
            mask = (r_vals >= x_lo) & (r_vals <= x_hi)
            ax.fill_between(r_vals[mask], 0, y_vals[mask], color=color, alpha=0.2)
            ax.axvline(mean_r, color=color, ls="--", lw=1)

            # pick a y position for the error bar, nudged up if it would
            # overlap a previously placed one (overlapping x-range and close y)
            y_marker = 1.05 * np.max(y_vals)
            while any(x_lo <= hi and x_hi >= lo and abs(y_marker - y) < step
                      for lo, hi, y in placed):
                y_marker += step
            placed.append((x_lo, x_hi, y_marker))

            ax.errorbar(mean_r, y_marker, xerr=sigma_r, fmt="o", color=color,
                        capsize=4, markersize=4)
            ax.annotate(rf"$\langle r\rangle$ = {mean_r:.2f} $\pm$ {sigma_r:.2f} $a_0$",
                        xy=(mean_r, y_marker), xytext=(5, 5),
                        textcoords="offset points", color=color, fontsize=8)

    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel(r"$r$ / $a_0$")
    ax.set_ylabel(r"$r^2 R_{nl}(r)^2$" if density else r"$R_{nl}(r)$")
    kind = "Radial probability density" if density else "Radial part"
    title = kind if Z == 1 else f"{kind} (Z = {Z})"
    ax.set_title(title)
    ax.legend()

    if save_as is not None:
        save_fig(save_as, fig=fig)
    plt.show()

def plot_polar(l, m, plane="xy", density=False, show_formula=False, n_points=400, size=(6, 6),
                save_as=None):
    '''
    Draw the 2D polar diagram of the angular part of an atomic orbital (the
    real spherical harmonic Z_lm, independent of n) in a plane containing
    the origin.

    A sweep angle gamma (0 to 2*pi) runs once around the chosen plane; at
    each gamma the corresponding direction (theta, phi) is computed, and
    the radius plotted is \|Z_lm(theta, phi)\| (or its square if
    density=True). The two signs of Z_lm are filled in different colors,
    so the lobes of p, d, ... orbitals and their phase change are visible;
    the radius goes to zero at a node (sign change).

    input:

    - l        = orbital angular momentum quantum number (l >= 0)
    - m        = magnetic quantum number (-l <= m <= l)
    - plane    = projection plane: "xy", "xz" or "yz" (default: "xy")
    - density  = if False (default), plot \|Z_lm\|; if True, plot the angular probability density Z_lm^2
    - show_formula = if True, display (via IPython) the symbolic expression of the quantity plotted (and its squared/density form if density=True); default: False
    - n_points = number of points used to sample the sweep angle
    - size     = (width, height) in inches for the figure
    - save_as  = file path to save the figure to (passed to pyphyschemtools.core's save_fig); not saved if None (default)

    output: None (the figure is drawn and shown, and saved if save_as is
    given)
    '''
    if plane.upper() not in ("XY", "XZ", "YZ"):
        raise ValueError('plane must be "xy", "xz" or "yz"')
    plane = plane.lower()

    theta_s, phi_s = sp.symbols('theta phi', real=True)
    Zlm_sym = Znm(l, m, theta_s, phi_s).expand(func=True)

    if show_formula:
        display(sp.Eq(sp.Symbol(f"Z_{{{l}}}^{{{m}}}"), sp.simplify(Zlm_sym)))
        if density:
            display(sp.Eq(sp.Symbol(f"(Z_{{{l}}}^{{{m}}})^2"), sp.simplify(Zlm_sym**2)))

    gamma = np.linspace(0, 2 * np.pi, n_points)
    if plane == "xy":
        x, y, z = np.cos(gamma), np.sin(gamma), np.zeros_like(gamma)
        axis_labels = ("x", "y")
    elif plane == "xz":
        x, y, z = np.cos(gamma), np.zeros_like(gamma), np.sin(gamma)
        axis_labels = ("x", "z")
    else:  # yz
        x, y, z = np.zeros_like(gamma), np.cos(gamma), np.sin(gamma)
        axis_labels = ("y", "z")

    theta_vals = np.arccos(np.clip(z, -1, 1))
    phi_vals = np.arctan2(y, x)
    Zlm_vals = np.array([complex(Zlm_sym.subs({theta_s: th, phi_s: ph}).evalf()).real
                          for th, ph in zip(theta_vals, phi_vals)])
    Zlm_vals = np.where(np.abs(Zlm_vals) < 1e-10, 0.0, Zlm_vals)
    Zlm_vals = np.where(np.abs(Zlm_vals) < 1e-10, 0.0, Zlm_vals)

    radius = Zlm_vals**2 if density else np.abs(Zlm_vals)
    positive = np.where(Zlm_vals >= 0, radius, 0.0)
    negative = np.where(Zlm_vals < 0, radius, 0.0)

    fig, ax = plt.subplots(figsize=size, subplot_kw={"projection": "polar"})
    ax.plot(gamma, positive, color="tab:blue", lw=1.5)
    ax.fill(gamma, positive, color="tab:blue", alpha=0.3)
    ax.plot(gamma, negative, color="tab:red", lw=1.5)
    ax.fill(gamma, negative, color="tab:red", alpha=0.3)

    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    ax.set_xticklabels([f"+{axis_labels[0]}", f"+{axis_labels[1]}",
                         f"-{axis_labels[0]}", f"-{axis_labels[1]}"])
    letter = ORBITAL_LETTERS.get(l, f"(l={l})")
    quantity = "angular density" if density else "angular part"
    ax.set_title(f"{letter} orbital (l={l}, m={m}) -- {quantity}, {plane} plane")
    ax.set_rlim(bottom=0)

    if save_as is not None:
        save_fig(save_as, fig=fig)
    plt.show()

def _sample_orbital_cloud(n, l, m, Z=1, n_points=20000, r_max=None, batch=50000):
    '''
    Sample n_points (x, y, z) points in units of a0, drawn with probability
    proportional to |psi_nlm(r, theta, phi)|^2 = R_nl(r)^2 * Z_l^m(theta, phi)^2,
    by rejection sampling inside a cube of half-side r_max. Also returns the
    sign of psi_nlm = R_nl(r) * Z_l^m(theta, phi) at each sampled point.

    output: (x, y, z, sign), each a 1D array of length n_points
    '''
    if r_max is None:
        r_max = 3 * n**2

    r = sp.symbols('r', positive=True)
    Rnl_func = sp.lambdify(r, R_nl(n, l, r, Z), "numpy")
    theta_s, phi_s = sp.symbols('theta phi', real=True)
    Zlm_func = sp.lambdify((theta_s, phi_s),
                            Znm(l, m, theta_s, phi_s).expand(func=True), "numpy")

    # rough upper bound of |psi|^2 to calibrate the acceptance threshold
    r_probe = np.linspace(1e-6, r_max, 2000)
    psi2_max = np.max(Rnl_func(r_probe)**2) * 1.0   # Z_lm^2 <= ~1 after norm
    th_probe = np.linspace(0, np.pi, 200)
    ph_probe = np.linspace(0, 2 * np.pi, 200)
    TH, PH = np.meshgrid(th_probe, ph_probe)
    Zlm_probe = np.real(np.asarray(Zlm_func(TH, PH), dtype=complex))
    psi2_max *= np.max(Zlm_probe**2)
    psi2_max *= 1.2        # safety margin

    xs, ys, zs, signs = [], [], [], []
    while len(xs) < n_points:
        x = np.random.uniform(-r_max, r_max, batch)
        y = np.random.uniform(-r_max, r_max, batch)
        z = np.random.uniform(-r_max, r_max, batch)
        rr = np.sqrt(x**2 + y**2 + z**2)
        rr = np.clip(rr, 1e-9, None)
        th = np.arccos(np.clip(z / rr, -1, 1))
        ph = np.arctan2(y, x)

        R_vals = Rnl_func(rr)
        Z_vals = np.real(np.asarray(Zlm_func(th, ph), dtype=complex))
        psi = R_vals * Z_vals
        psi2 = psi**2

        accept = np.random.uniform(0, psi2_max, batch) < psi2
        xs.extend(x[accept]); ys.extend(y[accept]); zs.extend(z[accept])
        signs.extend(np.sign(psi[accept]))

    xs, ys, zs, signs = (np.array(a[:n_points]) for a in (xs, ys, zs, signs))
    return xs, ys, zs, signs

def plot_electron_cloud(n, l, m, plane="xz", Z=1, n_points=20000,
                         point_size=3, alpha=0.25, color_pos="tab:blue",
                         color_neg="tab:red", size=(6, 6), show_formula=False,
                         save_as=None):
    '''
    Draw the electron cloud of a hydrogenic orbital psi_nlm, projected onto
    a plane, as a density-weighted point cloud: points are sampled with
    probability proportional to \|psi_nlm\|^2 (rejection sampling in 3D, see
    _sample_orbital_cloud), then projected onto the chosen plane. Both
    radial nodes (R_nl(r) = 0) and angular nodes (Z_l^m = 0) appear as
    empty regions, since no points are sampled there -- no separate
    geometric reconstruction is needed.

    input:

    - n, l, m   = quantum numbers of the orbital (0 <= l <= n-1, -l <= m <= l)
    - plane     = projection plane: "xy", "xz" or "yz" (default: "xz")
    - Z         = nuclear charge (default: 1 for hydrogen)
    - n_points  = number of sampled points
    - point_size, alpha = marker size and transparency of each point
    - color_pos, color_neg = colors for points where psi_nlm is positive / negative
    - size      = (width, height) in inches for the figure
    - show_formula = if True, display (via IPython) the symbolic expressions of R_nl(r) and Z_l^m(theta, phi)
    - save_as   = file path to save the figure to (passed to pyphyschemtools.core's save_fig); not saved if None (default)

    output: None (the figure is drawn and shown, and saved if save_as is
    given)
    '''
    if plane.upper() not in ("XY", "XZ", "YZ"):
        raise ValueError('plane must be "xy", "xz" or "yz"')
    plane = plane.lower()

    if show_formula:
        r = sp.symbols('r', positive=True)
        theta_s, phi_s = sp.symbols('theta phi', real=True)
        display(sp.Eq(sp.Symbol(f"R_{{{n},{l}}}"), R_nl(n, l, r, Z)))
        display(sp.Eq(sp.Symbol(f"Z_{{{l}}}^{{{m}}}"),
                       sp.simplify(Znm(l, m, theta_s, phi_s).expand(func=True))))

    x, y, z, signs = _sample_orbital_cloud(n, l, m, Z=Z, n_points=n_points)
    coords = {"x": x, "y": y, "z": z}
    u, v = plane[0], plane[1]

    fig, ax = plt.subplots(figsize=size)
    for sign, color in ((1, color_pos), (-1, color_neg)):
        mask = signs == sign
        ax.scatter(coords[u][mask], coords[v][mask], s=point_size,
                   alpha=alpha, color=color, linewidths=0)

    ax.set_xlabel(f"${u}$ / $a_0$")
    ax.set_ylabel(f"${v}$ / $a_0$")
    ax.set_aspect("equal")
    letter = ORBITAL_LETTERS.get(l, f"(l={l})")
    ax.set_title(f"{n}{letter} orbital (n={n}, l={l}, m={m}) -- "
                 f"electron cloud, {plane} projection")

    if save_as is not None:
        save_fig(save_as, fig=fig)
    plt.show()

def _probability_iso_level(psi, dV, prob_fraction):
    '''
    Return the amplitude level c such that the region |psi| >= c encloses
    prob_fraction of the total probability (integral of psi^2 dV).
    '''
    psi2 = np.sort((psi**2).ravel())[::-1]
    cum_prob = np.cumsum(psi2) * dV
    idx = min(np.searchsorted(cum_prob, prob_fraction * cum_prob[-1]),
               len(psi2) - 1)
    return np.sqrt(psi2[idx])
    
def plot_orbital_3D(n, l, m, Z=1, prob_fraction=0.7, show_nodes=True,
                     show_cloud=False, n_points=20000, point_size=2,
                     color_pos="blue", color_neg="red", grid_points=60,
                     width=900, height=900, show_formula=False, save_as=None):
    '''
    Draw the 3D shape of a hydrogenic orbital psi_nlm as isosurfaces: the
    surface(s) where \|psi_nlm\| = iso_fraction * max(\|psi_nlm\|), colored by
    the sign of psi_nlm. For orbitals with radial nodes (n-l-1 > 0), this
    naturally shows nested lobes -- a smaller inner surface inside a
    larger outer one (e.g. 2s, 3p, 4d, ...), with no extra work needed:
    the isosurface algorithm finds every disconnected region on its own.

    Optionally, the exact nodal surfaces (show_nodes=True) and/or the
    density-weighted electron cloud (show_cloud=True) can be overlaid.

    input:

    - n, l, m    = quantum numbers of the orbital (0 <= l <= n-1, -l <= m <= l)
    - Z          = nuclear charge (default: 1 for hydrogen)
    - prob_fraction = fraction of the total electron probability enclosed by the lobe surfaces (default: 0.7 -- higher shows fatter lobes)
    - show_nodes = if True, also draw the exact nodal surfaces (radial spheres, angular cones/planes); default: True
    - show_cloud = if True, also draw the sampled electron cloud; default: False
    - n_points   = number of sampled points for the cloud (only used if show_cloud=True)
    - point_size = marker size for the cloud points
    - color_pos, color_neg = colors for the lobes/cloud points where psi_nlm is positive / negative
    - grid_points = resolution (per axis) of the 3D grid used for the lobe isosurfaces
    - width, height = size of the figure, in pixels
    - show_formula = if True, display (via IPython) the symbolic expressions of R_nl(r) and Z_l^m(theta, phi)
    - save_as    = file path to save the figure to, as a static image (requires kaleido) or an interactive .html; format taken from the extension. Not saved if None (default)

    output: None (the figure is shown interactively, and saved if save_as
    is given)
    '''
    if show_formula:
        r_disp = sp.symbols('r', positive=True)
        theta_disp, phi_disp = sp.symbols('theta phi', real=True)
        display(sp.Eq(sp.Symbol(f"R_{{{n},{l}}}"), R_nl(n, l, r_disp, Z)))
        display(sp.Eq(sp.Symbol(f"Z_{{{l}}}^{{{m}}}"),
                       sp.simplify(Znm(l, m, theta_disp, phi_disp).expand(func=True))))

    r_max = 3 * n**2
    r_s = sp.symbols('r', positive=True)
    Rnl_func = sp.lambdify(r_s, R_nl(n, l, r_s, Z), "numpy")
    theta_s, phi_s = sp.symbols('theta phi', real=True)
    Zlm_func = sp.lambdify((theta_s, phi_s),
                            Znm(l, m, theta_s, phi_s).expand(func=True), "numpy")

    coords = np.linspace(-r_max, r_max, grid_points)
    X, Y, Zc = np.meshgrid(coords, coords, coords, indexing="ij")
    rr = np.clip(np.sqrt(X**2 + Y**2 + Zc**2), 1e-9, None)
    th = np.arccos(np.clip(Zc / rr, -1, 1))
    ph = np.arctan2(Y, X)
    psi = Rnl_func(rr) * np.real(np.asarray(Zlm_func(th, ph), dtype=complex))
    dV = (coords[1] - coords[0])**3
    iso_level = _probability_iso_level(psi, dV, prob_fraction)
    
    fig = go.Figure()
    
    if np.max(psi) > iso_level:
        fig.add_trace(go.Isosurface(
            x=X.flatten(), y=Y.flatten(), z=Zc.flatten(), value=psi.flatten(),
            isomin=iso_level, isomax=iso_level, surface_count=1,
            colorscale=[[0, color_pos], [1, color_pos]], showscale=False,
            opacity=0.5))
    if np.min(psi) < -iso_level:
        fig.add_trace(go.Isosurface(
            x=X.flatten(), y=Y.flatten(), z=Zc.flatten(), value=psi.flatten(),
            isomin=-iso_level, isomax=-iso_level, surface_count=1,
            colorscale=[[0, color_neg], [1, color_neg]], showscale=False,
            opacity=0.5))

    if show_nodes:
        # -- exact nodal surfaces: radial spheres + angular cones/planes --
        def _roots(f, x_min, x_max, n_scan=2000):
            xs = np.linspace(x_min, x_max, n_scan)
            vals = f(xs)
            roots = []
            for i in range(len(xs) - 1):
                if vals[i] == 0:
                    roots.append(xs[i])
                elif vals[i] * vals[i + 1] < 0:
                    roots.append(brentq(f, xs[i], xs[i + 1]))
            return roots

        radial_nodes = _roots(Rnl_func, 1e-6, r_max)

        def Z_eval(th_, ph_):
            val = np.real(np.asarray(Zlm_func(th_, ph_), dtype=complex))
            th_b, ph_b = np.broadcast_arrays(th_, ph_)
            return np.broadcast_to(val, th_b.shape)

        th_grid = np.linspace(0, np.pi, 300)
        ph_grid = np.linspace(0, 2 * np.pi, 300)
        TH, PH = np.meshgrid(th_grid, ph_grid)
        grid_vals = Z_eval(TH, PH)
        i_max, j_max = np.unravel_index(np.argmax(np.abs(grid_vals)), grid_vals.shape)
        theta_star, phi_star = TH[i_max, j_max], PH[i_max, j_max]
        f_theta = lambda t: Z_eval(t, phi_star)
        f_phi = lambda p: Z_eval(theta_star, p)
        eps = 1e-3
        cone_thetas = _roots(f_theta, eps, np.pi - eps)
        phi_roots = _roots(f_phi, 0, 2 * np.pi)
        plane_phis = []
        for root in phi_roots:
            canon = root % np.pi
            if not any(min(abs(canon - p), np.pi - abs(canon - p)) < 1e-3
                       for p in plane_phis):
                plane_phis.append(canon)

        gray = [[0, "gray"], [1, "gray"]]
        uu, vv = np.meshgrid(np.linspace(0, np.pi, 40), np.linspace(0, 2 * np.pi, 40))
        for radius in radial_nodes:
            fig.add_trace(go.Surface(
                x=radius * np.sin(uu) * np.cos(vv), y=radius * np.sin(uu) * np.sin(vv),
                z=radius * np.cos(uu), opacity=0.3, colorscale=gray, showscale=False))
        rr_g, pp_g = np.meshgrid(np.linspace(0, r_max, 2), np.linspace(0, 2 * np.pi, 60))
        for theta0 in cone_thetas:
            fig.add_trace(go.Surface(
                x=rr_g * np.sin(theta0) * np.cos(pp_g), y=rr_g * np.sin(theta0) * np.sin(pp_g),
                z=rr_g * np.cos(theta0), opacity=0.4, colorscale=gray, showscale=False))
        ss, zz = np.meshgrid(np.linspace(-r_max, r_max, 2), np.linspace(-r_max, r_max, 2))
        for phi0 in plane_phis:
            fig.add_trace(go.Surface(
                x=ss * np.cos(phi0), y=ss * np.sin(phi0), z=zz,
                opacity=0.3, colorscale=gray, showscale=False))

    if show_cloud:
        xs, ys, zs, signs = _sample_orbital_cloud(n, l, m, Z=Z,
                                                   n_points=n_points, r_max=r_max)
        for sign, color in ((1, color_pos), (-1, color_neg)):
            mask = signs == sign
            fig.add_trace(go.Scatter3d(
                x=xs[mask], y=ys[mask], z=zs[mask], mode="markers",
                marker=dict(size=point_size, color=color, opacity=0.4),
                name="+" if sign > 0 else "-"))

    letter = ORBITAL_LETTERS.get(l, f"(l={l})")
    fig.update_layout(
        title=f"{n}{letter} orbital (n={n}, l={l}, m={m})",
        scene=dict(xaxis=dict(title="x / a0", range=[-r_max, r_max]),
                   yaxis=dict(title="y / a0", range=[-r_max, r_max]),
                   zaxis=dict(title="z / a0", range=[-r_max, r_max]),
                   aspectmode="cube"),
        showlegend=False, width=width, height=height)

    if save_as is not None:
        from pathlib import Path
        Path(save_as).parent.mkdir(parents=True, exist_ok=True)
        if save_as.endswith(".html"):
            fig.write_html(save_as)
        else:
            fig.write_image(save_as)
    fig.show()