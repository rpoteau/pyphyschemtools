"""
hydrogen.py — hydrogen atom and hydrogenic ions

Bohr model / analytical solution of the Schrodinger equation for a
one-electron system (H, He+, Li2+, ...), with an optional finite
nuclear-mass (reduced-mass) correction.

Public functions:
- rydberg_Z        : Rydberg constant corrected for the reduced mass
- En_hydrogenic    : energy of level n (in joules)
- Ei_hydrogenic    : ionization energy (in eV)
"""

import scipy.constants as sc
from mendeleev import element
from .visualID_Eng import fg, hl, bg, color
from .core import save_fig
import matplotlib.pyplot as plt

def rydberg_Z(Z=1, A=None, M_atom=None):
    '''
    Rydberg constant (in m^-1) corrected for the finite nuclear mass of the
    given isotope: R = Rinf * M_nucleus / (me + M_nucleus).

    Note: the value depends on the nuclear mass only, not on Z. The Z^2 factor
    belongs to the level/transition formula, not to the Rydberg constant itself.

    input:

    - Z      = atomic number (default: 1 for hydrogen); used to look up the element and, together with M_atom, to get the nuclear mass
    - A      = mass number of the isotope to use. If None (default), the most abundant isotope of element Z (highest `abundance` among its isotopes with known abundance) is used
    - M_atom = atomic mass in kg; overrides the mendeleev lookup entirely (A is then ignored)

    output:

    - Rydberg constant corrected for the reduced mass, in m^-1
    '''
    me   = sc.value("electron mass")
    u    = sc.value("atomic mass constant")
    Rinf = sc.value("Rydberg constant")
    if M_atom is None:
        el = element(int(Z))
        if A is None:
            iso = max((i for i in el.isotopes if i.abundance),
                      key=lambda i: i.abundance)
        else:
            iso = next(i for i in el.isotopes if i.mass_number == A)
        M_atom = iso.mass * u
    M_nucleus = M_atom - Z*me
    mu = me*M_nucleus/(me + M_nucleus)
    return Rinf * mu/me


def En_hydrogenic(n, Z=1, A=None, M_atom=None, reduced_mass=True, unit='eV'):
    '''
    Energy of level n of a hydrogenic (hydrogen-like) atom or ion (Bohr model /
    analytical solution of the Schrodinger equation), in joules or eV.

    E_n = -h*c*R * Z^2 / n^2, with R the Rydberg constant. With reduced_mass=True
    (default) R = R_M is corrected for the finite nuclear mass of the isotope
    (see rydberg_Z); with reduced_mass=False R = Rinf is the infinite-nucleus
    limit. Equivalently E_n = -Ei * Z^2 / n^2 with Ei = h*c*R.

    input:

    - n            = principal quantum number (integer > 0)
    - Z            = atomic number (default: 1 for hydrogen)
    - A            = mass number of the isotope (for the mendeleev mass lookup)
    - M_atom       = atomic mass in kg (overrides the lookup)
    - reduced_mass = if True (default), use the reduced-mass-corrected R_M; if False, use the infinite-nucleus constant Rinf
    - unit         = 'J' or 'eV' (default)

    output:

    - energy of level n, in joules or eV (negative value, bound state)
    '''
    R = rydberg_Z(Z=Z, A=A, M_atom=M_atom) if reduced_mass else sc.value("Rydberg constant")
    Ei = sc.h * sc.c * R          # = ionization energy for Z=1; for general Z,
                                   # the ionization energy is Ei * Z^2 (see Ei_hydrogenic)
    E = -Ei * Z**2 / n**2
    if unit.upper() == "EV":
        return E / sc.e
    elif unit.upper() == "J":
        return E
    else:
        raise ValueError(f"unknown unit '{unit}': must be 'J' or 'eV'")


def Ei_hydrogenic(Z, A=None, M_atom=None, reduced_mass=True):
    '''
    Ionization energy (in eV) of a hydrogenic (hydrogen-like) ion of nuclear
    charge Z, i.e. the energy to remove its single electron from the ground
    state (n=1 to infinity). Ei = h*c*R*Z^2, the opposite of the ground-state
    energy.

    input:

    - Z            = nuclear charge / atomic number (1 = H, 2 = He+, 3 = Li2+, ...)
    - A            = mass number of the isotope (for the mendeleev mass lookup)
    - M_atom       = atomic mass in kg (overrides the lookup)
    - reduced_mass = if True (default), R = R_M is corrected for the finite nuclear mass of the isotope (see rydberg_Z); if False, R = Rinf is the infinite-nucleus limit (A and M_atom are then ignored)

    output:

    - ionization energy of the hydrogen-like ion, in eV

    note: this is a non-relativistic Bohr/Schrodinger result. The reduced-mass
    correction is tiny (~1e-4 relative) and shrinks as Z grows, whereas the gap
    to experiment grows with Z because relativistic and QED effects (absent
    here) scale steeply with Z. To compare with mendeleev, the Z-th ionization
    energy of the element matches the hydrogen-like ion of charge Z.
    '''
    R = rydberg_Z(Z=Z, A=A, M_atom=M_atom) if reduced_mass else sc.value("Rydberg constant")
    return sc.h * sc.c * R * Z**2 / sc.e     # eV

# Series names, keyed by the lower level n1 (historically defined for H, Z=1).
SERIES_NAMES = {
    1: "Lyman",
    2: "Balmer",
    3: "Paschen",
    4: "Brackett",
    5: "Pfund",
    6: "Humphreys",
}

# One color per series (by lower level n1), reused by plot_energy_levels and
# plot_line_spectrum so a given series always has the same color.
SERIES_COLORS = {
    1: "tab:blue",
    2: "tab:red",
    3: "tab:purple",
    4: "tab:green",
    5: "tab:brown",
    6: "tab:cyan",
}
 
# Short series names, for the alpha-line labels in plot_line_spectrum.
SERIES_SHORT_NAMES = {
    1: "Ly",
    2: "Ba",
    3: "Pa",
    4: "Br",
    5: "Pf",
    6: "Hu",
}

# Greek letters for the alpha, beta, gamma, ... transitions of a series.
GREEK_LETTERS = ["α", "β", "γ", "δ", "ε", "ζ",
                  "η", "θ", "ι"]

def spectral_series(Z=1, A=None, M_atom=None, reduced_mass=True, n2_max=12,
                     series_names=None):
    '''
    Print the spectral series of a hydrogenic atom/ion of nuclear charge Z:
    for each n1 in series_names, the series limit (as n2 tends to infinity)
    and every transition n2 --> n1 for n1 < n2 <= n2_max, ..., with the transition
    energy (eV), wavelength (nm) and frequency (Hz).

    The series names (Lyman, Balmer, ...) were historically defined for the
    hydrogen atom (Z=1). For Z>1 they are only used "by analogy" (the term
    structure is the same, just scaled by Z^2; e.g. the "Lyman-like" series of
    He+ is sometimes called its Lyman series in the literature) -- printed
    here with a "-like" suffix to make that explicit.

    input:

    - Z            = nuclear charge / atomic number (default: 1 for hydrogen)
    - A            = mass number of the isotope (for the mendeleev mass lookup)
    - M_atom       = atomic mass in kg (overrides the lookup)
    - reduced_mass = if True (default), use the reduced-mass-corrected R_M; if False, use the infinite-nucleus constant Rinf
    - n2_max       = highest principal quantum number n2 to print
    - series_names = {n1: name} mapping of lower levels to print; defaults to SERIES_NAMES (Lyman to Humphreys, n1 = 1 to 6)

    output:

    - None (prints to stdout)
    '''
    if series_names is None:
        series_names = SERIES_NAMES

    for n1 in range(1, n2_max):
        name = series_names.get(n1)
        if name is None:
            header = f"n1 = {n1} series"
        else:
            label = name if Z == 1 else f"{name}-like (Z={Z})"
            header = f"{label} series (n1 = {n1})"
        E1 = En_hydrogenic(n1, Z=Z, A=A, M_atom=M_atom, reduced_mass=reduced_mass)
        lambda_inf = -1e9 * sc.h * sc.c / E1 / sc.e    # series limit, n2 -> infinity
        print(f"\n{hl.BOLD}{fg.BLUE}{header}  lambda_inf = {lambda_inf:.5g} nm{hl.OFF}")
        for n2 in range(n1 + 1, n2_max + 1):
            E2 = En_hydrogenic(n2, Z=Z, A=A, M_atom=M_atom, reduced_mass=reduced_mass)
            Ediff = E2 - E1
            wavelength = 1e9 * sc.h * sc.c / Ediff / sc.e
            frequency = sc.c / (wavelength * 1e-9)
            print(f"{n2:2} --> {n1:2}   dE = {Ediff:6.3f} eV   "
                  f"lambda = {wavelength:8.2f} nm   nu = {frequency:.3e} Hz")


def plot_energy_levels(Z=1, A=None, M_atom=None, reduced_mass=True, n2_max=7,
                        series_names=None, to_scale=True,
                        transition_type="absorption", size=(9, 11),
                        show_level_labels=True, show_transition_labels=True,
                        save_as=None):
    '''
    Draw an energy-level diagram of a hydrogenic atom/ion of nuclear charge Z:
    a horizontal line for each level n = 1 .. n2_max (labeled "n = ..." on the
    right, against a true energy axis in eV on the left), a dashed line for
    the ionization limit at 0 eV with the continuum of free-electron states
    shaded above it, and the transitions n2 <-> n1 of each series in
    series_names, drawn as colored arrows and labeled with Greek letters
    (alpha, beta, ...) in the series' color.

    By default (to_scale=True) the levels are placed at their true energy
    (eV) on the left-hand axis: since E_n scales as 1/n^2, n=2..n2_max then
    crowd together near the ionization limit while n=1 sits far below --
    itself a good illustration of why the levels bunch up as n increases.
    With to_scale=False the levels are evenly spaced by rank instead (the
    usual simplified "schematic" textbook diagram), which keeps the n>=2
    levels readable but no longer reflects the true energy axis.

    transition_type controls both the arrow direction and, within a series,
    the left-to-right layout: "absorption" (default) draws arrows pointing
    up (n1 -> n2), ordered left to right by increasing transition size
    (alpha first); "emission" draws arrows pointing down (n2 -> n1, the
    electron falling back to n1), ordered left to right by decreasing
    transition size, all arriving at the same n1.

    input:

    - Z                = nuclear charge / atomic number (default: 1 for H)
    - A                = mass number of the isotope (for the mass lookup)
    - M_atom           = atomic mass in kg (overrides the lookup)
    - reduced_mass     = if True (default), use the reduced-mass-corrected R_M; if False, use the infinite-nucleus constant Rinf
    - n2_max           = highest principal quantum number shown
    - series_names     = {n1: name} mapping of lower levels to draw; defaults to SERIES_NAMES (Lyman to Humphreys, n1 = 1 to 6)
    - to_scale         = if True (default), levels are placed at their true energy on the left-hand eV axis; if False, they are evenly spaced by rank instead (schematic)
    - transition_type  = "absorption" (default) or "emission"; see above
    - show_level_labels      = if True (default), label every level "n = ... (... eV)" on the right; if False, only n=1 to 5 are labeled (the others still get their line, just no text). The ionization limit label is kept either way.
    - show_transition_labels = if True (default), label each transition with its Greek letter, alpha to epsilon only (the first five transitions of a series) -- there is no standard name for higher members, so none is invented and those stay unlabeled regardless of this setting. If False, no per-transition label is drawn at all (the arrows and series names are unaffected).
    - size             = (width, height) in inches for a new figure
    - save_as          = file path to save the figure to (e.g. "levels.png" or "levels.svg"), passed to pyphyschemtools.core's save_fig (format taken from the extension, missing directories created automatically). Not saved if None (default)

    '''
    if transition_type not in ("absorption", "emission"):
        raise ValueError('transition_type must be "absorption" or "emission"')
    if series_names is None:
        series_names = SERIES_NAMES
    _, ax = plt.subplots(figsize=size)

    levels = range(1, n2_max + 1)
    E_eV = {n: En_hydrogenic(n, Z=Z, A=A, M_atom=M_atom,
                             reduced_mass=reduced_mass)
            for n in levels}
    # y-position of each level: either the true energy, or an evenly-spaced
    # rank (1, 2, 3, ... from bottom to top) with the continuum one step above
    # the highest level shown
    if to_scale:
        y = dict(E_eV)
        y_continuum = 0.0
    else:
        y = {n: n for n in levels}
        y_continuum = n2_max + 1
    series = {n1: series_names.get(n1, f"n1 = {n1}") for n1 in range(1, n2_max)}

    # lay out series side by side, each one as wide as its number of transitions
    gap = 0.6
    x_start = 0.0
    x_positions = {}        # n1 -> {n2: x}
    for n1 in sorted(series):
        n2_list = [n2 for n2 in levels if n2 > n1]
        if transition_type == "emission":
            n2_list = n2_list[::-1]     # largest jump (series limit side) first, on the left
        x_positions[n1] = {n2: x_start + 0.5 + k for k, n2 in enumerate(n2_list)}
        x_start += len(n2_list) + gap
    x_max = x_start

    # energy levels, spanning the full width, labeled on the right with the
    # true energy (even in schematic mode). With show_level_labels=False,
    # only n=1..5 keep their text label; every level still gets its line.
    for n in levels:
        ax.hlines(y[n], 0, x_max, color="black", lw=1)
        if show_level_labels or n <= 5:
            ax.text(x_max + 0.2, y[n], f"n = {n}  ({E_eV[n]:.3f} eV)",
                     va="center", ha="left", fontsize=10)

    # ionization limit, with the continuum of free-electron states shaded above it
    label_offset = 0.04 * (y_continuum - y[1])
    continuum_top = y_continuum + 3 * label_offset
    ax.axhspan(y_continuum, continuum_top, color="0.90", zorder=0)
    ax.hlines(y_continuum, 0, x_max, color="black", lw=1, ls="--")
    ax.text(x_max + 0.2, y_continuum, "ionization limit (0 eV)",
             va="center", ha="left", fontsize=10)
    ax.text(x_max / 2, 0.5 * (y_continuum + continuum_top), "continuum",
             color="0.4", style="italic", ha="center", va="center", fontsize=10)

    # transitions, one colored column per series: arrows first, then -- in a
    # second pass, once every arrow for every series has been drawn -- the
    # Greek-letter and series-name labels, so the labels always sit on top
    # and are never covered by another series' arrow.
    #
    # Drawn with Axes.arrow (length_includes_head=True), not annotate's
    # arrowprops: annotate's FancyArrowPatch shrinks each end by a couple of
    # points by default, which is invisible for a long arrow (Lyman) but
    # leaves a visible gap at both the tail and the tip for a short one
    # (Brackett and beyond, where the level spacing is only a fraction of an
    # eV) -- exactly the "doesn't start/end exactly on the level" problem.
    # Axes.arrow has no such shrink: the tail sits exactly at the first level
    # and the tip exactly at the second.
    head_length_full = 0.018 * (y_continuum - y[1])
    head_width = 0.15
    text_labels = []        # (x, y, text, color, va, fontweight) to draw last
    for n1, name in sorted(series.items()):
        color = SERIES_COLORS.get(n1, "gray")
        xs = x_positions[n1]
        for n2, x in xs.items():
            greek_index = n2 - n1 - 1     # alpha for n2=n1+1, beta for n1+2, ...
            if transition_type == "absorption":
                start, end = y[n1], y[n2]        # arrow points up, n1 -> n2
            else:
                start, end = y[n2], y[n1]        # arrow points down, n2 -> n1
            dy = end - start
            head_length = min(head_length_full, 0.5 * abs(dy))
            ax.arrow(x, start, 0, dy, length_includes_head=True,
                     head_width=head_width, head_length=head_length,
                     fc=color, ec=color, lw=1.0)
            # Greek-letter labels stop at epsilon (the first five
            # transitions, index 0..4): there is no standard name for
            # higher members, so none is invented.
            if show_transition_labels and greek_index < 5:
                greek = GREEK_LETTERS[greek_index]
                text_labels.append((x, y[n1], greek, color, "top", "normal", 0))
        short = SERIES_SHORT_NAMES.get(n1, f"n1={n1}")
        series_label = short if Z == 1 else f"{short}-like"
        x_center = sum(xs.values()) / len(xs)
        rotation = 0 if n1 in SERIES_SHORT_NAMES else 90
        text_labels.append((x_center, continuum_top, series_label, color,
                             "bottom", "bold", rotation))

    for x, y_anchor, text, color, va, weight, rotation in text_labels:
        y_offset = -9 if va == "top" else 6
        ax.annotate(text, xy=(x, y_anchor), xytext=(0, y_offset),
                     textcoords="offset points", color=color, ha="center",
                     va=va, fontsize=12 if weight == "bold" else 11,
                     fontweight=weight, rotation=rotation)

    ax.set_xlim(-0.5, x_max + 3)
    y_bottom = y[1] - 2 * label_offset
    y_top = continuum_top + 4 * label_offset
    ax.set_ylim(y_bottom, y_top)

    # left axis = true energy scale (eV); no x-axis (x only encodes the
    # series layout, not a physical quantity). Only ticks at or below 0 eV
    # are kept: there is no bound state above the ionization limit, so a
    # tick like "2.5" in the margin above it is physically meaningless.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_position(("data", 0))
    ax.spines["left"].set_linewidth(2)
    ax.tick_params(axis="y", width=2)
    ax.xaxis.set_visible(False)
    ax.set_yticks([t for t in ax.get_yticks() if t <= 0])
    ax.set_ylabel("Energy / eV", fontsize=12)

    num_transitions = sum(len(xs) for xs in x_positions.values())
    title = (f"Energy-level diagram of the hydrogen atom "
              f"({transition_type}, {num_transitions} transitions)" if Z == 1
              else f"Energy-level diagram (Z={Z}, {transition_type}, "
              f"{num_transitions} transitions)")
    ax.set_title(title)

    if save_as is not None:
        save_fig(save_as, fig=ax.figure)

    plt.show()


def plot_line_spectrum(Z=1, A=None, M_atom=None, reduced_mass=True, n2_max=12,
                        series_names=None, show_visible=True, log_scale=True,
                        ax=None):
    '''
    Draw the line spectrum of a hydrogenic atom/ion of nuclear charge Z: a
    wavelength axis (nm) with one colored vertical line per transition n2 -->
    n1, colored by series, and the alpha line of each series labeled with an
    arrow (e.g. "Ly-alpha").

    input:

    - Z            = nuclear charge / atomic number (default: 1 for hydrogen)
    - A            = mass number of the isotope (for the mendeleev mass lookup)
    - M_atom       = atomic mass in kg (overrides the lookup)
    - reduced_mass = if True (default), use the reduced-mass-corrected R_M; if False, use the infinite-nucleus constant Rinf
    - n2_max       = highest principal quantum number n2 included
    - series_names = {n1: name} mapping of lower levels to draw; defaults to SERIES_NAMES (Lyman to Humphreys, n1 = 1 to 6)
    - show_visible = if True (default), shade the visible range (380-780 nm)
    - log_scale    = if True (default), use a log wavelength axis (the series span several decades, from ~100 nm to several microns)
    - ax           = matplotlib Axes to draw on; a new figure is created if None (default)

    output:

    - the matplotlib Axes used
    '''
    if series_names is None:
        series_names = SERIES_NAMES
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3))

    for n1, name in sorted(series_names.items()):
        if n1 >= n2_max:
            continue
        color = SERIES_COLORS.get(n1, "gray")
        E1 = En_hydrogenic(n1, Z=Z, A=A, M_atom=M_atom, reduced_mass=reduced_mass)
        wavelengths = []
        for n2 in range(n1 + 1, n2_max + 1):
            E2 = En_hydrogenic(n2, Z=Z, A=A, M_atom=M_atom, reduced_mass=reduced_mass)
            wavelengths.append(1e9 * sc.h * sc.c / (E2 - E1))

        ax.vlines(wavelengths, 0, 1, color=color, lw=1.2)
        short = SERIES_SHORT_NAMES.get(n1, name[:2])
        ax.annotate(f"{short}-{GREEK_LETTERS[0]}", xy=(wavelengths[0], 1),
                     xytext=(wavelengths[0], 1.2), ha="center", color=color,
                     fontsize=10, arrowprops=dict(arrowstyle="->", color=color))

    if show_visible:
        ax.axvspan(380, 780, color="0.88", zorder=0)
        ax.text(0.5 * (380 + 780), -0.15, "visible", ha="center", va="top",
                 fontsize=9)

    if log_scale:
        ax.set_xscale("log")
    ax.set_ylim(-0.35, 1.5)
    ax.set_yticks([])
    ax.set_xlabel("wavelength (nm)")
    title = ("Line spectrum of the hydrogen atom" if Z == 1
              else f"Line spectrum (Z={Z})")
    ax.set_title(title)
    return ax