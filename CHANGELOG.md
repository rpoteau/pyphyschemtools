<div style="text-align:center">
<img src="https://raw.githubusercontent.com/rpoteau/pyphyschemtools/main/pyphyschemtools/resources/svg/tools4pyPC_banner.png" alt="t4pyPCbanner" width="800"/>
</div>
<br>
<br>

**Table of Contents**
* [Changelog pyphyschemtools](#changelog-pyphyschemtools)
* [Changelog tools4Gaussian](#changelog-tools4gaussian)
* [Changelog tools4VASP](#changelog-tools4vasp)
* [Changelog pipManagement](#changelog-pipmanagement)

# Changelog pyphyschemtools

# [0.12.2] - 2026-10-20 "units.md"

### Fixed

- removed stray <div class="intro"> wrappers in **`theoryDocs/Units.md`** that broke the page layout from the Pressure section onward (unclosed div due to a missing blank line before </div>)

# [0.12.1] - 2026-10-20 "AO"

### Changed

- **`theoryDocs/AO.md`**: expanded the "Physical background" section (Hamiltonian, separation of variables, radial/angular equations, real vs. complex spherical harmonics) and added a link to the SymPy physics documentation.

- in **`AO.py`**, `plot_radial` can also overlay, for each orbital, the mean radius and its
dispersion, computed numerically from the already-normalized probability
density $P_{n\ell}(r) = r^2 R_{n\ell}(r)^2$:

    $$\langle r\rangle = \int_0^\infty r\,P_{n\ell}(r)\,dr, \qquad
    \sigma_r = \sqrt{\langle r^2\rangle - \langle r\rangle^2}$$
    
    Pass `density=True, show_mean_r=True` to `plot_radial` to draw, for each
    curve, a dashed vertical line at $\langle r\rangle$ and an error bar
    spanning $\langle r\rangle \pm \sigma_r$.

### Added

- `plot_radial()` (**`AO.py`**): new `show_mean_r` parameter (requires
  `density=True`) overlaying, for each orbital, a dashed vertical line at
  the mean radius $\langle r\rangle$ and an error bar spanning
  $\langle r\rangle \pm \sigma_r$, computed numerically from the plotted
  radial probability density $P_{n\ell}(r) = r^2 R_{n\ell}(r)^2$.

- example of `plot_radial(show_mean_r=True)` in **`Examples_pyPCT.ipynb`**

## [0.12.0] - 2026-10-19 "kinetics.py: ReactionNetwork, general reaction networks"

### Added
- **New `ReactionNetwork` class in `kinetics.py`**: a general chemical kinetics engine for an arbitrary network of (ir)reversible elementary mass-action reactions, declared as plain-text equations (`"A + 2 B -> C"`, `"A <=> B"`), numerically integrated with `scipy.integrate.solve_ivp` and, when fitting, wrapped in `scipy.optimize.curve_fit` exactly as `KORD.fit()` does. Unlike `kbasic`/`KORD` (closed-form integrated rate laws for a single A → B transformation of order 0, 1 or 2), it places no restriction on the number of species or reactions, covering parallel, consecutive and reversible reactions (and any combination) as one unified tool. Full API:
    - `add_reaction(equation, k_guess=..., kf_guess=..., kr_guess=..., order=..., order_back=...)`: parses the equation (species registered automatically, in order of first appearance); `order`/`order_back` decouple the kinetic order of the rate law from the written stoichiometry, species by species, for either direction of a reversible reaction.
    - Species names are **normalized** on parsing: accents are stripped, and a purely alphabetic name is capitalized (`"ethylene"`/`"ETHYLENE"`/`"éthylène"` all become `"Ethylene"`), while a name containing a digit is treated as a chemical formula and only its first character is forced to uppercase (`"O3"`, `"CO2"`, `"H2O"` keep their case) — so the same compound is always recognized as the same species regardless of how it was typed across different `add_reaction()` calls.
    - `set_initial_conditions({...})`, `simulate(t, k_values=None)` (pure simulation from given or guessed/fitted rate constants, no fit needed), `fit(t_exp, Y_exp, weights=..., observed_species=...)` (same returned-dict shape as `KORD.fit()`: `'k'`, `'RMSE'`, `'R2'`, `'AIC'`, `'BIC'`, `'weights'`), `summary()`.
    - `overall_equation()`: sums the stoichiometry of every declared reaction and cancels out whatever appears on both sides (typically the intermediate(s)), to recover the overall balanced equation a multi-step mechanism implies — an important sanity check on a mechanism.
    - `qssa_check(species, t)`: numerically tests the quasi-steady-state approximation (QSSA) for a given species, by comparing its net rate of change to its gross (production + consumption) rate at each requested time.
    - `plot(species=None, t_exp=None, Y_exp=None, t_max=None, log_scale=False, time_unit=None, save_img=None)`: `species=[...]` zooms in on a subset (raises a clear error on an unknown name); `log_scale=True` for a log y-axis; `time_unit="s"`/`"min"`/... labels the x-axis "Time / s" (cosmetic only — `ReactionNetwork` itself is unit-agnostic, see the new Kinetics.md 5.4); `save_img=...` saves the figure (asks before overwriting).
    - `plot_qssa(species, t, threshold=0.1, time_unit=None, save_img=None)`: two-panel diagnostic plot for the QSSA check (concentrations on top, QSSA ratio on the bottom, with the region under `threshold` shaded).
    - `copy()`: independent deep copy of the whole network (reactions, initial conditions, any fitted result), e.g. to fit the same declared network twice under different conditions without declaring every reaction twice or one `fit()` overwriting another's cached result — same rationale as `KORD.copy()`.
    - Same `__setattr__` typo guard as `kbasic`/`KORD`.
    - Validated against known analytical solutions (parallel-reaction branching ratio, consecutive-reaction Bateman equations, reversible-reaction equilibrium ratio) to ~1e-9 precision, plus a synthetic noisy-data fit-recovery test.
- New `colab_utils.py` module with a shared `colab_upload(single=True, verbose=True)` function (opens Google Colab's upload button, returns the uploaded file's path, ready for `pd.read_excel()`/`pd.read_csv()`), factoring out the code until now duplicated behind every `colab=True` parameter (`kbasic.read()`, `KORD.load_from_spreadsheet()`); exposed at the package root (`from pyphyschemtools import colab_upload`) via `__init__.py`'s lazy loader.

### Documentation
- **`Kinetics.md`**: new Section 5 "`ReactionNetwork`: general reaction networks" (5.1 Purpose and scope, 5.2 Quick Start, 5.3 Reaction syntax, 5.4 Units of rate constants, 5.5 Parallel/consecutive/reversible reactions, 5.6 Fitting rate constants, 5.7 Checking the overall equation, 5.8 Quasi-steady-state check, 5.9 Worked example: stratospheric ozone decomposition, 5.10 Worked example: consecutive reactions with a fit). The module intro now lists three classes (`kbasic`, `KORD`, `ReactionNetwork`) and one standalone function.
    - 5.4 documents that a rate constant's unit is fixed by the reaction's overall kinetic order ($[k]=\text{concentration}^{1-n}\cdot\text{time}^{-1}$), and that `time_unit=` on `plot()`/`plot_qssa()` is a cosmetic axis label only, not a unit conversion.
    - 5.8 clarifies that the QSSA ratio's numerator **is** $|d[\text{species}]/dt|$ (the first derivative), normalized by the gross (non-cancelling) flux through that species.
    - 5.9's ozone example notes that giving $\mathrm{O_2}$ a nonzero initial concentration (as it would have in a real stratospheric mixture) is what lets the fast pre-equilibrium actually equilibrate quickly; starting it at 0 instead gives the reverse step nothing to react against and the QSSA induction period becomes much longer.

## [0.11.0]-[0.11.1] - 2026-10-19 "kinetics.py: Y notation, binf identifiability, KORD comparison tools"

### Added
- **`KORD` in `kinetics.py`**:
    - `copy()`: returns an independent deep copy of a `KORD` instance, so the same loaded data can be fitted twice under different conditions (e.g. `fix_Y0`) without one fit overwriting the other's cache.
    - New module-level `superpose_plots(studies, order=1, show_experimental=True, labels=None, fix_Y0=False, fix_Yinf=False, save_img=None)` function (not a `KORD` method) to overlay the **fitted** curve of several `KORD` studies for the same order on one plot — e.g. the same data fitted freely vs. with `fix_Y0`/`fix_Yinf` — auto-labeling each curve with which parameters were held fixed.
    - `simulate_plot()`: new `order` parameter (`None` = all three orders as before, or an int to plot just one) and `show_experimental=True` by default, overlaying `self.t_exp`/`self.Y_exp` on the guess curve(s).
    - Typo guard: `kbasic` and `KORD` now override `__setattr__` to reject the creation of any new attribute once the instance is built, raising `AttributeError` (listing the existing attributes) instead of silently creating a stray, misspelled attribute that fit()/plots would then read with no warning.

### Changed
- **`kinetics.py`**: full rename from the `G` notation to `Y` (clearer, and consistent with the usual ML notation for a fitted/experimental quantity), including a "hat" on the fitted values, `Ŷ_THEO`, as in ML's ŷ for predictions. **Breaking change, no backward-compatible aliases.**
- **`KORD`**:
    - `plot_all_fits()`: an order already cached in `self.results` under different `fix_Y0`/`fix_Yinf` settings than requested is no longer silently reused as-is for the comparison plot; a small aligned warning table is printed instead, and the cache is never overwritten without the caller's explicit `fit()` call.
    - `fit()`: GUESS/OPTIM console printing now bolds the parameters actually being optimized (free) and prints the ones held fixed (`fix_Y0`/`fix_Yinf`, and `binf`, see Fixed below) in normal weight.
    - Renamed `Y_0_guess`/`Y_inf_guess`/`b_inf_guess` to `Y0_guess`/`Yinf_guess`/`binf_guess`, for consistency with the no-underscore style already used by `fix_Y0`/`fix_Yinf` and by the `binf` key in `results[order]`.
- Renamed **RMSD** to **RMSE** throughout `kinetics.py` (`kbasic` used RMSD, `KORD` already used RMSE for the exact same quantity); RMSE is now used consistently everywhere, matching scikit-learn's `root_mean_squared_error`.

### Fixed
- **`KORD.fit()`**: `binf` (the final product concentration) is no longer a free fit parameter for any order — it is now always derived from mass balance, `binf = a0*beta/alpha`, exactly as Order 1 already did. For Orders 0 and 2, leaving `binf` free alongside `k` introduced a non-identifiable, phantom degree of freedom: the model's shape depends on `k` and `binf` only through a single combined ratio, so infinitely many `(k, binf)` pairs gave the exact same curve and the exact same RMSE, and `curve_fit` converged to an arbitrary point on that family depending on the starting guess (`binf_guess`) rather than to a physically meaningful value. Deriving `binf` removes the degeneracy; the free-parameter count `p` used in AIC/BIC is now consistently 3 for every order (was 4 for Orders 0/2), removing a bias against those orders in BIC-based model selection.
- **`KORD.fit()`**: the returned `results[order]` dict no longer includes the full array of theoretical values (previously under `'G_theo'`/`'Y_theo'`), which was never read anywhere else in the module and only cluttered the printed/returned dictionary.

### Documentation
- **`Kinetics.md`**:
    - Reorganized Quick Start (new *Basic use*/*Specific fit* subsections) and new *Advanced use* section (comparing several fits with `superpose_plots`, `from_spreadsheet` vs. `load_from_spreadsheet`, reading a different sheet/experiment).
    - Section 1 rewritten ("Finding a Mathematical Model That Fits the Data"): general curve-fitting framing — the optimal parameters minimize the RMSE between $Y_\mathrm{EXP}$ and $\hat{Y}_\mathrm{THEO}$, which tends toward 0 only if the chosen model is the right one.
    - AIC formula documented alongside BIC's in Model Selection.
    - Added a note on why there is no `fix_binf` switch (see Fixed above).
    - Added a note that RMSD and RMSE are the same quantity, just the name preferred by different communities (crystallography/MD vs. statistics/ML).

## [0.10.1] - 2026-10-18 "Documentation, PeriodicTable"

### Added
- `PeriodicTable.py`
    - New `PeriodicTable(TableauPeriodique)` class: an English-language version, without the French localization patch (element/family names and `prop`/`isotopes`/`mononucleides`/`afficher` output are in English).
    - Refactor: extracted the French localization step into `_localize_names()`, overridable (no-op in `PeriodicTable`), so the rest of `patch_elements()` is shared between the two classes.

### Fixed
- `__init__.py`
    - Fixed the lazy-loading (`__getattr__`): a module and a class sharing the same name (e.g. `PeriodicTable.py` / `class PeriodicTable`) could get silently shadowed at import time (the module overwriting the class), making the class unimportable/uncallable.
    - Fix: the relevant attribute is now explicitly restored after each submodule import, so `from pyphyschemtools import TableauPeriodique, PeriodicTable` works reliably and repeatably.

### Changed
- Documentation (`PeriodicTable.md`): Added an English readthedocs page for `PeriodicTable.py`: Quick Start, method descriptions, and the two pedagogical warnings (radiogenic reclassification, CIAAW ranges).

## [0.10.0] - 2026-10-17 "AO, H, PT, calibration, HMO, easy_rdkit, MolSym"

### Added

- **New `hydrogen.py` module**: hydrogen atom / hydrogenic ions — Bohr
  model / analytical Schrödinger solution for a one-electron system
  (H, He⁺, Li²⁺, ...), with an optional finite-nuclear-mass (reduced-mass)
  correction.
    - `rydberg_Z(Z=1, A=None, M_atom=None)`: Rydberg constant (m⁻¹)
      corrected for the finite nuclear mass of a given isotope (looked up
      via `mendeleev`, or overridden with an explicit `M_atom`).
    - `En_hydrogenic(n, Z=1, A=None, M_atom=None, reduced_mass=True,
      unit='eV')`: energy of level n of a hydrogenic atom/ion, in J or eV,
      using either the reduced-mass-corrected R_M (default) or the
      infinite-nucleus constant R_inf.
    - `Ei_hydrogenic(Z, A=None, M_atom=None, reduced_mass=True)`:
      ionization energy (eV) of a hydrogenic ion of nuclear charge Z
      (ground-state → continuum); notes that the Z-th ionization energy of
      an element matches the hydrogen-like ion of charge Z, for comparison
      with `mendeleev`.
    - `spectral_series(Z=1, A=None, M_atom=None, reduced_mass=True,
      n2_max=12, series_names=None)`: prints, for each series, the series
      limit and every transition n2→n1 with its energy, wavelength and
      frequency (series named "...-like" for Z>1, since Lyman/Balmer/...
      are historically defined for H).
    - `plot_energy_levels(Z=1, A=None, M_atom=None, reduced_mass=True,
      n2_max=7, series_names=None, to_scale=True,
      transition_type="absorption", size=(9,11), show_level_labels=True,
      show_transition_labels=True, save_as=None)`: energy-level diagram —
      levels drawn either at their true 1/n² energy on an eV axis
      (`to_scale=True`, showing the levels crowd near the ionization limit)
      or evenly spaced by rank (`to_scale=False`, schematic); ionization
      limit at 0 eV with the continuum shaded above it; transitions of each
      series drawn as colored arrows labeled with Greek letters (α, β, …),
      as absorption (n1→n2, upward) or emission (n2→n1, downward,
      converging on n1).
    - `plot_line_spectrum(Z=1, A=None, M_atom=None, reduced_mass=True,
      n2_max=12, series_names=None, show_visible=True, log_scale=True,
      ax=None)`: line spectrum on a (by default log-scale) wavelength
      axis, one colored vertical line per transition, the visible range
      (380–780 nm) shaded, and each series' α-line labeled (e.g. "Ly-α").
    - `SERIES_NAMES`, `SERIES_SHORT_NAMES`, `SERIES_COLORS`,
      `GREEK_LETTERS`: module-level constants shared by
      `plot_energy_levels` and `plot_line_spectrum` so a given series
      always gets the same color/name.
    - Figure saving in `plot_energy_levels` goes through
      `pyphyschemtools.core.save_fig` (auto-creates missing directories).

- **New `AO.py` module**: hydrogenic atomic orbitals, built on sympy's radial
  wavefunctions (`R_nl`) and real spherical harmonics (`Znm`), with both
  static and interactive visualizations.
    - `plot_radial(orbitals, Z=1, density=False, show_formula=False, ...)`:
      radial part R_nl(r) or radial probability density r²R_nl(r)² for one
      or several orbitals on the same axes.
    - `plot_polar(l, m, plane="xy", density=False, show_formula=False, ...)`:
      2D polar diagram of the angular part (or angular density) in a chosen
      coordinate plane, colored by sign.
    - `plot_electron_cloud(n, l, m, plane="xz", ...)`: density-weighted
      point-cloud ("electron cloud") projection of a full orbital onto a
      chosen plane, showing both angular and radial nodes.
    - `plot_orbital_3D(n, l, m, prob_fraction=0.7, show_nodes=False,
      show_cloud=False, ...)`: interactive 3D (Plotly) rendering of an
      orbital's lobes as isosurfaces enclosing a chosen fraction of the
      total electron probability — correctly showing nested lobes for
      orbitals with radial nodes (e.g. 4d). Optionally overlays the exact
      nodal surfaces (radial nodes as spheres, angular nodes as cones/planes,
      found analytically via 1D root-finding — no extra dependency) and/or
      the sampled electron cloud.
    - `show_formula=True` (where applicable) displays the underlying R_nl
      and Z_l^m symbolic expressions via `IPython.display`.

- **New methods in `TableauPeriodique` (`PeriodicTable.py`)**:
    - `isotopes(elt_id)`: pandas table of an element's isotopes (mass
      number, isotopic mass in u, natural abundance, half-life), sorted by
      mass number; prints the element's molar mass; automatically flags
      (with an explanatory note) elements reclassified as binuclidic by
      IUPAC/CIAAW due to a trace radiogenic isotope (`exceptionsRadiogeniques`
      class attribute, currently thorium/²³⁰Th), and elements for which
      IUPAC/CIAAW gives a *range* rather than a single standard atomic
      weight because of naturally variable isotopic composition
      (`intervallesCIAAW` class attribute: H, Li, B, C, N, O, Mg, Si, S,
      Cl, Br, Tl).
    - `mononucleides(exclure_radiogeniques=True)`: pandas table of
      mononuclidic elements (elements with a single isotope of measurable
      natural abundance), with mass number, isotopic mass, molar mass, and
      half-life (or "stable"); computed from a single bulk query
      (`fetch_table("isotopes")` + `fetch_table("elements")`) instead of
      118 individual `element(Z)` ORM calls, for near-instant execution;
      `exclure_radiogeniques=False` reinstates elements like thorium
      (marked with a `*` in the symbol column) despite their trace
      radiogenic isotope.
- **`QuantitativeAnalysis` in `spectro.py`**:
    - RMSE (root mean square error) in the regression report and in `diagnostics['rmse']`.
    - `diagnostics['forced_intercept']` (bool).
    - The regression report flags a forced intercept with "(forced)".

- **New `HMO` class in the `huckel.py` module**: Simple Hückel (pi-MO) method applied directly to a molecule described by a **full SMILES** (sigma skeleton + pi system), with automatic detection of the conjugated pi system by RDKit (aromatic atoms, dienes, enones...) and Streitwieser parametrization for heteroatoms. The class aims at building and diagonalizing the Hückel matrix, fills the MOs by the Aufbau principle (+ Hund's rule for degenerate levels), and derives the usual set of quantities and diagrams:
    - energy-level diagram (`plot_energy_diagram()`), including `"FMO"` mode (diagram plus HOMO/LUMO panels assembled automatically at a fixed total height of 700 px, LUMO(s) on top and HOMO(s)/SOMO(s) on the bottom, handling degeneracies and radical/SOMO cases);
    - MO coefficients drawn on the molecule's actual 2D geometry (`plot_mo_on_graph()`, `plot_all_mo()`, `plot_mo_list()`);
    - pi charges and pi bond orders, computed relative to a neutral-carbon reference so a formally charged center (carbocation/carbanion) always carries a physically sensible sign even after delocalization (`pi_charge`, `bond_order`, `plot_pi_population()`);
    - density matrix (`density_matrix()`), symbolic or numeric Hückel matrix (`huckel_matrix()`), displayed without column truncation or scientific notation (fixed at 3 decimal places on screen; Excel/ODS export keeps full, unrounded values) and with `β` shown as the Greek letter;
    - total, atomization and conjugation pi energies (`total_pi_energy()`, `atomization_energy()`, `conjugation_energy()`, stored as `.E_total`/`.E_atomization`/`.E_conjugation`): atomization energy relative to fully isolated, non-interacting atoms; conjugation (resonance) energy relative to a localized Kekulé reference structure (RDKit Kekulization to identify the formal double bonds, each diagonalized locally as its own isolated 2-center bond, correctly generalizing to heteroatom-involving double bonds such as pyridine's C=N); both also reported by `summary()` and included in `properties()`/`hmo_properties_table()`;
    - reactivity descriptors (HOMO/LUMO energies, hardness, softness, chemical potential, electronegativity, electrophilicity) via `properties()`;
    - full molecular skeleton rendering (`show()`), highlighting the detected pi system and annotating hybridization;
    - multi-sheet export (MO table + density matrix) to Excel/ODS via `save_results()`.
    - Module-level functions: `parameter_table()` (Streitwieser / Van Catledge parameter tables, the latter extended to bromine, missing from the original 1980 publication, carried over from the historical Streitwieser parametrization), `hmo_properties_table()`, `save_fig()`, `save_data()`.
    - Support for **aromatic ions** (cyclopentadienyl, tropylium...) and **radicals**.
    - Ability to **override** any parameter (`h_overrides`, `k_overrides`, `electron_overrides`) or electron count instead of relying on the automatic heuristics.
    - **`charge`** keyword at instantiation (`HMO(smiles, charge=...)`) to add or remove pi electrons *globally*, without having to write an explicit carbocation/carbanion in the SMILES.
    - **Automatic inclusion of conjugated halogens** (F, Cl, Br) directly bonded to the pi system (lone pair folded into the pi system), with an explicit warning and an opt-out (`include_halogen_lone_pairs=False`).
    - **Hyperconjugation**: optional inclusion of a methyl/alkyl substituent attached to the pi system (`include_hyperconjugation`, `alkyl_scope`).
    - **Fixed-pixel C-C bond-length rendering** (via `bond_length_px`) for `show()` (80 px by default), `plot_pi_population()` (100 px) and the MO plots (70 px) — figure size adapts to the molecule's actual extent rather than the other way around, so neither ethylene (tiny) nor a large molecule like coronene or C60 renders poorly.

- **`easy_rdkit`, in `cheminformatics.py`**:
    - `from_name`: new classmethod to build an instance directly from a
      compound name via PubChem (e.g. `easy_rdkit.from_name("aspirin",
      lang="Fr")`), symmetrical to `from_cid`. Sets `self.cid` and
      `self.iupac_name` from the first PubChem match.

- **`molView` in `Chem3D.py`**:
    - now accepts `source='smiles'`: by default, queries PubChem
    first (pcp.get_compounds, consistent with `source='cid'`) for a
    curated 3D conformer, with automatic fallback to a local RDKit
    generation (ETKDG + MMFF94, UFF as fallback) if the molecule is not
    found there or the request fails. New parameter `prefer_local_3d`
    (default `False`); pass `True` to skip PubChem and always generate
    the conformer locally — fast and usable offline. Examples added to
    `Examples_pyPCT.ipynb`.
    - new `show_n` option to permanently display element symbol + atom index on the figure (instead of hover-only labels).
    - new `highlight_atoms` / `highlight_color` / `highlight_scale` options to overlay a translucent sphere on selected atoms without altering the base style.

- **New `symmetry.py` module**: molecular symmetry, from symmetry operations to IR / Raman activity.
    - Symmetry operations as 3×3 matrices: `identity()`, `inversion()`, `Cn(n, axis, k)`, `sigma(plane)`, `Sn(n, axis, k)`, `inv(M)`; `identify(M)` names any operation given as a matrix (`C3²(z)`, `σ(z,60°)`, `C2(xy,30°)`...); `show_matrix(M)` displays it in LaTeX, with exact entries (1/2, √3/2...) when possible.
    - Character tables of 55 point groups (`character_table(group, functions=True, complex=False)`, `available_groups()`): low-symmetry groups, Cn, S2n, Cnv, Cnh, Dn, Dnh, Dnd up to n = 8, cubic and icosahedral groups. Basis functions computed with the projection operators; complex conjugate pairs given as one real row (`*`) or, with `complex=True`, as their two components (ε notation); other names accepted with a message (S3 = C3h, C3i = S6...). All the tables are checked (classes, orthogonality, Σd² = h, Mulliken labels, Γxyz).
    - `reduce_representation(group, chi)` and `ir_raman_activity(group, chi)`: reduction of a representation given by its characters, and IR / Raman activity, without any molecule; warning and non-integer coefficients if the characters are not those of a representation.
    - `point_group_flowchart(group=None)`: decision tree of the point groups (matplotlib), with the path leading to a given group highlighted.
    - **`MolSym` class**: a molecule built from its 3D structure, read by `molView` (`MolSym.from_molView(molView(...))`: SMILES, xyz/pdb/mol files, PubChem CID...) or typed by hand (`MolSym(symbols, coords, group)`).
        - point group detected with pymatgen (`PointGroupAnalyzer`) or given; molecule placed in the standard orientation of its group (Mulliken conventions, as the "Standard orientation" of Gaussian) and symmetrized; message at creation (`verbose=True`): point group, stored in the attribute `group`, and orientation status; `orientation()`, `flowchart()`.
        - atom tracking: `track(op)`, `fixed_atoms(op)`, `is_symmetry(op)`, `equivalent_atoms()`; matrices of the operations (`matrices()`) and of their representation in a basis of atomic orbitals (`representation_matrix(op, basis)`, s, p, d, f shells).
        - representations and their reduction: `gamma_atoms()`, `gamma_orbitals(basis)` (SALCs for each set of equivalent atoms and each shell; valence basis by default, or shells such as `"pz"`, `"s pxy"`, `{"Fe": "spd", "C": "pz"}`), `gamma_bonds(bonds)` (bond stretchings), `gamma_3N()`, `gamma_vib()`, `reduce(chi)`, `activity(chi)`.
        - symmetry of molecular orbitals: `orbital_symmetry(C, atoms, shell)`, `orbital_symmetries(coeffs, energies, atoms, shell)`, e.g. with the Hückel MOs of `HMO` (`c_MOs`, `E_MOs`).
        - 3D views (py3Dmol): `show()`, `show_elements(op)` (axes, planes, inversion center), `show_operation(op, highlight)` (molecule before / after an operation, linked views).
        - messages in English or French (`lang="En"` / `"Fr"`).
    - Idealized xyz models in `data_examples/Symmetry/` (SF6, PF5, SF5Cl, cis/trans-SF4Cl2, mer/fac-SF3Cl3, B(OH)3, Fe(CO)5, eclipsed and staggered ferrocene, cerocene, ethylene, benzene), for the compounds whose 3D structure cannot be obtained from a SMILES.

### Changed
- **New examples in `Examples_pyPCT.ipynb`**: worked usage of the new modules; `demo-symmetry.ipynb`: demo of the `symmetry` module.
- **`QuantitativeAnalysis` in `spectro.py`**:
    - `QuantitativeAnalysis.fit_linear()`: new optional argument `intercept` (default `None`).
      - `None`: slope and intercept are both fitted with `scipy.stats.linregress` (unchanged behaviour).
      - float: the intercept is forced to this value (e.g. `intercept=0` for the Beer-Lambert law) and only the slope is fitted with `scipy.optimize.curve_fit`. The slope standard error is taken from the covariance matrix; the residual standard deviation `sy_x` uses n − 1 degrees of freedom instead of n − 2.
    - `QuantitativeAnalysis.model` is now a `types.SimpleNamespace` (attributes `slope`, `intercept`, `rvalue`, `stderr`, `intercept_stderr`) instead of a `LinregressResult`, so that both fitting modes expose the same interface to `plot_calibration()` and `predict()`.
    - R² is now computed as 1 − SS_res/SS_tot (SS_tot around the mean of y) in both modes, so that free and forced fits can be compared.
    - `QuantitativeAnalysis.plot_calibration()`: the 95 % prediction interval is now computed exactly, for a new single measurement:
      - free intercept: ŷ(x₀) ± t(0.975, n − 2) · s_y/x · √(1 + 1/n + (x₀ − x̄)²/Σ(xᵢ − x̄)²);
      - forced intercept: ŷ(x₀) ± t(0.975, n − 1) · s_y/x · √(1 + x₀²/Σxᵢ²).
    
      It replaces the former constant band ±1.96 · s_y/x, which ignored the Student factor (too narrow for small n) and the widening of the interval at the edges of the calibration range. The band is now drawn on a sorted grid of 200 points spanning the calibration range, so it is correct whatever the order of the input data.
- **`easy_rdkit`, in `cheminformatics.py`**:
    - `show_mol(show_Lewis=True)` no longer automatically adds explicit
      hydrogens (`show_H`) or mutates `self.mol`. The Lewis analysis now
      focuses on heavy atoms by default; pass `show_H=True` as well for full
      detail including H atoms.
    - `analyze_lewis`: new `heavy_only` parameter (default `True`). H/He atoms
      no longer appear as individual table rows; instead, a single summary
      message (✅/❌) reports whether all H atoms satisfy the duet rule,
      computed on an internal explicit-H copy without modifying `self.mol`.
    - `from_cid`, `from_name`: PubChem lookup failures (no match found, or
      network/API error) now raise `ValueError` / `ConnectionError` instead
      of printing an error and silently returning `None`. Since these
      classmethods have nothing valid to return on failure, a silent `None`
      used to cause confusing downstream errors (e.g. `AttributeError`) if
      the printed message was missed in notebook output.
- **push_pyPi.sh**
    - VERSION_XY / VERSION_XY_OLD computed once, right after NEW_VERSION, instead
      of being recomputed (or relied upon) inside separate conditional blocks.

### Fixed

- **`TableauPeriodique` in `PeriodicTable.py`**: `TableauPeriodique.__init__` crashed with
  SQLAlchemy ≥ 2.0 (`ObjectNotExecutableError`, from `mendeleev`'s
  `fetch_ionization_energies` passing an already-compiled statement to
  `pd.read_sql_query`). First ionization energies are now fetched via
  `mendeleev.fetch.fetch_table("ionizationenergies")` (filtered on
  `ion_charge == 0`), which does not have this issue.
- **`easy_rdkit`, in `cheminformatics.py`**:
    - `from_cid`: the `canonical` and `lang` parameters were not passed through
      to the constructor; an instance created via `from_cid(cid, lang="Fr")`
      always fell back to English. Both are now correctly propagated.
    - `get_isomers`: generated isomers did not inherit the `lang` of the parent
      molecule, always defaulting back to English. Fixed to inherit `self.lang`.
- **push_pyPi.sh**
    - GitHub Release creation now runs after `python -m build`, so it attaches
      the freshly built dist/ artifacts instead of the previous version's (or
      nothing, if dist/ was empty).
    - Unknown version-bump type (anything other than p/m/M) now aborts with
      exit 1 instead of silently continuing with NEW_VERSION = CURRENT_VERSION.
    - git commit, git tag, git push, git push --tags, python -m build, and
      twine upload are now checked; a failure stops the script with a message
      stating what already succeeded and what to re-run manually.
    - `read -r LEVEL` and `read -r CONFIRM` now read from /dev/tty, consistent
      with the initial REPLY prompt.

### Deprecated
- `source='rscb'` is deprecated in favor of `source='rcsb'`, in **`molView`** (correct RCSB PDB acronym).

### Documentation
- **`symmetry` module**: new documentation page `Symmetry.md` (principle, operations, character tables, `MolSym`, representations, visualization, conventions, limitations, API reference).
- **`easy_rdkit`, in `cheminformatics.py`**:
    - Added/clarified docstrings for `fetch_pubchem_data`, `from_cid`,
      `descriptors`, `show_descriptors`, `to_dict`, `analyze_stereochemistry`,
      `analyze_lewis`, and `show_mol` (behavior, side effects on `self.mol`,
      interactions between options).

## [0.9.11-0.9.12] - 2026-09-04 "centertxt"

### Changed

- `centertxt` (utils/core.py): added a `style` parameter with predefined presets (grey, title, warning, info); explicit arguments override the chosen style.

## [0.9.10] - 2026-09-02 "visualID"

### Added

- new solE style in visualID.css

## [0.9.9] - 2026-06-09 "molView fix"

### Fixed
- **`molView` class in `Chem3D.py`**:
    - `source='mol'` now accepts MDL molblocks (V2000/V3000) in addition to XYZ strings, with automatic format detection.
    - fixed an `AttributeError` on `source='mol'` and `source='cif'` where `self.server` was left unset, breaking the load confirmation message.

## [0.9.8] - 2026-06-08 "new kbasic class in kinetic.py"

### Added
- **New `kbasic` Class in `kinetics.py`**: a deliberately simple, fully commented pedagogical tool for finding a rate constant `k` by RMSD minimization, designed as a teaching step *upstream* of the more complete `KORD` class (it exposes the RMSD and the optimization rather than hiding them).
    - **Three-Operation Workflow**: a minimal, stateful interface kept to exactly three public methods.
        - `read()`: data ingestion from a two-column Excel file (time, G).
        - `fit()`: rate-constant determination by explicit RMSD minimization.
        - `plot()`: side-by-side comparison of data, fitted model, and RMSD landscape.
    - **Smart Data Ingestion**: `read()` automatically detects whether the first row is a header (string) or already data (numbers), and forces numeric (`float`) conversion of both columns.
        - **Google Colab Support**: a `colab=True` flag triggers `google.colab.files.upload()` for in-browser file selection, with the heavy import scoped locally so the class stays importable outside Colab.
        - **Pandas Recap**: prints a clean `DataFrame` summary of the loaded points (via `IPython.display` in notebooks, with a `print` fallback elsewhere).
    - **Multi-Order Models**: support for orders 0 (linear), 1 (exponential), and 2 (hyperbolic), each reduced to its simplest two-parameter form `G(t; k, G0)` decaying to zero, with `G0` defaulting to the first measured point or set manually.
    - **Explicit RMSD Minimization**: `fit()` uses `scipy.optimize.minimize` on `k` alone (with `G0` fixed), keeping the one-dimensional RMSD(k) landscape readable; the RMSD is written out by hand rather than delegated to `curve_fit`.
    - **Dual-Panel Visualization**: `plot()` generates a synchronized figure:
        - **Left Plot**: experimental points and the fitted model curve.
        - **Right Plot**: the RMSD(k) landscape with the located minimum highlighted, making visible what the optimizer explores.
    - **Pedagogical Documentation**: extensive inline comments aimed at chemistry students, including a cross-referenced explanation of why `minimize` expects a list (`x0=[k_depart]`) and why the objective therefore receives `k` as a one-element array (`k = k[0]`).
- **New Example**: added a worked `kbasic` example to `Examples_pyPCT.ipynb` (read, fit, plot on a simple kinetics dataset).

## [0.9.5-0.9.7] - 2026-04-22 "Source available in readthedocs"

### Added

- `'sphinx.ext.viewcode'` added as a sphinx option in `conf.py`

## [0.9.5] - 2026-04-16 "ReadIterVASP"

### Fixed

- `ReadIterVASP.py` was not found. Moved from `pyphyschemtools/quantumChem_corner` to `pyphyschemtools/quantumChem_corner/bash_scripts` 


## [0.9.4] - 2026-04-03 "QuantitativeAnalysis refinment"

### Added
- `QuantitativeAnalysis` Class in `spectro.py`
    - **Uncertainty Reporting**: The `fit_linear` method now calculates and displays standard errors ($\pm$) for both **Slope** (Analytical Sensitivity) and **Intercept**.
    - **Advanced Plotting Features**:
        - Added support for `figsize` in centimeters for publication-ready exports.
        - New `fontsize` parameter with automatic hierarchical scaling (-2 for legends, -1 for tick labels).
        - Implemented **bold font** weight for axis tick labels to ensure legibility in high-resolution PNG/SVG files.
    - **Global Scientific Notation**: Added a `scientific` boolean toggle to `plot_calibration` for easy switching between decimal and power-of-ten notation.
        - Integrated `ScalarFormatter` for both X and Y axes, ensuring consistency when dealing with trace concentrations (e.g., $10^{-6}$ M) or high-intensity signals.
        - Implemented `useMathText=True` to render axis multipliers as proper LaTeX exponents (e.g., $10^{7}$ instead of `1e7`).

## [0.9.0-0.9.3] - 2026-04-01 "spectro"

### Changed

- renamed `spectra.py` as `spectro.py`

### Added
- **New `QuantitativeAnalysis` Class in `spectro.py`**: complete workflow for analytical chemistry.
    - **Multi-format Data Ingestion**:
        - x and y arrays
        - `from_excel()` method using `pathlib` for robust path handling. 
        - full support for `.xlsx` (via `openpyxl`) and `.ods` LibreOffice files (via `odfpy`).
    - **Advanced Statistical Reporting**: The `fit_linear()` method now prints a professionally formatted ASCII table in the console including:
        - **Slope & Intercept** with scientific notation.
        - **Linearity Diagnostic**: $R^2$ calculation.
        - **Precision Metric**: Mean Absolute Error (**MAE**) in the signal's unit.
        - **Sensitivity Thresholds**: Automatic calculation of **LOD** (Limit of Detection) and **LOQ** (Limit of Quantification).
    - **Dual-Panel Visualization**: `plot_calibration()` generates a synchronized figure:
        - **Top Plot**: Regression line with a **95% Prediction Interval** (shaded area) to visualize measurement uncertainty.
        - **Bottom Plot**: Residual analysis to verify model homoscedasticity and detect non-linear trends.
    - **Quantification Engine**: `predict()` method to convert experimental signals (including replicates/triplicates) into concentrations with associated uncertainty.
    - - **Robust Image Export**: Enhanced `plot_calibration()` with a `save_img` parameter.
        - **Overwrite Protection**: Added an interactive user prompt (`y/n`) to prevent accidental overwriting of existing plots.
        - **Auto-Directory Creation**: The method now automatically creates the target folder hierarchy if it doesn't exist (using `pathlib`).
        - **High-Quality Defaults**: Standardized export with `300 DPI` and `bbox_inches='tight'` to ensure professional, non-cropped figures.
    - **New Documentation**: Updated `docs/spectro.md`, providing a full guide on this class:
        - Manual initialization (from lists/arrays).
        - Data ingestion from Excel/ODS.
        - Statistical interpretation of results (LOD, LOQ, MAE).
        - Quantification of unknown samples using.

## [0.8.14] - 2026-03-31 "view CN"

### Changed

- **New arguments in the `view_coordination()` function of `nano.py`: width (`w`) and height (`h`) of the window.

## [0.8.13] - 2026-03-31 "molView"

### Changed

- **New Core Attribute in the `molView` class of `Chem3D.py`: `self.structure`**: The full `ase.Atoms` object is now systematically stored upon initialization. This serves for all scientific calculations and ASE-based manipulations.

## [0.8.12] - 2026-03-23 "bye bye NanoparticleDistribution"

### Changed
`NanoparticleDistribution` class has been removed from `nano.py`. It is now part of the new **pyNanoMatBuilder** project

### Added
Link to the example in `pyproject.toml` => direct sidebar link in pyPi

### Fixed
Link to the example in `README.md` was wrong

## [0.8.11] - 2026-03-16 "svg plot of molecules"

### Added: `easy_rdkit.draw_consistent_svg()`

A new static method for generating high-quality molecular grids in SVG format with absolute scale consistency. This is specifically designed for publication-quality figures and chemical comparisons.

* **Bond Length Consistency:** Forcing `options.fixedBondLength = bond_length` ensures that a small molecule and a large one share the same scale.
* **Customizable Line Weight:** Full control over bond thickness (`options.bondLineWidth = bond_width`).
* **Vector Precision:** SVG output with outlined fonts for perfect resolution regardless of zoom.
* **Stereo Support:** Automatic rendering of R/S and E/Z annotations.
 
## [0.8.2] - 2026-03-04 "pipManagement"

### Changed
- updated list of dependencies in pyproject.toml

## [Unreleased] - 2026-02-26

### Added in `units.py`
- **Base class `PhysicalQuantity`**: Implemented a core class to handle shared logic for SI prefixes, NumPy array management, and string parsing across all physical units.
- **Molar units support**: Added automatic handling for `/mol` suffix in conversion factors (e.g., `kJ/mol`).
- **Standardized `parse()` method**: A class method to create objects directly from strings like `"1.5 atm"` or `"500 nm"`.
- **New `Pressure` class**: Full support for pressure units including `Pa`, `bar`, `atm`, `torr`, and `psi`.
- **New `Volume` class**: Full support for SI (`m3`, `L`) and Imperial units (`gallon`, `quart`, `pint`, `fluid ounce`, ...).
- **New `Length` class**: Full support for SI (`m`) and Imperial units (`inch`, `foot`, `yard`, `mile`, `nautical mile`, ...).
- **New `Mass` class**: Full support for SI (`g, u, t`) and Imperial units (`lb` , `oz`, ...).
- **New `MolarMass` class**
- **New `Density` class**

### added in the documentation `units.md`
- Quick Start section
- Sections for  `Pressure`, `Length`, `Area`, `Volume`, `Mass`, `MolarMass`, `Density`

### Changed in `units.py`
- **Refactored `Energy` class**: Now inherits from `PhysicalQuantity`. Same logic prevails for the other classes
- **Improved Arithmetic**:
    - All classes now support addition (`+`) and differences (`-`) between compatible quantities (eg Mass+Mass) with automatic internal SI conversion.
    - Addition between different types (e.g., Energy +- Pressure) is strictly prohibited to ensure physical consistency.
    - Scalar Scaling (*, /) is also implemented
- **Global SI Prefixes**: Moved `_PREFIX_DATA` outside of specific classes to a module-level global dictionary for better maintainability.
- **Smart unit representation**: `__repr__` uses scientific notation for values < 1e-4 or > 1e6
- `show_available_tools()` split into two different functions, `show_available_prefixes()` and `show_available_units()`

### Fixed in `units.py`
- **Attribute Resolution**: Fixed a bug where `show_available_tools()` failed when called as a class method.
- **Wavelength Inversion**: Improved the logic for adding energy objects when defined in wavelength units (e.g., `nm`), ensuring the result correctly returns to the expected unit.

## [0.8.0] - 2026-02-24 "FunctionalGroupExplorer by Lorris Toucouere"

### Added
- **New Module `orgachem.py`**: Introduced the `OrganicChemistryMaster` class (formerly `ChimieApp`), a comprehensive tool for learning organic chemistry. **Author**: Lorris Toucouere, PhD
    - **Interactive Explorer**: Real-time functional group detection using SMARTS patterns and RDKit.
    - **Dual Quiz System**: Structure recognition and IUPAC nomenclature training with fuzzy logic string validation.
    - **Advanced Visualization**: Integrated 2D SVG rendering with salmon-pink highlighting and 3D molecular modeling via `py3Dmol`.
    - **Bilingual Support**: Full French and English localization for all UI elements and chemical terminology.
    - **External Integration**: Automatic IUPAC name retrieval via PubChem API and structure optimization using UFF force fields.

### Dependencies
- Added `requests` (API calls)
- Added `ipywidgets` (Interactive UI)

## [0.7.6] - 2026-02-20 "Wrong link to Examples_pyPCT.ipynb"

Wrong link to `Examples_pyPCT.ipynb` in `README.md` and in `/docs/source/index.rst`, after renaming of the initial `Examples.ipynb`

## [0.7.5] - 2026-02-19 "CN in nano; Chem3D fix; chrono_show"

### Added
* **General Nano-Analysis Tools in `nano.py`:**
    * `get_coordination_numbers`: New utility to calculate atomic coordination (CN). Features an **automatic cutoff detection** algorithm based on 1.2x the 1st percentile of the interatomic distance distribution (robust for diverse scales).
    * `view_coordination`: Enhanced visualization tool that maps CN to discrete color palettes (e.g., `tab20`). 
    * **Integrated Legend:** Added a Matplotlib-based legend to `view_coordination` that dynamically displays the active CN values and the specific cutoff used in Angstroms.

### Fixed
* **3D Chemistry (`molView` class of `Chem3D.py`):** Resolved an issue where metadata from `ase.Atoms` objects caused ghost letters/labels to appear in the center of the 3D plot. 
    * Added `self.v.removeAllLabels()` during the finalization of the viewer to ensure a clean canvas.
    * Formally initialized `self.server` if ase objects.
    * Improved handling of the XYZ comment line when serializing ASE objects to prevent `py3Dmol` from interpreting header strings as 3D text.
* `chrono_show()` of `visualID_Eng.py` now given in the fast imports of `__init__.py`

## [0.7.4] - 2026-02-18 "Nano & GScan_Analyzis"

### Added
- New `nano` module. So far, contains only the `NanoparticleDistribution` class, precisely for nanoparticle size distribution analysis.
- One example has been added in the `Examples.ipynb` notebook
- (empty) file added for the documentation

## [unreleased version] - 2026-02-17

### Fixed in `GScan_Analyzis.py`: Geometry Stabilization & Alignment
- Trajectory Discontinuity: Resolved the issue where `_optimized_movie.xyz` files were not stable or smooth. This was caused by `cclib` defaulting to Gaussian's Standard Orientations, which change frame-to-frame.
- Molecular Jumping/Flickering: Implemented the Kabsch Algorithm (SVD-based Procrustes analysis, where SVD = Singular Value Decomposition) using `numpy.linalg.svd`. This corrects the "Standard Orientation" shifts by calculating the optimal rotation matrix between steps.
- Visual Flow: Output trajectory files are now stabilized, ensuring the molecule no longer rotates or shifts artificially between scan steps, making them suitable for analysis and presentations.

## [0.7.3] - 2026-02-15. "stereochem with easy-rdkit & sym with Chem3D"

### Added in `easy_rdkit`
- New `save_img` parameter in `easy_rdkit.show_mol` for SVG and PNG exports
- **Automated Stereochemistry Analysis**: New internal logic to detect and label $R/S$ chiral centers and $E/Z$ double bond geometry.
    - **Smart Warning System**: `show_mol()` now detects and reports undefined stereochemistry with context-specific messages:
        - Identifies unassigned chiral centers (atoms).
        - Identifies unassigned stereogenic double bonds (bonds).
    - **Dynamic Visualization**: 
        - Automatic activation of `addAtomIndices` and `addBondIndices` when undefined stereochemistry is detected to facilitate SMILES correction.
        - Support for `show_stereo=True` in 2D renderings using `PrepareMolForDrawing`.
    - **Achirality Detection**: Informative messages identifying molecules where stereochemistry is irrelevant (achiral).
    - **Isomer Enumeration**: 
        - Added `get_isomers()`: Recursively generates all possible discrete stereoisomers (enantiomers/diastereomers) for a given structure.
        - Added `show_isomers()`: Displays a side-by-side grid comparison of all valid stereochemical configurations.
    - **Grid Stereo Support**: `plot_grid_from_df()` now accepts a `show_stereo` argument to render CIP labels $(R/S, E/Z)$ across an entire molecular dataset.

### Added in `Chem3D`: **Advanced Symmetry & Analysis**
* **Pymatgen Integration**: Established a high-level coupling with **Pymatgen** to provide professional-grade symmetry analysis for both periodic and isolated systems => **Automated Symmetry Detection**: New `analyze_symmetry()` method:
  * **Crystals**: Identifies Space Groups, Crystal Systems, and Hermann-Mauguin notation.
    * **Molecules**: Identifies Schoenflies Point Groups and performs automated **Chirality detection**.
* **Geometry Persistence**: Atomic data is now encapsulated in a dedicated `XYZData` object, enabling geometric calculations (Center of Mass, Cavity Volume, Bounding Spheres) without reloading the source data.

### Changed
- Improved `show_mol()` drawing pipeline to prevent "ghosting" when combining highlights (aromatic/conjugation) with stereochemical labels.
- **Jupyter Integration**: Updated `show_isomers()` to use explicit `display()` calls, ensuring grids are not overwritten by subsequent plots in the same cell.

## [0.7.1, 0.7.2] - 2026-02-14. "zenodo"

### Added

- **CITATION.cff** file

### Changed
- synchronization between github and zenodo

## [0.7.0] - 2026-02-14. "NEW quantum chemistry corner"

### Added
- **Quantum Chemistry Corner**: Integration of a major suite of tools for VASP and Gaussian 16 calculations.
- **Unified Tool Changelog**: Added a specialized history section for computational chemistry tools with internal navigation:
    * [Changelog tools4Gaussian](#changelog-tools4gaussian)
    * [Changelog tools4VASP](#changelog-tools4vasp)
- **New Command Line Interfaces (CLI)**:
    - Added Python-native `GScan_Analyzis`, `pos2xyz`, `pos2cif`, and `cif2pos` (v20260213).
    - Integrated legacy Bash and Fortran tools (`GParser`, `GP2bw`, `cpG`, `ThermoWithVASP`, `vibVASP`, `selectLOBSTER`, `sel4vibVASP`, `RestartVASP`, `hVASP`, `cleanVASPf`, `VASPcv`, `ManipCell`) via Python entry point wrappers for better portability.
    - `get_qc_examples` utility for reference data retrieval.
        - Behavior: This command copies the compressed and tarred example archives (`.tar.bz2`) directly to your current working directory.
        - Usage: Requires an explicit argument (VASP or G16). Supports case-insensitive input and includes a -h help menu.

### Changed
- **Documentation**: New dedicated documentation for the Quantum Corner in `QuantumChemCorner.md`.

## [0.6.0] - 2026-02-10. "easy_rdkit, built-in data management, new core tools"

### Added
- **PubChem Integration in easy_rdkit**: Introduced `@classmethod` `easy_rdkit.from_cid(cid)` to instantiate molecular objects directly using PubChem Compound IDs.
- **easy_rdkit**: Added `descriptors` property and `show_descriptors()` method.
- **easy_rdkit**: Added `to_dict()` and `fetch_pubchem_data()` for seamless DataFrame integration.
- **Documentation & Tutorials**: Launched interactive **Google Colab** badges in the README for instant access to tutorials.
- **Documentation**: Integrated direct Google Colab links into the Sphinx documentation (`.rst` files) to allow users to jump from API docs to live examples.
- **easy_rdkit (Batch Visualization)**: 
    - Added `plot_grid_from_df()` static method.
    - Supports multi-line legends using a list of column names (e.g., `legend_cols=['Name', 'MW', 'LogP']`).
- **Data Reorganization**: Migrated `data_examples/` folder inside the `pyphyschemtools/` source directory. *Rationale*: This ensures that Excel files and graphical resources are correctly bundled and installed into the user's `site-packages` during a `pip install`.
- **Data Access Utility**: Introduced `get_ppct_data()` in `__init__.py`, a helper function to easily retrieve absolute paths for built-in example files (XYZ, Excel, etc.) available in pyphyschemtools/data_examples.
- **Modern Path Management**: Switched to `pathlib` and `importlib.resources` for robust, cross-platform file handling, ensuring compatibility with Python 3.11+.
- two powerful utilities to help organize the output results have been created in `.core.py`
- **`save_fig("path/to/plot.svg")`**: 
  - Automatically creates missing directories.
  - Supports both `plt` and `fig, ax` objects.
  - Automatically detects format (SVG, PNG, PDF, etc.) from the filename.
- **`save_data(df, "path/to/data.xlsx")`**: 
  - Save pandas DataFrames to CSV or Excel with one line.
  - Automatically handles folder creation to keep the workspace tidy.
- **Added analyze_lewis(lang='En')**: A full electronic audit of atoms (valence e-, bonding e-, formal charges, lone pairs, and vacancies).
- **Localization Engine**: Added a translation dictionary to support both English (lang="En") and French (lang="Fr") output for DataFrame headers and octet/duet status messages.
- **Smart Validation**: Standardized error messages to be language-aware (e.g., "Invalid molecule" vs "Molécule invalide").

### Fixed
- **Dependencies**: Added `ase`, `keras`, and `tensorflow` to `pyproject.toml` to ensure full environment compatibility (especially for Google Colab).
- **Lazy Loading System**: Enhanced `__init__.py` with a robust `__getattr__` mechanism. It now captures and reports specific `ImportError` messages (e.g., missing `ase` or `tensorflow`) during the module scan, providing clear diagnostic feedback to the user.
- **Data Bundling**: Updated `pyproject.toml` to ensure the `data_examples` folder (Excel, SVG) is physically included in the PyPI distribution.

## [0.5.6] - 2026-02-09. "SpectrumSimulator doc"

### Added
- **Documentation**: New introduction and technical sections in `Spectra.md`, related to the `SpectrumSimulator` class

### Fixed
- Resolved Sphinx build error: `Unknown target name: "qrcode"` by escaping underscores in docstrings.

## [0.5.5] - 2026-02-07. "qrcode and docs"

### Added
- New `QRCodeGenerator` class in `pyphyschemtools.misc` for generating branded QR codes with embedded logos.
- Integrated automatic surgical rollback logic in `push_pyPi.sh` to revert version numbers if a Git commit is cancelled.
- Documentation skeletons for core modules: `Misc`, `Chem3D`, `Spectra`, `Cheminformatics`, and `PeriodicTable`

### Changed
- Updated documentation typography to a modern **Calibri-like** (humanist sans-serif) stack via `visualID.css`.
- Refined MathJax configuration in `conf.py` to support left-alignment and improved font scaling for better readability of LaTeX formulas.
- Enhanced `push_pyPi.sh` with `tomllib` validation to catch syntax errors in `pyproject.toml` before processing.

### Fixed
- Corrected logic in the maintenance script to ensure `docs/source/conf.py` versioning is properly reverted during interrupted releases.
- Fixed font inheritance issues where LaTeX formulas were not scaling correctly with custom CSS.

## [0.5.4] - 2026-02-06. new units module

### Added
* **New `units` module**: Introduced the `Energy` class for high-level physical chemistry unit management.
    * **Spectroscopic Equivalence**: Native support for conversions between energy ($E$), wavelength ($\lambda$), and wavenumbers ($\bar{\nu}$) based on fundamental relationships:
    $$E = \frac{hc}{\lambda} \quad \text{and} \quad E = hc\bar{\nu}$$
    * **Thermal & Molar Scaling**: Integrated Boltzmann relationship ($E = k_B T$) and dynamic molar conversions for units like `kJ/mol` and `kcal/mol`.
    * **Recursive Prefix Manager**: Dynamic handling of SI prefixes ranging from **yocto** ($10^{-24}$) to **quetta** ($10^{30}$) across all supported base units.
    * **Vectorization**: Comprehensive support for **NumPy arrays**, enabling batch processing of energy levels, spectra, or thermodynamic datasets.
    * **Metadata & Inspection Tools**: 
        * `Energy.show_constants_metadata()`: Displays CODATA values and uncertainties for used physical constants.
        * `Energy.show_available_tools()`: Provides an interactive table of all supported units and prefixes.
    * **Robust Parsing**: Improved `Energy.parse()` regex logic to better support scientific notation, signed exponents, and varying whitespace.
    * **Unit Safety**: Added descriptive error messages and suggestions when a user provides a prefix (e.g., `k`) without a base unit (e.g., `kJ`).
    * **New associated documentation (`docs/source/theory/Units.md`)**
        * **ReadTheDocs Integration**: Updated `pyphyschemtools.rst` to include the new `units` module.
        * **Improved Discoverability**: Exposed the `Energy` class at the package root for simplified imports (`from pyphyschemtools import Energy`).
        * **User Manual**: Added a comprehensive Markdown-based manual featuring LaTeX-rendered physical principles and code examples.
        * **Interactive Code Blocks**: Integrated `sphinx-copybutton` to allow one-click copying of examples from the documentation.

## [0.5.0] - 2026-02-05. KORD. Kinetic Optimization & Visualization Overhaul & BIC

### Added
- **Smart Rate Constant Guessing**: Implemented an "Initial Slope" logic in the `fit` method. The starting $k$ is now dynamically calculated based on the first 10% of experimental data, ensuring improved convergence.
- **Mosaic Visualization**: Replaced single plots with a complex `subplot_mosaic` layout. It now displays a "Global Comparison" alongside individual concentration profiles for $[A]$ and $[B]$.
- **Interactive Overwrite Protection**: Added a console prompt with ANSI color coding to prevent accidentally overwriting existing `.svg` or `.png` files when saving plots.
- **Physical Constraints**: Integrated parameter `bounds` into `curve_fit` to enforce $k > 0$ and $b_{\infty} > 0$, preventing mathematically "correct" but physically impossible negative rates.
- **Information Criteria (AIC & BIC)**: Integrated Akaike and Bayesian Information Criteria into the fitting pipeline. These metrics allow for objective model selection by penalizing over-parameterization (complexity taken into account by BIC).
- **Statistical Verdict**: Added a "Confidence Verdict" in `get_best_order` based on the Kass & Raftery scale (ΔBIC), classifying model preference from "Weak" to "Decisive."

### Fixed
- **Order 1 Covariance Error**: Resolved the "rank deficient" error by correctly managing the redundancy of $b_{\infty}$ in first-order kinetic models.
- **F-string LaTeX Conflicts**: Fixed `SyntaxError` by using raw strings and doubling curly braces for LaTeX annotations (e.g., $t_{1/2}$ and $b_{\infty}$).
- **Zero-Order Stagnation**: Improved the sensitivity of Order 0 fits by scaling the initial guess to the experimental time window and signal magnitude.

### Changed
- **Docstring Standard**: Adopted NumPy/Google style docstrings in English for better code maintainability.
- **Export Quality**: Plot saving now defaults to `dpi=300` with `bbox_inches='tight'` for publication-ready graphics in both raster (.png) and vector (.svg) formats.

## [0.3.8] - 2026-02-03

### Fixed
- **Packaging**: Resolved `setuptools` warnings regarding "Package would be ignored" by replacing explicit package declaration with automatic discovery (`find_namespace`).
- **Metadata**: Fixed a syntax error in `pyproject.toml` (missing comma in dependencies) that prevented package building.

## [0.3.7] - 2026-02-03

### Added
- **Dependencies**: Added `bokeh`, `mendeleev`, and `ipykernel` to the core and documentation requirements.
- **Documentation Build**: Added `nbsphinx` and `pandoc` to `optional-dependencies[docs]` to support Jupyter Notebook integration on ReadTheDocs.

### Fixed
- **ReadTheDocs**: Updated `.readthedocs.yaml` to ensure all optional documentation dependencies are installed during the build process.

## [0.3.4] - 2026-02-03

### Added
- **Automatic Parameter Extraction**: `load_from_excel` now automatically parses $A_0$, $\alpha$, and $\beta$ from the Excel file (Rows 2, 3, and 4 in the $G$ column).
- **Dynamic Visualization**: Added color-matched horizontal dashed lines for fitted $G_0$ and $G_{\infty}$ values on the plots.
- **Dual Y-Axis**: Implemented a secondary y-axis on fits to display precise numerical values for the extrapolated limits ($G_0$ and $G_{\infty}$).
- **Documentation**: Added notes on the physical validity of Order 0 models regarding mathematical artifacts (e.g., negative absorbance) after reaction completion.

### Changed
- **Type Safety**: Forced explicit `float64` conversion for `t_exp` and `G_exp` arrays to resolve `ufunc` errors and "object" dtype issues from Excel imports.
- **Optimization Strategy**: Refactored the Order 0 fitting model to use a continuous linear function. This ensures non-zero gradients for the optimizer, significantly improving convergence when initial guesses are poor.
- **API Refactoring**: Grouped physical parameters into a dedicated tuple `(a0, alpha, beta)` in the `load_from_excel` return signature.
- **Docstring Relocation**: Moved parameter descriptions to the class level to ensure full visibility on **Read the Docs**.

### Fixed
- **Attribute Logic**: Corrected an assignment error where `self.alpha` was being used for both reactant and product coefficients (now correctly uses `self.beta`).
- **Stability**: Added safety checks in `plot_all_fits` to prevent `KeyError` crashes when specific kinetic orders fail to converge.
- **Gradient Stalling**: Fixed the "stuck" optimizer for Order 0 by removing the time-completion plateau during the fitting phase.

## [0.3.3] - 2026-02-03
### added
- `Examples.ipynb` notebook, new examples of use for:
    - **easy_rdkit** class of `cheminformatics.py`

## [0.3.0] - 2026-02-02
### changed
- `chem3D.py`
    -  Source Validation & Auto-detection: Improved `molView.__init__` to automatically detect local file paths if source is not specified. Added a strict validation check against a list of allowed sources ('file', 'mol', 'cif', 'cid', 'rscb', 'cod', 'ase') with an early exit and clear error messaging to prevent kernel crashes
    - Robust Data Extraction: Wrapped the ASE read() calls in `_load_and_display` with try/except blocks to specifically handle StopIteration and Exception errors. This prevents crashes when external APIs (like PubChem) return empty or malformed 3D data

### added
- `chem3D.py`
    - Documentation: Updated class docstrings to clarify the distinction between the `viewer` parameter (initializing the 3D engine) and the `display_now` parameter (controlling the immediate rendering of the widget).
- creation of the `Examples.ipynb` notebook, with examples of use for:
    - **molView** class of `chem3D.py`
    - **KORD** class of `kinetics.py`

## [0.2.1] - 2026-02-02
- `chem3D.py`. Enabled multi-line titles in `view_grid()` using HTML and CSS white-space properties 

## [0.2.0] - 2026-02-02
### changed
- `chem3D.py`. Wrong treatment of pdb coordinates by ase. Main change is `fmt="pdb"` replaced with `fmt="proteindatabank"` in `Chem3D.py`

## [0.1.0] - 2026-02-01
first commit

--- 

# Changelog tools4Gaussian

## 20240326. GParser. 
- Now saves without error the last optimized geometry if -S is activated, with the associated energy in the title. ZPE, H° and G° are now added
## 20240126. GParser
- Now saves the scan section of G16 is -S is activated
## 20230526. GParser.
- Now saves the last found coordinates in a `$prefix_OPT.xyz` file (Input orientation if nosymm, Standard orientation otherwise)
## 20230221. GParser.
- Minor bug fix
- `$OPTGS` instead of `$OPT`
- GS opt energies were no longer written
- NPA charges are saved twice in the `"$Prefix"_NPA.dat` file (JSMol/vChem3D requirement)
- Series of tddft vertical excitation energies calculated in the same run are all saved
in the `"$Prefix"_ExcStab.dat` file if the new -atd line command option is given
(could be useful to sum transitions of various geometries on the same simulated VUV plot)
## 20230219. GParser.
- New! Now reads tddft optimizations and frequency calculations
- Fixed. When saving TDDFT energies, wrote it twice on screen
- Fixed. Dipole moment search was wrong (| instead of || in a conditional test)
## 20230213. GParser.
- Dipole moments also searched
## 20230210. GParser.
- tddft excitations also saved in a `"$Prefix"_ExcS.dat` file
- NPA charges now saved as a single line in a `"$Prefix"_NPA.dat` file
(used in the MO page of vChem3D)
- NPA table also saved in a `"$Prefix"_NPAtab.dat` file
- `"$Prefix"_Te_fe.dat` renamed as `"$Prefix"_ExcStab.dat` 
- `"$Prefix"_NMR.dat` renamed as `"$Prefix"_NMRtab.dat` 
- all the created files are not saved anymore by default.
use the -S option to save them: `GParser -f XXX.log -S`
## 20230204. GParser.
- Now reads tddft calculations. And saves a `"$Prefix"_Te_fe.dat` file
## 20220210. Gparser.
- Bug fix with the new `-sp` functionality. Only 2/3 of the frequencies were printed and saved
## 20220210. GP2bw.
- Bug fix. some spurious characters (corresponding to the "clear" command) were not removed from the GParser log file
## 20220202. Gparser.
- New `-sp` option: all frequencies are printed in the standard output (provided that the "freq" keyword was used with Gaussian). Frequencies are now saved under column format in a `$prefix_freq.dat` file
- New GP2bw tool! Converts the colored standard output of GParser in black and white
- Commands: 
    - `GParser [options] > summary.dat`
    - `GP2bw summary.dat` creates a `summary.dat.bw` without color codes
## 20220111. GParser.
- NBO part. LP* > LP* and BD* > BD* were not all properly removed. Fixed
- new `-t threshold` option added in order to print only second-order NBO analysis > threshold
## 20220110. GParser. NBO part.
- Did not read long strings such as `BD*(   1)Mo   1 -Mo   2`
now searching form atom `$nbo` in columns 10-35 and 45-75 (formerly 17-21 & 53-58)
## 20220109. GParser.
- Bug when reading NMR isototropic chemical shieldings. 1st atom was skipped.
- Added reading of a scan calculation
- Start/End dates and time now printed
## 20220108. GParser & cpG.
- `-h` help option added 
## 20220105. GParser.
- Now reads an IRC calculation
- update: bug fix. when IRC only (all energies were printed in the OPT section)
## 20220103. GParser.
Now also reads NPA charges and NBO 2nd order PT analysis for an atom specified with `-nbo` option
## 20211230. GParser.
- Calculation of chemical shifts (use `-srX` options)
## 20211228. GParser.
- 1st version. Prints opt. iterations and energies, ZPE, thermodynamic values, chemical shieldings, low frequencies
## 20211231.
- New version of cpG

---

# Changelog tools4VASP

**Basic tools**

## 20260213.
- `pos2xyz`, `pos2cif`, `cif2pos` rewritten in python
- `hVASP` now calls `pos2xyz` and `pos2cif` instead of `pos2xyz.pl`
## 20240130. VASPcv.
- Now printing IDIPOL
## 20240130. RestartVASP.
- Now introducing various Restart schemes
## 20240127. VASPcv.
- `ReadIterVASP.py` adapted for python3
## 20230711. hVASP. New behaviour:
- All `CONTCAR` files will now recursively be saved in a `VASP-ARK/CONTCARs` folder
- `CONTCAR.xyz` files will now be saved in an XYZ folder
- New options: `-A`, `-LD`, `-O`, `-CIF`, `-POS`, `-h` (see documentation)
## 20230313. cpVASP:
- possibility to copy from a folder with a diffrerent path than the target folder
## 20220523. RestartVASP
- did not test the possible existence of a previous `OUTCAR.x.gz` file. Fixed 
## 20220515. RestartVASP:
- gzips the `OUTCAR` or `OUTCAR.x` file
## 20220515. VASPcv:
- gunzips input file (ie `OUTCAR[.x].gz`) if needed, parses the data and finally gzips the file (ie `OUTCAR[.x] > OUTCAR[.x].gz`) if and only if the input file was zipped 
## 20220419. ManipCell:
- `-fc` replication scheme added (surprisingly absent in the previous versions!)
## 20220122. RestartVASP:
- now checks if the `OUTCAR` and `CONTACR` files might be present in a `workdir`  (necessity to initialize the `tmpdir` variable at the beginning of the script)
## 20210510. ManipCell:
- new `-GA` option. Counts the number of molecules in a unitcell and returns the atom indexes/molecule (based on Graph Theory). See dedicated example
## 20210427. hVASP:
- content of `2ARK` file will be added to the cif and xyz title.
- Possibility to define another flag than `2ARK`
## 20210330. New tool: hVASP.
- Useful to create archives for SI or for data management.
## 20210312. cpVASP:
- bugfix when the trailing character of the target folder name was a "/"
## 20210311. ManipCell:
- bugfix. The AddAtoms and putInside routines did not  consider atoms lying outside the c direction
## 20210304. ManipCell:
- the delOutside routine did not consider atoms lying outside  the cell in the c direction
## 20210228. cpVASP
- updated with colors and setup of the job name in `LOBSTER.runjob`
## 20210225. ThermoWithVASP
- made compatible with the new `VASPcv` command
## 20210225. VASPcv
- now prints reference energy (i.e. 1st step) if `IBRION=5`


**selectLOBSTER**

Do not forget to put `selectLOBSTER.agr` & `selectLOBSTER-ColorMaps.README`
in a target folder (e.g. bin) and add the following line in your .bashrc:
`export PathToTables="$HOME/bin"`

## 20220719.
- Bug in c2i function when the input string did not contain trailing characters
(modification in toolbox_ABC)
## 20210314.
- Basis set and MOs are now analyzed, and written under molden format 
(writeBasisFunctions and printLCAORealSpaceWavefunction keywords of lobster >= 4.0.0)
## 20210311.
- Increased the input analyzis up to 600 characters (for long list selection)
## 20210304.
- Atomic symbols now written in the charges and magnetic moments output. 
- Fragment analysis also performed with LOBSTER original charges.
- bug fixed in the routine that saves *_EF.dat atomic d-band centers
## 20210303.
- Spilling in LOBSTER now printed in `selectLOBSTER.log`, as well as Mulliken 
- charges direcly calculated by LOBSTER (Mulliken charges of `CHARGES.lobster `
also saved as `CHARGESM.lobster`)
- minor bugs fixed in the atomic dbc printing (NaN now set to 0.d0 if atom not selected)
## 20210302.
- Improvement and minor bugs fixed. Add the possibility to use another input file than `selectLOBSTER.in`
## older versions
- pdos: calculation of the population and of the spin density/subshell/atom for open shell calculations
- pdos: addition of the "Search4EMax" option
- pdos: addition of the "Integrate" option
- cohpgenerator: selection of pair of atoms A/B according to the number of bonds between A andB ("bridge" option)
- cohpgenerator: selection of pair of atoms A/B according to the bonding of B with other atoms ("boundto" option)
- cohpgenerator: selection of metal atoms according to their core/surface type
- Up to DZ basis sets on each atom
- Possibility to start the integration of the pDOS for bandcenter calculations at a given energy (keyword: startBC)
- Correction of a serious bug: two sets with same type of atoms were merged into a single set
- Calculation of the center of mass and the width of the selected projected DOS
- Compatibility with the COHPBETWEEN command of Lobster
- Select COHP, COOP and DOS data from *.LOBSTER files

**Examples**

## 20210510.
- New `Tools4VASP-Examples/ManipCell/Ru13H17-solv-GA-option/` folder with an example of application of the new graph analysis option (`-GA`)
## 20210303.
- New `Tools4VASP-Examples/selectLOBSTER/Ru13IC-Ethanoate-H` folder with an example of analysis for a Ru13 cluster
## 20210302.
- `ManipCell-Cluster.sh` example in `Tools4VASP-Examples/ManipCell/` (application of ManipCell to supercells)

---

# Changelog pipManagement

## [0.8.4] - 2026-03-10 "pipManagement v.20260310"

### Fixed
- `systemCom = "cd " + p_home + ";\n ls -lrt *req*.txt |grep " + env` replaced with `systemCom = f"cd '{p_home}';\n ls -lrt *req*.txt | grep {env}"` (option 23)
- `systemCom = "cd " + p_home + ";\nvirtualenv " + NewEnvTxt + ";\n ls -lrt *req*.txt"` replaced with `systemCom = f"cd '{p_home}';\nvirtualenv {NewEnvTxt};\n ls -lrt *req*.txt"` (option 40)

## [0.8.2] - 2026-03-04 "pipManagement"

Integrated `pipManagement.py` as a core utility in `pyphyschemtools/utils`.

### Added
- Added automatic virtual environment detection (scans for `bin/activate` or `Scripts/activate`).
- Added cross-platform support (Linux, macOS, Windows).
- Added `pipManagement` entry point for terminal access.
- Added version check flag (`--version`).
- Replaced manual dictionary management with dynamic alphabetical sorting (case-insensitive).

### Fixed
- Fixed path handling using `pathlib` for better reliability.
