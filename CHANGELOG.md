<div style="text-align:center">
<img src="https://raw.githubusercontent.com/rpoteau/pyphyschemtools/main/pyphyschemtools/resources/svg/tools4pyPC_banner.png" alt="t4pyPCbanner" width="800"/>
</div>

# Changelog

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
