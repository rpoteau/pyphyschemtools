<div style="text-align:center">
<img src="https://raw.githubusercontent.com/rpoteau/pyphyschemtools/main/pyphyschemtools/icons_logos_banner/tools4pyPC_banner.png" alt="t4pyPCbanner" width="800"/>
</div>

# Changelog

## Version 0.3.4 - 2026-02-03

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

## Version 0.3.3 - 2026-02-03
### added
- `Examples.ipynb` notebook, new examples of use for:
    - **easy_rdkit** class of `cheminformatics.py`

## Version 0.3.0 - 2026-02-02
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

## Version 0.2.1 - 2026-02-02
- `chem3D.py`. Enabled multi-line titles in `view_grid()` using HTML and CSS white-space properties 

## Version 0.2.0 - 2026-02-02
### changed
- `chem3D.py`. Wrong treatment of pdb coordinates by ase. Main change is `fmt="pdb"` replaced with `fmt="proteindatabank"` in `Chem3D.py`

## Version 0.1.0 - 2026-02-01
first commit
