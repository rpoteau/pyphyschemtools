<div style="text-align:center">
<img src="https://raw.githubusercontent.com/rpoteau/pyphyschemtools/main/pyphyschemtools/icons_logos_banner/tools4pyPC_banner.png" alt="t4pyPCbanner" width="800"/>
</div>

# Changelog

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
