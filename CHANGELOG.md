<div style="text-align:center">
<img src="https://raw.githubusercontent.com/rpoteau/pyphyschemtools/main/pyphyschemtools/resources/svg/tools4pyPC_banner.png" alt="t4pyPCbanner" width="800"/>
</div>
<br>
<br>

**Table of Contents**
* [Changelog pyphyschemtools](#changelog-pyphyschemtools)
* [Changelog tools4Gaussian](#changelog-tools4gaussian)
* [Changelog tools4VASP](#changelog-tools4vasp)

# Changelog pyphyschemtools

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