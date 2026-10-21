# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Setup

Install in editable mode (required for working on the package):
```bash
pip install -e .
pip install -e ".[docs]"   # if working on documentation
pip install -e ".[ml]"     # if working on ML.py (tensorflow/keras/shap)
```

## Common Commands

```bash
# Build Sphinx documentation
cd docs && make clean && make html

# Build package distribution
python -m build

# Release to PyPI (interactive: bumps version, commits, tags, pushes, uploads)
bash push_pyPi.sh "short description of changes"
```

There is no automated test suite. Testing is done by running the notebooks (`Examples_pyPCT.ipynb`, `dev.ipynb`, `test.ipynb`).

## Architecture

### Package layout

```
pyphyschemtools/
├── __init__.py              # Fast imports + lazy-loader (__getattr__)
├── core.py                  # save_fig, save_data, crop_images, centerTitle
├── visualID_Eng.py          # Terminal/notebook styling: fg, bg, hl, color
├── units.py                 # PhysicalQuantity class hierarchy with SI prefixes
├── [~25 domain modules]
├── data_examples/           # Bundled data (molecules, spectra, kinetics, QC archives)
├── quantumChem_corner/      # VASP/Gaussian CLI tools and wrappers
├── resources/               # SVG/image assets
└── utils/                   # pipManagement utility
```

### Lazy-loading pattern

`__init__.py` eagerly imports only `visualID_Eng` and `core` (fast, no heavy deps). Everything else is deferred via `__getattr__`, which searches the `modules_to_search` list (19 modules) on first attribute access, importing each in order until one defines the name. This means `from pyphyschemtools import HMO` triggers import of `huckel.py` at that point.

**Critical subtlety**: when a module file and its main class share the same name (e.g., `PeriodicTable.py` / `class PeriodicTable`), Python's submodule import side-effect would bind the module object to the package namespace, shadowing the class. The `__getattr__` implementation explicitly detects and corrects this by rebinding the class over the module in `globals()`.

Notes:
- `quantumChem_corner/` is not in the lazy-loader list; it is a subpackage exposing console scripts (`[project.scripts]` in `pyproject.toml`, e.g. `ReadIterVASP`).
- `visualID.py` (French) and `*.bak` files in the package are legacy leftovers; modules import `visualID_Eng`.

### Accessing bundled data files

Use `get_ppct_data(filename, main_folder="data_examples")` from `__init__.py`:
```python
from pyphyschemtools import get_ppct_data
path = get_ppct_data("Molecules/betaCD-closed.xyz")
```
This uses `importlib.resources` and works in editable installs, wheel installs, and Google Colab.

### Adding a new module

1. Create `pyphyschemtools/mymodule.py`
2. Add `".mymodule"` to the `modules_to_search` list in `__init__.py`
3. Any public names in `mymodule.py` become auto-accessible as `pyphyschemtools.MyClass`

### Release workflow

`push_pyPi.sh` handles the full release cycle:
- Bumps version in `pyproject.toml`, `__init__.py`, `docs/source/conf.py`, and `CITATION.cff`
- Runs `docs/make html` as a pre-commit gate (fails fast on broken docstrings)
- Commits, tags `vX.Y.Z`, pushes branch and tags
- Builds with `python -m build`, uploads with `twine`, creates GitHub release with `gh`
- Reinstalls locally with `pip install -e .`

Version numbers must stay synchronized across all four files above — the script handles this automatically; manual edits should update all four. `CHANGELOG.md` and `push_pyPi_doc.md` document the release process.

## Key Domain Modules

| Module | Main class/function | Purpose |
|--------|--------------------|---------| 
| `AO.py` | `plot_radial`, `plot_electron_cloud`, `plot_orbital_3D` | Hydrogen-like atomic orbitals (SymPy + Plotly) |
| `huckel.py` | `HMO` | Hückel MO from SMILES; energies, charges, bond orders |
| `symmetry.py` | `Cn`, `sigma`, `Sn`, `identify` | 3×3 matrix symmetry operations, point group analysis |
| `cheminformatics.py` | `easy_rdkit` | SMILES → Lewis structure via RDKit; PubChem integration |
| `Chem3D.py` | — | 3D visualization (py3Dmol + ASE), geometry measurements |
| `kinetics.py` | `kbasic`, `ReactionNetwork` | Rate constant fitting; ODE-based reaction networks |
| `units.py` | `PhysicalQuantity` | SI-prefix-aware unit system for Energy, Pressure, Temperature… |
| `PeriodicTable.py` | `PeriodicTable`, `TableauPeriodique` | Interactive Bokeh periodic table (mendeleev backend) |
| `spectro.py` | `SpectrumSimulator` | Gaussian-broadened absorption spectra from oscillator strengths |
| `orgachem.py` | `OrgaChemQuiz` | Interactive bilingual organic chemistry quiz |

## Bilingual (FR/EN) Convention

Many modules expose both French and English class names with identical APIs (e.g., `TableauPeriodique` ↔ `PeriodicTable`). User-facing text in several modules (quizzes, surveys, errors) is also bilingual. Follow this convention when adding interactive or educational features.
