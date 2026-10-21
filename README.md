<div style="text-align:center">
<img src="https://raw.githubusercontent.com/rpoteau/pyphyschemtools/main/pyphyschemtools/resources/svg/tools4pyPC_banner.png" alt="t4pyPCbanner" width="800"/>
</div>

<br>

**pyphyschemtools** is a collection of Python utilities designed to facilitate data processing and analysis in the field of physical chemistry. In practice, it is essentially a collection of independent modules (units, atomic orbitals, Hückel MOs, symmetry, kinetics, spectroscopy, cheminformatics, 3D visualization, ...), each one providing its own classes and functions, which you import only when you need them

This library is an integral part of the broader [**pyPhysChem project**](https://github.com/rpoteau/pyPhysChem/), aimed at integrating Python programming into physical chemistry education, spanning from Bachelor's level to PhD students and early-career researchers. The pyPhysChem project is strategically positioned at the interface between higher education and cutting-edge research.

The pyPhysChem notebooks provide commented and illustrated examples covering fundamental topics — such as derivatives, integrals, NMR, or the hydrogen atom — while extending into advanced research applications. Among them, **ULR_DigitalChem1** (Université de La Réunion, L3 Chemistry) is a full semester-long TD/TP course built entirely on `pyphyschemtools`: fundamental constants and units, the hydrogen atom and atomic orbitals, loading and analyzing experimental data, measurement statistics, calibration and regression, and molecular symmetry (point groups, Lewis/VSEPR structures, SMILES, molecular orbitals). The repository also includes comprehensive resources for machine learning in chemistry, including artificial neural networks and autoencoders, bridging the gap between classroom theory and modern laboratory practice.

---

**Installation**

Activate your python environment and run:

```bash
pip install pyphyschemtools
```

[![PyPI version](https://img.shields.io/pypi/v/pyphyschemtools.svg)](https://pypi.org/project/pyphyschemtools/)
[![Documentation Status](https://readthedocs.org/projects/pyphyschemtools/badge/?version=latest)](https://pyphyschemtools.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/github/license/rpoteau/pyphyschemtools)](https://github.com/rpoteau/pyphyschemtools/blob/main/LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/pyphyschemtools)](https://pypi.org/project/pyphyschemtools/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18644730.svg)](https://doi.org/10.5281/zenodo.18644730)

**Main Tutorial**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rpoteau/pyphyschemtools/blob/main/Examples_pyPCT.ipynb)

---

## Quick start

### 1. Installation

**Jupyter / Google Colab** (first cell of your notebook). This cell hides the long, harmless pip warnings and simply prints a confirmation message:

```python
import subprocess, sys

result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "pyphyschemtools", "-q"],
    capture_output=True, text=True
)
print("✅ pyphyschemtools installed" if result.returncode == 0
      else f"⚠️ Installation problem:\n{result.stdout}\n{result.stderr}")
```

If you see **✅ pyphyschemtools installed**, everything is fine and you can go on to the next cell.

**Alternative** (to see the full pip output): `!pip install pyphyschemtools` in a notebook, or `pip install pyphyschemtools` in a terminal.

### 2. First steps

Modules are loaded on demand (lazy loading): there is no need for `import pyphyschemtools as ppct`, simply import the names you need.

```python
from pyphyschemtools import Energy, plot_radial, HMO

# Units: convert 1 eV to kJ/mol
print(Energy(1, "eV").to("kJ/mol"))                  # 96.4853 kJ/mol

# Hydrogen-like atomic orbitals: radial probability density of 1s, 2s, 2p
plot_radial([(1, 0), (2, 0), (2, 1)], density=True)

# Hückel molecular orbitals from a SMILES string (butadiene)
mol = HMO("C=CC=C")
mol.show()                  # molecule with its π system
mol.plot_energy_diagram()   # MO energy diagram with occupations
```

To organise the output of your notebook, two very convenient helpers are available: `centerTitle` (a centered, dark banner for section titles) and `centertxt` (centered text, with optional `style="grey"`, `"title"`, `"warning"` or `"info"`):

```python
from pyphyschemtools import centerTitle, centertxt

centerTitle("Hückel analysis of butadiene")
centertxt("Energies are given in units of β", style="info")
centertxt("Check your units!", style="warning")
```

Bundled example data are available through `get_ppct_data`:

```python
from pyphyschemtools import get_ppct_data
mol = get_ppct_data("Molecules/betaCD-closed.xyz")
```

The file can then be displayed as an interactive 3D structure (rotate, zoom) with `molView`:

```python
from pyphyschemtools import molView
molView(mol)
```

### 3. Kinetics on Google Colab, with your own data

```python
!pip install pyphyschemtools
```

```python
from pyphyschemtools import kbasic

kb = kbasic()
kb.read(colab=True)   # opens an upload dialog: pick your Excel file
                      # (2 columns: time, property G)
kb.fit(ordre=1)       # prints the optimal k and the RMSD
kb.plot()             # data + model, and the RMSD landscape
```

Here `kb.read(colab=True)` is a convenience built into `kbasic`: it opens the Colab upload button itself, then reads the file you picked.

The same thing can be done in two explicit steps, with the generic `colab_upload` function (which `kbasic.read` uses internally). It returns the path of the uploaded file, so you can pass it to `kb.read(...)` or to `pandas` directly:

```python
from pyphyschemtools import kbasic, colab_upload

path = colab_upload()   # opens the upload dialog, returns the file path
kb = kbasic()
kb.read(path)           # same as kb.read(colab=True)
kb.fit(ordre=1)
kb.plot()
```

This form is handy when you want to look at or process the file yourself first, e.g. `pd.read_excel(path)`, or when the function you use has no `colab=True` option.

Not on Colab, or just want to try it? Use the bundled example file:

```python
from pyphyschemtools import kbasic, get_ppct_data

kb = kbasic()
kb.read(get_ppct_data("Kinetics/kbasic.xlsx"))
kb.fit(ordre=1)
kb.plot()
```

### 4. Interactive tools (periodic table, organic chemistry quiz)

Some classes open a small interactive application directly in the notebook (Jupyter or Google Colab), with nothing to configure.

**Periodic table**: hover over a tile to see the properties of an element, then query one element in detail.

```python
from pyphyschemtools import PeriodicTable

pt = PeriodicTable()
pt.display()        # interactive periodic table (hover over the tiles)
pt.prop("O")        # detailed properties of oxygen (symbol or atomic number)
pt.isotopes("Cl")   # isotopes of chlorine
```

A French version with the same methods is available: `TableauPeriodique` (`afficher()` instead of `display()`).

**Organic chemistry**: the explorer and quiz opens as soon as the object is created. It has three tabs (functional group explorer, structure quiz, nomenclature quiz) and a button to switch between French and English.

```python
from pyphyschemtools import FunctionalGroupExplorer

app = FunctionalGroupExplorer()   # displays the interactive interface
```

The IUPAC names are retrieved from PubChem, so an internet connection is needed for that part.

### 5. Documentation

- **In Python**: every public class and function has a docstring, so `help()` works directly in a notebook or a terminal:

  ```python
  from pyphyschemtools import HMO
  help(HMO)
  ```

- **Online**: the full documentation is on [Read the Docs](https://pyphyschemtools.readthedocs.io/en/latest/). For each module you will find a few theoretical reminders and some context, followed by the API reference (all classes, functions and their parameters).

The full tutorial is in [`Examples_pyPCT.ipynb`](https://colab.research.google.com/github/rpoteau/pyphyschemtools/blob/main/Examples_pyPCT.ipynb).
