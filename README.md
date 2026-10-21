<div style="text-align:center">
<img src="https://raw.githubusercontent.com/rpoteau/pyphyschemtools/main/pyphyschemtools/resources/svg/tools4pyPC_banner.png" alt="t4pyPCbanner" width="800"/>
</div>

<br>

**pyphyschemtools** is a collection of Python utilities designed to facilitate data processing and analysis in the field of physical chemistry

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

**Jupyter / Google Colab** (first cell of your notebook):

```python
!pip install pyphyschemtools
```

**Environments without the `!` syntax** (claude.ai code sandbox, plain scripts):

```python
import subprocess, sys

result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "pyphyschemtools", "-q"],
    capture_output=True, text=True
)
print("✅ pyphyschemtools installed" if result.returncode == 0
      else f"⚠️ Installation problem:\n{result.stdout}\n{result.stderr}")
```

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

Bundled example data are available through `get_ppct_data`:

```python
from pyphyschemtools import get_ppct_data
path = get_ppct_data("Molecules/betaCD-closed.xyz")
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

Not on Colab, or just want to try it? Use the bundled example file:

```python
from pyphyschemtools import kbasic, get_ppct_data

kb = kbasic()
kb.read(get_ppct_data("Kinetics/kbasic.xlsx"))
kb.fit(ordre=1)
kb.plot()
```

The full tutorial is in [`Examples_pyPCT.ipynb`](https://colab.research.google.com/github/rpoteau/pyphyschemtools/blob/main/Examples_pyPCT.ipynb).
