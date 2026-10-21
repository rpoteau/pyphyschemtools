# Quick start

## 1. Installation

**Jupyter / Google Colab** (first cell of your notebook):

```ipython3
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

## 2. First steps

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

## 3. Kinetics on Google Colab, with your own data

```ipython3
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

The full tutorial is available as an [interactive notebook on Google Colab](https://colab.research.google.com/github/rpoteau/pyphyschemtools/blob/main/Examples_pyPCT.ipynb).
