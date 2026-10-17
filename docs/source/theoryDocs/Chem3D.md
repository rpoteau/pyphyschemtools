# Structural Chemistry - Chemistry in 3 dimensions

**`Chem3D.py` module**

So far, it only contains the `XYZData` and `molView` classes.

## Overview

`Chem3D` provides `molView`, an interactive 3D viewer and geometry-analysis
bridge between several molecular/crystal data sources and
[py3Dmol](https://pypi.org/project/py3Dmol/). It is designed for use in
Jupyter notebooks (including Google Colab).

## Supported input sources

| Source  | Accepts                                              |
|---------|-------------------------------------------------------|
| `file`  | Path to a local XYZ / PDB / CIF structure file         |
| `mol`   | Raw string (XYZ, PDB, or MDL mol/SDF block)             |
| `cif`   | CIF content as a string                                 |
| `cid`   | PubChem CID (`int` or `str`)                            |
| `rcsb`  | RCSB PDB ID                                             |
| `cod`   | Crystallography Open Database ID                        |
| `ase`   | An `ase.Atoms` object                                   |
| `smiles`| SMILES string (PubChem lookup by default, optional local RDKit 3D generation) |

Visualization styles: `'bs'` (ball-and-stick, default), `'cpk'`
(space-filling spheres), and `'cartoon'` (protein backbone ribbon).

## Quick start

```python
from pyphyschemtools.Chem3D import molView

# From a local file
molView("molecule.xyz", source="file")

# From PubChem (aspirin)
molView(2244, source="cid")

# From an ASE Atoms object
from ase.build import molecule
molView(molecule("H2O"), source="ase", style="cpk")

# Grid of several structures side by side
molView.view_grid([2244, 2519, 702], n_cols=3, source="cid")
```

`show_n=True` permanently labels atoms with their element symbol and index; `highlight_atoms=[...]` overlays a translucent sphere on selected atoms without altering the base style.

## Headless mode

Passing `viewer=False` skips py3Dmol initialization and only extracts
atomic data into `molView.data` (an `XYZData` instance), useful for batch
geometric analysis without the memory cost of a 3D engine:

```python
mv = molView("cage.xyz", viewer=False)
volume = mv.data.get_cage_volume()
```

## Cavity and bounding-sphere analysis

`XYZData` exposes geometric methods (center of mass/geometry, VdW bounding
sphere) and, via [CageCavityCalc](https://pypi.org/project/CageCavityCalc/),
internal cavity volume and principal dimensions (PCA-based) for molecular
cages. `molView.show_bounding_sphere()` and `molView.show_cage_cavity()`
render these directly in the viewer.

## Symmetry analysis

`molView.analyze_symmetry()` wraps [pymatgen](https://pymatgen.org/) to
return the space group (periodic structures) or point group and chirality
(isolated molecules).

## API reference

```{eval-rst}
.. autoclass:: pyphyschemtools.Chem3D.molView
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pyphyschemtools.Chem3D.XYZData
   :members:
   :undoc-members:
   :show-inheritance:
```

```{note}
`source='rscb'` is deprecated in favor of `source='rcsb'` (the correct
RCSB PDB acronym) as of the upcoming release.
```

