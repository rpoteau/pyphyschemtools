# Cheminformatics

**`cheminformatics.py` module**

So far, it only contains the `easy_rdkit` class

## 2D representation of organic and inorganic compounds

* [Principle](#principle)
* [Molecular Descriptors Reference](#molecular-descriptors-reference)
* [Practical examples](#practical-examples)

### Principle

The `easy_rdkit` class acts as a high-level wrapper around the RDKit library, specifically tailored for chemical education. It allows for the immediate visualization of chemical bond properties that are usually hidden in standard molecular representations. It is built on the SMILES (Simplified Molecular Input Line Entry System) standard. By providing a simple text string as input, the class automatically generates a full molecular object, calculates electronic properties and descriptors - useful for machine learning, and handles 2D coordinate generation. This approach allows students to bridge the gap between chemical notation and computational representation instantaneously.

### Molecular Descriptors Reference

Each descriptor provided by `easy_rdkit` captures a specific aspect of the molecule's nature, from its physical bulk to its potential as a drug candidate.

#### Physico-chemical Properties (Lipinski & Veber Criteria)

* **MW (Molecular Weight)**: The sum of atomic masses in the molecule (g/mol). It is a primary indicator of molecular size and its ability to diffuse across biological membranes.
* **LogP (Partition Coefficient)**: A measure of lipophilicity (octanol/water partition). A high LogP indicates a hydrophobic molecule, while a low LogP suggests hydrophilicity.
* **QED (Quantitative Estimate of Drug-likeness)**: A score between 0 and 1 that reflects how much a molecule "looks like" a drug based on its underlying properties.
* **HBA (Hydrogen Bond Acceptors)**: The number of electronegative atoms (typically N or O) that can accept a hydrogen bond.
* **HBD (Hydrogen Bond Donors)**: The number of hydrogen atoms attached to electronegative atoms (N-H or O-H) that can be donated to form a hydrogen bond.
* **RotB (Rotatable Bonds)**: A measure of molecular flexibility. Note that terminal groups like Methyl (-CH<sub>3</sub>) are **not** counted as they do not change the overall conformation upon rotation.
* **TPSA (Topological Polar Surface Area)**: The surface area (&#8491;<sup>2</sup>) contributed by polar atoms (N, O and their attached H). It is a key metric for the optimization of a drug's ability to permeate cells.

#### Topological & Structural Indices

* **Aromatic / Non-Aromatic Rings**: Distinguishes between stabilized aromatic systems (like benzene) and aliphatic or saturated rings (like cyclohexane).
* **Fsp3 (Fraction of $sp^3$ Carbons)**: The ratio of *sp*<sup>3</sup> hybridized carbons to the total carbon count. It measures the "3D-ness" or saturation of a molecule. Higher values often correlate with better solubility and success in clinical trials.
* **Connectivity Indices (Chi0 & Chi1)**:
    * **Chi0**: Focuses on individual atoms to estimate molecular volume and atomic crowding.
    * **Chi1**: Focuses on bonds and branching patterns. It is used in QSAR models to predict physical properties like boiling points and density.

---

### Practical examples

#### A. Basic use

While standard chemical drawings often omit conjugation or hybridization states, easy_rdkit allows for their explicit visualization, making it an ideal tool for teaching structural organic chemistry.

```python
import pyphyschemtools as t4pPC
from pyphyschemtools import easy_rdkit

# Initialize with a SMILES string (Paracetamol)
mol = easy_rdkit("CC(=O)NC1=CC=C(C=C1)O")

# 1. Highlight Aromaticity
t4pPC.centerTitle("Aromatic part")
mol.show_mol(plot_aromatic=True)

# 2. Highlight Conjugation and Lone Pairs
t4pPC.centerTitle("Conjugated part")
mol.show_mol(plot_conjugation=True, show_n=True)
```

#### B. Comprehensive Electronic View: Lewis structure

This example showcases how `easy_rdkit` simplifies the transition from a SMILES string to a detailed basic electronic analysis. By enabling the Lewis and hybridization flags, the library automatically calculates and overlays lone pairs, electron vacancies, and orbital states directly onto the molecular graph, providing a clear visual bridge between chemical structure and bonding theory.

```python
import pyphyschemtools as t4pPC
from pyphyschemtools import easy_rdkit

t4pPC.centerTitle("Lewis & hybridization")
mol = easy_rdkit("CC(=O)NC1=CC=C(C=C1)O", lang="Fr")
mol.show_mol(show_Lewis=True, show_n=True, show_hybrid=True, size=(600,400))
```

#### C. Fetching a SMILES from its PubChem CID

PubChem is the world's largest collection of freely accessible chemical information, maintained by the National Institutes of Health (NIH). It serves as a comprehensive authority for chemical structures, identifiers, and biological activities.

By integrating the PubChem Compound ID (CID) system, easy_rdkit allows you to bypass manual string entry. This ensures that the molecular structure is retrieved directly from a standardized, peer-reviewed source, significantly reducing errors for complex scaffolds.

```python
import pyphyschemtools as t4pPC
from pyphyschemtools import easy_rdkit

mol = easy_rdkit.from_cid(2519) #cafeine

t4pPC.centerTitle("Caffeine from PubChem")
mol.show_mol(plot_conjugation=True)
print(f"compound {mol.cid} = {mol.iupac_name}\nRDKit-canonicalized SMILES = {mol.smiles}")
```

#### D. Mastering Complex Geometry

Macrocycles (large rings) are notoriously difficult to represent clearly in 2D. Using the `macrocycle=True` flag triggers an optimized coordinate generation algorithm.

```python
import pyphyschemtools as t4pPC
from pyphyschemtools import easy_rdkit

# Cyclosporine: A challenging macrocycle for standard layout engines
smiles_cyc = "C/C=C/CC(C)C(O)C1C(=O)NC(CC)C(=O)N(C)CC(=O)N(C)C(CC(C)C)C(=O)NC(C(C)C)C(=O)N(C)C(CC(C)C)C(=O)NC(C)C(=O)NC(C)C(=O)N(C)C(CC(C)C)C(=O)N(C)C(CC(C)C)C(=O)N(C)C(C(C)C)C(=O)N1C"
mol = easy_rdkit(smiles_cyc)

t4pPC.centerTitle("Better representation of macrocycles")

t4pPC.centertxt("without the 'macrocycle' option")
mol.show_mol()

t4pPC.centertxt("with the 'macrocycle' option")
mol.show_mol(macrocycle=True)
```

#### E. Calculation of descriptors

This example highlights the seamless integration with PubChem and the automated extraction of molecular properties. By initializing a molecule via its Compound ID (CID), the library automatically retrieves the official IUPAC name and canonical SMILES, while providing a clear visualization of chemical features like &pi;-conjugation and a comprehensive table of calculated physicochemical descriptors.

```python
import pyphyschemtools as t4pPC
from pyphyschemtools import easy_rdkit

mol = easy_rdkit.from_cid(3672) #ibuprofen

t4pPC.centerTitle("Ibuprofen from PubChem")
mol.show_mol(plot_conjugation=True)
print(f"compound {mol.cid} = {mol.iupac_name}\nRDKit-canonicalized SMILES = {mol.smiles}\n")
mol.show_descriptors()
````

#### F. Store descriptors in a dataframe

This example demonstrates how to efficiently transform a chemical library into a structured dataset. By consolidating multiple `easy_rdkit` instances into a pandas DataFrame, you can automatically aggregate IUPAC names, PubChem IDs, and molecular descriptors into a single table, creating a powerful foundation for data analysis or machine learning workflows.

```python
import pyphyschemtools as t4pPC
from pyphyschemtools import easy_rdkit
import pandas as pd

# A list of molecules (some from CID, some from SMILES)
my_mols = [easy_rdkit.from_cid(3672), easy_rdkit("CC(C1=CC(=CC=C1)C(=O)C2=CC=CC=C2)C(=O)O"), easy_rdkit.from_cid(156391), easy_rdkit.from_cid(3033)]

t4pPC.centerTitle("Store descriptors in dataframes")
df = pd.DataFrame([m.to_dict(auto_fetch=True) for m in my_mols])
display(df)
````

#### G. Display the smiles of a dataframe on a grid, with legends

This final step illustrates the automated generation of a molecular gallery, translating your DataFrame back into a clean, visual grid. By specifying multiple legend columns, the method creates high-quality SVG panels with vertically stacked properties, providing an ideal format for data comparison and high-resolution export for scientific publications or student report.

```python
t4pPC.centerTitle("Draw molecules on a grid")
# Create a grid showing Name, Molecular Weight, and LogP
easy_rdkit.plot_grid_from_df(
    df, 
    smiles_col='SMILES', 
    legend_cols=['CID', 'MW', 'LogP'],
    mols_per_row=2,
    size=(350, 400),
    save_img="fig_examples/Molecules/gridMol.svg"
)
```
