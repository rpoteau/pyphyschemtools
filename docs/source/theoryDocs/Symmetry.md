# Molecular Symmetry

**`symmetry.py` module**

Symmetry operations as 3×3 matrices, character tables of 55 point groups, and the `MolSym` class: atom tracking, representations (atoms, atomic orbitals, bond stretchings, cartesian displacements), reduction into irreducible representations, symmetry of molecular orbitals, IR / Raman activity and 3D visualization of the symmetry elements.

* [Two ways of using the module](#two-ways)
* [1. Principle](#principle)
* [2. Symmetry operations](#operations)
* [3. Character tables](#tables)
* [4. The `MolSym` class](#molsym)
* [5. Representations and their reduction](#representations)
* [6. Visualization](#visualization)
* [7. Conventions](#conventions)
* [8. Limitations](#limitations)
* [API reference](#api)

---

<a id="two-ways"></a>
## Two ways of using the module

| | Without a molecular geometry | With a molecular geometry |
|---|---|---|
| **What you work on** | symmetry operations, point groups, character tables, representations given by their characters | a molecule: atoms, orbitals, bonds, vibrations |
| **Tools** | functions `Cn`, `sigma`, `Sn`, `inversion`, `identity`, `inv`, `identify`, `show_matrix`, `character_table`, `available_groups`, `point_group_flowchart`, `reduce_representation`, `ir_raman_activity` | the `MolSym` class (built from a SMILES, an xyz file, a PubChem structure... read by `molView`) |
| **Typical questions** | What is the product of two operations? What is the inverse of $C_3$? What does the character table of D<sub>6h</sub> look like? How does a given representation reduce? | Which operations exchange which atoms? What are the symmetries of the SALCs, of the π MOs, of the vibrations? How many IR bands? |
| **Sections** | [2](#operations), [3](#tables) | [4](#molsym), [5](#representations), [6](#visualization) |

No geometry, pymatgen or py3Dmol is needed for the first use.

---

<a id="principle"></a>
## 1. Principle

### 1.1 Symmetry operations as matrices

A symmetry operation leaves the molecule unchanged: each atom is sent onto an atom of the same element. With the origin at the center of mass, every operation is a linear transformation of the atomic coordinates, i.e. a 3×3 orthogonal matrix $R$:

$$\mathbf{r}' = R\,\mathbf{r}$$

* proper operations (identity $E$, rotations $C_n^k$): $\det R = +1$
* improper operations (reflections $\sigma$, inversion $i$, improper rotations $S_n^k$): $\det R = -1$

The product of two operations is the product of their matrices, and the inverse of an operation is the transpose of its matrix ($R^{-1} = R^{T}$).

### 1.2 Atom tracking

Applying $R$ to the coordinates and looking for the atom that sits at each new position gives a permutation of the atoms. This **atom tracking** is the heart of the module:

* the molecule belongs to the group if every operation of the group gives a permutation (no atom lands on an empty position)
* the number of atoms that stay in place is the character of the atomic representation $\Gamma_\mathrm{atoms}$
* the characters of all the other representations (atomic orbitals, bonds, displacements) follow from the same tracking

### 1.3 Reduction

A representation $\Gamma$ with characters $\chi(R)$ is decomposed into irreducible representations $\Gamma_i$ with the reduction formula:

$$n_i = \frac{1}{h}\sum_{\mathrm{classes}\ c} g_c\,\chi(c)\,\chi_i(c)^{*}$$

where $h$ is the order of the group and $g_c$ the number of operations in class $c$.

---

<a id="operations"></a>
## 2. Symmetry operations

*No molecular geometry is needed in this section.*

| Function | Operation |
|---|---|
| `identity()` | $E$ |
| `inversion()` | $i$ |
| `Cn(n, axis="z", k=1)` | rotation $C_n^k$ (angle $2\pi k/n$) |
| `sigma(plane="xy")` | reflection through a plane |
| `Sn(n, axis="z", k=1)` | improper rotation $S_n^k = \sigma_h C_n^k$ |
| `inv(M)` | inverse of an operation ($= M^T$) |
| `identify(M)` | name of an operation given as a matrix |
| `show_matrix(M)` | LaTeX display of a matrix, with exact entries ($\frac{1}{2}$, $\frac{\sqrt{3}}{2}$...) when possible |

* `axis`: `"x"`, `"y"`, `"z"` or any direction, e.g. `(1, 1, 1)`
* `plane`: `"xy"`, `"xz"`, `"yz"` or, for any other plane, its normal vector, e.g. `(1, -1, 0)` for the diagonal plane containing $z$ and the bisector of $x$ and $y$ ($\sigma_d$ of CH<sub>4</sub>, D<sub>4h</sub>...)

```python
from pyphyschemtools.symmetry import Cn, sigma, Sn, inv, identify, show_matrix

C3 = Cn(3)
identify(C3 @ C3)                    # 'C3²(z)'
identify(inv(C3), with_inverse=True) # 'C3²(z) = C3⁻¹(z)'
identify(sigma("xz") @ sigma("yz"))  # 'C2(z)'
identify(Sn(6) @ Sn(6) @ Sn(6))      # 'i'
show_matrix(C3)
```

Names given by `identify`:

* axes: `C3(z)`, `C2(x)`, `C3(1,1,1)`; an axis lying in the $xy$ plane at an angle $\varphi$ from $x$: `C2(xy,30°)`
* planes: `σ(xy)`, `σ(xz)`, `σ(yz)`; a vertical plane containing $z$ and the direction at $\varphi$ from $x$: `σ(z,60°)`; any other plane is given by its normal: `σ⊥(1,1,1)`
* powers: `C3²`, `S4³`, `S3⁵`...

---

<a id="tables"></a>
## 3. Character tables

*No molecular geometry is needed in this section.*

`character_table(group, functions=True, complex=False)` returns the character table as a pandas DataFrame, with the basis functions (linear functions and rotations, quadratic functions). `available_groups()` lists the 55 point groups:

| Family | Groups |
|---|---|
| low symmetry | C<sub>1</sub>, C<sub>s</sub>, C<sub>i</sub> |
| C<sub>n</sub> | C<sub>2</sub> ... C<sub>8</sub> |
| S<sub>2n</sub> | S<sub>4</sub>, S<sub>6</sub>, S<sub>8</sub> |
| C<sub>nv</sub>, C<sub>nh</sub>, D<sub>n</sub>, D<sub>nh</sub>, D<sub>nd</sub> | n = 2 ... 8 |
| cubic, icosahedral | T, T<sub>h</sub>, T<sub>d</sub>, O, O<sub>h</sub>, I, I<sub>h</sub> |

* other names are accepted with a message, e.g. S<sub>3</sub> = C<sub>3h</sub> (for odd $n$, $S_n$ contains $\sigma_h$ and $C_n$), C<sub>3i</sub> = S<sub>6</sub>
* the tables of the textbooks (n ≤ 6) are written explicitly; the other axial groups are built automatically, and the construction has been checked against the explicit tables
* every table has been checked: classes, orthogonality of the rows, $\sum d_i^2 = h$, Mulliken labels ($g/u$, $'/''$), reduction of $\Gamma_{xyz}$
* the basis functions are not copied from the textbooks: they are computed with the projection operators

<div class="rqE">

**Complex conjugate pairs.** In the groups C<sub>3</sub>...C<sub>8</sub>, S<sub>4</sub>, S<sub>6</sub>, S<sub>8</sub>, C<sub>3h</sub>...C<sub>8h</sub>, T and T<sub>h</sub>, some irreducible representations come in pairs with complex conjugate characters ($\varepsilon = e^{2\pi i/n}$). By default, each pair is given as a single real row (sum of the two characters, e.g. $2, -1, -1$ for C<sub>3</sub>), marked with `*`; it counts as one pair in the reductions. With `complex=True`, the two components ¹E and ²E are given separately.

</div>

**Finding the point group.** `point_group_flowchart(group=None, lang="En")` draws the usual decision tree (linear? several $C_n$ with $n \geq 3$? highest-order $C_n$? $n$ $C_2$ perpendicular to $C_n$? $\sigma_h$? $n$ $\sigma_d$ or $\sigma_v$? $S_{2n}$?...). If a group is given, the path leading to it is highlighted. `MolSym.flowchart()` does the same for the group of the molecule.

**Reduction without a molecule.** A representation given by its characters (one per class, in the order of the columns of the character table) can be reduced, and the IR / Raman activity of its components obtained, without any geometry:

```python
from pyphyschemtools.symmetry import reduce_representation, ir_raman_activity

reduce_representation("C3v", [4, 1, 2])          # Γ = 2A1 + E
ir_raman_activity("Td", [9, 0, 1, -1, 3])        # IR active: 2T2 | Raman active: A1 + E + 2T2
```

The methods `MolSym.reduce()` and `MolSym.activity()` do the same, with the point group of the molecule.

If the coefficients $n_i$ are not integers, the characters do not correspond to a representation of the group (typing error, wrong group...): a warning is printed, followed by the decomposition with its non-integer coefficients, e.g. `reduce_representation("Td", [9, 0, 1, -1, 5])` → `Γ = 1.5 A1 - 0.5 A2 + 1 E - 0.5 T1 + 2.5 T2`.

---

<a id="molsym"></a>
## 4. The `MolSym` class

### 4.1 Building a molecule

A `MolSym` molecule is built from its **3D structure** (position of each atom). `MolSym` does not read structures itself: it delegates this task to `molView` (Chem3D module), and gets the atoms and their coordinates from it.

| Input | Call | Typical use |
|---|---|---|
| SMILES | `MolSym.from_molView(molView(smiles, source="smiles", display_now=False))` | most molecules: 3D structure from PubChem, or generated by RDKit |
| file (xyz, pdb, mol...) | `MolSym.from_molView(molView(filename, source="file", display_now=False))` | your own structures, or the models of `data_examples/Symmetry/` |
| PubChem CID, COD... | `MolSym.from_molView(molView(962, source="cid", display_now=False))` | any structure read by `molView` |
| symbols and coordinates | `MolSym(symbols, coords, group)` | coordinates typed by hand, e.g. to see the effect of the orientation |

```python
from pyphyschemtools import get_ppct_data
from pyphyschemtools.symmetry import MolSym
from pyphyschemtools.Chem3D import molView

water = MolSym.from_molView(molView("O", source="smiles", display_now=False))
sf6 = MolSym.from_molView(molView(get_ppct_data("Symmetry/SF6_model.xyz"), source="file", display_now=False))
print(water.group, sf6.group)                     # C2v Oh
```

The SMILES → 3D step needs a geometry optimization (force fields MMFF94 or UFF of RDKit), which fails for hypervalent compounds (SF<sub>6</sub>, PF<sub>5</sub>...) and organometallic compounds (ferrocene...): use an xyz file. A SMILES does not describe conformations either (e.g. the orientation of the OH groups of B(OH)<sub>3</sub>). Idealized models are provided in `data_examples/Symmetry/`: SF<sub>6</sub>, PF<sub>5</sub>, SF<sub>5</sub>Cl, cis/trans-SF<sub>4</sub>Cl<sub>2</sub>, mer/fac-SF<sub>3</sub>Cl<sub>3</sub>, B(OH)<sub>3</sub>, Fe(CO)<sub>5</sub>, eclipsed and staggered ferrocene, cerocene, ethylene, benzene.

`from_molView(mv, group=None, display_now=False, **kwargs)`: `molView` already displays the molecule (its own `display_now` argument); `display_now=True` displays it again, in the standard orientation of its point group (see `show()`). The other arguments are those of `MolSym` (below).

### 4.2 Point group and orientation

* **Point group not given** (`group=None`, default): `MolSym` delegates the symmetry analysis to pymatgen (`PointGroupAnalyzer`), which gives the point group and the symmetry operations of the molecule. The molecule is then placed in the standard orientation of its point group and symmetrized. This is the simplest way to proceed, and the only one for a structure of unknown orientation, e.g. from PubChem.
* **Point group given** (e.g. `"C2v"`): the molecule is expected to be in the standard orientation of this group (see [Conventions](#conventions)): principal axis along $z$, molecular plane $yz$ for C<sub>2v</sub>, etc. In this case, pymatgen is not needed: the module only checks the symmetry and, among the equivalent standard orientations, chooses the conventional one (e.g. water in the $xz$ plane is moved to the $yz$ plane). If the molecule is not in a standard frame, the module calls pymatgen to reorient it (if pymatgen is installed); otherwise, a warning is printed: the molecule does not belong to the group, or it is not in the standard orientation.
* `orientation()` tells where the molecule is: shape (linear, planar...), molecular plane, principal axis.

The coordinates are always translated to the center of mass. With `standard_orientation=True` (default), they are also symmetrized (average of the images of each atom over the group): this is needed for the 3D structures of PubChem or RDKit, which are not exactly symmetric. The initial (centred) coordinates are kept in `coords_input`. With `standard_orientation=False`, the coordinates are kept as given (no reorientation, no symmetrization), which is useful to show what happens when a molecule is not in the standard orientation.

### 4.3 Arguments

| Argument | Meaning |
|---|---|
| `symbols`, `coords` | chemical symbols, cartesian coordinates (Å) |
| `group` | Schoenflies symbol; if `None`, detected with pymatgen |
| `standard_orientation=True` | places the molecule in the standard orientation of its point group, as the "Standard orientation" of Gaussian, and symmetrizes it |
| `sym_tolerance=0.3` | tolerance (Å) of the symmetry detection (pymatgen) and of the symmetrization |
| `tol=0.05` | tolerance (Å) used to decide that two atoms coincide |
| `lang="En"` | language of the messages (`"En"` or `"Fr"`) |
| `verbose=True` | prints, at the creation of the molecule, its point group and whether it was already in the standard orientation or has been placed in it |

**Attributes.** At its creation, the molecule prints its point group, e.g. *Point group C2v (detected by pymatgen), stored in the attribute group. Molecule placed in the standard orientation of C2v and symmetrized.* The results are stored in attributes:

| Attribute | Content |
|---|---|
| `group` | Schoenflies symbol of the point group, e.g. `"C2v"`: `print(water.group)` |
| `symbols`, `atom_labels` | chemical symbols; labels of the atoms (`"O0"`, `"H1"`...) |
| `coords` | coordinates (Å) used by the module, in the standard orientation |
| `coords_input` | initial coordinates, centred on the center of mass |
| `labels`, `order` | labels of the classes (columns of the character table); order $h$ of the group |

### 4.4 Main methods

| Method | Result |
|---|---|
| `orientation()` | shape (linear, planar...), molecular plane, principal axis |
| `flowchart()` | flowchart of the point groups, with the path leading to the group of the molecule |
| `track(op)` | atom tracking, e.g. `C2 : O0 → O0, H1 → H2, H2 → H1` |
| `fixed_atoms(op)`, `is_symmetry(op)` | atoms that stay in place; test of an operation |
| `equivalent_atoms()` | sets of symmetry-equivalent atoms |
| `matrices(op=None, all_operations=False)` | matrices of the operations of the group (LaTeX) |
| `representation_matrix(op, basis="s")` | matrix of an operation in a basis of atomic orbitals, and its trace (character); see [the syntax of the bases](#basis-syntax) |
| `character_table(gamma=True)` | character table, with the row $\Gamma_\mathrm{atoms}$ |
| `gamma_atoms()`, `gamma_orbitals()`, `gamma_bonds()`, `gamma_3N()`, `gamma_vib()` | representations (see below) |
| `reduce(chi)`, `activity(chi)` | reduction, IR / Raman activity of any representation |
| `orbital_symmetry(C)`, `orbital_symmetries(coeffs, energies)` | irreducible representation of molecular orbitals |
| `show()`, `show_elements(op)`, `show_operation(op)` | 3D views (py3Dmol) |

`op` is a class label of the character table (`"C2"`, `"σv(xz)"`, `"C2'"`...) or any 3×3 matrix.

<div class="rqE">

The methods that display a result (`orientation`, `reduce`, `gamma_*`, `activity`, `orbital_symmetry`...) return nothing with `show=True` (default), to avoid an echo in a notebook. Use `show=False` to get the result (characters, dict, DataFrame) without display.

</div>

---

<a id="representations"></a>
## 5. Representations and their reduction

### 5.1 Principle

A basis of functions attached to the molecule (atomic orbitals, bonds, displacements...) spans a representation of the point group. For an operation $R$, the character $\chi(R)$ is the trace of the matrix of $R$ in this basis. It can be obtained without writing the matrix: only the functions that stay in place count.

* $R\,u_i = +u_i$: contribution $+1$
* $R\,u_i = -u_i$: contribution $-1$
* $R\,u_i = u_j$, another function of the basis (e.g. the similar function of another atom): contribution $0$

In particular, an atom that moves contributes 0, and an atom that stays in place contributes:

| Basis | Method | Contribution of a fixed atom |
|---|---|---|
| atoms (one s function per atom) | `gamma_atoms()` | $1$ |
| atomic orbitals of angular momentum $l$ | `gamma_orbitals(basis)` | $\chi_l(R)$ |
| cartesian displacements | `gamma_3N()` | $\chi_{xyz}(R) = \mathrm{tr}\,R$ |

with, for a rotation of angle $\theta$:

$$\chi_l(C_\theta) = \frac{\sin\left((2l+1)\theta/2\right)}{\sin(\theta/2)}, \qquad \chi_l(R) = (-1)^l\,\chi_l(-R)\ \text{for an improper } R$$

($\chi_0 = 1$ for s, $\chi_1 = \mathrm{tr}\,R$ for p).

The representation is then reduced into irreducible representations with `reduce()` (see [1.3](#principle)).

<a id="atomic-orbitals"></a>
### 5.2 Atomic orbitals

`gamma_orbitals(basis)` gives, for each set of equivalent atoms and each shell, the characters and the symmetries of the symmetry-adapted linear combinations (SALCs), i.e. the building blocks of a molecular orbital diagram. Without argument, the basis is made of the valence orbitals of all the atoms:

```python
water.gamma_orbitals()                  # O 2s: A1; O 2p: A1 + B1 + B2; H 1s: A1 + B2
benzene.gamma_orbitals({"C": "pz"})     # π system: A2u + B2g + E1g + E2u
```

`representation_matrix(op, basis)` displays the matrix of an operation in the same basis, and its trace.

<a id="basis-syntax"></a>
**Syntax of the bases.** The basis is described by the **shells** carried by each atom:

| Shell | Orbitals |
|---|---|
| `s` | one s orbital |
| `p` | $p_x$, $p_y$, $p_z$ |
| `d` | $d_{z^2}$, $d_{xz}$, $d_{yz}$, $d_{x^2-y^2}$, $d_{xy}$ |
| `f` | $f_{z^3}$, $f_{xz^2}$, $f_{yz^2}$, $f_{z(x^2-y^2)}$, $f_{xyz}$, $f_{x(x^2-3y^2)}$, $f_{y(3x^2-y^2)}$ (real cubic set) |
| `px`, `py`, `pz` | a single p orbital, e.g. the p orbital perpendicular to a planar molecule (π basis) |
| `pxy`, `pxz`, `pyz` | two p orbitals, e.g. the p orbitals lying in the molecular plane (σ basis); `"px py"` is the same as `"pxy"` |

The shells are given:

| Form | Meaning | Example |
|---|---|---|
| `"valence"` (default of `gamma_orbitals`) | valence shells of each element: s for H and He; s, p for the main group elements; s, p, d for the d block; s, p, d, f for the f block | `water.gamma_orbitals()` |
| a string | the same shells on all the atoms; the shells can be written together or separated by spaces | `"s"`, `"sp"`, `"spd"`, `"pz"`, `"s pxy"`, `"s px py"` |
| a list | the same, as a list | `["s", "px", "py"]` |
| a dict | the shells of each element; an element that is absent (or `""`) carries no orbital | `{"C": "sp", "H": "s"}`, `{"Fe": "spd", "C": "pz"}` |

<div class="rqE">

**Partial p shells.** A set of p orbitals spans a representation only if no symmetry operation mixes it with the other p orbitals. In a planar molecule, the p orbital perpendicular to the molecular plane is always fine: `"pz"` for benzene (plane $xy$ in the standard orientation of D<sub>6h</sub>), `"px"` for water (plane $yz$ in C<sub>2v</sub>). Use `orientation()` to know the molecular plane. On the contrary, $p_x$ alone is mixed with $p_y$ by the $C_6$ rotation of benzene: `{"C": "s px"}` raises an error, and the in-plane orbitals must be given together, `"pxy"` or `"px py"`. In groups that do not mix them (e.g. C<sub>2v</sub>), `"px"` and `"py"` are kept separate: `water.gamma_orbitals(["s", "px", "py"])` gives one row for each.

</div>

<a id="molecular-orbitals"></a>
### 5.3 Molecular orbitals

`orbital_symmetry(C, atoms, shell)` gives the irreducible representation of a molecular orbital, and `orbital_symmetries(coeffs, energies, atoms, shell)` labels all the MOs of a set, the degenerate MOs being grouped.

The MOs must be expanded on **one atomic orbital per atom, all of the same type** (`shell` = `"s"`, `"px"`, `"py"` or `"pz"`). They may come from any calculation: Hückel π MOs on the pz orbitals, combinations of the 1s orbitals of hydrogen atoms, or the corresponding coefficients of an ab initio / DFT calculation. MOs that mix several orbitals on the same atom (s, p, d...) are not handled.

| Argument | Format |
|---|---|
| `C` (`orbital_symmetry`) | one MO: 1D array of length $n$; a set of degenerate MOs: 2D array of shape $(n, m)$, one column per MO |
| `coeffs` (`orbital_symmetries`) | 2D array of shape $(n, N_\mathrm{MO})$, one column per MO (e.g. `HMO.c_MOs`) |
| `energies` | one energy per column of `coeffs` (e.g. the Hückel $x$ values `HMO.E_MOs`), not necessarily sorted; MOs whose energies differ by less than `tol` are grouped as degenerate |
| `atoms` | indices (in the molecule) of the $n$ atoms carrying the basis orbitals, **in the order of the coefficients**; all the symmetry-equivalent atoms must be included |
| `shell` | `"s"`, `"px"`, `"py"` or `"pz"`; for a π system, the p orbital perpendicular to the molecular plane (see `orientation()`) |
| `tol=1e-4` | degeneracy threshold, and tolerance on the symmetry of numerical coefficients |

```python
from pyphyschemtools.huckel import HMO
h = HMO("c1ccccc1")
benzene.orbital_symmetries(h.c_MOs, h.E_MOs, atoms=[0, 1, 2, 3, 4, 5], shell="pz")
#   A2u (x = 2), E1g (x = 1), E2u (x = -1), B2g (x = -2)
```

A set of degenerate MOs must be given as a whole: a single component of a degenerate pair does not span a representation, and `orbital_symmetry` raises an error.

<a id="vibrations"></a>
### 5.4 Vibrations

`gamma_3N()` and `gamma_vib()` (non-linear molecules) give the representation of the 3N cartesian displacements and of the 3N − 6 vibrational modes, followed by their IR / Raman activity:

$$\Gamma_\mathrm{vib} = \Gamma_{3N} - \Gamma_\mathrm{trans} - \Gamma_\mathrm{rot}, \qquad \chi_\mathrm{trans}(R) = \mathrm{tr}\,R, \qquad \chi_\mathrm{rot}(R) = \det R\ \mathrm{tr}\,R$$

`gamma_bonds(bonds)` gives the representation spanned by bond stretchings (a bond sent onto itself contributes 1), e.g. the number of ν(CO) bands of a metal carbonyl: `feco5.gamma_bonds(["C-O"])` → $2A_1' + E' + A_2''$, 2 IR active bands. The bonds are detected from the distances (covalent radii), or given as element pairs (`["C-H"]`) or as atom indices (`[(0, 1), (0, 2)]`).

`activity(chi)`: an irreducible representation is IR active if it transforms as $x$, $y$ or $z$, Raman active if it transforms as a quadratic function.

```python
water.gamma_vib()        # 2A1 + B2; IR active: 2A1 + B2; Raman active: 2A1 + B2
ch4.gamma_vib()          # A1 + E + 2T2; IR active: 2T2; Raman active: A1 + E + 2T2
```

---

<a id="visualization"></a>
## 6. Visualization

* `show(labels=False, axes=False)`: the molecule alone, in its standard orientation
* `show_elements(op=None, labels=False, axes=True)`: molecule and symmetry elements, for one class (`"C3"`), several classes (`["S4", "C2"]`) or all the elements (proper axes, planes, inversion center)
* `show_operation(op)`: molecule before (left) and after (right) the operation, each atom keeping its label; the two views are linked

Colors: $C_2$ red, $C_3$ green, $C_4$ blue, $C_5$ orange, $C_6$ purple, $C_8$ brown; $S_n$ axes dashed; planes in translucent blue; inversion center in black; cartesian axes $x$, $y$, $z$ in grey.

---

<a id="conventions"></a>
## 7. Conventions

Standard orientation (Mulliken conventions):

| Group | Orientation |
|---|---|
| general | principal axis along $z$ |
| C<sub>2v</sub> | $C_2$ along $z$, molecular plane $yz$ |
| D<sub>2h</sub> | $C_2$ axes along $x$, $y$, $z$; $z$ through the largest number of atoms; $x$ perpendicular to the molecular plane |
| D<sub>nh</sub>, D<sub>nd</sub>, C<sub>nv</sub> | one $C_2'$ axis (or $\sigma_v$ plane) along $x$; $C_2'$ and $\sigma_v$ through the largest number of atoms |
| D<sub>2d</sub> | $S_4$ along $z$, $C_2'$ along $x$ and $y$, $\sigma_d$ planes on the diagonals |
| T, T<sub>h</sub>, T<sub>d</sub> | $C_2$ ($S_4$) axes along $x$, $y$, $z$, $C_3$ along $(1,1,1)$ |
| O, O<sub>h</sub> | $C_4$ axes along $x$, $y$, $z$ |
| I, I<sub>h</sub> | $C_2$ axes along $x$, $y$, $z$, $C_5$ along $(0, 1, \varphi)$ |

<div class="rqE">

These conventions decide the labels of some irreducible representations ($B_1$/$B_2$ in C<sub>2v</sub>, $B_{1u}$/$B_{2u}$/$B_{3u}$ in D<sub>2h</sub>, $B_{1u}$/$B_{2u}$ in D<sub>6h</sub>...). The textbooks do not all follow the same convention: check the orientation with `orientation()` before comparing with a table.

</div>

---

<a id="limitations"></a>
## 8. Limitations

* linear molecules (C<sub>∞v</sub>, D<sub>∞h</sub>) are not handled: use the subgroups C<sub>2v</sub> and D<sub>2h</sub>
* `gamma_vib()` assumes a non-linear molecule (3N − 6 modes)
* pymatgen is needed for the point group detection and for the standard orientation of a molecule that is not in a standard frame
* PubChem provides no 3D structure for some compounds (hypervalent compounds such as SF<sub>6</sub>, organometallic compounds): idealized models are provided in `data_examples/Symmetry/` (SF<sub>6</sub>, Fe(CO)<sub>5</sub>, ferrocene eclipsed and staggered, cerocene, ethylene, benzene)

<a id="api"></a>
## API reference

```{eval-rst}
.. automodule:: pyphyschemtools.symmetry
   :members:
   :undoc-members:
   :show-inheritance:
```
