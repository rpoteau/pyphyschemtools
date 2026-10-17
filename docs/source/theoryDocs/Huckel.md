# Hückel Molecular Orbital method

**`huckel.py` module**

It contains a single class, `HMO`, implementing the Simple Hückel Molecular Orbital (HMO) method for planar, fully conjugated organic pi-systems.

Given a molecule's SMILES string, `HMO` uses RDKit to parse the structure, identify its conjugated pi-system, build the Hückel secular matrix (with Streitwieser or Van Catledge parametrization for heteroatoms, and optional automatic treatment of methyl/alkyl hyperconjugation), and diagonalize it to obtain molecular orbital energies, coefficients, and derived properties (pi charges, bond orders, free valences).

```python
from pyphyschemtools import HMO

benzene = HMO("c1ccccc1")
benzene.summary()
benzene.plot_energy_diagram()
```

Beyond the core eigenvalue problem, `HMO` offers a complete toolkit for visualization and analysis: 2D structure display with pi-system highlighting, energy-level diagrams (including a frontier-orbital view), individual or batch molecular-orbital plots on the molecular graph, pi-electron population/charge maps, and export of results to spreadsheets.

## Theory

### σ/π separability

A planar conjugated molecule is treated in the LCAO-MO framework: each molecular orbital (MO) is written as a linear combination of atomic orbitals (AOs), normally the valence AOs. For a planar system, the AOs split cleanly into two sets with respect to the molecular plane: the in-plane (sp, sp², sp³) AOs building the σ skeleton, symmetric with respect to that plane, and the out-of-plane 2p AOs (one per conjugated atom, perpendicular to the plane) building the π system, antisymmetric with respect to it. Because the two sets transform differently under reflection through the molecular plane, the σ and π problems do not mix: they can be solved **separately**. Simple Hückel theory only tackles the π part; the σ framework is assumed fixed and is not treated explicitly.

For ethylene, the π AO basis is just the two 2p orbitals perpendicular to the C=C axis, $\{2p_{y_A}, 2p_{y_B}\}$, combining into a bonding $\pi$ and an antibonding $\pi^*$ MO. For benzene, it is the six out-of-plane 2p orbitals, one per ring carbon, $\{2p_1, \dots, 2p_6\}$: delocalization is simply the π electrons hopping from one such p AO to its neighbors around the ring.

### The two Hückel parameters

Solving the π problem means solving the one-electron Schrödinger-like equation $\hat{h}_\pi \varphi_\pi = \varepsilon_\pi \varphi_\pi$. Rather than evaluate the Hamiltonian matrix elements from first principles, simple Hückel theory introduces two **empirical, phenomenological parameters** per conjugated system:

- the **Coulomb integral** $\alpha_r = \langle 2p_r | \hat{h} | 2p_r \rangle$, the diagonal matrix element of the effective one-electron Hamiltonian for the 2p AO of atom $r$: in this model it directly *is* that atom's orbital energy;
- the **resonance (or hopping) integral** $\beta_{rs} = \langle 2p_r | \hat{h} | 2p_s \rangle$: the energy associated with an electron delocalizing between two neighboring 2p AOs $r$ and $s$. It is what lets a single secular matrix describe delocalization at all, rather than independent, localized p electrons on each atom.

On top of this, Hückel theory makes two simplifying approximations that turn the general secular problem $\mathbf{Hc} = E\mathbf{Sc}$ into an ordinary eigenvalue problem:

1. **Zero differential overlap**: the AO overlap matrix $\mathbf{S}$ is replaced by the identity, $\langle 2p_r | 2p_s \rangle = \delta_{rs}$, even though real, neighboring p orbitals do overlap physically;
2. **Nearest-neighbor, topological β**: $\beta_{rs}$ is taken to be a single constant $\beta$ (identical for every formal pi bond) when atoms $r$ and $s$ are directly bonded, and exactly zero otherwise, regardless of the actual through-space distance or geometry. The resonance matrix is therefore purely **topological**: it only encodes the molecule's connectivity (its sigma-bond graph), not its 3D shape.

### Secular matrix and the benzene example

With these two approximations, the π Hamiltonian matrix elements for an $N$-atom π system reduce to

$$
H_{rr} = \alpha_r \qquad
H_{rs} = \begin{cases} \beta & \text{if } r, s \text{ directly pi-bonded} \\ 0 & \text{otherwise} \end{cases}
$$

($\alpha_r = \alpha$ for a plain carbon, shifted for a heteroatom; see below), and the secular equation $\mathbf{Hc} = E\mathbf{c}$ ($\mathbf{S} = \mathbf{I}$) becomes an ordinary matrix eigenvalue problem. For benzene's six 2p AOs this is, explicitly,

$$
\mathbf{H} =
\begin{pmatrix}
\alpha & \beta & 0 & 0 & 0 & \beta \\
\beta & \alpha & \beta & 0 & 0 & 0 \\
0 & \beta & \alpha & \beta & 0 & 0 \\
0 & 0 & \beta & \alpha & \beta & 0 \\
0 & 0 & 0 & \beta & \alpha & \beta \\
\beta & 0 & 0 & 0 & \beta & \alpha
\end{pmatrix}
$$

the connectivity/adjacency matrix of the ring, dressed with $\alpha$ on the diagonal and $\beta$ on each nearest-neighbor off-diagonal entry. Diagonalizing it directly yields the familiar six π MOs and the textbook $x = 2, 1, 1, -1, -1, -2$ energy pattern (with $E = \alpha + x\beta$, $\beta < 0$).

In practice, the whole problem is made dimensionless by substituting $x = (E - \alpha)/\beta$, so that diagonalizing $\mathbf{H}$ amounts to diagonalizing the purely topological matrix $\mathbf{M}$ ($\mathbf{H} = \alpha\mathbf{I} + \beta\mathbf{M}$, so $\mathbf{M}$ is exactly $\mathbf{H}$ above with $\alpha \to 0$ and $\beta \to 1$): this is exactly what `HMO.huckel_matrix()` returns, and what the module reports every energy as, in units of $|\beta|$ relative to $\alpha$.

### Pi population and bond orders

Each occupied MO $i$, with coefficients $c_{ri}$ on atom $r$ and occupation number $n_i$ (0, 1 or 2), contributes $n_i c_{ri}^2$ electrons to atom $r$. Summing over all occupied MOs gives the **π electron density** on atom $r$, $q_r = \sum_i n_i c_{ri}^2$, and comparing it to the number of π electrons that atom formally contributed to the system gives its net **π charge**. The same coefficients, taken pairwise, give the **π bond order** between two bonded atoms $r$ and $s$, $p_{rs} = \sum_i n_i c_{ri} c_{si}$, a measure of π bonding character on that bond (close to 1 for a full double bond, close to 0 for a purely single one). Both are standard Hückel population-analysis quantities, and both are available directly from `HMO` (`pi_charge`, `bond_order`, `density_matrix()`, `plot_pi_population()`).

### Delocalization and conjugation energies

Two complementary measures of pi delocalization are available directly from `HMO`.

`atomization_energy()` (and the stored `.E_atomization`) compares the total pi energy $E_\pi$ to a reference where every pi electron stays localized on its own atom, non-interacting, at that atom's own Coulomb level $\alpha_r$:

$$
E_{at} = E_\pi - \sum_r n_r \alpha_r
$$

where $n_r$ is the number of pi electrons atom $r$ contributes. For an all-carbon system ($h=0$ everywhere) this reference is zero, so $E_{at} = E_\pi$; the gap between the two widens as soon as a heteroatom ($h \neq 0$) is involved.

`conjugation_energy()` (and `.E_conjugation`) uses a different, more specific reference: a localized Kekulé structure, obtained from RDKit's own Kekulization of the input SMILES. Each formal pi double bond of that structure is treated as its own isolated, localized 2-center bond: diagonalizing just that $2\times 2$ block,

$$
\begin{pmatrix} \alpha_r & \beta_{rs} \\ \beta_{rs} & \alpha_s \end{pmatrix},
$$

and occupying the bonding level with 2 electrons (this reduces to the familiar $\alpha+\beta$ level for a plain C=C, and generalizes correctly to a heteroatom-involving double bond such as pyridine's C=N, whose two atoms have different $\alpha$). Every atom left unpaired by that Kekulé structure (a lone pair localized on a single center, e.g. pyrrole-type N, furan-type O, a conjugated halogen) keeps the same isolated-atom reference used in `atomization_energy()`:

$$
E_{conj} = E_\pi - \left[ \sum_{\text{double bonds}} 2(\alpha + x_{loc}\beta) + \sum_{\text{unpaired } r} n_r \alpha_r \right]
$$

This is the classic textbook "resonance energy" ($2|\beta|$ for benzene). It is a more specific, usually smaller quantity than the atomization energy, since it isolates the stabilization gained by delocalizing beyond simple, already-bonded, localized double bonds, rather than relative to fully isolated atoms. Cyclobutadiene is the instructive edge case: its localized Kekulé structure (two isolated C=C double bonds) is already as stable as the delocalized, antiaromatic ring, so its conjugation energy comes out to (near) zero. For a non-alternant or unsymmetrical system admitting several valid Kekulé structures, the result follows whichever one RDKit happens to produce, a known limitation of the classical concept itself, not of this implementation.

### From the toy model to `HMO`

Classic Hückel theory, as sketched above, only has one kind of atom (carbon) and one kind of bond. `HMO` extends this to heteroatom-containing systems the standard way: a heteroatom $X$ keeps the same secular-matrix machinery, but its Coulomb integral is shifted relative to carbon's, $\alpha_X = \alpha_C + h_X \beta_{CC}$, and a C-X bond's resonance integral is scaled, $\beta_{CX} = k_{CX} \beta_{CC}$, with the dimensionless $h_X$/$k_{CX}$ pulled from tabulated parameter sets chosen from each heteroatom's local bonding pattern (pyridine- vs. pyrrole-type N, ether vs. carbonyl O...). Everything downstream (diagonalization, Aufbau/Hund filling, energies, coefficients, charges, bond orders) follows exactly the same recipe as plain benzene above.

Two parameter sets are bundled with the module:

- **Streitwieser**: A. Streitwieser, *Molecular Orbital Theory for Organic Chemists*, Wiley, 1961.
- **Van Catledge**: F. A. Van-Catledge, *A Pariser-Parr-Pople-based set of Hückel molecular orbital parameters*, *J. Org. Chem.* **1980**, *45* (24), 4801-4802.

## API reference

```{eval-rst}
.. automodule:: pyphyschemtools.huckel
   :members:
   :undoc-members:
   :show-inheritance:
```
