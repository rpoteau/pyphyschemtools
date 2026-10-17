"""
huckel.py
---------
Simple Hückel (HMO) method applied to molecules described by a SMILES
string, with Streitwieser parametrization for heteroatoms.

Overview
========
1. Start from a complete SMILES (sigma skeleton + pi system).
2. RDKit identifies the atoms belonging to the conjugated pi system
   (aromatic atoms, plus atoms linked by non-aromatic conjugated bonds:
   dienes, enones, etc.).
3. Each pi atom is given a Coulomb parameter h_X and each pi bond a
   resonance parameter k_C-X (Streitwieser), relative to the reference
   values alpha_C and beta_CC::

        alpha_X = alpha_C + h_X * beta_CC
        beta_CX = k_CX   * beta_CC

   Automatic detection (degree, number of H atoms, aromaticity) picks
   the heteroatom type (pyridine-like/pyrrole-like N, ether/furan or
   carbonyl O, etc.); the caller can always override these values with
   `h_overrides` / `k_overrides` / `electron_overrides`.
4. The Hückel matrix is diagonalized, the MOs are filled by the Aufbau
   principle (+ Hund's rule for degenerate levels), and a number of
   diagrams / descriptors can be derived from the result: the energy
   diagram, MO coefficients drawn on the real 2D geometry of the
   molecule, frontier-orbital (HOMO/LUMO) figures, pi charges and pi
   bond orders, the density matrix, and reactivity
   descriptors (HOMO/LUMO energies, hardness, electronegativity...).

Limitations: pure pi-electron model, no explicit self-consistency
(no extended Hückel, no charge-dependent VSIP); formal charge only
enters through its effect on the number of pi electrons contributed by
each atom.
"""

from __future__ import annotations

import warnings

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D

try:
    import pandas as pd
except ImportError:  # pandas is an optional, soft dependency
    pd = None

try:
    import sympy
except ImportError:  # sympy is an optional, soft dependency
    sympy = None

try:
    from IPython.display import display
except ImportError:  # outside Jupyter/IPython: fall back to plain printing
    def display(obj):
        print(obj)

__all__ = [
    "HMO", "STREITWIESER_TABLE", "VAN_CATLEDGE_TABLE",
    "parameter_table", "hmo_properties_table", "save_fig", "save_data",
]


# ---------------------------------------------------------------------------
# Figure/data saving. If huckel.py lives inside the pyphyschemtools package
# next to a `core` module providing `save_fig`/`save_data`, those are reused
# as-is (single source of truth, same behavior everywhere in the package).
# Otherwise (e.g. huckel.py used standalone) a local, functionally identical
# fallback is used instead, so this module never hard-depends on `.core`.
# ---------------------------------------------------------------------------
try:
    from .core import save_fig, save_data  # type: ignore
except ImportError:
    def save_fig(path_with_name: str, fig=None, dpi: int = 300, **kwargs):
        """Save a figure to a path, automatically creating missing
        directories. Works with the current plt figure or a specific
        figure object."""
        from pathlib import Path
        full_path = Path(path_with_name)
        if full_path.parent:
            full_path.parent.mkdir(parents=True, exist_ok=True)
        target = fig if fig is not None else plt.gcf()
        target.savefig(full_path, dpi=dpi, bbox_inches="tight", **kwargs)
        print(f"Figure saved to: {full_path}")

    def save_data(df, path_with_name: str, **kwargs):
        """Save a pandas DataFrame to CSV or Excel, automatically
        creating missing directories."""
        from pathlib import Path
        full_path = Path(path_with_name)
        if full_path.parent:
            full_path.parent.mkdir(parents=True, exist_ok=True)
        ext = full_path.suffix.lower()
        index_option = kwargs.pop("index", False)
        if ext == ".csv":
            df.to_csv(full_path, index=index_option, **kwargs)
        elif ext in (".xlsx", ".xls"):
            df.to_excel(full_path, index=index_option, **kwargs)
        else:
            raise ValueError(f"Unsupported file extension: {ext}. Use .csv, .xlsx, or .xls")
        print(f"Data saved to: {full_path}")


# ---------------------------------------------------------------------------
# Streitwieser parameter table (h_X, k_C-X), standard textbook values
# (Streitwieser, "Molecular Orbital Theory for Organic Chemists", 1961;
# common pedagogical variants). h and k are dimensionless.
# Key: (symbol, "kind") -> (h, k, n_pi_electrons_contributed)
# ---------------------------------------------------------------------------
STREITWIESER_TABLE = {
    ("C", "sp2"):      (0.0, 1.0, 1),   # ordinary pi carbon
    ("C", "anion"):    (0.0, 1.0, 2),   # carbanion: lone pair in the p orbital (e.g. cyclopentadienyl)
    ("C", "cation"):   (0.0, 1.0, 0),   # carbocation: empty p orbital (e.g. tropylium)
    ("N", "pyridine"):    (0.5, 1.0, 1),   # =N-, lone pair in-plane (not in the pi system)
    ("N", "pyrrole"):     (1.5, 0.8, 2),   # -NH- (or -NR-), lone pair in the p orbital
    ("N", "pyridinium"):  (2.0, 1.0, 1),   # aromatic N+ (e.g. N-alkylpyridinium): same pi
                                            # bonding pattern as neutral pyridine-type N (one
                                            # pi electron, h raised by the + charge). h_N+=2
                                            # is Streitwieser's own recommended value (1961,
                                            # Table 5.3, p. 135); k kept at the aromatic k_CN=1
                                            # (same table), as for neutral pyridine.
    ("O", "furan"):    (2.0, 0.8, 2),   # -O- ether/furan type
    ("O", "carbonyl"): (1.0, 1.0, 1),   # =O ketone/aldehyde type
    ("O", "oxocarbenium"): (2.5, 1.0, 1),  # >C=O+- (oxocarbenium), one pi electron.
                                            # h_O+=2.5 is Streitwieser's own recommended
                                            # value (1961, Table 5.3, p. 135).
    ("S", "thiophene"): (1.5, 0.6, 2),  # -S- thioether/thiophene type
    ("F", "halogen"):  (3.0, 0.7, 2),
    ("Cl", "halogen"): (2.0, 0.4, 2),
    ("Br", "halogen"): (1.5, 0.3, 2),   # h_Br=1.5, k_C-Br=0.3: Streitwieser's own
                                         # recommended values (1961, Table 5.3, p. 135).
    # "Me" / exocyclic methyl group, modelled as a 2-pi-electron donor
    # pseudo-heteroatom (Matsen's and Stevenson's "heteroatom model" of
    # hyperconjugation, see Streitwieser 1961 Sec. 5.7 p. 132-133): the
    # C-H bonding electrons of the methyl group are treated, to a crude
    # approximation, as a single lone pair sitting on one pseudo-atom,
    # exactly like -O- or -NR-. CONFIRMED in Streitwieser (1961), Table
    # 5.3 (p. 135, "Methyl, heteroatom model = C-X"): h_X=2, k_C-X=0.7.
    ("C", "methyl_donor"): (2.0, 0.7, 2),
}


# ---------------------------------------------------------------------------
# Van Catledge (1980) parameter table: derived from Pariser-Parr-Pople (PPP)
# calculations, with a wider and more internally-consistent heteroatom
# coverage than the classic Streitwieser set -- including boron, silicon,
# phosphorus, and (unlike Streitwieser, which only tabulates k_C-X) a full
# two-center k_XY matrix between any two heteroatom types. It also carries
# a reference "free valence" Fx per atom type, a classic HMO radical-
# reactivity index (see `HMO.free_valence`).
#
# Reference: F. A. Van-Catledge, "A Pariser-Parr-Pople-Based Set of Hückel
# Molecular Orbital Parameters", J. Org. Chem. 1980, 45 (24), 4801-4802.
# (Table I: one-center h_X, Fx; Table II: two-center k_XY.)
#
# Key: (symbol, "kind") -> (h, k_to_carbon, n_pi_electrons_contributed, Fx)
# ---------------------------------------------------------------------------
VAN_CATLEDGE_TABLE = {
    ("C", "sp2"):           (0.00, 1.00, 1, 1.732),
    # Carbanion/carbocation centers (e.g. cyclopentadienyl anion,
    # tropylium cation): same trivial h=0, k=1 as a neutral sp2 carbon,
    # only the pi-electron count changes (2 for a filled lone pair, 0
    # for an empty p orbital). Not a PPP-derived Van-Catledge (1980)
    # value -- their paper does not cover these -- but carried over
    # here, unchanged, so both parameter sets handle the same simple
    # Hückel textbook cases. No Fx (same as "sp2").
    ("C", "anion"):         (0.00, 1.00, 2, 1.732),
    ("C", "cation"):        (0.00, 1.00, 0, 1.732),
    ("B", "vacant"):        (-0.45, 0.73, 0, 1.705),
    ("N", "pyridine"):      (0.51, 1.02, 1, 1.393),
    ("N", "pyrrole"):       (1.37, 0.89, 2, 1.583),
    ("O", "carbonyl"):      (0.97, 1.06, 1, 0.909),
    ("O", "furan"):         (2.09, 0.66, 2, 0.942),
    ("F", "halogen"):       (2.71, 0.52, 2, 0.179),
    ("Si", "sp2"):          (0.00, 0.75, 1, 1.732),
    ("P", "phosphaalkene"): (0.19, 0.77, 1, 1.409),
    ("P", "phosphole"):     (0.75, 0.76, 2, 1.666),
    ("S", "thioketone"):    (0.46, 0.81, 1, 0.962),
    ("S", "thiophene"):     (1.11, 0.69, 2, 1.229),
    ("Cl", "halogen"):      (1.48, 0.62, 2, 0.321),
    # Br is not part of Van-Catledge (1980) itself -- their PPP paper
    # simply does not cover it -- so, exactly like the carbanion/
    # carbocation and "Me" entries above/below, Streitwieser's own
    # recommended h/k are carried over here UNCHANGED (1961, Table 5.3,
    # p. 135) as the closest available approximation, rather than
    # guessing a PPP-style rescaling with no actual calculation behind
    # it. No Fx (free valence undefined for it, same as those entries).
    ("Br", "halogen"):      (1.5, 0.3, 2),
    # "Me" (methyl/alkyl pseudo-heteroatom, see STREITWIESER_TABLE) is not
    # part of Van-Catledge (1980) itself -- there is no PPP-derived value
    # for it -- but it is carried over here with the SAME h/k as the
    # Streitwieser entry, exactly as in the user's own earlier HMO tool
    # (which reused Streitwieser's h=2.00, k=0.70 unchanged for its
    # "van Catledge" table too). No Fx (free valence undefined for it).
    ("C", "methyl_donor"):  (2.00, 0.70, 2),
}

# Van Catledge's own one-letter/number kind codes (their Table II column
# headers), used only to look up the two-center k_XY matrix below.
_VAN_CATLEDGE_CODE = {
    ("C", "sp2"): "C", ("C", "anion"): "C", ("C", "cation"): "C",
    ("B", "vacant"): "B",
    ("N", "pyridine"): "N1", ("N", "pyrrole"): "N2",
    ("O", "carbonyl"): "O1", ("O", "furan"): "O2",
    ("F", "halogen"): "F", ("Si", "sp2"): "Si",
    ("P", "phosphaalkene"): "P1", ("P", "phosphole"): "P2",
    ("S", "thioketone"): "S1", ("S", "thiophene"): "S2",
    ("Cl", "halogen"): "Cl",
    ("Br", "halogen"): "Br",
    ("C", "methyl_donor"): "Me",
}

# Table II of Van-Catledge (1980): k_XY = beta_XY / beta_CC. Symmetric
# half-matrix as printed (upper triangle); `_van_catledge_kxy` mirrors it.
# "Me" (not part of the original Table II -- see VAN_CATLEDGE_TABLE) only
# ever bonds to an ordinary carbon in practice (a methyl/alkyl substituent
# is always attached directly to the conjugated system, never to another
# heteroatom), so only k_C-Me=0.70 is given; the symmetric lookup in
# `_van_catledge_kxy` falls back to it from either side. "Br" is the same
# situation (not part of the original table, see VAN_CATLEDGE_TABLE): only
# k_C-Br=0.3 (Streitwieser's own value, carried over unchanged) is given,
# since a halogen substituent is likewise always attached directly to a
# carbon of the conjugated system in practice.
_VAN_CATLEDGE_KXY = {
    "C":  {"C": 1.00, "B": 0.73, "N1": 1.02, "N2": 0.89, "O1": 1.06, "O2": 0.66, "F": 0.52, "Si": 0.75, "P1": 0.77, "P2": 0.76, "S1": 0.81, "S2": 0.69, "Cl": 0.62, "Br": 0.3, "Me": 0.70},
    "B":  {"B": 0.87, "N1": 0.66, "N2": 0.53, "O1": 0.60, "O2": 0.35, "F": 0.26, "Si": 0.57, "P1": 0.53, "P2": 0.54, "S1": 0.51, "S2": 0.44, "Cl": 0.41},
    "N1": {"N1": 1.09, "N2": 0.99, "O1": 1.14, "O2": 0.80, "F": 0.65, "Si": 0.72, "P1": 0.78, "P2": 0.81, "S1": 0.83, "S2": 0.78, "Cl": 0.77},
    "N2": {"N2": 0.98, "O1": 1.13, "O2": 0.89, "F": 0.77, "Si": 0.43, "P1": 0.55, "P2": 0.64, "S1": 0.68, "S2": 0.73, "Cl": 0.80},
    "O1": {"O1": 1.26, "O2": 1.02, "F": 0.92, "Si": 0.65, "P1": 0.75, "P2": 0.82, "S1": 0.84, "S2": 0.85, "Cl": 0.88},
    "O2": {"O2": 0.95, "F": 0.94, "Si": 0.24, "P1": 0.31, "P2": 0.39, "S1": 0.43, "S2": 0.54, "Cl": 0.70},
    "F":  {"F": 1.04, "Si": 0.17, "P1": 0.21, "P2": 0.22, "S1": 0.28, "S2": 0.32, "Cl": 0.51},
    "Si": {"Si": 0.64, "P1": 0.62, "P2": 0.52, "S1": 0.61, "S2": 0.40, "Cl": 0.34},
    "P1": {"P1": 0.63, "P2": 0.58, "S1": 0.65, "S2": 0.48, "Cl": 0.35},
    "P2": {"P2": 0.63, "S1": 0.65, "S2": 0.60, "Cl": 0.55},
    "S1": {"S1": 0.68, "S2": 0.65, "Cl": 0.52},
    "S2": {"S2": 0.63, "Cl": 0.59},
    "Cl": {"Cl": 0.68},
}


def _van_catledge_kxy(class_i: tuple[str, str], class_j: tuple[str, str]) -> float:
    """Symmetric lookup of k_XY in the Van Catledge two-center table for
    two atom classes (each a key of VAN_CATLEDGE_TABLE)."""
    ci, cj = _VAN_CATLEDGE_CODE[class_i], _VAN_CATLEDGE_CODE[class_j]
    row = _VAN_CATLEDGE_KXY.get(ci, {})
    if cj in row:
        return row[cj]
    return _VAN_CATLEDGE_KXY[cj][ci]


def parameter_table(parameter_set: str = "van_catledge", as_dataframe: bool = True):
    """The full one-center Hückel parameter table for `parameter_set`
    ("streitwieser" or "van_catledge"): one row per (symbol, kind) with
    its h, k_to_carbon and n_pi_electrons, plus Fx (reference free
    valence) for "van_catledge" (Streitwieser does not define one).

    Returns a pandas DataFrame if as_dataframe=True and pandas is
    available, otherwise a list of dicts.

    References
    ----------
    streitwieser: A. Streitwieser, Jr., "Molecular Orbital Theory for
        Organic Chemists", John Wiley & Sons, 1961, p. 135.
    van_catledge: F. A. Van-Catledge, "A Pariser-Parr-Pople-Based Set of
        Hückel Molecular Orbital Parameters", J. Org. Chem. 1980, 45
        (24), 4801-4802.
    """
    table = _parameter_table_for(parameter_set)
    rows = []
    for (sym, kind), vals in table.items():
        row = {"symbol": sym, "kind": kind, "h": vals[0], "k_to_C": vals[1], "n_pi": vals[2]}
        if len(vals) > 3:
            row["Fx"] = vals[3]
        rows.append(row)
    if as_dataframe and pd is not None:
        return pd.DataFrame(rows)
    return rows


def _parameter_table_for(parameter_set: str):
    if parameter_set == "streitwieser":
        return STREITWIESER_TABLE
    if parameter_set == "van_catledge":
        return VAN_CATLEDGE_TABLE
    raise ValueError(
        f"Unknown parameter_set {parameter_set!r}: use 'streitwieser' or "
        "'van_catledge'."
    )


def _classify_heteroatom(atom: "Chem.Atom", parameter_set: str = "van_catledge") -> tuple[str, str]:
    """Guess the (symbol, kind) key into the active parameter table from
    the local RDKit structure. This is a heuristic: check/override it
    for ambiguous cases via the HMO overrides."""
    sym = atom.GetSymbol()
    charge = atom.GetFormalCharge()
    if sym == "C":
        if charge < 0:
            return ("C", "anion")
        if charge > 0:
            return ("C", "cation")
        return ("C", "sp2")

    degree = atom.GetTotalDegree()  # heavy-atom bonds + H

    if sym == "N":
        if charge > 0 and degree == 3:
            # Aromatic N+ (e.g. N-alkylpyridinium): same pi-bonding
            # pattern as the neutral pyridine-type N (one pi electron),
            # just with a higher h due to the formal + charge.
            return ("N", "pyridinium")
        if degree == 3:
            return ("N", "pyrrole")
        return ("N", "pyridine")

    if sym == "O":
        if charge > 0:
            return ("O", "oxocarbenium")
        # Oxygen with a double bond (carbonyl): total degree 1 (=O).
        if degree == 1:
            return ("O", "carbonyl")
        return ("O", "furan")

    if sym == "S":
        if parameter_set == "van_catledge":
            return ("S", "thioketone") if degree == 1 else ("S", "thiophene")
        if degree == 1:
            return ("O", "carbonyl")  # thiocarbonyl, same electron count
        return ("S", "thiophene")

    if sym in ("F", "Cl", "Br"):
        return (sym, "halogen")

    if sym == "B" and parameter_set == "van_catledge":
        return ("B", "vacant")

    if sym == "Si" and parameter_set == "van_catledge":
        return ("Si", "sp2")

    if sym == "P" and parameter_set == "van_catledge":
        return ("P", "phosphole") if degree == 3 else ("P", "phosphaalkene")

    raise ValueError(
        f"No default {parameter_set} parameters for element {sym} "
        f"(atom idx {atom.GetIdx()}). Provide h_overrides / k_overrides / "
        f"electron_overrides for this atom"
        + (", or parameter_set='van_catledge' (covers B, Si, P)."
           if parameter_set == "streitwieser" and sym in ("B", "Si", "P") else ".")
    )


class HMO:
    """Hückel pi system built from a SMILES string.

    Parameters
    ----------
    smiles : str
        SMILES of the complete molecule (sigma skeleton + pi system).
    h_overrides, k_overrides : dict[int, float], optional
        Force h_X for the given RDKit atom index, or k_C-X for the pi
        bond(s) touching that atom (key = index of one of the two atoms
        of the bond; use a (i, j) tuple key to target one specific bond).
    electron_overrides : dict[int, int], optional
        Force the number of pi electrons contributed by the given atom
        index.
    charge : int, optional
        Defaults to 0. A global formal charge added to the WHOLE pi
        system, on top of whatever individual atoms already contribute
        -- the easy way to study a system-wide reduction/oxidation
        (e.g. the benzene dication, `HMO("c1ccccc1", charge=2)`, or
        the naphthalene dianion, `HMO("c1ccc2ccccc2c1", charge=-2)`)
        WITHOUT having to spell out a carbocation/carbanion center
        anywhere in the SMILES (which -- see `_detect_pi_system` --
        also ties that extra/missing electron to one particular atom,
        rather than letting it delocalize over the whole system the
        way a real, symmetric di-ion does). Ordinary chemistry sign
        convention: a positive `charge` REMOVES that many pi electrons
        (cation), a negative one ADDS that many (anion); `pi_charge`'s
        per-atom values still sum to exactly this `charge` however it
        ends up delocalized.
    parameter_set : str, optional
        "van_catledge" (default) or "streitwieser" -- which one-/two-
        center parameter table to use for heteroatoms (see
        `parameter_table`). "van_catledge" additionally covers B, Si and
        P, defines a full heteroatom-heteroatom k_XY matrix (Streitwieser
        only tabulates k_C-X), and enables `free_valence`.
    include_hyperconjugation : bool, optional
        Defaults to True. A methyl or alkyl group attached directly to
        the conjugated system (e.g. toluene's -CH3, or a benzylic -CH2-
        of indane/tetralin) is included as one extra pseudo-heteroatom
        contributing a lone pair (Matsen's and Stevenson's "heteroatom
        model" of hyperconjugation, (sym, kind)=("C", "methyl_donor")
        -- see STREITWIESER_TABLE / VAN_CATLEDGE_TABLE), the whole
        substituent collapsed to its FIRST attached carbon only (its own
        further substitution, e.g. the rest of an ethyl or isopropyl
        group, is not modeled any further). Only a substituent with
        exactly ONE attachment point to the conjugated system qualifies
        -- a -CH2- bridging two separate pi systems is left alone, as a
        genuine break in conjugation, not a substituent. The literature
        value (h=2.0, k=0.7) is strictly only derived for a true methyl;
        for any larger alkyl it is reused as the closest available
        approximation, and a warning is raised naming the atom (see
        `alkyl_scope` to restrict this to true methyls only, or pass
        h_overrides/k_overrides/electron_overrides for that atom index
        to supply a better value by hand). Set to False to restore the
        previous behavior (these substituents are plain, inert sigma
        groups, outside the pi system entirely).
    alkyl_scope : str, optional
        "all" (default) or "methyl_only" -- with `include_hyperconjugation`,
        whether a larger alkyl substituent (-CH2R, -CHR2, -CR3) is
        included too (reusing the methyl parameter, with a warning) or
        only a true methyl (-CH3) is. Ignored if
        include_hyperconjugation=False.
    include_halogen_lone_pairs : bool, optional
        Defaults to True. A halogen (F, Cl or Br) attached directly to
        the conjugated system (e.g. fluorobenzene's -F, vinyl
        chloride's -Cl) is included in the pi system as one extra
        2-electron lone-pair donor, parametrized exactly like any other
        heteroatom (see STREITWIESER_TABLE / VAN_CATLEDGE_TABLE,
        (symbol, "halogen")) -- the standard Hückel textbook treatment
        of aryl/vinyl halides. This is genuinely RDKit's own blind spot
        (its `GetIsConjugated()` is also False for an aryl/vinyl-
        halogen bond, the same class of issue as the ethylene/allyl
        cases documented in `_detect_pi_system`), yet whether to model
        a halogen's lone pair at all is a modelling CHOICE rather than
        an unambiguous correction -- some Hückel treatments leave
        halogens out entirely. A `UserWarning` is raised for every
        halogen folded in this way, naming the atom. Set to False to
        restore the previous behavior (halogens left out, as plain,
        inert sigma substituents, outside the pi system entirely).
    """

    def __init__(
        self,
        smiles: str,
        h_overrides: dict[int, float] | None = None,
        k_overrides: dict[int, float] | None = None,
        electron_overrides: dict[int, int] | None = None,
        charge: int = 0,
        parameter_set: str = "van_catledge",
        include_hyperconjugation: bool = True,
        alkyl_scope: str = "all",
        include_halogen_lone_pairs: bool = True,
    ):
        self.smiles = smiles
        self.parameter_set = parameter_set
        self._table = _parameter_table_for(parameter_set)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles!r}")
        self.mol = mol
        AllChem.Compute2DCoords(self.mol)

        self._h_over = h_overrides or {}
        self._k_over = k_overrides or {}
        self._e_over = electron_overrides or {}

        core_pi = self._detect_pi_system()
        if len(core_pi) < 2:
            raise ValueError(
                "Fewer than 2 pi atoms detected: nothing to diagonalize. "
                "Check the SMILES (a conjugated system is expected)."
            )
        core_pi_set = set(core_pi)

        self._hyperconjugating_idx = set()
        if include_hyperconjugation:
            self._hyperconjugating_idx = self._detect_hyperconjugating_substituents(
                core_pi_set, alkyl_scope=alkyl_scope,
            )
        self._halogen_idx = set()
        if include_halogen_lone_pairs:
            self._halogen_idx = self._detect_halogen_substituents(core_pi_set)
        self.pi_atom_idx = sorted(core_pi_set | self._hyperconjugating_idx | self._halogen_idx)

        n = len(self.pi_atom_idx)
        self._idx_map = {ridx: k for k, ridx in enumerate(self.pi_atom_idx)}

        self.h = np.zeros(n)
        self.labels = []
        self.classes = []
        self.contributed_pi_electrons = np.zeros(n)
        self._fx_ref = np.full(n, np.nan)  # Van Catledge reference free valence
        electrons = 0
        for ridx in self.pi_atom_idx:
            atom = self.mol.GetAtomWithIdx(ridx)
            if ridx in self._hyperconjugating_idx:
                sym, kind = ("C", "methyl_donor")
            else:
                sym, kind = _classify_heteroatom(atom, parameter_set)
            if (sym, kind) not in self._table:
                raise ValueError(
                    f"({sym}, {kind}) has no entry in the {parameter_set!r} "
                    f"parameter table (atom idx {atom.GetIdx()}). This "
                    "combination may only be defined for the other "
                    "parameter_set, or needs h_overrides/k_overrides/"
                    "electron_overrides here."
                )
            vals = self._table[(sym, kind)]
            h_default, _, e_default = vals[:3]
            k_loc = self._idx_map[ridx]
            self.h[k_loc] = self._h_over.get(ridx, h_default)
            n_e = self._e_over.get(ridx, e_default)
            self.contributed_pi_electrons[k_loc] = n_e
            if len(vals) > 3:
                self._fx_ref[k_loc] = vals[3]
            electrons += n_e
            label_prefix = "Me" if (sym, kind) == ("C", "methyl_donor") else atom.GetSymbol()
            self.labels.append(f"{label_prefix}{ridx}")
            self.classes.append((sym, kind))
        self.charge = charge
        self.n_pi_electrons = electrons - charge

        # Reference used specifically for per-atom pi CHARGE analysis
        # (see `pi_charge`) -- deliberately NOT the same array as
        # `contributed_pi_electrons`. The latter must keep the true
        # electron count each atom actually brings into the system (0
        # for a carbocation center, 2 for a carbanion center), since
        # `n_pi_electrons`/`atomization_energy` depend on it; but
        # comparing a charge-bearing carbon's electron DENSITY against
        # its OWN already-ionized count makes that same center look
        # like it is carrying extra charge on top of its delocalized
        # share of the system's real formal charge (its reference
        # silently "pre-pays" for the + or - once, and delocalization
        # then distributes that same unit of charge all over again --
        # e.g. tropylium's nominally cationic carbon coming out
        # strongly NEGATIVE). The correct reference for a ("C","anion")
        # /("C","cation") center is the same as any OTHER carbon (1 pi
        # electron, same as "C","sp2") -- comparing every position
        # against a plain NEUTRAL carbon is what correctly spreads the
        # molecule's one formal charge unit over every position (e.g.
        # tropylium: +1/7 on each of the 7 carbons by symmetry, not
        # -6/7 on the nominally "+" one and +1/7 elsewhere).
        self._neutral_ref_electrons = np.array([
            1.0 if cls in (("C", "anion"), ("C", "cation")) else e
            for cls, e in zip(self.classes, self.contributed_pi_electrons)
        ])

        # Matrix M such that H = alpha_C * I + beta_CC * M. No extra check
        # of the bond's own conjugated/aromatic/double flags is needed
        # here: every atom in `self._idx_map` (self.pi_atom_idx) already
        # got there for a reason that makes it part of the pi network
        # (the core system from `_detect_pi_system`, or a hyperconjugating
        # substituent with exactly one bond INTO that core), so a direct
        # bond between two such atoms is always a pi bond, whatever
        # RDKit's own per-bond flags say about it in isolation (a plain
        # isolated double bond -- ethylene, a lone C=O -- is a case in
        # point: RDKit reports it as "not conjugated", by its own
        # definition of that term, yet it is still the textbook two-atom
        # pi system of simple Hückel theory).
        M = np.diag(self.h)
        for bond in self.mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if i in self._idx_map and j in self._idx_map:
                ki, kj = self._idx_map[i], self._idx_map[j]
                sym_i, _ = self.classes[ki]
                sym_j, _ = self.classes[kj]
                if parameter_set == "van_catledge":
                    k_default = _van_catledge_kxy(self.classes[ki], self.classes[kj])
                elif self.classes[ki] == ("C", "methyl_donor") or self.classes[kj] == ("C", "methyl_donor"):
                    # Both atoms carry plain symbol "C" (methyl/alkyl is
                    # keyed under ("C", "methyl_donor")), so the generic
                    # "both carbon -> k=1.0" rule below must NOT apply
                    # here: use the pseudo-heteroatom's own k_to_C=0.7.
                    methyl_cls = self.classes[ki] if self.classes[ki][1] == "methyl_donor" else self.classes[kj]
                    k_default = self._table[methyl_cls][1]
                else:
                    k_default = 1.0 if (sym_i == "C" and sym_j == "C") else self._default_k(
                        self.classes[ki], self.classes[kj]
                    )
                k_val = self._k_over.get((i, j), self._k_over.get((j, i)))
                if k_val is None:
                    k_val = self._k_over.get(i, self._k_over.get(j, k_default))
                M[ki, kj] = M[kj, ki] = k_val
        self.M = M
        self.graph_pi = self._build_networkx(M)

        # Diagonalization: eigh returns eigenvalues in ascending order.
        w, v = np.linalg.eigh(M)
        order = np.argsort(w)[::-1]           # descending x <-> ascending energy (beta < 0)
        self.E_MOs = w[order]                  # coefficient x in E = alpha + x*beta (dimensionless, units of |beta|)
        self.c_MOs = v[:, order]               # columns = MOs, same atom order as self.labels
        self.occupation = self._fill_electrons()

        # Energetics, stored as plain attributes (dimensionless "units of
        # |beta|" convention: alpha=0, beta=-1) so they can be reused
        # directly -- e.g. collected into a DataFrame across molecules --
        # without having to call total_pi_energy()/atomization_energy()
        # again. Call those methods directly for other alpha/beta values.
        self.E_total = self.total_pi_energy()
        self.E_atomization = self.atomization_energy()
        self.E_conjugation = self.conjugation_energy()

    # -- construction ----------------------------------------------------
    @staticmethod
    def _default_k(class_i, class_j):
        """Default k_C-X when the bond does not involve two carbons."""
        for sym, kind in (class_i, class_j):
            if sym != "C":
                _, k, _ = STREITWIESER_TABLE[(sym, kind)]
                return k
        return 1.0

    def _detect_pi_system(self) -> list[int]:
        """Aromatic atoms + sp2 atoms linked by a non-aromatic conjugated
        bond (polyenes, enones...) OR a plain, isolated double bond
        (ethylene, a lone C=O/C=N not next to anything else) -- RDKit's
        `GetIsConjugated()` is False for the latter (by definition it
        requires a NEIGHBORING multiple bond or lone pair to be
        "conjugated" with), yet a single, isolated double bond is still
        the textbook two-atom pi system of simple Hückel theory, so it
        must not be dropped.

        A carbon carrying a formal charge or an unpaired (radical)
        electron, directly bonded to one of the atoms above, extends
        this SAME pi system rather than being a separate hyperconjugating
        substituent -- e.g. the allyl radical's terminal >CH2 in
        "C=C[CH2]", or an allylic/benzylic cation or anion written the
        same explicit way. This is genuinely RDKit's own blind spot, not
        a limitation of its conjugation detector being applied
        incompletely: it perceives that lone =CH2 radical carbon as sp3
        and the bond to it as not conjugated, because nothing in a plain
        SMILES like this tells it the radical sits in a p orbital rather
        than an sp3 lone valence -- unlike a lone pair spelled out
        aromatically (pyrrole-type N, "[cH-]1cccc1"...), which RDKit
        already recognizes and marks aromatic on its own.

        Restricted to the largest connected conjugated component."""
        candidates = set()
        for atom in self.mol.GetAtoms():
            if atom.GetIsAromatic():
                candidates.add(atom.GetIdx())
        for bond in self.mol.GetBonds():
            if bond.GetIsAromatic():
                continue
            if bond.GetIsConjugated() or bond.GetBondType() == Chem.BondType.DOUBLE:
                candidates.add(bond.GetBeginAtomIdx())
                candidates.add(bond.GetEndAtomIdx())

        for atom in self.mol.GetAtoms():
            ridx = atom.GetIdx()
            if ridx in candidates or atom.GetSymbol() != "C":
                continue
            if atom.GetFormalCharge() == 0 and atom.GetNumRadicalElectrons() == 0:
                continue
            if any(nb.GetIdx() in candidates for nb in atom.GetNeighbors()):
                candidates.add(ridx)

        if not candidates:
            return []

        # Keep only the largest connected component of the pi subgraph.
        # Any direct bond between two already-accepted candidates counts
        # here (not just an aromatic/conjugated/double one): each atom
        # only entered `candidates` for a reason that already makes it
        # part of the pi network, so a bond directly linking two of them
        # is a pi bond too, whatever RDKit's own bond-level flags say
        # about it in isolation (this is also what lets the allylic
        # radical/charge extension above actually join its neighbor's
        # component instead of forming its own, singleton one).
        g = nx.Graph()
        g.add_nodes_from(candidates)
        for bond in self.mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if i in candidates and j in candidates:
                g.add_edge(i, j)
        largest = max(nx.connected_components(g), key=len)
        return sorted(largest)

    def _detect_hyperconjugating_substituents(self, core_pi_set, alkyl_scope="all"):
        """Methyl/alkyl substituents attached directly to `core_pi_set`
        (see `include_hyperconjugation`/`alkyl_scope` on __init__):
        an sp3 carbon, not itself in `core_pi_set`, with EXACTLY ONE
        heavy-atom neighbour that is -- so a -CH2- bridging two separate
        pi systems (two such neighbours) is correctly left out, as a
        genuine break in conjugation rather than a substituent.

        Returns the set of RDKit atom indices to add to the pi system
        (each to be classified as ("C", "methyl_donor")); emits a
        `UserWarning` for every substituent that is not a true, terminal
        methyl (3 H's, no further heavy-atom substitution) when
        alkyl_scope="all", since the literature h/k values are strictly
        only derived for methyl itself."""
        found = set()
        for atom in self.mol.GetAtoms():
            ridx = atom.GetIdx()
            if ridx in core_pi_set or atom.GetSymbol() != "C":
                continue
            if atom.GetHybridization() != Chem.HybridizationType.SP3:
                continue
            heavy_neighbors = [nb for nb in atom.GetNeighbors()]
            pi_neighbors = [nb for nb in heavy_neighbors if nb.GetIdx() in core_pi_set]
            if len(pi_neighbors) != 1:
                continue  # 0: unrelated: >=2: a bridge between two pi systems, not a substituent
            bond = self.mol.GetBondBetweenAtoms(ridx, pi_neighbors[0].GetIdx())
            if bond.GetIsAromatic() or bond.GetIsConjugated():
                continue  # already part of the core system via the normal route
            is_true_methyl = atom.GetTotalNumHs() == 3 and len(heavy_neighbors) == 1
            if not is_true_methyl:
                if alkyl_scope == "methyl_only":
                    continue
                warnings.warn(
                    f"Atom idx {ridx} ({atom.GetSymbol()}, {atom.GetTotalNumHs()} H, "
                    f"{len(heavy_neighbors)} heavy-atom neighbour(s)) is a larger alkyl "
                    "substituent, not a plain methyl -- it is being parametrized with "
                    "the methyl ('Me') pseudo-heteroatom values (h=2.0, k=0.7), the "
                    "closest literature-backed approximation available, since no "
                    "dedicated value exists for larger alkyls. Pass h_overrides/"
                    "k_overrides/electron_overrides for this atom index if you have a "
                    "better value, or alkyl_scope='methyl_only' to exclude it instead "
                    "(and treat it as an inert sigma substituent, outside the pi "
                    "system, as before)."
                )
            found.add(ridx)
        return found

    def _detect_halogen_substituents(self, core_pi_set):
        """Halogen atoms (F, Cl, Br) bonded directly to `core_pi_set`
        (an aryl- or vinyl-halogen, e.g. fluorobenzene, vinyl chloride)
        -- see `include_halogen_lone_pairs` on __init__ for why these
        default to being folded INTO the pi system, as a 2-electron
        lone-pair donor parametrized exactly like any other heteroatom
        (STREITWIESER_TABLE / VAN_CATLEDGE_TABLE, (symbol, "halogen")),
        rather than being silently ignored as a plain, inert sigma
        substituent.

        This is the SAME class of RDKit blind spot documented in
        `_detect_pi_system`'s own docstring (ethylene, the allyl
        radical): `bond.GetIsConjugated()` is False for an aryl/vinyl-
        halogen bond too (verified empirically against fluorobenzene,
        chlorobenzene, bromobenzene -- RDKit never marks that bond
        "conjugated"), yet a halogen's in-plane lone pair conjugating
        into an adjacent pi system is standard, textbook Hückel
        treatment.

        Returns the set of RDKit atom indices to add to the pi system
        (each then classified normally by `_classify_heteroatom`, which
        already maps F/Cl/Br to (sym, "halogen")); emits a
        `UserWarning` for every atom folded in this way, since -- unlike
        the ethylene/allyl fixes, which corrected a case the chosen
        parametrization always intends to cover -- modelling a
        halogen's lone pair at all is a genuine CHOICE: some Hückel
        treatments leave halogens out of the pi system entirely."""
        found = set()
        for atom in self.mol.GetAtoms():
            ridx = atom.GetIdx()
            if ridx in core_pi_set or atom.GetSymbol() not in ("F", "Cl", "Br"):
                continue
            pi_neighbors = [nb for nb in atom.GetNeighbors() if nb.GetIdx() in core_pi_set]
            if len(pi_neighbors) != 1:
                continue  # not directly attached to the conjugated system
            warnings.warn(
                f"Atom idx {ridx} ({atom.GetSymbol()}) is a halogen bonded "
                "directly to the conjugated pi system -- its in-plane lone "
                "pair is being folded into the pi system as a 2-electron "
                f"donor, using the {self.parameter_set!r} parameter set's "
                "'halogen' h/k values. This follows the standard Hückel "
                "treatment of aryl/vinyl halides, but is a modelling "
                "choice, not every textbook treatment includes it -- pass "
                "include_halogen_lone_pairs=False to exclude halogens "
                "instead (as plain, inert sigma substituents, outside the "
                "pi system entirely)."
            )
            found.add(ridx)
        return found

    def _build_networkx(self, M) -> nx.Graph:
        g = nx.Graph()
        for k, ridx in enumerate(self.pi_atom_idx):
            g.add_node(k, label=self.labels[k])
        n = len(self.pi_atom_idx)
        for a in range(n):
            for b in range(a + 1, n):
                if M[a, b] != 0:
                    g.add_edge(a, b, k=M[a, b])
        return g

    def _fill_electrons(self) -> np.ndarray:
        """Aufbau filling, with Hund's rule applied within degenerate
        levels (numerical tolerance on energy equality)."""
        n = len(self.E_MOs)
        occ = np.zeros(n)
        remaining = self.n_pi_electrons
        i = 0
        tol = 1e-6
        while i < n and remaining > 0:
            j = i
            while j + 1 < n and abs(self.E_MOs[j + 1] - self.E_MOs[i]) < tol:
                j += 1
            degeneracy = j - i + 1
            capacity = 2 * degeneracy
            put = min(remaining, capacity)
            # Hund: first 1 electron per degenerate orbital, then pair up.
            per_orbital = [0] * degeneracy
            singles = min(put, degeneracy)
            for k in range(singles):
                per_orbital[k] += 1
            left = put - singles
            for k in range(left):
                per_orbital[k] += 1
            for k, val in enumerate(per_orbital):
                occ[i + k] = val
            remaining -= put
            i = j + 1
        if remaining > 0:
            raise ValueError(
                "Not enough pi MOs to host all the pi electrons "
                f"({self.n_pi_electrons} e- for {n} orbitals)."
            )
        return occ

    def _group_levels(self, tol=1e-6):
        """Group MO indices whose x values are numerically equal
        (degeneracy), using an absolute tolerance."""
        groups = []
        n = len(self.E_MOs)
        i = 0
        while i < n:
            j = i
            while j + 1 < n and abs(self.E_MOs[j + 1] - self.E_MOs[i]) < tol:
                j += 1
            groups.append(list(range(i, j + 1)))
            i = j + 1
        return groups

    def _frontier_indices(self):
        """Return (homo_index, lumo_index): a single representative MO
        index for the HOMO and the LUMO (the highest-index member of the
        HOMO group / lowest-index member of the LUMO group). Use
        `_frontier_groups` instead when a degenerate HOMO or LUMO must be
        handled as a whole (several MOs sharing the same x)."""
        homo_group, lumo_group = self._frontier_groups()
        homo = homo_group[-1] if homo_group else None
        lumo = lumo_group[0] if lumo_group else None
        return homo, lumo

    def _frontier_groups(self, tol=1e-6):
        """Return (homo_group, lumo_group): the full list of MO indices
        in the highest-energy occupied group (HOMO, or SOMO group for a
        radical) and in the lowest-energy virtual group (LUMO),
        respecting degeneracy -- e.g. both members of a degenerate HOMO
        pair are returned, not just one representative."""
        groups = self._group_levels(tol=tol)
        occ_groups = [g for g in groups if any(self.occupation[k] > 0 for k in g)]
        unocc_groups = [g for g in groups if all(self.occupation[k] == 0 for k in g)]
        homo_group = occ_groups[-1] if occ_groups else []
        lumo_group = unocc_groups[0] if unocc_groups else []
        return homo_group, lumo_group

    @staticmethod
    def _clean_near_zero(x, tol=1e-9):
        """Snap floating-point noise around zero (e.g. -4.5e-17 coming
        out of eigh on a symmetric system) to a clean 0.0 for display."""
        return 0.0 if abs(x) < tol else float(x)

    @staticmethod
    def _display_df(df):
        """Display a DataFrame (or, for `as_dataframe=False`, whatever
        plain object was built instead) for interactive inspection,
        with two Jupyter-friendly tweaks that only apply to THIS
        on-screen rendering -- `save_results()` always writes the
        full, unrounded numeric values to Excel/ODS, regardless of
        what is set here:

        - No column truncation ("..." folding the middle columns away,
          as pandas does past its default `display.max_columns`): a
          larger molecule's Hückel/density matrix (one column per pi
          atom) is shown in full.
        - Plain fixed-point numbers rounded to 3 decimals, never
          pandas' default repr -- which, for very small values (e.g.
          -5.07e-16 numerical noise off a large diagonalization),
          falls back to scientific notation whose exponent can wrap
          onto its own line in a narrow cell, exactly the kind of
          unreadable output this is meant to avoid.

        Non-numeric content (e.g. the sympy/Greek-letter entries of
        `huckel_matrix(symbolic=True)`, or the "role"/string columns
        of `mo_table`/`used_parameters`) is simply left as pandas
        would otherwise render it -- `display.float_format` only ever
        touches float-dtype columns."""
        if pd is not None and isinstance(df, pd.DataFrame):
            with pd.option_context(
                "display.max_columns", None,
                "display.max_rows", None,
                "display.width", None,
                "display.float_format", lambda x: f"{x:.3f}",
            ):
                display(df)
        else:
            display(df)

    # -- Hückel population analysis --------------------------------------
    @property
    def electron_density(self) -> np.ndarray:
        """Pi electron density per atom (local index, same order as
        self.labels): q_r = sum_i n_i * c_ri^2."""
        return (self.c_MOs ** 2) @ self.occupation

    @property
    def pi_charge(self) -> np.ndarray:
        """Net formal pi charge per atom = (pi electrons a NEUTRAL atom
        of that same type would contribute) - (electron density
        actually found there). Positive means electron-poor relative
        to the neutral reference, negative means electron-rich.

        For almost every atom type this neutral reference IS
        `contributed_pi_electrons` (a neutral pyrrole-type N really
        does contribute 2, a neutral pyridine-type N really does
        contribute 1 -- the heteroatom's charge state, if any, is
        already encoded through h instead, e.g. ("N","pyridinium")
        keeps n_pi=1, same as neutral pyridine). The one exception is a
        carbon folded into the pi system as ("C","anion")/("C","cation")
        (e.g. the allyl/benzyl cation or anion, cyclopentadienyl anion,
        tropylium): `contributed_pi_electrons` is 2 or 0 there -- the
        ion's OWN electron count, needed elsewhere for `n_pi_electrons`
        -- but the neutral reference used HERE is 1, same as any other
        carbon, so the molecule's one formal charge unit is correctly
        spread over every position by the delocalization itself,
        instead of being double-counted (pre-paid once at that single
        atom's reference, then distributed again by its density). This
        is why, e.g., tropylium comes out with +1/7 on EVERY carbon
        (by symmetry), never a spuriously negative nominal "+" center.

        Sums to exactly the pi system's total formal charge (0 for a
        neutral molecule; +1 for a cation such as tropylium or the
        allyl cation; -1 for an anion such as the cyclopentadienyl
        anion) -- not always 0."""
        return self._neutral_ref_electrons - self.electron_density

    def bond_order(self, r: int, s: int) -> float:
        """Pi bond order between local pi-atom indices r and s (same
        indexing as self.labels / self.c_MOs rows)::

            p_rs = sum_i n_i * c_ri * c_si

        input:
        - r = local pi-atom index (row of self.c_MOs)
        - s = local pi-atom index (row of self.c_MOs)

        output:
        - the pi bond order p_rs between atoms r and s
        """
        return float(np.sum(self.occupation * self.c_MOs[r, :] * self.c_MOs[s, :]))

    @property
    def free_valence(self) -> np.ndarray:
        """Free valence (radical-reactivity) index per atom::

            F_r = Fx_r - sum_s p_rs   (sum over the pi-bonded neighbours s)

        where Fx_r is Van Catledge's reference maximum total bond order
        for that atom's type/hybridization (e.g. Fx=1.732=sqrt(3) for an
        ordinary sp2 carbon). A larger F_r means a more reactive site
        towards radical attack.

        Only defined for parameter_set="van_catledge" -- the Streitwieser
        table does not provide reference Fx values. Reference: Van-
        Catledge, J. Org. Chem. 1980, 45 (24), 4801-4802 (Table I).

        output:
        - free valence F_r per atom, as a numpy array (one value per pi atom, same order as self.labels)
        """
        if np.any(np.isnan(self._fx_ref)):
            raise ValueError(
                "free_valence requires parameter_set='van_catledge' "
                "(Streitwieser parameters do not define a reference Fx)."
            )
        n = len(self.labels)
        total_bond_order = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i != j and self.M[i, j] != 0:
                    total_bond_order[i] += self.bond_order(i, j)
        return self._fx_ref - total_bond_order

    def density_matrix(self, as_dataframe: bool = True):
        """First-order pi density (charge/bond-order) matrix::

            P = C . diag(n) . C^T

        where C holds the MO coefficients (columns) and n the MO
        occupations. P_rr is the electron density on atom r, P_rs
        (r != s) is the pi bond order between atoms r and s.

        Displayed immediately (works regardless of where this call
        sits in a cell -- see `show()` for why) and cached as
        `self.density_matrix_df`; returns None (read the attribute, or
        call again, for the value itself).

        input:
        - as_dataframe = if True (default), cache/display the matrix as a labeled pandas DataFrame; if False, as a plain numpy array

        output:
        - None (the matrix is displayed and cached as self.density_matrix_df)
        """
        P = self.c_MOs @ np.diag(self.occupation) @ self.c_MOs.T
        if as_dataframe and pd is not None:
            result = pd.DataFrame(P, index=self.labels, columns=self.labels)
        else:
            result = P
        self.density_matrix_df = result
        self._display_df(result)
        return None

    def huckel_matrix(self, symbolic: bool = False, as_dataframe: bool = True):
        """The Hückel matrix itself, such that H = alpha*I + beta*M.

        symbolic=False (default): returns the numeric, dimensionless
        matrix M (diagonal = Coulomb parameters h_r, off-diagonal =
        resonance parameters k_rs for bonded pi atoms, 0 otherwise). For
        a simple all-carbon system (all h_r=0, all k_rs=1) this is just
        0/1 -- e.g. ethylene C=C gives [[0, 1], [1, 0]].

        symbolic=True: returns the same matrix with literal Greek-letter
        alpha/beta entries instead of numbers -- a diagonal entry is
        alpha when h_r=0, else alpha + h*beta with the numeric h_r
        substituted in; an off-diagonal bonded entry is beta when
        k_rs=1, else k*beta with the numeric k_rs substituted in;
        unbonded pairs are 0. Uses real sympy symbols (nicer, typeset
        rendering in Jupyter) if sympy is installed, otherwise plain
        Greek-letter strings ("α", "α + 0.5·β", ...).

        Displayed immediately (works regardless of where this call sits
        in a cell -- see `show()` for why) and cached as
        `self.huckel_matrix_df` -- a labeled pandas DataFrame
        (rows/cols = self.labels) if as_dataframe=True and pandas is
        available, otherwise a raw numpy array (symbolic=False) or a
        list of lists (symbolic=True). Returns None (read the
        attribute, or call again, for the value itself).
        """
        if not symbolic:
            if as_dataframe and pd is not None:
                result = pd.DataFrame(self.M, index=self.labels, columns=self.labels)
            else:
                result = self.M
            self.huckel_matrix_df = result
            self._display_df(result)
            return None

        n = len(self.labels)

        if sympy is not None:
            alpha, beta = sympy.symbols("α β", real=True)  # alpha, beta

            def diag_entry(h):
                h = self._clean_near_zero(h)
                return alpha if h == 0.0 else alpha + h * beta

            def offdiag_entry(k):
                if k == 0.0:
                    return sympy.Integer(0)
                return beta if k == 1.0 else k * beta
        else:
            # Plain-string fallback (no sympy installed): same literal
            # Greek-letter rendering, built as formatted text instead.
            def diag_entry(h):
                h = self._clean_near_zero(h)
                if h == 0.0:
                    return "α"  # alpha
                sign = "+" if h > 0 else "-"
                return f"α {sign} {abs(h):g}·β"  # alpha +/- h*beta

            def offdiag_entry(k):
                if k == 0.0:
                    return "0"
                if k == 1.0:
                    return "β"  # beta
                return f"{k:g}·β"

        rows = []
        for i in range(n):
            row = []
            for j in range(n):
                val = self.M[i, j]
                row.append(diag_entry(val) if i == j else offdiag_entry(val))
            rows.append(row)

        if as_dataframe and pd is not None:
            result = pd.DataFrame(rows, index=self.labels, columns=self.labels)
        else:
            result = rows
        self.huckel_matrix_df = result
        self._display_df(result)
        return None

    def used_parameters(self, as_dataframe: bool = True):
        """The parameter-table rows actually used for the atom classes
        present in THIS molecule -- a convenient subset of
        `parameter_table(self.parameter_set)` to sanity-check what was
        applied (one row per distinct (symbol, kind), in the order first
        encountered). Each row also repeats `self.parameter_set` (so the
        table is self-identifying -- "streitwieser" or "van_catledge"
        today, whatever other parameter sets may be added later -- even
        when read out of context, e.g. printed on its own or saved to a
        file).

        Displayed immediately (works regardless of where this call
        sits in a cell -- see `show()` for why) and cached as
        `self.used_parameters_df` -- a pandas DataFrame if
        as_dataframe=True and pandas is available, otherwise a list of
        dicts. Returns None (read the attribute, or call again, for
        the value itself)."""
        seen = []
        rows = []
        for cls in self.classes:
            if cls in seen:
                continue
            seen.append(cls)
            sym, kind = cls
            vals = self._table[cls]
            row = {"parameter_set": self.parameter_set, "symbol": sym,
                   "kind": kind, "h": vals[0], "k_to_C": vals[1], "n_pi": vals[2]}
            if len(vals) > 3:
                row["Fx"] = vals[3]
            rows.append(row)
        if as_dataframe and pd is not None:
            result = pd.DataFrame(rows)
        else:
            result = rows
        self.used_parameters_df = result
        self._display_df(result)
        return None

    # -- energetics --------------------------------------------------------
    def total_pi_energy(self, alpha: float = 0.0, beta: float = 1.0) -> float:
        """Total Hückel pi-electron energy::

            E_pi = sum_i n_i * (alpha + x_i * beta)

        where n_i is the occupation of MO i and x_i its coefficient
        (self.E_MOs). Defaults to the dimensionless "units of \|beta\|"
        convention used when quoting these energies in textbooks
        (e.g. "E_pi(benzene) = 6*alpha + 8*beta"): alpha=0, beta=+1 --
        the *magnitude* of beta, reported as a plain coefficient, not
        beta's true physical sign. Since the lowest-energy (most
        bonding) MOs are filled first and have the largest x, this
        dimensionless total is positive for any normal bonded pi
        system. Pass the real, physically negative beta (e.g. in eV)
        to get the true, physical (and then negative) energy.

        input:
        - alpha = Coulomb reference energy (default: 0.0)
        - beta  = resonance reference energy; magnitude convention by default -- pass the real, physically negative value for a true physical energy (default: 1.0)

        output:
        - total pi-electron energy E_pi, in the chosen alpha/beta units
        """
        total_n = float(np.sum(self.occupation))
        return total_n * alpha + beta * float(np.sum(self.occupation * self.E_MOs))

    def atomization_energy(self, alpha: float = 0.0, beta: float = 1.0) -> float:
        """Hückel atomization (delocalization) energy::

            Eat = E_pi(Hückel) - sum_r (alpha_r * contributed_r)

        i.e. the total pi-electron energy of the real, delocalized
        molecule minus the reference energy of the same pi electrons
        sitting, non-interacting, on their own atoms (each atom r
        contributing `contributed_r` electrons at its own Coulomb level
        alpha_r = alpha + h_r * beta).

        Same dimensionless "units of \|beta\|" convention as
        `total_pi_energy` by default (alpha=0, beta=+1, i.e. beta's
        magnitude): a positive Eat means the delocalized pi system is
        more stable than the isolated-atom reference (the usual case:
        conjugation/aromatic stabilization). Pass the real, physically
        negative beta (e.g. in eV) for a true physical energy -- the
        sign then flips, since a system that is MORE stable has a
        LOWER (more negative) energy once beta's true sign is used.

        input:
        - alpha = Coulomb reference energy (default: 0.0)
        - beta  = resonance reference energy; magnitude convention by default -- pass the real, physically negative value for a true physical energy (default: 1.0)

        output:
        - atomization energy Eat, in the chosen alpha/beta units
        """
        e_pi = self.total_pi_energy(alpha=alpha, beta=beta)
        total_contributed = float(np.sum(self.contributed_pi_electrons))
        reference = total_contributed * alpha + beta * float(
            np.sum(self.contributed_pi_electrons * self.h)
        )
        return e_pi - reference

    def _kekule_pairing(self):
        """Pair up the pi atoms into formal (localized) double bonds,
        from a Kekulized copy of `self.mol` -- the classic "localized
        structure" reference used by `conjugation_energy` (isolated
        double bonds, e.g. ethylene-like, rather than isolated atoms).

        Returns (pairs, unpaired): `pairs` is a list of (ki, kj) LOCAL
        indices (into self.pi_atom_idx/self.h/self.M) for each formal
        pi double bond; `unpaired` is the set of local indices left
        without a double-bond partner in that same Kekule structure
        (a lone pair localized on a single atom: pyrrole-type N,
        furan-type O, a conjugated halogen, a hyperconjugating methyl/
        alkyl pseudo-atom, or a formally charged/radical center such as
        an allyl cation's empty p orbital).

        RDKit already stores a specific Kekule structure internally for
        any aromatic input (`Chem.Kekulize` only makes it explicit), so
        no guessing is needed for benzenoid systems; for a molecule with
        several inequivalent Kekule structures (most non-alternant or
        unsymmetrical cases), RDKit's own choice is used, exactly like
        picking one particular Lewis structure by hand -- a known
        limitation of the "resonance energy" concept itself, not of
        this implementation.
        """
        mol_kek = Chem.Mol(self.mol)
        try:
            Chem.Kekulize(mol_kek, clearAromaticFlags=True)
        except Chem.KekulizeException as exc:
            raise ValueError(
                "Could not find a Kekule (localized single/double bond) "
                "structure for this molecule, so `conjugation_energy` has "
                "no localized reference to compare against."
            ) from exc

        pairs = []
        paired = set()
        for bond in mol_kek.GetBonds():
            if bond.GetBondType() != Chem.BondType.DOUBLE:
                continue
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if i in self._idx_map and j in self._idx_map:
                ki, kj = self._idx_map[i], self._idx_map[j]
                pairs.append((ki, kj))
                paired.add(ki)
                paired.add(kj)
        unpaired = set(range(len(self.pi_atom_idx))) - paired
        return pairs, unpaired

    def conjugation_energy(self, alpha: float = 0.0, beta: float = 1.0) -> float:
        """Hückel conjugation (resonance/delocalization) energy, relative
        to a LOCALIZED (Kekule) reference structure rather than to fully
        isolated atoms::

            Econj = E_pi(Hückel) - [ sum over formal pi double bonds of
                                        2*(localized bonding level)
                                      + sum over unpaired atoms of
                                        (contributed_r * alpha_r) ]

        This is the classic textbook quantity (e.g. "benzene's resonance
        energy is 2|beta|"): each formal double bond of a Kekule
        structure (from a Kekulized copy of the input SMILES, see
        `_kekule_pairing`) is treated as its own isolated, localized
        2-center pi bond -- diagonalizing just that 2x2 block (which
        reduces to the familiar alpha+beta level for a plain C=C, but
        generalizes correctly to a heteroatom-involving double bond such
        as pyridine's C=N, whose two atoms have different alpha) -- and
        occupied with 2 electrons; an atom left unpaired by that Kekule
        structure (a lone pair localized on a single center: pyrrole-
        type N, furan-type O, a conjugated halogen, a hyperconjugating
        methyl/alkyl pseudo-atom, or a formally charged/radical center)
        keeps the same isolated-atom reference as `atomization_energy`.

        Unlike `atomization_energy` (reference: fully isolated, non-
        interacting atoms, no bonding of any kind), this isolates only
        the stabilization gained by DELOCALIZING beyond simple, already-
        bonded, localized double bonds -- the smaller, more specific
        number usually meant by "conjugation energy" or "resonance
        energy" in textbooks.

        Same dimensionless "units of \|beta\|" convention as
        `atomization_energy` by default (alpha=0, beta=+1): a positive
        Econj means the delocalized pi system is more stable than its
        own localized Kekule structure.

        input:
        - alpha = Coulomb reference energy (default: 0.0)
        - beta  = resonance reference energy; magnitude convention by default -- pass the real, physically negative value for a true physical energy (default: 1.0)

        output:
        - conjugation energy Econj, in the chosen alpha/beta units
        """
        e_pi = self.total_pi_energy(alpha=alpha, beta=beta)
        pairs, unpaired = self._kekule_pairing()

        reference = 0.0
        for ki, kj in pairs:
            h_i, h_j, m_ij = self.h[ki], self.h[kj], self.M[ki, kj]
            # Bonding (lower-energy, larger x) root of the isolated 2x2
            # block [[h_i, m_ij], [m_ij, h_j]] -- same dimensionless
            # (alpha=0, beta=1) convention as self.h/self.M themselves.
            x_loc = (h_i + h_j) / 2 + np.sqrt(((h_i - h_j) / 2) ** 2 + m_ij ** 2)
            reference += 2 * (alpha + x_loc * beta)
        for k in unpaired:
            reference += self.contributed_pi_electrons[k] * (alpha + self.h[k] * beta)

        return e_pi - reference

    # -- frontier-orbital descriptors -------------------------------------
    def properties(self, alpha: float = 0.0, beta: float = 1.0, as_series: bool = False):
        """Frontier-orbital energies and simple conceptual-DFT reactivity
        descriptors, handy to collect into a table (DataFrame) across
        several molecules.

        `alpha`/`beta` control E_HOMO, E_LUMO, gap, hardness, softness,
        chemical_potential, electronegativity and electrophilicity.
        They default to alpha=0, beta=+1: E_HOMO, E_LUMO,
        chemical_potential and electronegativity are then reported as
        their COEFFICIENT of beta (E = alpha + value*beta), the same
        convention as `mo_table` (x_beta) -- e.g. E_HOMO = 0.618 means
        alpha + 0.618*beta. gap, hardness and electrophilicity are
        intrinsically positive quantities and are always returned as
        positive values, in units of \|beta\| (softness in 1/\|beta\|).
        Pass the real, physically negative beta (e.g. in eV) to get all
        of these in physical units.

        total_pi_energy, atomization_energy and conjugation_energy are
        NOT affected by `alpha`/`beta` here -- they are always reported
        in the dimensionless "units of \|beta\|" convention (beta's
        magnitude, always positive for a normal bonded pi system; see
        `total_pi_energy`/`atomization_energy`/`conjugation_energy`, or
        `self.E_total` / `self.E_atomization` / `self.E_conjugation`).
        Call those methods directly, with your own alpha/beta, for a
        physical-unit total/atomization/conjugation energy.

        input:
        - alpha     = Coulomb reference energy (default: 0.0)
        - beta      = resonance reference energy; dimensionless magnitude convention by default -- pass the real, physically negative value (e.g. in eV) for physical units (default: 1.0)
        - as_series = if True, return a pandas Series instead of a dict (requires pandas); default: False

        output:
        - dict (or pandas Series if as_series=True) with: smiles, n_pi_atoms, n_pi_electrons, homo_index, lumo_index, E_HOMO, E_LUMO, gap, hardness, softness, chemical_potential, electronegativity, electrophilicity, total_pi_energy, atomization_energy, conjugation_energy
        """
        homo, lumo = self._frontier_indices()
        if homo is None or lumo is None:
            raise ValueError(
                "Cannot compute frontier descriptors: the pi system has "
                "no occupied MO or no virtual MO."
            )
        x_homo, x_lumo = float(self.E_MOs[homo]), float(self.E_MOs[lumo])
        e_homo = alpha + x_homo * beta
        e_lumo = alpha + x_lumo * beta
        gap = abs(e_lumo - e_homo)
        hardness = gap / 2.0
        softness = (1.0 / (2.0 * hardness)) if hardness != 0 else float("inf")
        mu = (e_homo + e_lumo) / 2.0          # chemical potential
        chi = -mu                              # Mulliken electronegativity
        omega = (mu ** 2) / (2.0 * hardness) if hardness != 0 else float("inf")

        data = {
            "smiles": self.smiles,
            "n_pi_atoms": len(self.pi_atom_idx),
            "n_pi_electrons": self.n_pi_electrons,
            "homo_index": homo,
            "lumo_index": lumo,
            "E_HOMO": e_homo,
            "E_LUMO": e_lumo,
            "gap": gap,
            "hardness": hardness,
            "softness": softness,
            "chemical_potential": mu,
            "electronegativity": chi,
            "electrophilicity": omega,
            "total_pi_energy": self.E_total,
            "atomization_energy": self.E_atomization,
            "conjugation_energy": self.E_conjugation,
        }
        if as_series and pd is not None:
            return pd.Series(data)
        return data

    # -- results / MO listing ---------------------------------------------
    def mo_table(self, as_dataframe: bool = True, tol: float = 1e-6,
                 with_coefficients: bool = True, coeff_tol: float = 1e-9):
        """Table of all pi MOs (index, x in units of beta, occupation,
        role, and -- by default -- the MO coefficient on every pi atom),
        ordered from the least stable (top) to the most stable (bottom)
        MO -- same order as the energy diagram. A degenerate HOMO or
        LUMO (several MOs sharing the same x) is fully reflected in the
        `role` column, not just a single representative MO. Values of x
        within `tol` of zero are cleaned up to exactly 0.0 (floating-
        point noise from the diagonalization), and likewise for the
        coefficients within `coeff_tol`.

        `with_coefficients=True` (default) adds one column per pi atom
        (named after `self.labels`, e.g. "C0", "N3", ...) holding c_ri,
        the coefficient of that atom in MO i -- set it to False for a
        compact table with just index/energy/occupation/role (this is
        what `summary()` uses, to keep the printed report short for
        large systems).

        Displayed immediately (works regardless of where this call
        sits in a cell -- see `show()` for why) and cached as
        `self.mo_table_df` -- a pandas DataFrame if as_dataframe=True
        and pandas is available, otherwise a list of dicts. Returns
        None (read the attribute, or call again, for the value
        itself)."""
        rows = self._build_mo_table(tol=tol, with_coefficients=with_coefficients,
                                     coeff_tol=coeff_tol)
        if as_dataframe and pd is not None:
            result = pd.DataFrame(rows)
        else:
            result = rows
        self.mo_table_df = result
        self._display_df(result)
        return None

    def _build_mo_table(self, tol: float = 1e-6, with_coefficients: bool = True,
                         coeff_tol: float = 1e-9):
        """Row-building logic shared by `mo_table` (public, displays and
        caches) and anything else that just needs the raw rows (list of
        dicts) without triggering a display -- `summary()` and
        `save_results()`."""
        homo_group, lumo_group = self._frontier_groups(tol=tol)
        rows = []
        for k in reversed(range(len(self.E_MOs))):
            if k in homo_group:
                role = "SOMO" if self.occupation[k] < 2 else "HOMO"
            elif k in lumo_group:
                role = "LUMO"
            elif self.occupation[k] > 0:
                role = "occupied"
            else:
                role = "virtual"
            row = {
                "mo_index": k,
                "x_beta": self._clean_near_zero(self.E_MOs[k]),
                "occupation": float(self.occupation[k]),
                "role": role,
            }
            if with_coefficients:
                for label, c in zip(self.labels, self.c_MOs[:, k]):
                    row[label] = self._clean_near_zero(c, tol=coeff_tol)
            rows.append(row)
        return rows

    def summary(self):
        """Print a short text report: SMILES, pi atoms (with the number
        of pi electrons each one contributes), total pi electrons, total
        Hückel pi-energy, atomization energy, conjugation (Kekule-
        referenced resonance) energy, and the full MO table (see
        `mo_table` to get it as a DataFrame instead of printed text)."""
        per_atom = ", ".join(
            f"{label}:{int(n_e)}"
            for label, n_e in zip(self.labels, self.contributed_pi_electrons)
        )
        print(f"SMILES: {self.smiles}")
        print(f"Pi atoms ({len(self.pi_atom_idx)}): {per_atom}")
        if self.charge != 0:
            print(f"Global charge (charge=): {self.charge:+d}")
        print(f"Pi electrons: {self.n_pi_electrons}")
        n_e = int(round(float(np.sum(self.occupation))))
        x_tot = self._clean_near_zero(self.E_total)
        x_at = self._clean_near_zero(self.E_atomization)
        x_conj = self._clean_near_zero(self.E_conjugation)
        sign = "+" if x_tot >= 0 else "-"
        print(f"Total pi energy: E_pi = {n_e}α {sign} {abs(x_tot):.4f}β")
        print(f"Atomization energy: E_at = {x_at:.4f}β")
        print(f"Conjugation energy: E_conj = {x_conj:.4f}β")
        print()
        rows = self._build_mo_table(with_coefficients=False)
        if pd is not None:
            print(pd.DataFrame(rows).to_string(index=False))
        else:
            print(f"{'MO':>4} {'x (beta)':>10} {'occupation':>10}  role")
            for row in rows:
                print(f"{row['mo_index']:>4} {row['x_beta']:>10.4f} "
                      f"{row['occupation']:>10.1f}  {row['role']}")

    # -- visualization -----------------------------------------------------
    @staticmethod
    def _points_to_data_dx(ax, points):
        """Horizontal distance, in data-coordinate units, equivalent to
        a FIXED physical size of `points` points (1/72 inch) for `ax`
        as it is currently laid out. Used so a visual gap (e.g. between
        two spin arrows) stays the same size on screen regardless of
        how wide the containing figure/axes happens to be -- unlike a
        plain fraction of the data range, which widens right along
        with the figure."""
        dpi = ax.figure.dpi
        inv = ax.transData.inverted()
        d0 = inv.transform(ax.transData.transform((0.0, 0.0)))
        d1 = inv.transform(ax.transData.transform((0.0, 0.0)) + (points * dpi / 72.0, 0.0))
        return abs(float(d1[0] - d0[0]))

    # Reference pixel<->inch conversion for every "N pixels" sizing
    # request below (`bond_length_px` on `show()`/`plot_pi_population()`/
    # `plot_mo_on_graph()`, the 700 px frontier-orbital figure height...).
    # matplotlib figures are sized in inches, not pixels, so SOME
    # reference scale is needed to turn a pixel target into a figsize;
    # 100 is simply that reference -- it does NOT cap or change the
    # actual exported resolution, which is set independently by
    # `save_path`'s own `dpi` (a higher save dpi just rasterizes the
    # exact same inch-sized figure, and everything in it, at more dots
    # per inch -- all the proportions computed from this constant are
    # unaffected).
    _DISPLAY_DPI = 100
    # Target total height (px, at `_DISPLAY_DPI`) for the frontier-
    # orbital composite figure (`plot_energy_diagram(mode="FMO")`) AND
    # for the plain, standalone energy diagram -- the two are kept at
    # the same height on purpose, so the diagram panel looks identical
    # (not just same-width, see `_energy_diagram_width_in`) whichever
    # way it's drawn.
    _ENERGY_DIAGRAM_HEIGHT_PX = 700
    # Top/bottom margin (inches) of the energy-diagram axes, standalone
    # AND in the frontier-orbital composite (`_plot_frontier_orbitals`
    # reuses these exact values so the diagram panel keeps the same
    # look either way). Kept small: there is no title to leave room for
    # any more (removed -- it was taking a visible bite out of the
    # requested `_ENERGY_DIAGRAM_HEIGHT_PX`, on top of matplotlib's own
    # default top/bottom margins, so the figure's actual *drawn*
    # content -- what a tight bbox crop, e.g. Jupyter's own inline
    # display, keeps -- ended up noticeably shorter than requested), and
    # the data itself already carries a 0.5-unit padding above/below
    # the lowest/highest level (see `ymin`/`ymax` in `plot_energy_diagram`),
    # so the axes box can hug the figure's top/bottom edge closely
    # without clipping anything.
    _DIAGRAM_TOP_MARGIN_IN = 0.08
    _DIAGRAM_BOTTOM_MARGIN_IN = 0.08

    @staticmethod
    def _bond_unit_length():
        """`AllChem.Compute2DCoords` -- used throughout this module to
        lay out every molecule (see `__init__`) -- generates 2D
        coordinates with an ordinary single bond very close to 1.5
        "RDKit units" long, REGARDLESS of the molecule (verified
        empirically: ethylene, benzene, a fused-ring PAH all come out
        the same). This is a fixed RDKit depiction convention, not a
        real physical bond length, but it is the one constant that
        lets every fixed-pixel-bond-length sizing below (`show()`,
        `plot_pi_population()`, `plot_mo_on_graph()`/the frontier-
        orbital panels) convert "N pixels per bond" into an actual
        figure/canvas size from nothing but the molecule's own 2D
        layout -- so a molecule's rendered size always comes out
        proportional to its real spatial extent, atom for atom and
        bond for bond, never to a fixed canvas that a small molecule
        (ethylene) would otherwise swim in and a large one (coronene)
        would otherwise overflow."""
        return 1.5

    def _pi_bbox(self):
        """(width, height) of `self.pi_atom_idx`'s 2D layout, in
        RDKit's own coordinate units (see `_bond_unit_length`) -- the
        bounding box of the PI SYSTEM only, not the whole molecule's
        sigma skeleton (which `show()` uses its own full-molecule bbox
        for instead, since it draws everything, substituents included).
        A degenerate (zero-width or zero-height) dimension -- e.g.
        ethylene's single, perfectly horizontal bond -- is floored to
        one bond length, so a panel never collapses to a sliver."""
        conf = self.mol.GetConformer()
        xs = [conf.GetAtomPosition(ridx).x for ridx in self.pi_atom_idx]
        ys = [conf.GetAtomPosition(ridx).y for ridx in self.pi_atom_idx]
        unit = self._bond_unit_length()
        return (max(max(xs) - min(xs), unit), max(max(ys) - min(ys), unit))

    def _figsize_for_bond_px(self, bond_px, margin_in=0.9, min_side_in=1.6):
        """Figure size (inches) such that one bond-length unit of
        `self.pi_atom_idx`'s real 2D layout (`_pi_bbox`) renders as
        `bond_px` screen pixels at the `_DISPLAY_DPI` reference scale
        -- e.g. ethylene's lone, short bond and coronene's sprawling
        ring system each get a panel sized to match their own actual
        extent, instead of sharing one fixed figsize that leaves one
        of them either tiny or clipped. `margin_in`: flat inch margin
        added on every side, for labels/titles and (`plot_mo_on_graph`)
        the extra padding it adds itself for lobe overflow.
        `min_side_in`: floor so a very small bbox (after the
        `_pi_bbox` flooring, still possible for e.g. a 2-atom system)
        never collapses to an unreadably small panel."""
        bbox_w, bbox_h = self._pi_bbox()
        in_per_unit = (bond_px / self._bond_unit_length()) / self._DISPLAY_DPI
        width_in = max(bbox_w * in_per_unit + 2 * margin_in, min_side_in)
        height_in = max(bbox_h * in_per_unit + 2 * margin_in, min_side_in)
        return width_in, height_in

    @staticmethod
    def _energy_diagram_width_in(max_deg):
        """Total width, in inches, that `plot_energy_diagram` auto-sizes
        its own standalone figure to for a largest level degeneracy of
        `max_deg` (see the comments in `plot_energy_diagram` for what
        each zone is). Factored out so `_plot_frontier_orbitals` can
        give its energy-diagram column that exact same width, instead
        of an unrelated fixed fraction of its own composite figure."""
        seg_w_in, seg_gap_in = 0.22, 0.04        # one segment, and gap within a group
        left_margin_in, mid_gap_in = 0.4, 0.1     # before segments, and before labels
        label_w_in, right_margin_in = 1.15, 0.1   # label text, and after it
        seg_zone_in = max_deg * seg_w_in + (max_deg - 1) * seg_gap_in
        return (left_margin_in + seg_zone_in + mid_gap_in
                + label_w_in + right_margin_in)

    @staticmethod
    def _draw_spin_arrow(ax, x, y, up, amp, color="red"):
        # The y axis is inverted (beta < 0): a smaller y value is drawn
        # HIGHER on screen. A visually "upward" arrow must therefore
        # point (xy) towards y - amp.
        y_tail, y_head = (y + amp, y - amp) if up else (y - amp, y + amp)
        ax.annotate(
            "", xy=(x, y_head), xytext=(x, y_tail),
            arrowprops=dict(arrowstyle="-|>,head_width=0.2,head_length=0.35",
                             color=color, lw=0.9, shrinkA=0, shrinkB=0),
            zorder=3,
        )

    def plot_energy_diagram(self, mode=None, ax=None, figsize=None,
                             tol=1e-6, frontier_figsize=None,
                             save_path=None, dpi=300):
        """Energy-level diagram. The x axis is oriented so that the
        energy E = alpha + x*beta increases upward (beta < 0: the
        largest, most stabilizing x values end up at the bottom).

        mode="FMO" (or "fmo"): instead of the diagram alone, build and
        return a full figure with the diagram plus the HOMO(s)/LUMO(s)
        (frontier orbitals) next to it, so the subplots do not have to
        be composed by hand. Handles degenerate HOMO/LUMO sets and
        radical (SOMO) cases.

        `figsize` defaults to None, which auto-scales the diagram's
        *width* with the largest level degeneracy (e.g. benzene, max
        degeneracy 2, stays as compact as before; a highly degenerate
        spectrum such as C60's, up to degeneracy 9, gets enough room
        for every individual segment) -- pass an explicit figsize to
        override.

        `save_path`: if given, the figure is also saved there (see
        `save_fig`); `dpi` is passed through.

        input:
        - mo_index      = index of the MO to draw (column of self.c_MOs)
        - ax            = matplotlib Axes to draw on; a new figure is created if None (default)
        - scale         = marker-size scale factor for the largest coefficient
        - figsize       = (width, height) in inches; auto-computed from bond_length_px if None (default) and ax is None
        - bond_length_px = target on-screen bond length in pixels, used to auto-size figsize (default: 70)
        - lobe_offset   = separation of the two lobes, as a fraction of each atom's marker radius (default: 0.2); see above
        - save_path     = file path to save the figure to (passed to save_fig); not saved if None (default)
        - dpi           = resolution passed to save_fig when save_path is given (default: 300)

        output:
        - None (the figure is drawn and shown, and saved if save_path is given)
        """
        if mode is not None and str(mode).upper() == "FMO":
            return self._plot_frontier_orbitals(tol=tol, figsize=frontier_figsize,
                                                 save_path=save_path, dpi=dpi)

        groups = self._group_levels(tol=tol)
        max_deg = max(len(g) for g in groups)

        # The diagram has two zones: the segments themselves (should
        # widen with the largest degeneracy, so e.g. C60's 9-fold level
        # still has individually-legible segments) and the text labels
        # to their right (a short, roughly fixed-width string -- "x=
        # 0.618  g=5" -- whatever the degeneracy). Each zone's width is
        # set in inches FIRST, and the whole figure sized to match;
        # expressing both as fixed *fractions* of the axis (the old
        # x_left/x_right=0.08/0.92, independent of figsize) made the
        # label zone grow right along with the figure as max_deg grew,
        # leaving most of a wide, highly-degenerate diagram (C60) blank.
        # `_energy_diagram_width_in` is shared with `_plot_frontier_orbitals`
        # so its diagram column matches this exactly -- its own internal
        # zone widths are duplicated here (not re-derived from it) only
        # because the per-level drawing loop below also needs them.
        seg_w_in, seg_gap_in = 0.22, 0.04
        left_margin_in, mid_gap_in = 0.4, 0.1
        label_w_in, right_margin_in = 1.15, 0.1
        seg_zone_in = max_deg * seg_w_in + (max_deg - 1) * seg_gap_in
        total_width_in = self._energy_diagram_width_in(max_deg)

        created_fig = ax is None
        if created_fig:
            if figsize is None:
                # Same height as the frontier-orbital composite figure
                # (`_ENERGY_DIAGRAM_HEIGHT_PX`, see `_plot_frontier_orbitals`)
                # -- so the diagram panel looks identical whether it is
                # drawn alone or as part of `mode="FMO"`.
                figsize = (total_width_in, self._ENERGY_DIAGRAM_HEIGHT_PX / self._DISPLAY_DPI)
            fig, ax = plt.subplots(figsize=figsize)
            fig_h = figsize[1]
            fig.subplots_adjust(
                top=1 - self._DIAGRAM_TOP_MARGIN_IN / fig_h,
                bottom=self._DIAGRAM_BOTTOM_MARGIN_IN / fig_h,
            )

        spread = float(self.E_MOs.max() - self.E_MOs.min())
        amp = max(0.025 * spread, 0.06)

        # Important: fix the axis limits and orientation BEFORE drawing
        # the annotate() arrows. Otherwise some arrows end up with a
        # corrupted transform if set_xlim/invert_yaxis are called after
        # they were created (a matplotlib caching side effect).
        ymax = float(self.E_MOs.max()) + 0.5
        ymin = float(self.E_MOs.min()) - 0.5
        ax.set_xlim(0, 1)
        ax.set_ylim(ymin, ymax)
        ax.invert_yaxis()

        # The two zones above, expressed as fractions of `total_width_in`
        # (not of whatever figsize was actually passed in -- if the
        # caller overrides figsize, the proportions still match what a
        # figure of `total_width_in` would have looked like, rather than
        # stretching or squashing either zone).
        x_left = left_margin_in / total_width_in
        x_right = x_left + seg_zone_in / total_width_in
        label_x = x_right + mid_gap_in / total_width_in
        gap = seg_gap_in / total_width_in

        # One short segment per degenerate MO (not a single full-width
        # line), side by side. All segments share the SAME width across
        # the whole diagram -- sized so that the most degenerate group
        # still fits the available width -- and a group is centered
        # rather than stretched to fill the width itself. Otherwise a
        # non-degenerate level (1 wide segment) and a highly degenerate
        # one (many narrow segments) end up with wildly different,
        # visually inconsistent segment widths -- most noticeable on
        # high-symmetry spectra such as icosahedral C60 (degeneracies up
        # to 5, or more for "accidental" degeneracies).
        seg_w = max((x_right - x_left - (max_deg - 1) * gap) / max_deg, 0.02)

        # Gap between the two spin arrows of a filled (occ=2) level:
        # a FIXED on-screen size (a few points), not a plain fraction of
        # the data-coordinate width -- the same data-unit gap would
        # otherwise render wider in pixels as soon as the containing
        # figure/axes gets wider (e.g. a bigger figsize used for a very
        # degenerate spectrum such as C60). Converting a fixed point
        # size through the axes' actual, current transData keeps the
        # rendered gap the same size as in a narrow, benzene-sized
        # diagram; `seg_w * 0.28` is kept only as a safety cap so the
        # two arrows never overlap when segments themselves are very
        # narrow (high degeneracy).
        try:
            pair_offset = min(self._points_to_data_dx(ax, 3.0), seg_w * 0.28)
        except Exception:
            pair_offset = min(0.02, seg_w * 0.28)

        for group in groups:
            deg = len(group)
            y = self._clean_near_zero(np.mean(self.E_MOs[group]))  # equal up to tolerance

            group_w = deg * seg_w + (deg - 1) * gap
            group_x0 = x_left + (x_right - x_left - group_w) / 2  # center the group
            for j, k in enumerate(group):
                x0 = group_x0 + j * (seg_w + gap)
                x1 = x0 + seg_w
                center = (x0 + x1) / 2
                ax.hlines(y, x0, x1, color="black", lw=1.5, zorder=1)
                occ = self.occupation[k]
                if occ >= 2:
                    self._draw_spin_arrow(ax, center - pair_offset, y, True, amp)
                    self._draw_spin_arrow(ax, center + pair_offset, y, False, amp)
                elif occ == 1:
                    self._draw_spin_arrow(ax, center, y, True, amp)

            label = f"x={y:.3f}" + (f"  g = {deg}" if deg > 1 else "")
            ax.text(label_x, y, label, va="center", fontsize=8)

        ax.axhline(0, color="grey", lw=0.5, ls="--")
        ax.set_xticks([])
        ax.set_ylabel(r"$x$  (with $E = \alpha + x\,\beta$,  $\beta<0$)")
        if save_path is not None:
            save_fig(save_path, fig=ax.figure, dpi=dpi)
        if created_fig:
            # Show (and, under the inline backend, close) the figure
            # right here -- see `show()` for why, and why `None` (not
            # `ax`) is returned in this branch: a closed Axes isn't
            # useful to the caller anyway, and returning it would
            # trigger an extra, bare "<Axes: ...>" text line if this
            # call happens to be a cell's last statement. When an `ax`
            # was passed in (composing into the caller's own figure),
            # neither applies: that figure isn't necessarily finished
            # yet, and the caller still needs its `ax` back.
            plt.show()
            return None
        return ax

    def _plot_frontier_orbitals(self, tol=1e-6, figsize=None,
                                 mo_bond_length_px=70, save_path=None, dpi=300):
        """Figure with the energy diagram plus the HOMO(s)/LUMO(s)
        (frontier orbitals), handling degeneracy and radical cases.

        Layout: the energy diagram on the left, at its own standalone
        width (`_energy_diagram_width_in`) and the SAME total height as
        that standalone figure (`_ENERGY_DIAGRAM_HEIGHT_PX`, 700 px by
        default) -- to its right, the LUMO(s) occupy the TOP half and
        the HOMO(s)/SOMO(s) the BOTTOM half (350 px each), matching the
        diagram's own vertical sense: with beta<0, a filled, bonding
        MO (larger x) is drawn toward the BOTTOM of the diagram and an
        empty, antibonding one (smaller x) toward the TOP -- so HOMO-
        bottom/LUMO-top lines the MO panels up with where their own
        level actually sits on the diagram beside them. A degenerate
        HOMO or LUMO gets one column per member, side by side within
        its own row; the two rows share one column grid, so e.g. a
        single HOMO next to a doubly-degenerate LUMO still lines up
        (one empty cell under the second LUMO column). Each MO panel's
        WIDTH is sized, as in standalone `plot_mo_on_graph`, so a C-C
        bond renders at roughly `mo_bond_length_px` pixels -- the same
        guard against a small molecule's orbitals looking artificially
        huge (ethylene) or a large one's looking cramped (coronene)."""
        homo_group, lumo_group = self._frontier_groups(tol=tol)
        if not homo_group and not lumo_group:
            raise ValueError(
                "Cannot determine HOMO/LUMO (pi system is completely "
                "empty or completely full of pi electrons)."
            )

        n_cols = max(len(homo_group), len(lumo_group), 1)

        # The diagram column gets exactly the same width (inches) that
        # `plot_energy_diagram` would auto-size its own standalone figure
        # to for this molecule -- not an unrelated fraction of this
        # composite figure's width -- so the two look identical whether
        # or not the frontier orbitals are shown alongside it.
        groups = self._group_levels(tol=tol)
        max_deg = max(len(g) for g in groups)
        diagram_width_in = self._energy_diagram_width_in(max_deg)
        mo_width_in, _ = self._figsize_for_bond_px(mo_bond_length_px, margin_in=0.5)
        fig_h_in = self._ENERGY_DIAGRAM_HEIGHT_PX / self._DISPLAY_DPI
        if figsize is None:
            figsize = (diagram_width_in + mo_width_in * n_cols, fig_h_in)
        fig_w, fig_h = figsize

        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(
            nrows=2, ncols=1 + n_cols,
            width_ratios=[diagram_width_in] + [mo_width_in] * n_cols,
            height_ratios=[1, 1],
        )
        ax_diagram = fig.add_subplot(gs[:, 0])
        self.plot_energy_diagram(ax=ax_diagram, tol=tol)

        for col in range(n_cols):
            if col < len(lumo_group):
                k = lumo_group[col]
                lab = "LUMO"
                if len(lumo_group) > 1:
                    lab += f" ({col + 1}/{len(lumo_group)})"
                ax_k = fig.add_subplot(gs[0, 1 + col])
                self.plot_mo_on_graph(k, ax=ax_k)
                ax_k.set_title(f"{lab}\n(x = {self._clean_near_zero(self.E_MOs[k]):.3f})", fontsize=10)
            if col < len(homo_group):
                k = homo_group[col]
                lab = "HOMO" if self.occupation[k] >= 2 else "SOMO"
                if len(homo_group) > 1:
                    lab += f" ({col + 1}/{len(homo_group)})"
                ax_k = fig.add_subplot(gs[1, 1 + col])
                self.plot_mo_on_graph(k, ax=ax_k)
                ax_k.set_title(f"{lab}\n(x = {self._clean_near_zero(self.E_MOs[k]):.3f})", fontsize=10)

        fig.tight_layout()

        # A matching gridspec *column* width (above) is not quite enough:
        # `tight_layout()` trims the diagram axes' own margins inside
        # that column differently than the (margin-only, no tight_layout)
        # standalone figure does, so the rendered diagram still came out
        # a bit larger here. Force the diagram axes' physical box (in
        # inches) to be identical to what the standalone figure gives it
        # -- same default rcParams margins, applied to the same
        # `diagram_width_in` and the same `fig_h` -- so the two are
        # pixel-for-pixel the same regardless of what else tight_layout
        # did to the MO panels beside it.
        left_in = plt.rcParams["figure.subplot.left"] * diagram_width_in
        right_in = (1 - plt.rcParams["figure.subplot.right"]) * diagram_width_in
        # Top/bottom: the same small fixed margins the standalone diagram
        # now uses (`_DIAGRAM_TOP_MARGIN_IN`/`_DIAGRAM_BOTTOM_MARGIN_IN`),
        # not matplotlib's default rcParams fractions -- those were sized
        # to leave room for a title, which neither diagram has anymore.
        top_in = self._DIAGRAM_TOP_MARGIN_IN
        bottom_in = self._DIAGRAM_BOTTOM_MARGIN_IN
        ax_diagram.set_position([
            left_in / fig_w,
            bottom_in / fig_h,
            (diagram_width_in - left_in - right_in) / fig_w,
            (fig_h - top_in - bottom_in) / fig_h,
        ])

        if save_path is not None:
            save_fig(save_path, fig=fig, dpi=dpi)
        # Show (and close) right here rather than returning `fig` for
        # Jupyter to auto-display: that only works when this call is a
        # cell's *last* statement, and leaves nothing shown at all
        # otherwise (see `show()` for the detailed rationale, which
        # applies here identically).
        plt.show()
        return None

    def plot_mo_on_graph(self, mo_index, ax=None, scale=800, figsize=None,
                         bond_length_px=70, lobe_offset=0.2, save_path=None, dpi=300):
        """Coefficients of MO `mo_index` drawn on the real 2D geometry
        of the molecule, each pi atom rendered as a stylized two-lobed
        p orbital rather than a flat disc: a front lobe (circle size =
        \|c_i\|, color = sign of c_i, same convention as before) shifted
        slightly toward the bottom-left along the diagonal, and a same-
        size, opposite-color back lobe shifted the same amount toward
        the top-right, drawn behind it. The atom label stays at the
        true atom position, between the two lobes.

        `figsize` defaults to None, which auto-sizes the panel from the
        molecule's own 2D layout so a C-C bond renders at roughly
        `bond_length_px` screen pixels (see `_figsize_for_bond_px`) --
        this is what keeps ethylene's one short bond from filling the
        whole panel the way it would at a fixed figsize, while a much
        larger pi system (coronene...) still gets the room it needs;
        pass an explicit figsize to override. Only applies when this
        call creates its OWN figure (`ax=None`) -- when composed into
        an existing `ax` (`plot_all_mo`/`plot_mo_list`/the frontier-
        orbital panels), the panel's size is already decided by the
        caller instead.

        `lobe_offset`: how far apart the two lobes are drawn, as a
        fraction of EACH atom's own rendered circle radius (so a big
        coefficient's lobes are offset more than a small one's, and the
        two always overlap sensibly regardless of marker size) -- 0
        collapses back to a single disc per atom (the pre-existing
        look); 1 offsets each lobe by a full radius (touching at the
        atom's true position); the default (0.2) gives a subtle,
        mostly-overlapping two-lobed look -- raise it for a more
        pronounced dumbbell shape.

        `save_path`: if given, the figure is also saved there (see
        `save_fig`); `dpi` is passed through.

        input:
        - mo_index      = index of the MO to draw (column of self.c_MOs)
        - ax            = matplotlib Axes to draw on; a new figure is created if None (default)
        - scale         = marker-size scale factor for the largest coefficient
        - figsize       = (width, height) in inches; auto-computed from bond_length_px if None (default) and ax is None
        - bond_length_px = target on-screen bond length in pixels, used to auto-size figsize (default: 70)
        - lobe_offset   = separation of the two lobes, as a fraction of each atom's marker radius (default: 0.2); see above
        - save_path     = file path to save the figure to (passed to save_fig); not saved if None (default)
        - dpi           = resolution passed to save_fig when save_path is given (default: 300)

        output:
        - None (the figure is drawn and shown, and saved if save_path is given)
        """
        created_fig = ax is None
        if created_fig:
            if figsize is None:
                figsize = self._figsize_for_bond_px(bond_length_px)
            fig, ax = plt.subplots(figsize=figsize)
        conf = self.mol.GetConformer()
        pos = {k: (conf.GetAtomPosition(ridx).x, conf.GetAtomPosition(ridx).y)
               for k, ridx in enumerate(self.pi_atom_idx)}

        nx.draw_networkx_edges(self.graph_pi, pos, ax=ax, edge_color="grey")
        ax.set_aspect("equal")
        ax.autoscale_view()  # establishes a valid transData from the atom
                              # positions alone, before any lobe offset

        c = self.c_MOs[:, mo_index]
        sizes = scale * (c ** 2) / max((c ** 2).max(), 1e-9)
        colors = ["#d62728" if ci > 0 else "#1f77b4" for ci in c]
        anti_colors = ["#1f77b4" if ci > 0 else "#d62728" for ci in c]

        # Each atom's own offset, from ITS marker radius (points -> data
        # units via the axes' current transData -- same trick used for
        # the energy-diagram spin-arrow gap, see `_points_to_data_dx`),
        # not a single molecule-wide distance: a big lobe is shifted
        # more than a small one, and the pair always overlaps the same
        # way regardless of the coefficient's actual magnitude.
        radii_pts = np.sqrt(sizes / np.pi)
        offsets = np.array([self._points_to_data_dx(ax, r) for r in radii_pts])
        d = lobe_offset * offsets / np.sqrt(2)  # diagonal x/y components

        pos_front = {k: (pos[k][0] - d[k], pos[k][1] - d[k]) for k in pos}  # bottom-left
        pos_back = {k: (pos[k][0] + d[k], pos[k][1] + d[k]) for k in pos}   # top-right

        back_nodes = nx.draw_networkx_nodes(
            self.graph_pi, pos_back, ax=ax, node_size=sizes,
            node_color=anti_colors, alpha=0.85, edgecolors="black",
        )
        back_nodes.set_zorder(1)
        front_nodes = nx.draw_networkx_nodes(
            self.graph_pi, pos_front, ax=ax, node_size=sizes,
            node_color=colors, alpha=0.85, edgecolors="black",
        )
        front_nodes.set_zorder(2)

        labels = {k: self.labels[k] for k in pos}
        nx.draw_networkx_labels(self.graph_pi, pos, labels, ax=ax, font_size=8)
        ax.set_title(f"MO #{mo_index}  (x = {self._clean_near_zero(self.E_MOs[mo_index]):.3f},"
                     f" occ = {self.occupation[mo_index]:.0f})")
        ax.set_axis_off()
        ax.set_aspect("equal")

        # `autoscale_view()` above only accounted for the atom positions
        # themselves, not how far a marker's own radius (let alone a
        # shifted lobe) extends past them -- a big coefficient's circle
        # would otherwise get clipped at the axes' edge, worse once a
        # lobe is also shifted outward. Pad the view by the largest
        # lobe's actual reach (its own radius plus how far it was
        # shifted) so nothing this method draws is ever cut off,
        # whatever `scale`/`lobe_offset` or how small the containing
        # axes happens to be (e.g. a `plot_all_mo` grid cell).
        pad = float((offsets + np.abs(d)).max()) * 1.15 if len(offsets) else 0.0
        if pad > 0:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            ax.set_xlim(xlim[0] - pad, xlim[1] + pad)
            ax.set_ylim(ylim[0] - pad, ylim[1] + pad)
        if save_path is not None:
            save_fig(save_path, fig=ax.figure, dpi=dpi)
        if created_fig:
            plt.show()  # see `show()` for why, and why `None` is returned here
            return None
        return ax

    def plot_all_mo(self, ncols=3, figsize_per=None, bond_length_px=70,
                     save_path=None, dpi=300):
        """Grid with every pi MO drawn on the molecular graph. See
        `plot_mo_list` for a grid restricted to a chosen subset of MOs
        (e.g. just the frontier ones, or a hand-picked list)."""
        return self.plot_mo_list(range(len(self.E_MOs)), ncols=ncols,
                                  figsize_per=figsize_per,
                                  bond_length_px=bond_length_px,
                                  save_path=save_path, dpi=dpi)

    def plot_mo_list(self, mo_indices, ncols=3, figsize_per=None,
                      bond_length_px=70, save_path=None, dpi=300):
        """Grid with a chosen list of pi MOs drawn on the molecular
        geometry (see `plot_mo_on_graph` for a single one) -- e.g.
        `mol.plot_mo_list([0, 1, 7, 8])` to compare a few MOs of
        interest side by side, in exactly the order given, without
        plotting the whole set as `plot_all_mo` does.

        `figsize_per` (the per-panel size) defaults to None, which
        auto-sizes EVERY panel, uniformly, from the molecule's own 2D
        layout so a C-C bond renders at roughly `bond_length_px` screen
        pixels -- same idea, and the same `_figsize_for_bond_px` helper,
        as the standalone `plot_mo_on_graph`'s own `figsize`/
        `bond_length_px` (all panels here show the SAME molecule, just
        a different MO each, so one computed size applies to all of
        them). Pass an explicit `figsize_per` to override.

        `save_path`: if given, the figure is also saved there (see
        `save_fig`); `dpi` is passed through.

        input:
        - mo_index      = index of the MO to draw (column of self.c_MOs)
        - ax            = matplotlib Axes to draw on; a new figure is created if None (default)
        - scale         = marker-size scale factor for the largest coefficient
        - figsize       = (width, height) in inches; auto-computed from bond_length_px if None (default) and ax is None
        - bond_length_px = target on-screen bond length in pixels, used to auto-size figsize (default: 70)
        - lobe_offset   = separation of the two lobes, as a fraction of each atom's marker radius (default: 0.2); see above
        - save_path     = file path to save the figure to (passed to save_fig); not saved if None (default)
        - dpi           = resolution passed to save_fig when save_path is given (default: 300)

        output:
        - None (the figure is drawn and shown, and saved if save_path is given)
        """
        mo_indices = list(mo_indices)
        n = len(mo_indices)
        if n == 0:
            raise ValueError("mo_indices is empty: nothing to plot.")
        if figsize_per is None:
            figsize_per = self._figsize_for_bond_px(bond_length_px)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols,
                                  figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows))
        axes = np.atleast_1d(axes).ravel()
        for ax, mo_index in zip(axes, mo_indices):
            self.plot_mo_on_graph(mo_index, ax=ax)
        for ax in axes[n:]:
            ax.set_axis_off()
        fig.tight_layout()
        if save_path is not None:
            save_fig(save_path, fig=fig, dpi=dpi)
        plt.show()  # see `show()` for why, and why `None` is returned here
        return None

    def plot_pi_population(self, ax=None, figsize=None, bond_length_px=100,
                            charge_fmt="{:+.2f}", bond_fmt="{:.2f}",
                            atom_fontsize=10, bond_fontsize=10,
                            save_path=None, dpi=300):
        """Classic Hückel 'charge and bond-order diagram': the pi
        system graph annotated with the net pi charge on each atom and
        the pi bond order on each bond (see `pi_charge` / `bond_order`
        / `density_matrix`).

        `figsize` defaults to None, which auto-sizes the figure from
        the molecule's own 2D layout so a C-C bond renders at roughly
        `bond_length_px` screen pixels (see `_figsize_for_bond_px`) --
        larger than `plot_mo_on_graph`'s own default (100 vs. 70) since
        the bond-order labels sitting right on top of each bond need
        more room to stay legible than a plain p-orbital lobe does;
        pass an explicit figsize to override. `atom_fontsize`/
        `bond_fontsize` control the label sizes directly -- bump them
        further for a very crowded graph, or when exporting to a
        larger figure.

        `save_path`: if given, the figure is also saved there (see
        `save_fig`); `dpi` is passed through.

        input:
        - mo_index      = index of the MO to draw (column of self.c_MOs)
        - ax            = matplotlib Axes to draw on; a new figure is created if None (default)
        - scale         = marker-size scale factor for the largest coefficient
        - figsize       = (width, height) in inches; auto-computed from bond_length_px if None (default) and ax is None
        - bond_length_px = target on-screen bond length in pixels, used to auto-size figsize (default: 70)
        - lobe_offset   = separation of the two lobes, as a fraction of each atom's marker radius (default: 0.2); see above
        - save_path     = file path to save the figure to (passed to save_fig); not saved if None (default)
        - dpi           = resolution passed to save_fig when save_path is given (default: 300)

        output:
        - None (the figure is drawn and shown, and saved if save_path is given)
        """
        created_fig = ax is None
        if created_fig:
            if figsize is None:
                figsize = self._figsize_for_bond_px(bond_length_px, margin_in=1.1)
            fig, ax = plt.subplots(figsize=figsize)

        conf = self.mol.GetConformer()
        pos = {k: (conf.GetAtomPosition(ridx).x, conf.GetAtomPosition(ridx).y)
               for k, ridx in enumerate(self.pi_atom_idx)}

        charges = self.pi_charge
        nx.draw_networkx_edges(self.graph_pi, pos, ax=ax, edge_color="grey")
        # red = electron-poor (positive pi charge), blue = electron-rich
        # (negative), white = essentially neutral (|q| < 0.01).
        node_colors = [
            "white" if abs(q) < 0.01 else ("#d62728" if q > 0 else "#1f77b4")
            for q in charges
        ]
        # Base size bumped up from the original 600/1500 (slightly larger
        # atom radius -- was reading as too small, especially with labels).
        sizes = 900 + 1800 * np.abs(charges)
        nx.draw_networkx_nodes(self.graph_pi, pos, ax=ax, node_size=sizes,
                                node_color=node_colors, alpha=0.75, edgecolors="black")

        for k in range(len(self.pi_atom_idx)):
            x, y = pos[k]
            ax.annotate(f"{self.labels[k]}\n{charge_fmt.format(charges[k])}",
                        (x, y), ha="center", va="center", fontsize=atom_fontsize,
                        zorder=4)

        edge_labels = {
            (a, b): bond_fmt.format(self.bond_order(a, b))
            for a, b in self.graph_pi.edges()
        }
        nx.draw_networkx_edge_labels(self.graph_pi, pos, edge_labels=edge_labels,
                                      ax=ax, font_size=bond_fontsize,
                                      font_color="darkgreen")

        ax.set_title("Pi charges (on atoms) and pi bond orders (on bonds)")
        ax.set_axis_off()
        # `adjustable="datalim"` (not the default "box"): for a perfectly
        # collinear pi system (ethylene's lone C=C, any isolated double
        # bond...) the y-extent of the atom positions is ~0, and "box"
        # tries to shrink the AXES BOX itself to match that aspect ratio
        # -- which can collapse to a zero-size box (a singular transform,
        # raised as `LinAlgError` deep in matplotlib/networkx's edge-label
        # placement) rather than just padding the data limits instead.
        ax.set_aspect("equal", adjustable="datalim")
        if save_path is not None:
            save_fig(save_path, fig=ax.figure, dpi=dpi)
        if created_fig:
            plt.show()  # see `show()` for why, and why `None` is returned here
            return None
        return ax

    def show(self, ax=None, figsize=None, size=None, bond_length_px=80,
             title=None, show_indices=True, show_hybridization=False,
             highlight_pi=True, save_path=None, dpi=300):
        """Draw the molecule's FULL 2D skeleton (sigma + pi, with
        implicit H's, wedge/dash stereo if any...) using RDKit's own
        depiction engine -- unlike `plot_mo_on_graph` / `plot_pi_population`,
        which only draw the pi subsystem detected for the Hückel
        treatment. Handy as a quick sanity check that the SMILES was
        parsed/drawn (and the conjugated system detected) as intended.

        `show_indices` (default True): label each atom with its RDKit
        atom index -- the same index used in `self.pi_atom_idx` and in
        the "C0", "N3"... entries of `self.labels`.
        `show_hybridization`: annotate each atom with its hybridization
        (sp, sp2, sp3...) as given by RDKit.
        `highlight_pi` (default True): highlight the atoms/bonds of the
        conjugated pi system actually used for the Hückel treatment
        (`self.pi_atom_idx`) -- the rest of the molecule (sigma-only
        substituents, sp3 carbons...) is left unhighlighted.

        `size` defaults to None, which sizes RDKit's own drawing canvas
        from the molecule's actual 2D layout (its full sigma+pi
        skeleton, not just the pi system) so a C-C bond renders at a
        FIXED `bond_length_px` (80 by default) -- RDKit's own
        `fixedBondLength` drawing option (verified empirically: it is
        NOT a 1:1 pixel setting on this RDKit build, so the conversion
        is calibrated against `_bond_unit_length`). This is what keeps
        ethylene's one bond from filling the whole canvas the way a
        fixed size would, while a large, crowded molecule (coronene...)
        still gets a canvas sized to actually fit it, hybridization
        labels and all. Pass an explicit `size` (pixels, RDKit's own
        legacy behavior: the molecule is auto-scaled, in either
        direction, to fill that fixed canvas) to go back to a plain
        fixed-size drawing instead -- `bond_length_px` is then ignored.
        The drawing's blank margin is cropped automatically either way.

        `figsize` defaults to None, which sizes the matplotlib figure
        to match the (now-cropped) drawing 1:1 in pixels when `size`
        was auto-computed (so the bond-length guarantee above survives
        all the way to the actual on-screen rendering, not just the
        intermediate RDKit image), or to the drawing's aspect ratio at
        a fixed 4-inch height when an explicit `size` was given
        instead (the previous behavior). Pass an explicit `figsize` to
        override either way (e.g. to make it larger, or to match a
        figure you're composing by hand).
        `title` defaults to the SMILES. `save_path`/`dpi`: see
        `save_fig`."""
        import io
        from PIL import Image

        auto_size = size is None
        unit = self._bond_unit_length()
        if auto_size:
            conf = self.mol.GetConformer()
            xs = [conf.GetAtomPosition(a.GetIdx()).x for a in self.mol.GetAtoms()]
            ys = [conf.GetAtomPosition(a.GetIdx()).y for a in self.mol.GetAtoms()]
            bbox_w = max(max(xs) - min(xs), unit)
            bbox_h = max(max(ys) - min(ys), unit)
            px_per_unit = bond_length_px / unit
            # Flat pixel margin around the bbox -- generous on purpose
            # (atom-index/hybridization labels extend past the bare
            # bond skeleton, especially at the molecule's own edges):
            # any excess is trimmed away by the whitespace crop below
            # anyway, so overshooting here is harmless, but too SMALL a
            # canvas would make RDKit silently shrink `fixedBondLength`
            # to fit it instead (verified empirically), defeating the
            # whole point.
            margin_px = 90
            size = (int(round(bbox_w * px_per_unit)) + 2 * margin_px,
                    int(round(bbox_h * px_per_unit)) + 2 * margin_px)

        notes_set = []
        if show_hybridization:
            # A radical/charged terminal carbon folded into the pi system
            # by `_detect_pi_system` (allyl's >CH2, etc. -- see its own
            # docstring) is chemically sp2 -- that is the whole point of
            # it joining the pi system -- but RDKit's own hybridization
            # perception still reports it as sp3, exactly like it
            # reports that bond as unconjugated: nothing in a plain
            # SMILES like this tells RDKit the radical/charge sits in a p
            # orbital. Shown as "sp2" here instead of repeating that same
            # blind spot on screen; a genuine sp3 hyperconjugating
            # substituent (an actual methyl/alkyl group) keeps its real
            # "sp3" label, since that one is correct.
            core_pi_idx = set(self.pi_atom_idx) - self._hyperconjugating_idx
            for atom in self.mol.GetAtoms():
                ridx = atom.GetIdx()
                hyb = str(atom.GetHybridization()).rsplit(".", 1)[-1].lower()
                if (hyb == "sp3" and ridx in core_pi_idx
                        and (atom.GetFormalCharge() != 0 or atom.GetNumRadicalElectrons() > 0)):
                    hyb = "sp2"
                if hyb not in ("unspecified", "other"):
                    atom.SetProp("atomNote", hyb)
                    notes_set.append(atom)

        highlight_atoms, highlight_bonds = [], []
        highlight_atom_colors, highlight_bond_colors = {}, {}
        if highlight_pi:
            pi_color = (0.68, 0.85, 0.90)  # light blue
            highlight_atoms = list(self.pi_atom_idx)
            highlight_atom_colors = {idx: pi_color for idx in highlight_atoms}
            # Every bond directly between two pi_atom_idx members is a pi
            # bond -- see the matching comment in `__init__` (the Hückel
            # matrix M construction) for why no further per-bond check is
            # needed here either.
            for bond in self.mol.GetBonds():
                i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                if i in self._idx_map and j in self._idx_map:
                    highlight_bonds.append(bond.GetIdx())
                    highlight_bond_colors[bond.GetIdx()] = pi_color

        try:
            drawer = rdMolDraw2D.MolDraw2DCairo(*size)
            if auto_size:
                drawer.drawOptions().fixedBondLength = bond_length_px / unit
            drawer.drawOptions().addAtomIndices = show_indices
            rdMolDraw2D.PrepareAndDrawMolecule(
                drawer, self.mol,
                highlightAtoms=highlight_atoms, highlightBonds=highlight_bonds,
                highlightAtomColors=highlight_atom_colors,
                highlightBondColors=highlight_bond_colors,
            )
            drawer.FinishDrawing()
            img = Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGB")
        finally:
            for atom in notes_set:
                atom.ClearProp("atomNote")

        # RDKit always renders into the full, square `size` canvas, with
        # the molecule centered and padded to fit -- for anything that
        # isn't itself roughly square (two rings side by side, a long
        # chain...) that leaves a lot of blank margin. Crop down to the
        # actual drawing (plus a small margin) before it goes anywhere
        # near matplotlib.
        arr = np.asarray(img)
        non_white = np.any(arr < 250, axis=-1)
        rows, cols = np.where(non_white)[0], np.where(non_white.any(axis=0))[0]
        if rows.size and cols.size:
            pad = 12
            top, bottom = max(rows.min() - pad, 0), min(rows.max() + pad, arr.shape[0] - 1)
            left, right = max(cols.min() - pad, 0), min(cols.max() + pad, arr.shape[1] - 1)
            img = img.crop((left, top, right + 1, bottom + 1))

        created_fig = ax is None
        if created_fig:
            full_bleed = False
            if figsize is None:
                w, h = img.size
                if auto_size:
                    # The canvas was already built so a bond renders at
                    # exactly `bond_length_px` pixels -- show it 1:1 at
                    # `_DISPLAY_DPI` instead of re-scaling into an
                    # unrelated fixed-height figure, so that guarantee
                    # survives all the way to the actual on-screen size.
                    figsize = (w / self._DISPLAY_DPI, h / self._DISPLAY_DPI)
                    full_bleed = True
                else:
                    # Explicit `size` given (legacy fixed-canvas mode):
                    # size the figure to the drawing's own (now-cropped)
                    # aspect ratio, instead of forcing a plain square
                    # that leaves an elongated molecule small and
                    # surrounded by whitespace.
                    height = 4.0
                    width = max(2.5, min(10.0, height * w / h))
                    figsize = (width, height)
            fig = plt.figure(figsize=figsize)
            if full_bleed:
                # `plt.subplots()`'s default axes box only covers a
                # fraction of the figure (matplotlib's own default
                # margins), so `imshow` would shrink the already-1:1
                # `img` down to fit that smaller box -- defeating the
                # whole point of sizing the canvas to the pixel. A
                # full-bleed axes (no margin) keeps the image, and the
                # bond length baked into it, at the real `bond_length_px`
                # on screen; the title (added below) still gets its own
                # room since it is drawn outside the axes box, and a
                # tight bbox crop (e.g. Jupyter's own inline display)
                # simply grows to include it instead of eating into the
                # image.
                ax = fig.add_axes([0, 0, 1, 1])
            else:
                ax = fig.add_subplot(1, 1, 1)
        ax.imshow(img)
        ax.set_title(title if title is not None else self.smiles)
        ax.set_axis_off()
        if save_path is not None:
            save_fig(save_path, fig=ax.figure, dpi=dpi)
        if created_fig:
            # Force the figure to display right here (inline backend:
            # render + close immediately), instead of leaving it queued
            # for Jupyter's end-of-cell flush -- which would otherwise
            # make it appear *after* anything printed later in the same
            # cell (e.g. a subsequent `.summary()` call), however the
            # two calls are ordered in the code. Only done when `show`
            # created its own standalone figure: an `ax` passed in by
            # the caller belongs to a larger figure of theirs that may
            # not be fully drawn yet. `None` (not `ax`) is returned in
            # that case too -- a closed Axes isn't useful to the caller
            # anyway, and returning it would trigger an extra, bare
            # "<Axes: ...>" text line if this call happens to be a
            # cell's last statement.
            plt.show()
            return None
        return ax

    def save_results(self, path_with_name: str, tol: float = 1e-6, **kwargs):
        """Save the MO table (with coefficients) and the density matrix
        together in one multi-sheet Excel/ODS file (one sheet each:
        "MO", "density_matrix"), via `pandas.ExcelWriter` -- the engine
        (openpyxl for .xlsx, odfpy for .ods) is picked automatically
        from the file extension, and must be installed. For a single
        table (e.g. `properties_table`'s own result), just call the
        module-level `save_data(df, path)` directly instead.

        `tol`: passed to the MO table (numerical cleanup of near-zero
        values -- not the coefficient tolerance, which keeps its own
        default; pass `coeff_tol` via a direct `mo_table()` call if you
        need to change that one too).

        Computes the two tables directly (rather than going through
        `mo_table`/`density_matrix`) so saving a file stays silent --
        those public methods now always display their result
        immediately, which isn't wanted here. Both tables are still
        cached as `self.mo_table_df`/`self.density_matrix_df`, same as
        calling each method would."""
        if pd is None:
            raise ImportError("save_results requires pandas (and openpyxl "
                               "or odfpy, depending on the file extension).")
        from pathlib import Path
        full_path = Path(path_with_name)
        if full_path.suffix.lower() not in (".xlsx", ".xls", ".ods"):
            raise ValueError(
                f"Unsupported file extension: {full_path.suffix!r}. "
                "Use .xlsx or .ods for a multi-sheet export."
            )
        if full_path.parent:
            full_path.parent.mkdir(parents=True, exist_ok=True)

        mo_rows = self._build_mo_table(tol=tol, with_coefficients=True)
        mo_df = pd.DataFrame(mo_rows)
        P = self.c_MOs @ np.diag(self.occupation) @ self.c_MOs.T
        density_df = pd.DataFrame(P, index=self.labels, columns=self.labels)

        with pd.ExcelWriter(full_path, **kwargs) as writer:
            mo_df.to_excel(writer, sheet_name="MO", index=False)
            density_df.to_excel(writer, sheet_name="density_matrix")

        self.mo_table_df = mo_df
        self.density_matrix_df = density_df
        print(f"Results saved to: {full_path}  (sheets: MO, density_matrix)")


def hmo_properties_table(smiles_list, alpha: float = 0.0, beta: float = 1.0, **huckel_kwargs):
    """Build a table of frontier-orbital descriptors (see
    `HMO.properties`) for a list of molecules, ready to use
    as a pandas DataFrame.

    Parameters
    ----------
    smiles_list : list
        Each element is either a bare SMILES string, or a
        (smiles, overrides_dict) tuple for per-molecule keyword
        overrides (h_overrides, k_overrides, electron_overrides,
        charge).
    alpha, beta : float
        Passed to `HMO.properties` for every molecule.
    **huckel_kwargs :
        Default keyword arguments passed to every `HMO(...)`
        construction (overridden per-molecule by the tuple form above).

    Returns a pandas DataFrame if pandas is available, otherwise a list
    of dicts.
    """
    rows = []
    for item in smiles_list:
        if isinstance(item, tuple):
            smi, extra = item
        else:
            smi, extra = item, {}
        mol = HMO(smi, **{**huckel_kwargs, **extra})
        rows.append(mol.properties(alpha=alpha, beta=beta))
    if pd is not None:
        return pd.DataFrame(rows)
    return rows
