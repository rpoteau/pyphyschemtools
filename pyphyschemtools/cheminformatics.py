############################################################
#                       easy_rdkit
############################################################
from .visualID_Eng import fg, bg, hl
from .core import centerTitle, centertxt

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, GetPeriodicTable, Draw, rdCoordGen
from rdkit.Chem import Descriptors, QED, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
import pandas as pd
from IPython.display import SVG
from PIL import Image
import os, math
from pathlib import Path

class easy_rdkit():
    """
    A helper class to analyze and visualize molecules using RDKit.
    Provides tools for Lewis structure analysis and advanced 2D drawing.
    Initialize the molecule object from a SMILES string.
    
    Args:
        smiles (str): The SMILES representation of the molecule.
        canonical (bool): If True, converts the SMILES to its canonical form 
                          to ensure consistent atom numbering and uniqueness.
        lang (str): Language for headers and messages of the Lewis analyzis("En" (default) or "Fr"). 
    """

    def __init__(self,smiles, canonical=True, lang="En"):
        from rdkit import Chem

        self.cid = None

        self._descriptors_cache = None
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")
            
        if canonical:
            # Generate canonical isomeric SMILES
            self.smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
            # Re-load the molecule from the canonical SMILES to sync atom indices
            self.mol = Chem.MolFromSmiles(self.smiles)
        else:
            self.mol=mol
            self.smiles = smiles
        self.lang = lang.lower().capitalize()
    
    @classmethod
    def from_cid(cls, cid, canonical=True, lang="En"):
        """
        Builds an easy_rdkit instance directly from a PubChem CID.
    
        Fetches the compound from PubChem, extracts its connectivity SMILES,
        and uses it to construct the molecule (equivalent to
        `easy_rdkit(smiles)` but starting from a known CID instead).
        self.cid and self.iupac_name are set from the PubChem record.
    
        Args:
            cid (int or str): The PubChem Compound ID.
            canonical (bool): Passed through to __init__.
            lang (str): Passed through to __init__ ("En" or "Fr").
    
        Returns:
            easy_rdkit: The constructed instance.
    
        Raises:
            ValueError: if the CID does not correspond to a valid PubChem
                compound.
            ConnectionError: if the PubChem request fails (network/API issue).
        """
        import pubchempy as pcp
        try:
            compound = pcp.Compound.from_cid(cid)
        except Exception as e:
            raise ConnectionError(f"Error fetching CID {cid} from PubChem: {e}") from e
    
        if compound is None:
            raise ValueError(f"No PubChem compound found for CID: {cid}")
    
        print(f"✅ Successfully retrieved: {compound.iupac_name}")
        instance = cls(compound.connectivity_smiles, canonical=canonical, lang=lang)
        instance.cid = cid
        instance.iupac_name = compound.iupac_name
        return instance
        
    def fetch_pubchem_data(self):
        """
        Queries PubChem using the molecule's current SMILES to retrieve its
        PubChem CID and IUPAC name, and stores them on the instance
        (self.cid, self.iupac_name).
    
        This is the reverse of `from_cid`: instead of building the molecule
        from a known CID, it looks up an already-built molecule (e.g. created
        via `easy_rdkit("CCO")`) to see if it exists in PubChem.
    
        Only the first match returned by PubChem is used. If the SMILES is
        ambiguous (e.g. corresponds to several close PubChem entries such as
        tautomers or salts), this may not be the expected compound — check
        self.cid against the PubChem page if in doubt.
    
        Notes:
            - Requires an internet connection (PubChem REST API via pubchempy).
            - Does not raise on failure: if no match is found, or if the
              request fails, a warning/error is printed and self.cid /
              self.iupac_name are left unchanged.
        """
        import pubchempy as pcp
        try:
            # We search PubChem using the SMILES string
            results = pcp.get_compounds(self.smiles, namespace='smiles')
            if results:
                compound = results[0]
                self.cid = compound.cid
                self.iupac_name = compound.iupac_name
                print(f"✅ PubChem match found: {self.iupac_name} (CID: {self.cid})")
            else:
                print(f"⚠️ No PubChem match found for SMILES: {self.smiles}")
        except Exception as e:
            print(f"❌ Error syncing with PubChem: {e}")

    @classmethod
    def from_name(cls, name, canonical=True, lang="En"):
        """
        Builds an easy_rdkit instance directly from a compound name, by
        querying PubChem (e.g. "aspirin", "caffeine", "benzene").
    
        Args:
            name (str): Compound name to search for on PubChem.
            canonical (bool): Passed through to __init__.
            lang (str): Passed through to __init__ ("En" or "Fr").
    
        Returns:
            easy_rdkit: The constructed instance, with self.cid and
            self.iupac_name set from the first PubChem match.
    
        Raises:
            ValueError: if no PubChem compound matches the given name.
            ConnectionError: if the PubChem request fails (network/API issue).
    
        Note:
            If several compounds match the name, only the first PubChem
            result is used — check self.cid if the match seems ambiguous.
        """
        import pubchempy as pcp
        try:
            compounds = pcp.get_compounds(name, namespace='name')
        except Exception as e:
            raise ConnectionError(f"Error fetching '{name}' from PubChem: {e}") from e
    
        if not compounds:
            raise ValueError(f"No PubChem match found for name: {name}")
    
        compound = compounds[0]
        print(f"✅ Successfully retrieved: {compound.iupac_name}")
        instance = cls(compound.connectivity_smiles, canonical=canonical, lang=lang)
        instance.cid = compound.cid
        instance.iupac_name = compound.iupac_name
        return instance
    
    @property
    def descriptors(self):
        """
        Computes (once, then caches) a set of standard molecular descriptors.
    
        Returns:
            dict: MW, LogP, QED, HBA, HBD, RotB (rotatable bonds), TPSA,
            aromatic/non-aromatic ring counts, Fsp3, and connectivity
            indices (Chi0, Chi1). Values are rounded for display.
    
        Note:
            Computed lazily on first access and cached in
            self._descriptors_cache; if self.mol changes afterward (e.g. via
            show_mol adding explicit Hs), the cache is NOT automatically
            invalidated — delete self._descriptors_cache if you need a refresh.
        """
        if not hasattr(self, '_descriptors_cache') or self._descriptors_cache is None:
            mol = self.mol
            self._descriptors_cache = {
                "MW": round(Descriptors.MolWt(mol), 2),
                "LogP": round(Descriptors.MolLogP(mol), 2),
                "QED": round(QED.qed(mol), 3),
                "HBA": Descriptors.NumHAcceptors(mol),
                "HBD": Descriptors.NumHDonors(mol),
                "RotB": Descriptors.NumRotatableBonds(mol),
                "TPSA": round(Descriptors.TPSA(mol), 1),
                "Aromatic Rings": rdMolDescriptors.CalcNumAromaticRings(mol),
                "Non-Aromatic Rings": rdMolDescriptors.CalcNumRings(mol) - rdMolDescriptors.CalcNumAromaticRings(mol),
                "Fsp3": round(rdMolDescriptors.CalcFractionCSP3(mol), 3),
                "Connectivity (Chi0)": round(Descriptors.Chi0(mol), 3),
                "Connectivity (Chi1)": round(Descriptors.Chi1(mol), 3)
            }
        return self._descriptors_cache

    def show_descriptors(self):
        """
        Prints the molecular descriptors (see `descriptors`) as a simple
        formatted table to stdout. Display-only; returns nothing.
        """
        d = self.descriptors
        print(f"--------- Molecular Descriptors ---")
        print(f"{'Property':<20} | {'Value':<10}")
        print("-" * 35)
        for prop, value in d.items():
            print(f"{prop:<20} | {value:<10}")
        print("-" * 35)

    def to_dict(self, auto_fetch=False):
        """
        Exports identity and descriptors as a flat dictionary, ready for use
        in a DataFrame row (e.g. with plot_grid_from_df).
    
        Args:
            auto_fetch (bool): If True and self.cid is not already set,
                calls fetch_pubchem_data() first to try to retrieve the
                CID/IUPAC name before building the dictionary.
    
        Returns:
            dict: {"IUPAC Name", "CID", "SMILES"} plus all keys from
            `descriptors` (MW, LogP, QED, ...). CID/IUPAC Name are "N/A"
            if unavailable.
        """
        if auto_fetch and self.cid is None:
            self.fetch_pubchem_data()

        data = {
            "IUPAC Name": getattr(self, 'iupac_name', "N/A"),
            "CID": self.cid if self.cid else "N/A",
            "SMILES": self.smiles
        }
        # Merges the descriptors (MW, LogP, Chi, etc.)
        data.update(self.descriptors)
        return data
        
    def analyze_lewis(self, heavy_only=True):
        """
        Performs a simplified Lewis-structure analysis of the molecule, atom
        by atom, based on RDKit's valence/bond perception (no 3D geometry
        involved).
    
        For each atom, derives:
            - valence electrons (from the periodic table)
            - bonding electrons (RDKit's total valence)
            - formal charge
            - lone pairs (nonbonding electrons / 2)
            - vacancies (missing electrons to reach octet/duet, if any)
            - total electron count and whether the octet (or duet, for H/He)
              rule is satisfied, electron-deficient, or hypervalent
    
        This is a teaching aid, not a rigorous electronic-structure
        calculation: it relies on RDKit's bond-order perception from the
        SMILES, so it reflects whatever Lewis structure RDKit infers and will
        not be meaningful for molecules with unusual/ambiguous valence (e.g.
        hypervalent or radical species) beyond the simple flags above.
    
        Args:
            heavy_only (bool): If True (default), only heavy atoms (Z > 2)
                get a row in the returned table. H atoms are checked for
                duet compliance on an internal copy of the molecule with
                explicit Hs (self.mol itself is not modified), and a single
                summary line is printed after the table (✅ if all H satisfy
                the duet rule, ❌ with the offending atom indices otherwise).
                If False, H/He atoms are included in the table with full
                LP/vacancy detail, like heavy atoms.
    
        Returns:
            pd.DataFrame: One row per analyzed atom, with columns depending
            on self.lang ("En": Atom Index, Symbol, Valence e-, Bonding e-,
            Formal Charge, Lone Pairs (LP), Vacancies ([]), Number of Bonds,
            Total e- (octet?), Octet Status (O/H/D); "Fr": French equivalents).
            H/He atoms are excluded from the rows when heavy_only is True.
    
        Raises:
            ValueError: if self.mol is None (invalid SMILES), with a message
            in English or French depending on self.lang.
        """
        if self.mol is None and self.lang == "En":
            raise ValueError(f"Invalid molecule for {self.smiles} (Check if SMILES is correct) ")
        if self.mol is None and self.lang == "Fr":
            raise ValueError(f"Molécule invalide pour {self.smiles} (Vérifier si le SMILES est correct) ")
    
        pt = GetPeriodicTable()
        rows = []
    
        # Use an explicit-H copy for the loop so implicit H are visible,
        # but only when heavy_only needs to check them separately;
        # self.mol is never modified here.
        work_mol = Chem.AddHs(self.mol) if heavy_only else self.mol
        h_duet_ok = []  # [(atom_idx, bool), ...] for H atoms, only used if heavy_only
    
        for atom in work_mol.GetAtoms():
            Z = atom.GetAtomicNum()
            valence_e = pt.GetNOuterElecs(Z)
            bonding_e = atom.GetTotalValence()
            formal_charge = atom.GetFormalCharge()
            num_bonds = int(sum(bond.GetBondTypeAsDouble() for bond in atom.GetBonds()))
            nonbonding = valence_e - bonding_e - formal_charge
            lone_pairs = max(0, nonbonding // 2)
    
            target = 2 if Z in (1, 2) else 8
            missing_e = max(0, target/2 - (bonding_e + 2*lone_pairs))
            vacancies = int(missing_e)
            total_e = 2*(lone_pairs + bonding_e)
    
            if heavy_only and Z in (1, 2):
                h_duet_ok.append((atom.GetIdx(), total_e == 2))
                continue  # H excluded from the table
    
            if self.lang == 'En':
                if total_e > 8:
                    octet_msg = "❌ hypervalent"
                elif total_e < 8 and Z > 2:
                    octet_msg = "❌ electron-deficient"
                elif total_e == 8:
                    octet_msg = "✅ octet"
                elif total_e == 2 and (Z == 1 or Z == 2):
                    octet_msg = "✅ duet"
                else:
                    octet_msg = "🤔"
    
                rows.append({
                    "Atom Index": atom.GetIdx(),
                    "Symbol": atom.GetSymbol(),
                    "Valence e-": valence_e,
                    "Bonding e-": bonding_e,
                    "Formal Charge": formal_charge,
                    "Lone Pairs (LP)": lone_pairs,
                    "Vacancies ([])": vacancies,
                    "Number of Bonds": num_bonds,
                    "Total e- (octet?)": total_e,
                    "Octet Status (O/H/D)": octet_msg
                })
            elif self.lang == "Fr":
                if total_e > 8:
                    octet_msg = "❌ hypercoordiné"
                elif total_e < 8 and Z > 2:
                    octet_msg = "❌ électron-déficient"
                elif total_e == 8:
                    octet_msg = "✅ octet"
                elif total_e == 2 and (Z == 1 or Z == 2):
                    octet_msg = "✅ duet"
                else:
                    octet_msg = "🤔"
                rows.append({
                    "index atome": atom.GetIdx(),
                    "symbole": atom.GetSymbol(),
                    "e- valence": valence_e,
                    "e- liants": bonding_e,
                    "charge formelle": formal_charge,
                    "doublets non-liants (DNL)": lone_pairs,
                    "lacunes ([])": vacancies,
                    "nombre de liaisons": num_bonds,
                    "e- total (octet ?)": total_e,
                    "O/H/D ?": octet_msg
                })
    
        df = pd.DataFrame(rows)
    
        if heavy_only and h_duet_ok:
            all_ok = all(ok for _, ok in h_duet_ok)
            bad = [idx for idx, ok in h_duet_ok if not ok]
            if self.lang == "Fr":
                if all_ok:
                    print(f"{fg.GREEN}✅ Tous les atomes H ({len(h_duet_ok)}) respectent la règle du duet.{fg.OFF}")
                else:
                    print(f"{fg.RED}❌ Atomes H ne respectant pas le duet (indices) : {bad}{fg.OFF}")
            else:
                if all_ok:
                    print(f"{fg.GREEN}✅ All H atoms ({len(h_duet_ok)}) satisfy the duet rule.{fg.OFF}")
                else:
                    print(f"{fg.RED}❌ H atoms not satisfying the duet rule (indices): {bad}{fg.OFF}")
    
        return df
            
    def show_mol(self,
                 size: tuple=(400,400),
                 show_Lewis: bool=False,
                 plot_conjugation: bool=False,
                 plot_aromatic: bool=False,
                 show_n: bool=False,
                 show_hybrid: bool=False,
                 show_H: bool=False,
                 show_stereo: bool=False,
                 rep3D: bool=False,
                 macrocycle: bool=False,
                 highlightAtoms: list=[],
                 legend: str='',
                 save_img: str=None
                ):
        """
        Renders the molecule as a 2D (SVG or PNG) depiction, with optional
        annotations, and displays it inline (Jupyter/Colab). Can also save
        the image to disk.
    
        This is the main "do everything" drawing method of the class: most
        arguments toggle an independent overlay or computation that gets
        combined onto the same drawing. A few interactions to know about:
    
            - show_Lewis implies show_H (explicit Hs are added so lone pairs/
              vacancies can be computed and displayed) and internally calls
              analyze_lewis(), which also prints/displays its DataFrame.
            - show_Lewis and show_hybrid share the same atom-note area (the
              little text under each atom); both can be shown together.
            - rep3D adds explicit Hs and computes a 3D-like embedding
              (AllChem.EmbedMolecule) before generating a (still 2D) drawing
              from it — mainly useful to get a more realistic bond geometry.
            - plot_conjugation and plot_aromatic each highlight bonds/atoms
              in a different color (blue / red) but use the same
              highlighting machinery, so combining both silently uses
              whichever was computed last (not an additive overlay).
            - show_stereo perceives and labels R/S and E/Z. If any chiral
              center or double bond is unassigned ("?"), it prints a warning
              and automatically turns on atom (and, for double bonds, bond)
              index display so you can identify what needs fixing in the
              SMILES.
            - Side effect: several options (show_H, show_Lewis, rep3D,
              macrocycle) mutate self.mol in place (adding Hs, 3D coords,
              atom notes), so self.mol may differ after calling show_mol
              even without save_img.
    
        Args:
            size (tuple): Drawing dimensions in pixels (width, height).
            show_Lewis (bool): Annotate atoms with lone pairs / vacancies
                (implies show_H; runs analyze_lewis() and displays its table).
            plot_conjugation (bool): Highlight conjugated bonds in blue.
            plot_aromatic (bool): Highlight aromatic atoms/bonds in red.
            show_n (bool): Display atom indices.
            show_hybrid (bool): Display atom hybridization (sp, sp2, sp3...).
            show_H (bool): Add explicit hydrogens to the drawing.
            show_stereo (bool): Compute and display R/S, E/Z labels; warns
                and auto-enables index display if stereochemistry is
                undefined in the input SMILES.
            rep3D (bool): Embed a 3D-like conformation (via AllChem) before
                drawing, for more realistic bond angles.
            macrocycle (bool): Use CoordGen for better layout of large rings
                (e.g. cyclodextrins).
            highlightAtoms (list): Atom indices to highlight (ignored if
                plot_conjugation or plot_aromatic is used).
            legend (str): Caption displayed under the structure.
            save_img (str, optional): Output file path; extension decides
                format ('.svg' or '.png'). Image is also always displayed
                inline regardless of this argument.
    
        Returns:
            None. Displays the image (IPython.display.SVG) as a side effect;
            if save_img ends in '.png', also regenerates and displays an SVG
            version for screen viewing (via a recursive call).
        """

        if show_H:
            mol = Chem.AddHs(self.mol)
            self.mol = mol
        else:
            mol = self.mol
        
        if show_Lewis:
            # heavy_only=True par défaut (show_H=False) : seuls les atomes
            # lourds sont détaillés, les H sont juste vérifiés (duet) et
            # résumés en un message. Si show_H=True, détail complet H inclus.
            df = self.analyze_lewis(heavy_only=not show_H)
            if self.lang == "Fr":
                lewis_info = {row["index atome"]: (row["doublets non-liants (DNL)"], row["lacunes ([])"])
                              for _, row in df.iterrows()}
            elif self.lang == "En":
                lewis_info = {row["Atom Index"]: (row["Lone Pairs (LP)"], row["Vacancies ([])"])
                              for _, row in df.iterrows()}
        else:
            df = None
            
        if rep3D:
            mol = Chem.AddHs(self.mol)
            self.mol = mol
            AllChem.EmbedMolecule(mol)

        if macrocycle:
            rdCoordGen.AddCoords(self.mol)
                
        # 2. Define Extension and Drawer
        ext = Path(save_img).suffix.lower() if save_img else ".svg"
        
        if ext == ".png":
            d2d = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
        else:
            d2d = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
        
        atoms = list(mol.GetAtoms())
    
        if plot_conjugation:
            from collections import defaultdict
            Chem.SetConjugation(mol)
            colors = [(0.0, 0.0, 1.0, 0.4)]
            athighlights = defaultdict(list)
            arads = {}
            bndhighlights = defaultdict(list)
            for bond in mol.GetBonds():
                aid1 = bond.GetBeginAtomIdx()
                aid2 = bond.GetEndAtomIdx()
            
                if bond.GetIsConjugated():
                    bid = mol.GetBondBetweenAtoms(aid1,aid2).GetIdx()
                    bndhighlights[bid].append(colors[0])
            
        if plot_aromatic:
            from collections import defaultdict
            colors = [(1.0, 0.0, 0.0, 0.4)]
            athighlights = defaultdict(list)
            arads = {}
            for a in atoms:
                if a.GetIsAromatic():
                    aid = a.GetIdx()
                    athighlights[aid].append(colors[0])
                    arads[aid] = 0.3
                    
            bndhighlights = defaultdict(list)
            for bond in mol.GetBonds():
                aid1 = bond.GetBeginAtomIdx()
                aid2 = bond.GetEndAtomIdx()
            
                if bond.GetIsAromatic():
                    bid = mol.GetBondBetweenAtoms(aid1,aid2).GetIdx()
                    bndhighlights[bid].append(colors[0])
            
        if show_hybrid or show_Lewis:
            for i,atom in enumerate(atoms):
                # print(i,atom.GetDegree(),atom.GetImplicitValence())
                note_parts = []
                if show_hybrid and(atom.GetValence(rdkit.Chem.rdchem.ValenceType.IMPLICIT) > 0 or atom.GetDegree() > 1):
                    note_parts.append(str(atom.GetHybridization()))
                if show_Lewis and i in lewis_info:
                    lp, vac = lewis_info[i]
                    if lp > 0:
                        if self.lang == "Fr":
                            note_parts.append(f" {lp}DNL")
                        elif self.lang == "En":
                            note_parts.append(f" {lp}LP")
                    if vac > 0:
                        note_parts.append(f" {vac}[]")
                if note_parts:
                    mol.GetAtomWithIdx(i).SetProp('atomNote',"".join(note_parts))
                # print(f"Atom {i+1:3}: {atom.GetAtomicNum():3} {atom.GetSymbol():>2} {atom.GetHybridization()}")
            if show_Lewis:
                display(df)
                
        ##### Drawing Block
        # if show_n:
        #     d2d.drawOptions().addAtomIndices=show_n
    
        # if plot_aromatic or plot_conjugation:
        #     d2d.DrawMoleculeWithHighlights(mol,legend,dict(athighlights),dict(bndhighlights),arads,{})
        # else:
        #     d2d.DrawMolecule(mol,legend=legend, highlightAtoms=highlightAtoms)

        # if show_stereo:
        #     # This prepares the molecule for stereo display (R/S and E/Z labels)
        #     Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
        #     # Force RDKit to calculate E/Z for the drawing engine
        #     d2d.drawOptions().addStereoAnnotation = True
        #     rdMolDraw2D.PrepareAndDrawMolecule(d2d, mol, legend=legend, highlightAtoms=highlightAtoms)
            
        # d2d.FinishDrawing()

        # 1. SET GLOBAL OPTIONS
        opts = d2d.drawOptions()
        opts.addAtomIndices = show_n
        
        if show_stereo:
            # 1. Perception (You have this, keep it!)
            Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
            Chem.FindPotentialStereoBonds(mol)

            # 2. Find chiral centers
            chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
            # print(f"DEBUG: Chiral Centers found: {chiral_centers}")
            
            # 3. Find potential stereogenic double bonds
            potential_db_indices = []
            unassigned_db = []
            
            for bond in mol.GetBonds():
                if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                    # After FindPotentialStereoBonds, RDKit marks stereogenic bonds
                    # even if they aren't assigned yet.
                    stereo = bond.GetStereo()
                    
                    # If it's STEREONONE but the atoms have enough neighbors, it's a candidate
                    if bond.GetBeginAtom().GetDegree() > 1 and bond.GetEndAtom().GetDegree() > 1:
                        potential_db_indices.append(bond.GetIdx())
                        
                        # If RDKit hasn't found a specific E or Z, it's unassigned
                        if stereo in [Chem.rdchem.BondStereo.STEREONONE, Chem.rdchem.BondStereo.STEREOANY]:
                            unassigned_db.append(bond.GetIdx())

            # for idx in potential_db_indices:
            #     b = mol.GetBondWithIdx(idx)
            #     print(f"DEBUG: Bond {idx} Type: {b.GetBondType()} | Stereo: {b.GetStereo()}")

            # --- CASE 0: IRRELEVANT ---
            if not chiral_centers and not potential_db_indices:
                msg = "ℹ️  Note: Stereochemistry is irrelevant for this molecule."
                print(f"{fg.CYAN}{msg}{fg.OFF}")

            # --- CASE 1: WARNING ---
            else:
                unassigned_chiral = [idx for idx, config in chiral_centers if config == "?"]
                
                if unassigned_chiral or unassigned_db:
                    msg = "⚠️  Warning: This SMILES contains undefined stereochemistry.\n"
                    if unassigned_chiral:
                        msg += f"   - Unassigned Chiral Centers (atoms): {unassigned_chiral}\n"
                    if unassigned_db:
                        msg += f"   - Unassigned Double Bond geometry (bonds): {unassigned_db}\n"
                    
                    # --- DYNAMIC FOOTER MESSAGE ---
                    if unassigned_chiral and unassigned_db:
                        msg += "   Labels (R/S, E/Z) cannot be displayed for undefined centers and bonds."
                    elif unassigned_chiral:
                        msg += "   Labels (R/S) cannot be displayed for undefined centers."
                    else: # only unassigned_db
                        msg += "   Labels (E/Z) cannot be displayed for undefined bonds."
                    
                    print(f"{fg.RED}{msg}{fg.OFF}")
                    opts.addAtomIndices = True
                    if unassigned_db:
                        opts.addBondIndices = True # Show bond numbers for E/Z geometry
                        print(f"{fg.CYAN}   -> Atom and Bond indices enabled to help you locate issues.{fg.OFF}")
                else:
                    print(f"{fg.GREEN}✅ Stereochemistry is fully defined.{fg.OFF}")

            # Always apply the annotation option if the user asked for it
            opts.addStereoAnnotation = True
            rdMolDraw2D.PrepareMolForDrawing(mol)
        # 3. SELECT THE DRAWING COMMAND
        if plot_aromatic or plot_conjugation:
            # This method supports stereo labels if opts.addStereoAnnotation is True
            d2d.DrawMoleculeWithHighlights(
                mol, legend, dict(athighlights), dict(bndhighlights), arads, {}
            )
        else:
            # Standard drawing (also supports stereo labels)
            d2d.DrawMolecule(mol, legend=legend, highlightAtoms=highlightAtoms)

        d2d.FinishDrawing()
        
        ##### Save Image Block
        if save_img:
            save_path = Path(save_img)
            if save_path.parent:
                save_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_img, 'wb' if ext == ".png" else 'w') as f:
                content = d2d.GetDrawingText()
                f.write(content)
            print(f"✅ Image saved to: {save_img}")

        # 3. Affichage Jupyter (toujours en SVG pour la qualité)
        if ext == ".png":
            # Si on a sauvé en PNG, on regénère un SVG pour l'affichage écran
            return self.show_mol(size=size, show_Lewis=show_Lewis, show_H=show_H, legend=legend)
        else:
            display(SVG(d2d.GetDrawingText()))
        return

    @staticmethod
    def plot_grid_from_df(df, smiles_col='SMILES', legend_cols='IUPAC Name',
                          mols_per_row=4, size=(250, 250), show_stereo=False, save_img=None):
        """
        Generates a grid image of molecular structures from a pandas DataFrame.

        This method extracts SMILES strings from the specified column, generates 2D 
        representations, and arranges them in a grid. It supports multi-line legends 
        by passing a list of column names, and can export the result to external files.

        Args:
            df (pd.DataFrame): DataFrame containing the molecular data.
            smiles_col (str): Name of the column containing SMILES strings. 
                Defaults to 'SMILES'.
            legend_cols (str or list): Column name(s) to display as legends 
                below each molecule. If a list is provided, each value is 
                displayed on a new line (e.g., ['IUPAC Name', 'MW', 'LogP']). 
                Defaults to 'IUPAC Name'.
            mols_per_row (int): Number of molecules to display in each row. 
                Defaults to 4.
            size (tuple): Dimensions (width, height) in pixels for each 
                individual molecule panel. Defaults to (250, 250).
            save_img (str, optional): File path to save the resulting image. 
                Supports '.svg' (vector) and '.png' (raster) extensions. 
                Defaults to None.

        Returns:
            IPython.display.SVG: An SVG object for rich display in Jupyter/Colab 
            environments.

        Note:
            PNG export requires the Cairo backend to be available in the 
            RDKit installation.
        """
        from rdkit.Chem import Draw, rdCoordGen
        from rdkit.Chem.Draw import rdMolDraw2D
        
        mols = []
        legends = []
        
        if isinstance(legend_cols, str):
            cols_to_use = [legend_cols]
        else:
            cols_to_use = legend_cols

        for _, row in df.iterrows():
            smiles = row[smiles_col]
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                rdCoordGen.AddCoords(mol)
                # --- NEW STEREO PERCEPTION FOR GRID ---
                if show_stereo:
                    Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
                    Chem.FindPotentialStereoBonds(mol)
                    # This ensures the R/S and E/Z labels are calculated
                    
                mols.append(mol)
                
                # Build legend: use a separator that helps RDKit spacing
                lines = []
                for col in cols_to_use:
                    if col in df.columns:
                        val = row[col]
                        val_str = f"{val:.2f}" if isinstance(val, float) else f"{val}"
                        lines.append(f"{col}: {val_str}")
                
                # Joining with a double newline or space can sometimes help
                legends.append("\n".join(lines))
            else:
                print(f"⚠️ Skipping invalid SMILES: {smiles}")

        # Set Draw Options for better legibility
        dopts = rdMolDraw2D.MolDrawOptions()
        dopts.legendFontSize = 14  # Adjust this to change legend size
        dopts.padding = 0.15      # Adds room around the molecule
        if show_stereo:
            dopts.addStereoAnnotation = True

        # --- NEW SAVING LOGIC STARTS HERE ---
        if save_img:
            ext = os.path.splitext(save_img)[1].lower()
            # 1. Ensure the directory exists (Works for Colab and Local)
            save_path = Path(save_img)
            if save_path.parent:
                save_path.parent.mkdir(parents=True, exist_ok=True)
            
            ext = save_path.suffix.lower()
            
            # Generate SVG specifically for saving
            svg_to_save = Draw.MolsToGridImage(
                    mols, molsPerRow=mols_per_row, subImgSize=size, 
                    legends=legends, useSVG=True, drawOptions=dopts
            )
            svg_data = svg_to_save.data if hasattr(svg_to_save, 'data') else svg_to_save
                
            if ext == '.svg':
                with open(save_img, 'w') as f:
                    f.write(svg_data)
                print(f"✅ SVG saved to: {svg_to_save}")

            elif ext == '.png':
                import math
                # 1. Calculate the grid dimensions
                n_rows = math.ceil(len(mols) / mols_per_row)
                
                # 2. Initialize the Cairo drawer with PANEL sizes (the 3rd and 4th arguments)
                # width, height, panelWidth, panelHeight
                drawer = rdMolDraw2D.MolDraw2DCairo(
                    mols_per_row * size[0], 
                    n_rows * size[1], 
                    size[0], 
                    size[1]
                )
                
                # 3. Apply your exact same drawing options
                drawer.SetDrawOptions(dopts)
                
                # 4. Draw the molecules - RDKit now knows it's a grid because of the panelWidth/Height
                drawer.DrawMolecules(mols, legends=legends)
                drawer.FinishDrawing()
                
                # 5. Write the binary data
                with open(save_img, 'wb') as f:
                    f.write(drawer.GetDrawingText())
                print(f"✅ PNG saved to: {save_img}")

        grid_img = Draw.MolsToGridImage(
            mols, 
            molsPerRow=mols_per_row, 
            subImgSize=size, 
            legends=legends,
            useSVG=True,
            drawOptions=dopts # Apply the spacing options here
        )

        return grid_img
    
    def analyze_stereochemistry(self):
        """
        Summarizes the stereochemical features of the molecule as currently
        perceived by RDKit (chiral centers and stereogenic double bonds).
    
        Unlike `get_isomers`, this does not enumerate alternative
        configurations — it only reports what is already encoded (or left
        unassigned, i.e. "?") in self.mol.
    
        Returns:
            dict: Dictionary with keys ``chiral_centers_count`` (int),
            ``chiral_centers`` (list of (atom_index, "R"/"S"/"?")),
            ``stereo_bonds_count`` (int), and ``stereo_bonds``
            (list of (begin_atom_idx, end_atom_idx, "STEREOE"/"STEREOZ"/...)).
        """
        # Find chiral centers (includes centers with unassigned stereo)
        chiral_centers = Chem.FindMolChiralCenters(self.mol, includeUnassigned=True)
        
        # Identify stereogenic bonds (E/Z)
        stereo_bonds = []
        for bond in self.mol.GetBonds():
            st = bond.GetStereo()
            if st != Chem.rdchem.BondStereo.STEREONONE:
                stereo_bonds.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), st.name))
        
        return {
            "chiral_centers_count": len(chiral_centers),
            "chiral_centers": chiral_centers, # List of (atom_index, "R/S/?")
            "stereo_bonds_count": len(stereo_bonds),
            "stereo_bonds": stereo_bonds
        }

    def get_isomers(self, max_isomers=12, verbose=True):
        """
        Explores the stereochemical space of the molecule by enumerating all possible stereoisomers.
        
        This method identifies all unassigned or flexible stereocenters (chiral centers 
        and double bonds) and generates a complete set of discrete stereochemical 
        configurations. It is particularly useful for resolving "flat" SMILES strings 
        into their constituent enantiomers and diastereomers.

        Args:
            max_isomers (int): The maximum number of isomers to generate. This prevents 
                computational explosion for molecules with many stereocenters (2^n). 
                Defaults to 12.
            verbose (bool): If True, prints a summary of the number of isomers found 
                using a colored terminal message. Defaults to True.

        Returns:
            list: A list of easy_rdkit instances, each representing a unique, 
                fully-defined stereoisomer of the parent molecule.
        """
        from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
    
        opts = StereoEnumerationOptions(tryEmbedding=True)
        isomers_mols = list(EnumerateStereoisomers(self.mol, options=opts))
    
        if verbose:
            print(f"{fg.CYAN}✨ {len(isomers_mols)} stereoisomers found for {self.smiles}{fg.OFF}")
    
        return [easy_rdkit(Chem.MolToSmiles(iso), lang=self.lang) for iso in isomers_mols[:max_isomers]]

    def show_isomers(self, mols_per_row=4, size=(250, 250), save_img=None):
        """
        Generates, labels, and displays a grid of all possible stereoisomers.
        
        This method automates the transition from a single chemical identity to a 
        visual comparative analysis of its stereoisomers. Each isomer is rendered 
        with its specific SMILES string (including @ markers and / directionals) 
        and an isomer index. It utilizes the class's internal grid-plotting logic 
        to ensure high-quality SVG output and optional file export.

        Args:
            mols_per_row (int): Number of isomer structures to display per row 
                in the grid. Defaults to 4.
            size (tuple): The (width, height) in pixels for each individual 
                molecular panel. Defaults to (250, 250).
            save_img (str, optional): File path (e.g., 'isomers.svg' or 'isomers.png') 
                to export the grid. Directories are created automatically. 
                Defaults to None.

        Returns:
            IPython.display.SVG: A grid image displayed directly in the Jupyter/Colab 
                environment.
        """
        # 1. Generate the objects
        isomers = self.get_isomers(verbose=False)
        
        # 2. Build a temporary DataFrame to use our grid visualizer
        # We include the SMILES so the user can see the slashes/chiral markers
        data = []
        for i, iso in enumerate(isomers):
            d = iso.to_dict()
            d["Isomer #"] = i
            data.append(d)
        
        df_isomers = pd.DataFrame(data)
        
        # 3. Use the static method to plot
        grid = self.plot_grid_from_df(
            df_isomers, 
            legend_cols=['Isomer #'], 
            mols_per_row=mols_per_row, 
            size=size, 
            show_stereo=True,
            save_img=save_img,
        )
        display(grid)
        
        # 3. Return the object so the user can still use it if they want
        return grid
        
    @staticmethod
    def draw_consistent_svg(smiles_list, legends=None, size_per_mol=300, mols_per_row=3,
                            bond_length=35.0, bond_width=1.2, font_size=12,
                            save_img="export/molecules.svg",
                           ):
        """
        Generates an SVG file with a grid of molecules having consistent bond lengths.
        Ideal for high-quality printing and comparison.
        
        Args:
            smiles_list (list): List of SMILES strings to draw.
            legends (list, optional): List of strings to display under each molecule.
            size_per_mol (int): Size of the square panel for each molecule (Default: 300).
            mols_per_row (int): Number of molecules per row (Default: 3).
            bond_length (float): Fixed length for bonds in pixels (Default: 35.0).
            bond_width (float): Thickness of the bonds (Default: 1.2).
            font_size (int): Font size for atom labels (Default: 12).
            save_img (str): Output SVG file path (Default: "export/molecules.svg").
        """
        mols = []
        for s in smiles_list:
            m = Chem.MolFromSmiles(s)
            if m:
                # Use CoordGen for a more aesthetic rendering if available
                rdCoordGen.AddCoords(m)
                mols.append(m)

        if not mols:
            print("⚠️ No valid molecules to draw.")
            return
            
        if legends is None:
            legends = [""] * len(mols)
        elif len(legends) < len(mols):
            legends.extend([""] * (len(mols) - len(legends)))

        # Calculate canvas dimensions
        n_mols = len(mols)
        n_rows = math.ceil(n_mols / mols_per_row)
        width = size_per_mol * mols_per_row
        height = size_per_mol * n_rows

        # Initialize the SVG drawer (Width, Height, PanelWidth, PanelHeight)
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height, size_per_mol, size_per_mol)
        options = drawer.drawOptions()

        # --- Consistency parameters ---
        options.fixedBondLength = bond_length
        options.bondLineWidth = bond_width
        options.addStereoAnnotation = True
        options.prepareMolsBeforeDrawing = True # Handle aromaticity and labels properly
        options.legendFontSize = font_size
        drawer.SetFontSize(font_size)

        # Draw the complete grid
        drawer.DrawMolecules(mols, legends=legends)
        drawer.FinishDrawing()

        # Save the file
        save_path = Path(save_img)
        if save_path.parent:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
        with open(save_img, "w") as f:
            f.write(drawer.GetDrawingText())
        
        print(f"✅ Consistent SVG saved to: {save_img}")
        
        # display in Jupyter
        return SVG(drawer.GetDrawingText())