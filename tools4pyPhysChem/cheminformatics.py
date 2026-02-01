############################################################
#                       easy_rdkit
############################################################
from .visualID_Eng import fg, bg, hl
from .core import centerTitle, centertxt

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, GetPeriodicTable, Draw, rdCoordGen
import pandas as pd
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import SVG
from PIL import Image

class easy_rdkit():
    """
    A helper class to analyze and visualize molecules using RDKit.
    Provides tools for Lewis structure analysis and advanced 2D drawing.
    """

    def __init__(self,smiles, canonical=True):
        """
        Initialize the molecule object from a SMILES string.
        
        Args:
            smiles (str): The SMILES representation of the molecule.
            canonical (bool): If True, converts the SMILES to its canonical form 
                              to ensure consistent atom numbering and uniqueness.
        """
        from rdkit import Chem
        
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
        
    def analyze_lewis(self):
        """
        Performs a Lewis structure analysis for each atom in the molecule.
        Calculates valence electrons, lone pairs, formal charges, and octet rule compliance.
        
        Returns:
            pd.DataFrame: A table containing detailed Lewis electronic data per atom.
        """
        if self.mol is None:
            raise ValueError(f"Molécule invalide pour {self.smiles} (SMILES incorrect ?)")
        
        pt = GetPeriodicTable()
        rows = []
    
        for atom in self.mol.GetAtoms():
            Z = atom.GetAtomicNum()
            valence_e = pt.GetNOuterElecs(Z)
            bonding_e = atom.GetTotalValence()
            formal_charge = atom.GetFormalCharge()
            num_bonds = int(sum(bond.GetBondTypeAsDouble() for bond in atom.GetBonds()))
            # hybridization = atom.GetHybridization()
            nonbonding = valence_e - bonding_e - formal_charge
    
            lone_pairs = max(0, nonbonding // 2)
    
            if Z==1 or Z==2:  # règle du duet
                target = 2
            else:       # règle de l’octet
                target = 8
    
            missing_e = max(0, target/2 - (bonding_e + 2*lone_pairs))
            vacancies = int(missing_e)
            total_e = 2*(lone_pairs + bonding_e)
    
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
        return pd.DataFrame(rows)    
            
    def show_mol(self,
                 size: tuple=(400,400),
                 show_Lewis: bool=False,
                 plot_conjugation: bool=False,
                 plot_aromatic: bool=False,
                 show_n: bool=False,
                 show_hybrid: bool=False,
                 show_H: bool=False,
                 rep3D: bool=False,
                 macrocycle: bool=False,
                 highlightAtoms: list=[],
                 legend: str=''
                ):
        """
        Renders the molecule in 2D SVG format with optional property overlays.
        
        Args:
            size (tuple): Drawing dimensions in pixels.
            show_Lewis (bool): Annotates atoms with Lone Pairs and Vacancies.
            plot_conjugation (bool): Highlights conjugated bonds in blue.
            plot_aromatic (bool): Highlights aromatic rings in red.
            show_n (bool): Displays atom indices.
            show_hybrid (bool): Displays atom hybridization (sp3, sp2, etc.).
            show_H (bool): Adds explicit Hydrogens to the drawing.
            rep3D (bool): Computes a 3D-like conformation before drawing.
            macrocycle (bool): Uses CoordGen for better rendering of large rings (e.g., Cyclodextrins).
            highlightAtoms (list): List of indices to highlight.
            legend (str): Title or legend text for the drawing.
        """

        def safe_add_hs():
            try:
                return Chem.AddHs(self.mol)
            except Exception as e:
                print(f"[Warning] Impossible d'ajouter les H pour {self.smiles} ({e}), on garde la version brute.")
                return mol      
        
        if show_H and not show_Lewis:
            mol = Chem.AddHs(self.mol)
        else:
            mol = self.mol
        if show_Lewis:
            mol = safe_add_hs()
            self.mol = mol
            df = self.analyze_lewis()
            lewis_info = {row["index atome"]: (row["doublets non-liants (DNL)"], row["lacunes ([])"])
                          for _, row in df.iterrows()}
        else:
            df = None
            
        if rep3D:
            mol = Chem.AddHs(self.mol)
            self.mol = mol
            AllChem.EmbedMolecule(mol)

        if macrocycle:
            rdCoordGen.AddCoords(self.mol)
                
        d2d = rdMolDraw2D.MolDraw2DSVG(size[0],size[1])
        
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
                        note_parts.append(f" {lp}DNL")
                    if vac > 0:
                        note_parts.append(f" {vac}[]")
                if note_parts:
                    mol.GetAtomWithIdx(i).SetProp('atomNote',"".join(note_parts))
                # print(f"Atom {i+1:3}: {atom.GetAtomicNum():3} {atom.GetSymbol():>2} {atom.GetHybridization()}")
            if show_Lewis:
                display(df)
    
        if show_n:
            d2d.drawOptions().addAtomIndices=show_n
    
        if plot_aromatic or plot_conjugation:
            d2d.DrawMoleculeWithHighlights(mol,legend,dict(athighlights),dict(bndhighlights),arads,{})
        else:
            d2d.DrawMolecule(mol,legend=legend, highlightAtoms=highlightAtoms)
            
        d2d.FinishDrawing()
        display(SVG(d2d.GetDrawingText()))

        return

