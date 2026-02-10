############################################################
#                       easy_rdkit
############################################################
from .visualID_Eng import fg, bg, hl
from .core import centerTitle, centertxt

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, GetPeriodicTable, Draw, rdCoordGen
from rdkit.Chem import Descriptors, QED, rdMolDescriptors
import pandas as pd
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import SVG
from PIL import Image
import os, math

class easy_rdkit():
    """
    A helper class to analyze and visualize molecules using RDKit.
    Provides tools for Lewis structure analysis and advanced 2D drawing.
    Initialize the molecule object from a SMILES string.
    
    Args:
        smiles (str): The SMILES representation of the molecule.
        canonical (bool): If True, converts the SMILES to its canonical form 
                          to ensure consistent atom numbering and uniqueness.
    """

    def __init__(self,smiles, canonical=True):
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

    @classmethod
    def from_cid(cls, cid):
        """
        Create an easy_rdkit instance directly from a PubChem CID.
        
        Parameters:
        -----------
        cid : int or str
            The PubChem Compound ID.
        """
        import pubchempy as pcp
        try:
            compound = pcp.Compound.from_cid(cid)
            smiles = compound.connectivity_smiles
            print(f"✅ Successfully retrieved: {compound.iupac_name}")
            # In your from_cid method
            instance = cls(smiles)
            instance.cid = cid
            instance.iupac_name = compound.iupac_name
            return instance
        except Exception as e:
            print(f"❌ Error fetching CID {cid} from PubChem: {e}")
            return None

    def fetch_pubchem_data(self):
        """
        Retrieves CID and IUPAC name from PubChem based on the current SMILES.
        Useful for molecules initialized directly via SMILES.
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
            
    @property
    def descriptors(self):
        """Compute and return a dictionary of key molecular descriptors."""
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
        """Print all computed descriptors in a formatted table."""
        d = self.descriptors
        print(f"--------- Molecular Descriptors ---")
        print(f"{'Property':<20} | {'Value':<10}")
        print("-" * 35)
        for prop, value in d.items():
            print(f"{prop:<20} | {value:<10}")
        print("-" * 35)

    def to_dict(self, auto_fetch=False):
        """
        Export identity and descriptors as a flat dictionary.
        If auto_fetch is True, it will attempt to find missing CID/Name on PubChem.
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

    @staticmethod
    def plot_grid_from_df(df, smiles_col='SMILES', legend_cols='IUPAC Name', mols_per_row=4, size=(250, 250), save_img=None):
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

        # --- NEW SAVING LOGIC STARTS HERE ---
        if save_img:
            ext = os.path.splitext(save_img)[1].lower()
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

        return Draw.MolsToGridImage(
            mols, 
            molsPerRow=mols_per_row, 
            subImgSize=size, 
            legends=legends,
            useSVG=True,
            drawOptions=dopts # Apply the spacing options here
        )