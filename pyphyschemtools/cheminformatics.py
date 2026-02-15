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
        if self.mol is None and self.lang == "En":
            raise ValueError(f"Invalid molecule for {self.smiles} (Check if SMILES is correct) ")
        if self.mol is None and self.lang == "Fr":
            raise ValueError(f"Molécule invalide pour {self.smiles} (Vérifier si le SMILES est correct) ")
        
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
            
        return pd.DataFrame(rows)    
            
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
        Renders the molecule in 2D SVG format with optional property overlays.
        
        Args:
            size (tuple): Drawing dimensions in pixels.
            show_Lewis (bool): Annotates atoms with Lone Pairs and Vacancies.
            plot_conjugation (bool): Highlights conjugated bonds in blue.
            plot_aromatic (bool): Highlights aromatic rings in red.
            show_n (bool): Displays atom indices.
            show_hybrid (bool): Displays atom hybridization (sp3, sp2, etc.).
            show_H (bool): Adds explicit Hydrogens to the drawing.
            show_stereo (bool): Shows R,S,Z,E labels - if relevant
            rep3D (bool): Computes a 3D-like conformation before drawing.
            macrocycle (bool): Uses CoordGen for better rendering of large rings (e.g., Cyclodextrins).
            highlightAtoms (list): List of indices to highlight.
            legend (str): Title or legend text for the drawing.
            save_img (str):  File path to save the resulting image. 
                Supports '.svg' (vector) and '.png' (raster) extensions. 
                Defaults to None.
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
        Identifies chiral centers and stereogenic double bonds.
        Returns a dictionary with counts and specific assignments.
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
        
        # Options to ensure we explore the full space
        opts = StereoEnumerationOptions(tryEmbedding=True)
        isomers_mols = list(EnumerateStereoisomers(self.mol, options=opts))
        
        if verbose:
            print(f"{fg.CYAN}✨ {len(isomers_mols)} stereoisomers found for {self.smiles}{fg.OFF}")

        # Wrap the resulting RDKit molecules back into easy_rdkit objects
        return [easy_rdkit(Chem.MolToSmiles(iso)) for iso in isomers_mols[:max_isomers]]

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