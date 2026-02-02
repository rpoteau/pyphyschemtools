############################################################
#                       3D Chemistry
############################################################
from .visualID_Eng import fg, bg, hl
from .core import centerTitle, centertxt

import py3Dmol
import io, os
from ase import Atoms
from ase.io import read, write
from ase.data import vdw_radii, atomic_numbers
import requests
import numpy as np
from ipywidgets import GridspecLayout, VBox, Label, Layout
import CageCavityCalc as CCC

# ============================================================
# Jmol-like element color palette
# ============================================================
JMOL_COLORS = {
    'H':  '#FFFFFF',
    'C':  '#909090',
    'N':  '#3050F8',
    'O':  '#FF0D0D',
    'F':  '#90E050',
    'Cl': '#1FF01F',
    'Br': '#A62929',
    'I':  '#940094',
    'S':  '#FFFF30',
    'P':  '#FF8000',
    'B':  '#FFB5B5',
    'Si': '#F0C8A0',

    'Li': '#CC80FF',
    'Na': '#AB5CF2',
    'K':  '#8F40D4',
    'Mg': '#8AFF00',
    'Ca': '#3DFF00',

    'Fe': '#E06633',
    'Co': '#F090A0',
    'Ni': '#50D050',
    'Cu': '#C88033',
    'Zn': '#7D80B0',

    'Ru': '#248F8F',   # Ruthenium (Jmol faithful)
    'Rh': '#E000E0',
    'Pd': '#A0A0C0',
    'Ag': '#C0C0C0',
    'Pt': '#D0D0D0',
    'Au': '#FFD123',
    'Ir': '#175487',
    'Os': '#266696',
}

class XYZData:
    """
    Object containing molecular coordinates and symbols extracted by molView.
    Allows for geometric calculations without reloading data.
    """
    def __init__(self, symbols, positions):
        self.symbols = np.array(symbols)
        self.positions = np.array(positions, dtype=float)

    def get_center_of_mass(self):
        return np.mean(self.positions, axis=0)

    def get_center_of_geometry(self):
        """
        Calculates the arithmetic mean of the atomic positions (Centroid).
        """
        return np.mean(self.positions, axis=0)
        
    def get_bounding_sphere(self, include_vdw=True, scale=1.0):
        """
        Calculates the center and radius of the bounding sphere using ASE.
        scale: multiplication factor (e.g., 0.6 to match a reduced CPK style).
        """
        center = np.mean(self.positions, axis=0)
        distances = np.linalg.norm(self.positions - center, axis=1)
        
        if include_vdw:
            z_numbers = [atomic_numbers[s] for s in self.symbols]
            radii = vdw_radii[z_numbers] * scale
            radius = np.max(distances + radii)
        else:
            radius = np.max(distances)
            
        return center, radius

    def get_cage_volume(self, grid_spacing=0.5, return_spheres=False):
        """
        Calculates the internal cavity volume of a molecular cage using CageCavityCalc.
        
        This method interfaces with the CageCavityCalc library by generating a 
        temporary PDB file of the current structure. It can also retrieve the 
        coordinates of the 'dummy atoms' (points) that fill the detected void.

        Parameters
        ----------
        grid_spacing : float, optional
            The resolution of the grid used for volume integration in Å. 
            Smaller values provide higher precision (default: 0.5).
        return_spheres : bool, optional
            If True, returns both the volume and an ase.Atoms object 
            containing the dummy atoms representing the cavity (default: False).

        Returns
        -------
        volume : float or None
            The calculated cavity volume in Å³. Returns None if the 
            calculation fails.
        cavity_atoms : ase.Atoms, optional
            Returned only if return_spheres is True. An ASE Atoms object 
            representing the internal void space.
        """
        import tempfile
        import os
        from ase import Atoms
        from ase.io import read as ase_read, write as ase_write
            
        try:
            from CageCavityCalc.CageCavityCalc import cavity
            
            # 1. Fichier temporaire pour la cage
            with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
                cage_tmp = tmp.name
                temp_atoms = Atoms(symbols=self.symbols, positions=self.positions)
                ase_write(cage_tmp, temp_atoms)

            cav = cavity()
            cav.read_file(cage_tmp)
            cav.grid_spacing = float(grid_spacing)
            cav.dummy_atom_radii = float(grid_spacing)
            volume = cav.calculate_volume()
            
            cavity_atoms = None
            if return_spheres:
                with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp2:
                    cav_tmp = tmp2.name
                
                cav.print_to_file(cav_tmp) 
                
                # --- NOUVEAU : Correction pour ASE (remplace ' D ' par ' H ') ---
                with open(cav_tmp, 'r') as f:
                    content = f.read().replace(' D ', ' H ') # On transforme les Dummy en Hydrogène
                with open(cav_tmp, 'w') as f:
                    f.write(content)
                
                # Maintenant ASE peut lire le fichier sans erreur
                cavity_atoms = ase_read(cav_tmp)
                
                if os.path.exists(cav_tmp): 
                    os.remove(cav_tmp)
            
            # ... (fin de la fonction) ...
            if return_spheres:
                return volume, cavity_atoms
            return volume        
            
        except Exception as e:
            print(f"Erreur CageCavityCalc : {e}")
            return None

    def get_cavity_dimensions(self, cavity_atoms):
        """
        Calculates the principal dimensions (Length, Width, Height) of the cavity points.
        
        This method uses Principal Component Analysis (PCA) to find the natural 
        axes of the cavity, making it independent of the molecule's orientation.
        Percentiles are used instead of absolute Max-Min to filter out 
        potential outliers or 'leaking' points at the openings.

        Parameters
        ----------
        cavity_atoms : ase.Atoms
            The Atoms object containing the 'dummy atoms' generated 
            by the cavity calculation.

        Returns
        -------
        tuple (float, float, float)
            The dimensions (L, W, H) sorted from largest to smallest.
        """
        import numpy as np
        
        # On récupère les positions des dummy atoms (les points de vide)
        points = cavity_atoms.get_positions()
        
        if len(points) < 2:
            return 0, 0, 0

        # Center the points at the origin (Arithmetic Mean)
        centered_points = points - np.mean(points, axis=0)
        
        # Compute the Covariance Matrix to find the spread direction
        cov = np.cov(centered_points, rowvar=False)
        
        # Compute Eigenvalues and Eigenvectors
        # Eigenvectors represent the principal axes of the cavity
        evals, evecs = np.linalg.eigh(cov)
        
        # Project the points onto the principal axes (PCA transformation)
        projections = np.dot(centered_points, evecs)
        
        dims = []
        for i in range(3):
            # Calculate the spread using percentiles (2% to 98%)
            # This is more robust than np.ptp() as it ignores outliers
            p_min = np.percentile(projections[:, i], 2)
            p_max = np.percentile(projections[:, i], 98)
            dims.append(p_max - p_min)
        
        # Sort dimensions from largest to smallest
        dims = sorted(dims, reverse=True)
        
        return dims[0], dims[1], dims[2]
    
    def __repr__(self):
        return f"<XYZData: {len(self.symbols)} atoms>"        
        
class molView:
    """
    Initializes a molecular/crystal viewer and coordinate extractor.

    This class acts as a bridge between various molecular data sources and 
    the py3Dmol interactive viewer. It can operate in 'Full' mode (display + 
    analysis) or 'Headless' mode (analysis only) by toggling the `viewer` parameter.

    The class automatically extracts geometric data into the `self.data` attribute 
    (an XYZData object), allowing for volume, dimension, and cavity calculations.
        
    Display molecular and crystal structures in py3Dmol from various sources:
    
    - XYZ/PDB/CIF local files
    - XYZ-format string
    - PubChem CID
    - ASE Atoms object
    - COD ID
    - RSCB PDB ID

    Three visualization styles are available:
    
    - 'bs'     : ball-and-stick (default)
    - 'cpk'    : CPK space-filling spheres (with adjustable size)
    - 'cartoon': protein backbone representation
      
    Upon creation, an interactive 3D viewer is shown directly in a Jupyter notebook cell, unless the headless viewer parameter is set to False.

    Parameters
    ----------
        mol : str or ase.Atoms
            The molecular structure to visualize.
            
            - If `source='file'`, this should be a path to a structure file (XYZ, PDB, etc.)
            - If `source='mol'`, this should be a string containing the structure (XYZ, PDB...)
            - If `source='cif'`, this should be a cif file (string)
            - If `source='cid'`, this should be a PubChem CID (string or int)
            - If `source='rscb'`, this should be a RSCB PDB ID (string)
            - If `source='cod'`, this should be a COD ID (string)
            - If `source='ase'`, this should be an `ase.Atoms` object
        source : {'file', 'mol', 'cif', 'cid', 'rscb', 'ase'}, optional
            The type of the input `mol` (default: 'file').
        style : {'bs', 'cpk', 'cartoon'}, optional
            Visualization style (default: 'bs').
            
            - 'bs'  → ball-and-stick
            - 'cpk' → CPK space-filling spheres
            - 'cartoon' → draws a smooth tube or ribbon through the protein backbone
                         (default for pdb structures)
        displayHbonds : plots hydrogen bonds (default: True)
        cpk_scale : float, optional
            Overall scaling factor for sphere size in CPK style (default: 0.5).
            Ignored when `style='bs'`.
        supercell : tuple of int
            Repetition of the unit cell (na, nb, nc). Default is (1, 1, 1).
        w : int, optional
            Width of the viewer in pixels (default: 600).
        h : int, optional
            Height of the viewer in pixels (default: 400).
        detect_BondOrders : bool, optional
            If True (default) and input is XYZ, uses RDKit to perceive connectivity 
            and bond orders (detects double/triple bonds). 
            Requires the `rdkit` library. If False, fallback to standard 3Dmol 
            distance-based single bonds.
        viewer : bool, optional
                If True (default), initializes the py3Dmol viewer and renders the 
                molecule. If False, operates in 'headless' mode: only coordinates 
                are processed for calculations (default: True).
        zoom : None, optional
            scaling factor

    Attributes
    ----------
        data : XYZData or None
            Container for atomic symbols and positions, used for geometric analysis.
        v : py3Dmol.view or None
            The 3Dmol.js viewer instance (None if viewer=False).

    Examples
    --------
        >>> molView("molecule.xyz", source="file")
        >>> molView(xyz_string, source="mol")
        >>> molView(2244, source="cid")   # PubChem aspirin
        >>> from ase.build import molecule
        >>> molView(molecule("H2O"), source="ase")
        >>> molView.view_grid([2244, 2519, 702], n_cols=3, source='cid', style='bs')
        >>> molView.view_grid(xyzFiles, n_cols=3, source='file', style='bs', titles=titles, w=500, sync=True)
        >>> # Headless mode for high-throughput volume calculations
        >>> mv = molView("cage.xyz", viewer=False)
        >>> vol = mv.data.get_cage_volume()
    """

    def __init__(self, mol, source='file', style='bs', displayHbonds=True, cpk_scale=0.6, w=600, h=400,\
                 supercell=(1, 1, 1), display_now=True, detect_BondOrders=True, viewer=True, zoom=None):
        self.mol = mol
        self.source = source
        self.style = style
        self.cpk_scale = cpk_scale
        self.displayHbonds = displayHbonds
        self.w = w
        self.h = h
        self.detect_bonds = detect_BondOrders # Store the option
        self.supercell = supercell
        self.viewer = viewer
        self.zoom = zoom
        self.v = py3Dmol.view(width=self.w, height=self.h) # Création du viewer une seule fois
        self._load_and_display(show=display_now)

    @classmethod
    def view_grid(cls, mol_list, n_cols=3, titles=None, **kwargs):
        """
        Displays a list of molecular structures in an interactive n_rows x n_cols grid.
        
        This method uses ipywidgets.GridspecLayout to organize multiple 3D viewers 
        into a clean matrix. It automatically calculates the required number of rows 
        based on the length of the input list.

        Parameters
        ----------
        mol_list : list
            A list containing the molecular data to visualize. Elements should 
            match the expected 'mol' input for the class (paths, CIDs, strings, etc.).
        n_cols : int, optional
            Number of columns in the grid (default: 3).
        titles : list of str, optional
            Custom labels for each cell. If None, the string representation 
            of the 'mol' input is used as the title.
        **kwargs : dict
            Additional arguments passed to the molView constructor:
            - source : {'file', 'mol', 'cif', 'cid', 'rscb', 'ase'}
            - style : {'bs', 'cpk', 'cartoon'}
            - displayHbonds : plots hydrogen bonds (default: True)
            - w : width of each individual viewer in pixels (default: 300)
            - h : height of each individual viewer in pixels (default: 300)
            - supercell : tuple (na, nb, nc) for crystal structures
            - cpk_scale : scaling factor for space-filling spheres

        Returns
        -------
        ipywidgets.GridspecLayout
            A widget object containing the grid of molecular viewers.
            
        Examples
        --------
        >>> files = ["mol1.xyz", "mol2.xyz", "mol3.xyz", "mol4.xyz"]
        >>> labels = ["Reactant", "TS", "Intermediate", "Product"]
        >>> molView.view_grid(files, n_cols=2, titles=labels, source='file', w=400)
        """
        from ipywidgets import GridspecLayout, VBox, Label, Layout, Output
        from IPython.display import display

        # 1. Gestion des dimensions
        w_cell = kwargs.get('w', 300)
        h_cell = kwargs.get('h', 300)
        
        n_mol = len(mol_list)
        n_rows = (n_mol + n_cols - 1) // n_cols # Calcul automatique du nombre de lignes
        
        # Largeur totale pour éviter le scroll horizontal
        total_width = n_cols * (w_cell + 25) 
        grid = GridspecLayout(n_rows, n_cols, layout=Layout(width=f'{total_width}px'))
        
        kwargs['w'] = w_cell
        kwargs['h'] = h_cell
        kwargs['display_now'] = False # Indispensable pour garder le contrôle

        # 2. Remplissage de la grille
        for i, mol in enumerate(mol_list):
            row, col = i // n_cols, i % n_cols
            t = titles[i] if titles and i < len(titles) else str(mol)

            # Création de l'instance (charge les données et styles)
            obj = cls(mol, **kwargs)
            
            # Widget de sortie pour capturer le rendu JS de py3Dmol
            out = Output(layout=Layout(
                width=f'{w_cell}px', 
                height=f'{h_cell}px', 
                overflow='hidden'
            ))
            
            with out:
                display(obj.v)
            
            # Assemblage Titre + Molécule dans la cellule
            grid[row, col] = VBox([
                Label(value=t, layout=Layout(display='flex', justify_content='center', width='100%')),
                out
            ], layout=Layout(
                width=f'{w_cell + 15}px', 
                align_items='center', 
                overflow='hidden',
                margin='5px'
            ))
            
        return grid

    def _get_ase_atoms(self, content, fmt):
        """Helper to convert string content to ASE Atoms and apply supercell."""
        # Use ASE to parse the structure (more robust for symmetry)
        atoms = read(io.StringIO(content), format=fmt)
        if self.supercell != (1, 1, 1):
            atoms = atoms * self.supercell
        return atoms

    def _draw_cell_vectors(self, cell, origin=(0, 0, 0),
                           radius=0.12, head_radius=0.25, head_length=0.6,
                           label_offset=0.15):
        """
        Draw crystallographic vectors a, b, c as colored arrows
        and add labels a, b, c at their tips.
        
        a = red, b = blue, c = green
        """
        a, b, c = np.array(cell, dtype=float)
        o = np.array(origin, dtype=float)
    
        vectors = {
            "a": (a, "red"),
            "b": (b, "blue"),
            "c": (c, "green")
        }
    
        for name, (vec, color) in vectors.items():
            end = o + vec
    
            # Arrow
            self.v.addArrow({
                "start": {
                    "x": float(o[0]), "y": float(o[1]), "z": float(o[2])
                },
                "end": {
                    "x": float(end[0]), "y": float(end[1]), "z": float(end[2])
                },
                "radius": float(radius),
                "radiusRatio": head_radius / radius,
                "mid": 0.85,
                "color": color
            })
    
            # Label slightly beyond the arrow tip
            label_pos = end + label_offset * vec / np.linalg.norm(vec)
    
            self.v.addLabel(
                name,
                {
                    "position": {
                        "x": float(label_pos[0]),
                        "y": float(label_pos[1]),
                        "z": float(label_pos[2])
                    },
                    "fontColor": color,
                    "backgroundColor": "white",
                    "backgroundOpacity": 0.,
                    "fontSize": 16,
                    "borderThickness": 0
                }
            )
    def _draw_lattice_wireframe(self, cell, reps, color="black", radius=0.05):
        """
        Draw all unit cells of a supercell lattice as wireframes.
        
        Parameters
        ----------
        cell : ase.Cell
            Primitive cell.
        reps : tuple(int,int,int)
            Supercell repetitions (na, nb, nc).
        """
        a, b, c = np.array(cell, dtype=float)
        na, nb, nc = reps
    
        for i in range(na):
            for j in range(nb):
                for k in range(nc):
                    origin = i*a + j*b + k*c
                    self._draw_cell_wireframe(
                        cell,
                        color=color,
                        radius=radius,
                        origin=origin
                    )

    def _draw_cell_wireframe(self, cell, color="black", radius=0.05, origin=(0, 0, 0)):
        """
        Draw a unit cell as a wireframe using py3Dmol lines.
        Works with XYZ or CIF models.
        """
        a, b, c = np.array(cell)
        o = np.array(origin)
    
        corners = [
            o,
            o + a,
            o + b,
            o + c,
            o + a + b,
            o + a + c,
            o + b + c,
            o + a + b + c
        ]
    
        edges = [
            (0,1), (0,2), (0,3),
            (1,4), (1,5),
            (2,4), (2,6),
            (3,5), (3,6),
            (4,7), (5,7), (6,7)
        ]
    
        for i, j in edges:
            self.v.addCylinder({
                "start": {
                    "x": float(corners[i][0]),
                    "y": float(corners[i][1]),
                    "z": float(corners[i][2]),
                },
                "end": {
                    "x": float(corners[j][0]),
                    "y": float(corners[j][1]),
                    "z": float(corners[j][2]),
                },
                "color": color,
                "radius": float(radius),
                "fromCap": True,
                "toCap": True
                })

    def _add_h_bonds(self, atoms, dist_max=2.5, angle_min=120):
        """
        Detects and renders realistic H-bonds using ASE neighbor list.
        Criteria: d(H...A) < dist_max & Angle(Donor-H...Acceptor) > angle_min
        """
        from ase.neighborlist import neighbor_list
        import numpy as np
    
        # 1. Identify donors: Find Hydrogens covalently bonded to N or O
        # i_cov: indices of H, j_cov: indices of parent atoms (N, O)
        i_cov, j_cov = neighbor_list('ij', atoms, cutoff=1.2)
        donors = {
            idx_h: idx_d for idx_h, idx_d in zip(i_cov, j_cov)
            if atoms[idx_h].symbol == 'H' and atoms[idx_d].symbol in ['N', 'O']
        }
    
        # 2. Search for potential acceptors near these Hydrogens
        # i_h: indices of H, j_acc: indices of potential acceptors (N, O)
        i_h, j_acc, d_ha = neighbor_list('ijd', atoms, cutoff=dist_max)
    
        for idx_h, idx_acc, dist in zip(i_h, j_acc, d_ha):
            # Validate: H is a known donor, Target is N or O, and not its own parent
            if idx_h in donors and atoms[idx_acc].symbol in ['N', 'O']:
                idx_d = donors[idx_h]
                if idx_acc == idx_d: 
                    continue
                
                # 3. Angle check: Donor-H...Acceptor
                try:
                    # ASE get_angle returns the angle in degrees
                    angle = atoms.get_angle(idx_d, idx_h, idx_acc)
                    
                    if angle >= angle_min:
                        p_h = atoms[idx_h].position
                        p_a = atoms[idx_acc].position
                        
                        self.v.addCylinder({
                            'start': {'x': float(p_h[0]), 'y': float(p_h[1]), 'z': float(p_h[2])},
                            'end': {'x': float(p_a[0]), 'y': float(p_a[1]), 'z': float(p_a[2])},
                            'radius': 0.06,
                            'color': '#00FFFF', # Cyan
                            'dashed': True,
                            'fromCap': 1,
                            'toCap': 1
                        })
                except Exception:
                    continue
    
    def _load_and_display(self, show):

        content = ""
        fmt = "xyz"

        # --- 1. Handle External API Sources ---
        if self.source == 'cid':
            if self.viewer: self.v = py3Dmol.view(query=f'cid:{self.mol}', width=self.w, height=self.h)
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{self.mol}/SDF?record_type=3d"
            response = requests.get(url)
            if response.status_code == 200:
                content = response.text
                fmt = "sdf" 
                
        elif self.source == 'rscb':
            if self.viewer: self.v = py3Dmol.view(query=f'pdb:{self.mol}', width=self.w, height=self.h)
            url = f"https://files.rcsb.org/view/{self.mol}.pdb"
            response = requests.get(url)
            if response.status_code == 200:
                content = response.text
                fmt = "pdb"
                
        elif self.source == 'cod':
            url = f"https://www.crystallography.net/cod/{self.mol}.cif"
            response = requests.get(url)
            if response.status_code == 200:
                self.mol = response.text
                self.source = 'cif'
            else:
                raise ValueError(f"Could not find COD ID: {self.mol}")

        # --- FIX 1: Initialisation par défaut pour les sources non-API ---
        if self.viewer and self.v is None:
            self.v = py3Dmol.view(width=self.w, height=self.h)
            
        # --- 2. Handle Logic for Files and Data ---
        if self.source == 'file':
            if not os.path.exists(self.mol):
                raise FileNotFoundError(f"File not found: {self.mol}")
            ext = os.path.splitext(self.mol)[1].lower().replace('.', '')
            fmt = 'cif' if ext == 'cif' else ext
            with open(self.mol, 'r') as f:
                content = f.read()
        
        elif self.source == 'cif':
            content = self.mol
            fmt = 'cif'
        
        elif self.source == 'mol':
            content = self.mol
            fmt = 'xyz'

        # --- EXTRACTION XYZData (Interne) ---
        # On extrait les données ici avant toute modification (RDKit ou Supercell)
        try:
            if self.source == 'ase':
                temp_atoms = self.mol
            else:
                temp_atoms = read(io.StringIO(content), format=fmt)
            
            self.data = XYZData(
                symbols=temp_atoms.get_chemical_symbols(),
                positions=temp_atoms.get_positions()
            )
        except Exception as e:
            print(f"Note: Extraction des coordonnées impossible ({e})")
            self.data = None            
            
        # --- Modern Bond Perception with RDKit ---
        if self.detect_bonds and self.source in ['file', 'mol', 'xyz'] and fmt == 'xyz':
            try:
                from rdkit import Chem
                from rdkit.Chem import rdDetermineBonds
                
                raw_mol = Chem.MolFromXYZBlock(content)
                rdDetermineBonds.DetermineConnectivity(raw_mol)
                rdDetermineBonds.DetermineBondOrders(raw_mol, charge=0)
                
                content = Chem.MolToMolBlock(raw_mol)
                fmt = "sdf"
            except ImportError:
                # Silent skip if RDKit is missing
                pass
            except Exception as e:
                # Small warning if the geometry is the problem
                print(f"Note: Bond perception failed for {self.mol}. Falling back to standard XYZ.")
                
        # --- 3. Rendering Logic ---
        if fmt == 'cif' or self.supercell != (1, 1, 1) or self.source == 'ase':
            # Create ASE atoms object
            if self.source == 'ase':
                atoms = self.mol
            else:
                atoms = read(io.StringIO(content), format=fmt)
            
            # --- CRYSTAL LOGIC (Jmol packed-like) ---
            
            # 1. Read primitive cell (before supercell)
            atoms0 = atoms.copy()
            
            # 2. Apply supercell if requested
            if self.supercell != (1, 1, 1):
                atoms = atoms * self.supercell
            
            # 3. Send atoms to py3Dmol (XYZ, robust)
            xyz_buf = io.StringIO()
            write(xyz_buf, atoms, format="xyz")

            if self.viewer: 
                self.v.addModel(xyz_buf.getvalue(), "xyz")
            
                # 4. Draw supercell (optional, thick & gray)
                if self.supercell != (1, 1, 1):
                    self._draw_lattice_wireframe(
                        atoms0.cell,
                        self.supercell,
                        color="gray",
                        radius=0.015
                    )
                    self._draw_cell_wireframe(
                        atoms.cell,
                        color="gray",
                        radius=0.015
                    )
                
                # 5. Draw primitive cell (Jmol packed equivalent)
                self._draw_cell_wireframe(
                    atoms0.cell,
                    color="black",
                    radius=0.03
                )
                # Vecteurs a, b, c
                self._draw_cell_vectors(
                    atoms0.cell,
                    radius=0.04
                )
 
        else:
            # Standard molecule (non-crystal)
            if self.viewer:
                self.v.addModel(content, fmt)
            # FIX: Create the atoms object for standard molecules here
            atoms = read(io.StringIO(content), format=fmt)

            
        # Finalize
        if self.viewer:
            self._apply_style()
            self._add_interactions()
            # Detect and add H-bonds if hydrogens are present
            symbols = atoms.get_chemical_symbols()
            if 'H' in symbols and self.displayHbonds:
                self._add_h_bonds(atoms)
            self.v.zoomTo()
            if self.zoom is not None:
                self.v.zoom(self.zoom)
            elif self.source != 'cif':
                self.v.zoom(0.9) # Zoom par défaut pour ne pas coller aux bords
            if show: self.v.show()

    def _apply_element_colors(self, color_table):
        """
        Override element colors without breaking the current style (bs / cpk).
        """
        for elem, color in color_table.items():
            if self.style == 'bs':
                self.v.setStyle(
                    {'elem': elem},
                    {
                        'sphere': {'color': color, 'scale': 0.25},
                        'stick':  {'color': color, 'radius': 0.15}
                    }
                )
    
            elif self.style == 'cpk':
                self.v.setStyle(
                    {'elem': elem},
                    {
                        'sphere': {'color': color, 'scale': self.cpk_scale}
                    }
                )

    def _apply_style(self):
        """Apply either ball-and-stick, cartoon or CPK style."""

        if self.style == 'bs':
            self.v.setStyle({'sphere': {'scale': 0.25, 'colorscheme': 'element'},
                        'stick': {'radius': 0.15, 'multibond': True}})
            self._apply_element_colors(JMOL_COLORS)
        elif self.style == 'cpk':
            self.v.setStyle({'sphere': {'scale': self.cpk_scale,
                                   'colorscheme': 'element'}})
            self._apply_element_colors(JMOL_COLORS)
        elif self.style == 'cartoon':
            self.v.setStyle({'cartoon': {'color': 'spectrum', 'style': 'rectangle', 'arrows': True}})
        else:
            raise ValueError("style must be 'bs', 'cpk' or 'cartoon'")

    def _add_interactions(self):
        """Add basic JavaScript Hover labels for atom identification."""
        label_js = "function(atom,viewer) { viewer.addLabel(atom.elem+atom.serial,{position:atom, backgroundColor:'black'}); }"
        reset_js = "function(atom,viewer) { viewer.removeAllLabels(); }"
        self.v.setHoverable({}, True, label_js, reset_js)

    def show_bounding_sphere(self, color='gray', opacity=0.2, scale=1.0):
        """Calculates and displays the VdW bounding sphere in one go."""
        if self.data:
            center, radius = self.data.get_bounding_sphere(include_vdw=True, scale=scale)
            self.v.addSphere({
                'center': {'x': float(center[0]), 'y': float(center[1]), 'z': float(center[2])},
                'radius': float(radius),
                'color': color,
                'opacity': opacity
            })
            print(f"Bounding Sphere: Radius = {radius:.2f} Å | Volume = {(4/3)*np.pi*radius**3:.2f} Å³")
        return self.v.show()

    def show_cage_cavity(self, grid_spacing=0.5, color='cyan', opacity=0.5):
        """Calculates cavity with CageCavityCalc and displays it as a single model."""
        if self.data:
            result = self.data.get_cage_volume(grid_spacing=grid_spacing, return_spheres=True)
            if result:
                volume, spheres = result
                L, W, H = self.data.get_cavity_dimensions(spheres)
                # Création du modèle optimisé pour éviter le gel du navigateur
                xyz_cavity = f"{len(spheres)}\nCavity points\n"
                for pos in spheres.get_positions():
                    xyz_cavity += f"He {pos[0]:.3f} {pos[1]:.3f} {pos[2]:.3f}\n"
                
                self.v.addModel(xyz_cavity, "xyz")
                # On applique le style au dernier modèle ajouté
                self.v.setStyle({'model': -1}, {
                    'sphere': {'radius': grid_spacing/2, 'color': color, 'opacity': opacity}
                })
                print(f"Cavity Volume (CageCavityCalc): {volume:.2f} Å³")
                print(f"Dimensions: {L:.2f} x {W:.2f} x {H:.2f} Å")
                print(f"Aspect Ratio (L/W): {L/W:.2f}")
        return self.v.show()
