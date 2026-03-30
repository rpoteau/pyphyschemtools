############################################################
#                    Nano tools
############################################################
from .visualID_Eng import fg, bg, hl
from .core import centerTitle, centertxt

import numpy as np
import matplotlib.pyplot as plt
import os, io
from pathlib import Path


# =====================================================================================================
#                           general tools
# =====================================================================================================

import numpy as np
import py3Dmol
from ase import Atoms
from ase.io import write
from ase.neighborlist import NeighborList
from matplotlib.patches import Patch

def get_coordination_numbers(mol: Atoms, cutoff: float = None):
    """
    Calculates the coordination number (CN) for each atom in an ASE Atoms object.
    
    This function determines how many neighbors each atom has based on a distance 
    threshold. If no cutoff is provided, it automatically estimates one based on 
    the 1st percentile of the interatomic distance distribution.

    Args:
        mol (ase.Atoms): The structural model (nanoparticle, molecule, or crystal).
        cutoff (float, optional): The distance threshold (in Angstroms) to define 
            a chemical bond. Defaults to None (automatic detection).

    Returns:
        tuple: A tuple containing:
            - cn (numpy.ndarray): An array of integers representing the CN of each atom.
            - used_cutoff (float): The actual cutoff value used for the calculation.
    """
    nat = len(mol)
    
    if cutoff is None:
        # Automatic cutoff detection: 1.2x the 1st percentile of bond distances
        dist = mol.get_all_distances()
        non_zero_dist = dist[dist > 0]
        if len(non_zero_dist) == 0:
            used_cutoff = 3.0 # Fallback for single-atom systems
        else:
            used_cutoff = np.percentile(non_zero_dist, 1) * 1.2
    else:
        used_cutoff = cutoff

    # Initialize NeighborList with a flat cutoff radius for all atoms
    cutoffs = [used_cutoff / 2.0] * nat
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(mol)

    cn = np.array([len(nl.get_neighbors(i)[0]) for i in range(nat)], dtype=int)
    return cn, used_cutoff

def view_coordination(mol: Atoms, cutoff: float = None, stick_radius: float = 0.1, sphere_scale: float = 0.6, color_map = "YlOrRd"):
    """
    Visualizes a structure using py3Dmol with atoms color-coded by coordination number.
    
    This function computes the coordination environment and generates a 3D ball-and-stick 
    model where colors represent the connectivity (e.g., surface vs. bulk atoms). 
    A Matplotlib legend is displayed alongside the 3D view.

    The color logic is optimized for nanoparticles:
    - CN < 5 : Pastel (low coordination/isolated)
    - CN 5-13: Sequential Gradient (surface to bulk transition)
    - CN > 13: Deep Dark (high density/interstitials)

    Args:
        mol (ase.Atoms): The structural model to visualize.
        cutoff (float, optional): Distance threshold for bond detection. 
            Defaults to None (auto-detect).
        stick_radius (float, optional): The thickness of the bonds in the 3D view. 
            Defaults to 0.1.
        sphere_scale (float, optional): The size multiplier for the atomic spheres. 
            Defaults to 0.6.
        color_map (str, optional): The Matplotlib colormap to use for discrete CN values.
            Defaults to "YlOrRd" (recommended!).

    Returns:
        py3Dmol.view: The interactive viewer object.
    """
    # --- color map for CN 1→20 (just in case...) ---
    def cn_palette():
        from matplotlib import colormaps, colors
        low_map = colormaps.get_cmap("Pastel1")    # Distinct pastels
        mid_map = colormaps.get_cmap(color_map)    # Sequential gradient. YlOrRd is recommended
        high_map = colormaps.get_cmap("Dark2")     # Distinct dark colors
        palette = {}
        for cn in range(1, 21):
            if cn <= 4:
                # Zone 1: Unique Pastel for each (1, 2, 3, 4)
                palette[cn] = colors.to_hex(low_map(cn - 1))
                
            elif 5 <= cn <= 12:
                # Zone 2: Sequential Gradient for surface-to-bulk
                # --- THE HIGH-CONTRAST HACK ---
                # We map specific CNs to hardcoded positions on the YlGnBu scale:
                # 5-6: Bright Yellow/Green (Start of scale)
                # 7-8: Teal/Turquoise (Middle)
                # 9-11: Strong Blue (Upper middle)
                # 12-13: Dark Navy (End of scale)
                
                anchors = {
                    5:  0.00, # Pale Yellow
                    6:  0.15, # Yellow-Green (Vertices)
                    7:  0.30, # Green
                    8:  0.45, # Teal/Cyan (Edges)
                    9:  0.60, # Bright Blue (Facets)
                    10: 0.75, # Royal Blue
                    11: 0.90, # Deep Blue
                    12: 1.00  # Midnight Blue (Bulk)
                }
                # Use .get() to find the anchor, or interpolate if missing
                val = anchors.get(cn, 0.5) 
                palette[cn] = colors.to_hex(mid_map(val))
                
            else:
                # Zone 3: Unique Dark colors for high coordination (> 13)
                # We use cn-14 to restart the index for the high_map
                palette[cn] = colors.to_hex(high_map((cn - 14) % 8))
        
        return palette
    
    def colors_for_cn(cn, palette):
        return [palette[val] for val in cn]

    # 1. Compute CN using the helper function
    cn, used_cutoff = get_coordination_numbers(mol, cutoff=cutoff)
    unique_cns = sorted(np.unique(cn))
    
    # 2. Setup Palette (using tab20 for discrete, distinct categories)
    palette = cn_palette()
    colors = colors_for_cn(cn, palette)

    # 3. py3Dmol Visualization Logic
    buf = io.StringIO()
    write(buf, mol, format="xyz")
    xyz_str = buf.getvalue()
    buf.close()

    v = py3Dmol.view(width=600, height=400)
    v.addModel(xyz_str, "xyz")

    for i, color in enumerate(colors):
        v.setStyle({"serial": i},
                   {"sphere": {"color": color, "scale": sphere_scale},
                    "stick": {"radius": stick_radius, "color": "gray"}})
    
    v.zoomTo()
    v.zoom(0.9)
    
    # 4. Legend rendering using Matplotlib
    legend_elements = [Patch(facecolor=palette[val], edgecolor="k", label=f"CN = {val}")
                       for val in unique_cns]

    fig, ax = plt.subplots(figsize=(3, len(unique_cns) * 0.4))
    ax.axis("off")
    ax.legend(handles=legend_elements, loc="center left", frameon=False, 
              title=f"Coordination (Cutoff: {used_cutoff:.2f}Å)")
    plt.show()
    v.show()