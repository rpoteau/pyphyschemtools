#!/usr/bin/env python3
import sys

# --- VERSIONING ---
__version__ = "20260211"

# ANSI escape codes for colors
RED = "\033[1;31m"
GREEN = "\033[1;32m"
BLUE = "\033[1;34m"
RESET = "\033[0m"
CYAN = "\033[1;36m"

# Print Header
print(f"{CYAN}{'='*60}{RESET}")
print(f"{CYAN} GScan_Analyzis - Version {__version__}{RESET}".center(60))
print(f"{CYAN}{'='*60}{RESET}\n")

# --- DEPENDENCY CHECK ---
missing_packages = []

try:
    import cclib
except ImportError:
    missing_packages.append("cclib")

try:
    import matplotlib.pyplot as plt
except ImportError:
    missing_packages.append("matplotlib")

try:
    import numpy as np
except ImportError:
    missing_packages.append("numpy")

if missing_packages:
    # Everything inside this print will be RED
    print(RED)
    print("!" * 60)
    print(" ERROR: Python environment is not configured correctly ".center(60, "!"))
    print("!" * 60)
    print(f"\nMissing libraries: {', '.join(missing_packages)}")
    print(RESET) # Switch back to normal color for the instructions
    print(f"The following libraries are missing: {', '.join(missing_packages)}")
    print("\nSuggested Actions:")
    print("1. Check if you activated your environment (conda activate ...)")
    print("2. Or install the dependencies using:")
    print(f"   pip install {' '.join(missing_packages)}")
    print(RED)
    print("-" * 60 + "\n")
    print(RESET) # Switch back to normal color for the instructions
    sys.exit(1)

####################################################################################################

import os
import cclib
import matplotlib.pyplot as plt
import numpy as np
from cclib.io import ccopen, ccwrite

def main():
    # 1. Handle File Input
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        file_path = filedialog.askopenfilename(title="Select Gaussian Log File")

    if not file_path or not os.path.exists(file_path):
        print(f"{RED}Error: File not found.{RESET}"); return

    base_name = os.path.splitext(file_path)[0]
    
    print(f"{BLUE}Scan parsing of: {os.path.basename(file_path)}{RESET}")
    print(f"cclib version: {cclib.__version__}")
    data = ccopen(file_path).parse()

    # --- TALKATIVE REPORT ---
    print("\n" + "="*50)
    print("CCLIB REPORT. FOUND IN THE LOG...")
    print("="*50)
    
    attributes = data.getattributes() if hasattr(data, 'getattributes') else []
    print(f"Attributes found: {', '.join(attributes)}")
    
    metadata = getattr(data, 'metadata', {})
    print(f"Software: {metadata.get('methods', ['Unknown'])[0]} version {metadata.get('package_version', 'Unknown')}")
    print(f"Total SCF Energies: {len(data.scfenergies)}")
    print(f"Total Geometries: {len(data.atomcoords)}")
    print("="*50 + "\n")

    # --- SCAN FILTER LOGIC ---
    indices = []
    is_relaxed = True # Default to True for this analysis

    # 1. Strategy A: Check optstatus 
    if 'optstatus' in attributes:
        status = np.array(data.optstatus)
        # s & 2: Stationary point; s & 4: Last step of opt
        indices = [i for i, s in enumerate(status) if (s & 2 or s & 4)]
        
    # 2. Strategy B: Energy Valley fallback if optstatus is empty/weird
    if len(indices) < 2:
        print("-> optstatus was inconclusive. Falling back to Energy Valley detection...")
        energies = data.scfenergies
        for i in range(1, len(energies) - 1):
            if energies[i] < energies[i-1] and energies[i] < energies[i+1]:
                # Broadened threshold to ensure we don't miss steps
                if energies[i+1] > energies[i] + 0.005: 
                    indices.append(i)
        
    # 3. Strategy C: Clean up duplicates (Keep only the LAST index of a cluster)
    if len(indices) > 1:
        refined = []
        indices = sorted(list(set(indices))) # Remove any accidental duplicates
        for j in range(len(indices) - 1):
            # If next index is much further away, current index is the end of a step
            if indices[j+1] - indices[j] > 1:
                refined.append(indices[j])
        refined.append(indices[-1])
        indices = refined

    # D. FALLBACK: If nothing found, it's Rigid
    if not indices or len(indices) < 2:
        indices = list(range(len(data.scfenergies)))
        is_relaxed = False
        print("-> Treating as a standard Rigid Scan.")
    else:
        print(f"-> Detection Result: Found {len(indices)} optimized scan points.")

    # --- FEATURE: Last-Point Recovery ---
    last_idx = len(data.scfenergies) - 1
    has_recovered_point = False
    if last_idx not in indices:
        indices.append(last_idx)
        has_recovered_point = True
        print(f"-> Added unconverged last point (index {last_idx}) for analysis.")

    # --- DATA EXTRACTION ---
    energies_ev = np.array([data.scfenergies[i] for i in indices])
    coords = np.array([data.atomcoords[i] for i in indices])
    delta_e_kcal = (energies_ev - np.min(energies_ev)) * 23.0605

    # --- FIND GLOBAL MINIMUM ---
    min_idx_in_subset = np.argmin(energies_ev) 
    absolute_min_idx = indices[min_idx_in_subset] 
    min_energy_kcal = delta_e_kcal[min_idx_in_subset]
    min_coords = coords[min_idx_in_subset]

    print(f"{BLUE}Lowest energy found at scan point {min_idx_in_subset} (Log Index {absolute_min_idx}){RESET}")

    # --- SAVE XYZ MOVIE ---
    xyz_name = f"{base_name}_optimized_movie.xyz"
    try:
        filtered = cclib.parser.data.ccData()
        filtered.atomcoords = coords
        filtered.atomnos = data.atomnos
        ccwrite(filtered, xyz_name, outputtype='xyz')
        print(f"{GREEN}\n[Success]{RESET} Trajectory saved: {xyz_name}")
    except:
        with open(xyz_name, 'w') as f:
            for i, frame in enumerate(coords):
                f.write(f"{len(data.atomnos)}\nStep {i} (Index {indices[i]})\n")
                for j, at_num in enumerate(data.atomnos):
                    sym = cclib.parser.utils.PeriodicTable().element[at_num]
                    f.write(f"{sym} {frame[j][0]:12.8f} {frame[j][1]:12.8f} {frame[j][2]:12.8f}\n")
        print(f"{GREEN}\n[Success]{RESET} Trajectory saved (Manual Writer): {xyz_name}")

    # --- SAVE MINIMUM GEOMETRY ---
    min_xyz_name = f"{base_name}_lowest_energy.xyz"
    with open(min_xyz_name, 'w') as f:
        f.write(f"{len(data.atomnos)}\n")
        f.write(f"Lowest Energy Geometry - Index {absolute_min_idx} - Rel Energy: {min_energy_kcal:.4f} kcal/mol\n")
        for j, at_num in enumerate(data.atomnos):
            sym = cclib.parser.utils.PeriodicTable().element[at_num]
            f.write(f"{sym:2} {min_coords[j][0]:12.8f} {min_coords[j][1]:12.8f} {min_coords[j][2]:12.8f}\n")

    print(f"{GREEN}[Success]{RESET} Lowest energy structure saved: {min_xyz_name}")

    # --- PLOTTING ---
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(delta_e_kcal)), delta_e_kcal, '-', color='gray', alpha=0.3)
    
    if has_recovered_point:
        plt.scatter(range(len(delta_e_kcal)-1), delta_e_kcal[:-1], color='#1f77b4', label='Converged Step')
        plt.scatter(len(delta_e_kcal)-1, delta_e_kcal[-1], color='red', s=100, label='Crashed Point', zorder=5)
    else:
        plt.scatter(range(len(delta_e_kcal)), delta_e_kcal, color='#1f77b4', label='Scan Step')

    plt.title(f"PES Profile ({'Relaxed' if is_relaxed else 'Rigid'})\n{os.path.basename(file_path)}")
    plt.xlabel("Scan Point Index")
    plt.ylabel("Relative Energy (kcal/mol)")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    
    plot_name = f"{base_name}_plot.png"
    plt.savefig(plot_name, dpi=300)
    print(f"{GREEN}[Done]{RESET} Plot saved: {plot_name}")
    plt.show()

if __name__ == "__main__":
    main()
