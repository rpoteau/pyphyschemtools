#!/usr/bin/env python3
import sys
import io
import os
import re
from ase.io import read, write

VERSION = "20260213"

def clean_symbol(label):
    """Extrait le symbole pur sans les chiffres pour ASE."""
    match = re.match(r"([a-zA-Z]+)", label)
    if match:
        sym = match.group(1)
        # On retourne H même pour D car ASE ne connaît pas D à la lecture
        return 'H' if sym == 'D' else sym
    return label

def convert():
    name = os.path.basename(sys.argv[0])
    
    # Gestion de l'option version
    if "-v" in sys.argv:
        print(f"{name} version {VERSION}")
        sys.exit(0)

    if len(sys.argv) < 2:
        print(f"Usage: {name} [-v] <file.cif>"); sys.exit(1)
    
    file_path = sys.argv[1]
    base_name = os.path.splitext(file_path)[0]
    output_name = f"{base_name}_POSCAR"
    
    try:
        # 1. Analyse du CIF original pour garder L'ORDRE EXACT
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        full_symbols = []
        for line in lines:
            parts = line.split()
            # Détection des lignes d'atomes (H, D, Pd, C, N...)
            if len(parts) >= 5 and parts[0].strip().startswith(('H', 'D', 'Pd', 'C', 'N')):
                label, element = parts[0], parts[1]
                if label.startswith('D') or element == 'D':
                    full_symbols.append('D')
                else:
                    full_symbols.append(clean_symbol(element))

        # 2. Lecture par ASE (Masquage D -> H)
        cif_clean = "".join(lines).replace(' D ', ' H ').replace('\nD ', '\nH ')
        atoms = read(io.StringIO(cif_clean), format='cif')
        
        # 3. On assure la correspondance des symboles standards pour ASE
        atoms.set_chemical_symbols(['H' if s == 'D' else s for s in full_symbols])

        # 4. Écriture du POSCAR via ASE (format vasp)
        write(output_name, atoms, format='vasp')

        # 5. RECONSTRUCTION MANUELLE DES LIGNES 6 ET 7 (Respect de l'ordre)
        unique_groups = []
        counts = []
        
        if full_symbols:
            current_sym = full_symbols[0]
            count = 0
            for s in full_symbols:
                if s == current_sym:
                    count += 1
                else:
                    unique_groups.append(current_sym)
                    counts.append(count)
                    current_sym = s
                    count = 1
            unique_groups.append(current_sym)
            counts.append(count)

        with open(output_name, 'r') as f:
            pos_lines = f.readlines()

        # Injection des groupes incluant les isotopes D
        pos_lines[5] = "  ".join(unique_groups) + "\n"
        pos_lines[6] = "  ".join(map(str, counts)) + "\n"

        with open(output_name, 'w') as f:
            f.writelines(pos_lines)
        
        print(f"✅ {name} (v{VERSION}) terminé -> {output_name}")
        print(f"   Séquence : {' '.join(unique_groups)} ({' '.join(map(str, counts))})")

    except Exception as e:
        print(f"❌ Erreur : {e}")

if __name__ == "__main__":
    convert()
