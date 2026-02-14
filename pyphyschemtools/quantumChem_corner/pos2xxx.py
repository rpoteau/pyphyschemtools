#!/usr/bin/env python3
import sys, io, os
from ase.io import read, write

VERSION = "20260213"

def fix_xyz(output, full_symbols):
    with open(output, 'r') as f:
        lines = f.readlines()
    for i, s in enumerate(full_symbols):
        if s == 'D': lines[i+2] = lines[i+2].replace('H', 'D', 1)
    with open(output, 'w') as f: f.writelines(lines)

def fix_cif(output, full_symbols):
    with open(output, 'r') as f:
        lines = f.readlines()
    start = 0
    for idx, line in enumerate(lines):
        if "_atom_site_occupancy" in line:
            start = idx + 1
            break
    for i, s in enumerate(full_symbols):
        if s == 'D': lines[start+i] = lines[start+i].replace('H', 'D')
    with open(output, 'w') as f: f.writelines(lines)

def main():
    name = os.path.basename(sys.argv[0])
    
    # Gestion de l'option version
    if "-v" in sys.argv:
        print(f"{name} version {VERSION}")
        sys.exit(0)

    fmt = "cif" if "cif" in name else "xyz"
    if len(sys.argv) < 2:
        print(f"Usage: {name} [-v] <POSCAR>"); sys.exit(1)
    
    path = sys.argv[1]
    try:
        with open(path, 'r') as f: lines = f.readlines()
        raw_syms = lines[5].split()
        counts = [int(c) for c in lines[6].split()]
        full_syms = []
        for s, c in zip(raw_syms, counts): full_syms.extend([s] * c)

        # Masquage pour ASE
        new_lines = list(lines)
        new_lines[5] = "  ".join(["H" if s == "D" else s for s in raw_syms]) + "\n"
        atoms = read(io.StringIO("".join(new_lines)), format='vasp')

        out = f"{path}.{fmt}"
        write(out, atoms, format=fmt)

        if fmt == 'xyz': fix_xyz(out, full_syms)
        else: fix_cif(out, full_syms)
        print(f"✅ {name} (v{VERSION}) terminé -> {out}")
    except Exception as e: print(f"❌ Erreur : {e}")

if __name__ == "__main__": main()
