# Quantum Chemistry Corner

## 🚀 Brief introduction, dependencies and examples

A collection of specialized tools for **VASP** and **Gaussian** post-processing and job management are also provided with the `pyphyschemtools` library.

> [!IMPORTANT]
> Except for a few Python utilities (`pos2cif`, `pos2xyz`, `cif2pos`), these tools are designed for **Unix-based systems** and have been primarily tested on openSUSE.

**Dependencies:**

- python 3.8+
- the `ase` (for the VASP tools) and `cclib` (for the Gaussian tools) python libraries
- `bash`
- linux fortran library
- the `bc` unix calculator

**Examples**

To help you get started, several reference datasets (including `POSCAR`, `OUTCAR`, and `Gaussian` .log files) are included in the package distribution. Rather than cluttering your workspace, these are stored as compressed archives.

You can retrieve these examples using the built-in `get_qc_examples` CLI utility:

```bash
# To get VASP and LOBSTER examples:
get_qc_examples VASP

# To get Gaussian 16 examples:
get_qc_examples G16
```

---

## ⚛️ Tools for VASP

### 📖 Citation
Some of these tools are mentioned in:
> L. Cusinato, I. del Rosal, R. Poteau (**2017**). Shape, electronic structure and steric effects of organometallic nanocatalysts: relevant tools to improve the synergy between theory and experiments. *Dalton Trans.* **46**: 378-395. [DOI: 10.1039/C6DT04207D](https://dx.doi.org/10.1039/C6DT04207D)

### ✅ **VASPcv** (bash & python)
Reads `OUTCAR` files and quickly prints an overview of VASP output. Uses the `grad2.py` tool of Peter Larsson.
* **Command:** <span style="color:darkcyan; font-family:monospace; font-weight:bold;">VASPCV OUTCAR_file[.gz]</span>

    Automatically gunzips `OUTCAR` files, prints the overview, and gzips it at the end of the process
* **Current version:** 20240311

### ✅ **cpVASP** (bash)
Copies the necessary input files (`INCAR`, `POTCAR`, `POSCAR`, `KPOINTS`, `VASP.runjob`) of a VASP folder into a new one.
* **Command:** <span style="color:darkcyan; font-family:monospace; font-weight:bold;">cpVASP SOURCE_FOLDER DESTINATION_FOLDER *n*</span>
* **Note:** The destination folder is created automatically and the job name in the runjob is updated (*edit the script to rename VASP.runjob to your own runjob command*)
* **Option *n* (optional, default: 0):**
    * 0: Copy `POSCAR` of the source as `POSCAR` in the destination.
    * 1: Copy `CONTCAR` of the source as `POSCAR` in the destination.
* **Current version**: 20240307

### ✅ **RestartVASP** (bash)
Run it from a job folder that has abruptly stopped (time limit, number of optimization steps exceeded etc.). Makes an incremental copy of the `CONTCAR` and `OUTCAR` files.

The tool also includes logic to check for the presence of `OUTCAR` and `CONTCAR` files within a temporary working directory. This is controlled by the `tmpdir` variable at the start of the script - <span style="color:red; font-family:monospace; font-weight:bold;">don't forget to change it</span>. 
**Why this is useful**: On many HPC (High Performance Computing) clusters, calculations run on a local "scratch" or "tmp" disk for better performance. If a job crashes or times out, the files might remain in that temporary location. This update allows RestartVASP to "rescue" those files even if they haven't been moved back to your home/main directory yet. **If left empty or undefined**, `RestartVASP` defaults to standard behavior, processing files only within the local directory.

* **Command:** <span style="color:darkcyan; font-family:monospace; font-weight:bold;">RestartVASP n</span> (where `n` is the restart index)
* **n=1:** Copies `POSCAR`, `CONTCAR`, `OUTCAR` as `POSCAR.0`, `CONTCAR.1`, `OUTCAR.1`. `CONTCAR.1` becomes the new `POSCAR`.
* **n <> 1:** Copies `CONTCAR` as `CONTCAR.n` and the new `POSCAR`. `OUTCAR` is saved as `OUTCAR.n`.
* **Current version**: 20240130

### ✅ **cleanVASPf** (bash & python)
Run it from a job folder you intend to archive. Keeps only essential files (`INCAR`, `KPOINTS`, `OUTCAR`, `POSCAR`, `POSCAR.0`) and creates `.xyz` and `.cif` versions of the coordinates files. Gzips `OUTCAR`, and also handles a previously gzipped `OUTCAR.gz` present in the folder.
* **Command:** <span style="color:darkcyan; font-family:monospace; font-weight:bold;">cleanVASPf</span>
* **Tip (Recursive gzip):** <span style="color:red; font-family:monospace; font-weight:bold;">find . -type f -name OUTCAR -exec gzip {} \;</span>
* **Current version**: 20211115

### ✅ **ManipCell** (fortran binary)
Rotation around z/c and size adjustment of any unit cell with c orthogonal to a and b.
* **Command:** <span style="color:darkcyan; font-family:monospace; font-weight:bold;">ManipCell -POS POSCAR_file [Options]</span>, where options are <span style="color:darkcyan; font-family:monospace; font-weight:bold;">-a angle -fa value -fb value -fc value -S value -sha value -shb value -shc value -c c_new -T value -OC a_new b_new c_new -GA</span>
* **Returns:** new `POSCAR_file_MC` and `POSCAR_file_MC.cif` files
* **Options:**
    * `-a float`: rotation angle /c (degree, default: 0.0).
    * `-fa float, -fb float, -fc float`: size of the target unitcell (in units of a, b and c, default: 1.0).
    * `-S float`: scale all a,b final positions by S factor (should be close to 1.00, default: 1.0).
    * `-sha float`: shift atoms along a axis by sha factor (in reduced units, default: 0.0)
    * `-shb float`: shift atoms along b axis by shb factor (in reduced units, default: 0.0)
    * `-shc float`: shift atoms along c axis by shc factor (in reduced units, default: 0.0)
    * `-c float`: change c to the new value (given in Å; all previous operations are done w.r.t. the initial a,b,c basis).
    * `-T float`: threshold to remove atoms that lie outside the unitcell (try to lower it to 1e-10 to check the accuracy of your data, default: 0.02)
    * `-OC a(float) b(float) c(float)`: change size of an orthorombic unitcell to a_new & b_new & c_new (whilst -c involves only the c parameter).
    * `-GA`: topological analysis based on Graph Theory. Returns the number of molecules in a unitcell and the atom indexes/molecule (please provide a `radii.dat` file. One line per atom type. Same order as in the CONTCAR file. Radii in pm).
* **Current version**: 20241102

### ✅ **sel4vibVASP** (fortran binary)
Set up of an harmonic frequencies and modes calculation (selective dynamics).  Recommended INCAR file options: `IBRION = 5; NFREE = 2; NSW = 1; POTIM=0.0005; ediff=1.e-7`.

* **Command:** <span style="color:darkcyan; font-family:monospace; font-weight:bold;">sel4vibVASP -POS POSCAR_file -V SELECTCAR_file</span>
 
    Selects the atoms to consider in a `SELECTCAR file`: one line per atom. Each line is made of two values: a central atom (integer number) followed by the radius (float number) to select atoms within this radius.
* **Returns:** a `POSCAR_filev` file.
* **Current version**: 20190407

### ✅ **vibVASP** (fortran binary)
Returns all calculated modes as a single `modes.xyz` file (xyzvib format), whilst vibrational frequencies (in cm-1) are saved in a `freq.dat` file.
* **Command:** <span style="color:darkcyan; font-family:monospace; font-weight:bold;">vibVASP -POS POSCAR_file -O OUTCAR_file [-m int float]</span>

    (warning: creates a temporary `MODES` folder, deleted at the end of the command)
* **Option [-m]:** for transition states (TS), can be interesting to do a very basic reaction path following. In this case select the imaginary normal mode #k, and change the coordinates with a small amplitude, e.g. ±0.5 (`vibVASP -m 1 0.5 or vibVASP -m 1 -0.5`).
* **Current version**: 20190407

### ✅ **ThermoWithVASP** (fortran binary)
Computes thermochemical values after a normal modes calculation with VASP (same equations as used in the Gaussian software).
* **Command:** <span style="color:darkcyan; font-family:monospace; font-weight:bold;">ThermoWithVASP [Options]</span>
* **Options:**
    * `-T value`: temperature (default: 298.15K)
    * `-P value`: pressure (default: 1atm)
    * `-F file`: name of the frequencies file (default: freq.dat)
        - 1 freq/line (in cm-1). Lines that start with '!' won't be considered
        - VASP format: `frequency value  'cm^{-1}'   '...'  extremum index`
        - you can write as many informations as you want after each extremum index value (such as the description of the mode nu_CO etc...)
    * `-C file`: name of the coordinates file (default: CONTCAR.xyz)
    * `-O file`: name of the OUTCAR file (default: OUTCAR)
    * `-t T/F`: contribution from translation        (default: F(alse))
    * `-r T/F`: contribution from rotation           (default: F(alse))
    * `-e T/F`: contribution from electronic motion  (default: F(alse))
    * `-v T/F`: contribution from vibrational motion (default: T(rue))
    * `-S value`: rotational symmetry number (default: 1.0)
        * <span style="font-family:monospace"> C1, Ci, Cs, Cinfv: 1  
        *          Cn, Cnv, Cnh: n  
        *                 Dinfh: 2  
        *          Dn, Dnh, Dnd: 2n 
        *                 T, Td: 12 
        *                    Sn: n/2
        *                    Oh: 24 
        *                    Ih: 60
</span>

* **Current version**: 2021202411312

### ✅ **selectLOBSTER** (fortran binary)
Reads LOBSTER files and helps selecting relevant data to create COHP, COOP and DOS diagrams under the xmgrace format.

LOBSTER does crystal orbital Hamilton population (COHP) analysis. A COHP diagram indicates bonding and antibonding contributions to the band-structure energy, and it is usually plotted alongside the DOS. Regarding VASP, the added-value is the projection of the PW wavefunction into a local slater-type orbital basis set.

* **Command:** <span style="color:darkcyan; font-family:monospace; font-weight:bold;">selectLOBSTER -F input_file</span>
* **Requires:**
    * files provided by LOBSTER: `ICOOPLIST.lobster`, `ICOHPLIST.lobster`, `COOPCAR.lobster`, `COHPCAR.lobster`, `DOSCAR.LOBSTER`, and `CONTCAR.xyz`.
    * `COBTCAR. xyz`
    * a `CoreSurfaceAtoms.dat` file is needed if the keywords core or surface are used in the selectLOBSTER.in input file
* **Current version**: 20250121

### ✅ **hVASP** (bash)
Useful to create archives for SI or for data management. All subfolders of the input folder will recursively be saved in a `VASP-ARK` folder, after transformation into xyz and cif files. xyz and cif files will be named after the name of the parent VASP folder. Energies will also be written in the title (xyz) or data_ (cif) sections.

* **Command:** <span style="color:darkcyan; font-family:monospace; font-weight:bold;">hVASP -F FolderName</span>
* **Options:**
    - `-flag 2ARK-FileName`: name of the 2ARK flag (default: empty `2ARK` file in each folder to be archived; do not forget to create it in each subfolder you want to archive the coordinates, by simply running the `touch 2ARK` bash command in each target folder).

      If not empty, the content of the `2ARK-FileName` will be added to the xyz and cif titles
    - `-A`: systematic recursive archiving, ie without considering the presence of a 2ARK file (default: false)
    - `-F name`: name of the parent folder that contains subfolders that will be archived (default: current folder)
    - `-LD boolean value` : if true, rename the `CONTCAR` and `OUTCAR` files after the name of *all* their parent folders

      *i.e.* `Folder1/Folder2/OUTCAR` will be saved as `Folder1_Folder2_OUTCAR.gz` if true, otherwise it will simply be saved as `Folder2_OUTCAR.gz` (default: false) 
    - `-O`: save OUTCAR files as well (will be saved in an `OUTCARs` subfolder; default: false)
    - `-CIF`: create a cif file from the `CONTCAR` and save it in a `CIF` folder (default: false)
    - `-POS`: save the `POSCAR` file as well in a `POSCARs` folder (default: false)
    - `-h`: print only this help and exit
* **Current version**: 20260213

---

## ⚛️ Tools for Gaussian

### ✅ **GParser** (bash)
Reads Gaussian log files and quickly prints an overview of its content.
* **Command:** <span style="color:darkcyan; font-family:monospace; font-weight:bold;">GParser -f log_file [Options]</span>
* **General command:** <span style="color:darkcyan; font-family:monospace; font-weight:bold;">GParser -f file.log -s float_number -srH float_number -srC float_number -srN float_number -srO float_number" -nbo atom_number -t float_number -pf -S -atd</span>
    * `-f file`: name of the log file (default: default.log)
    * `-s float`: scaling factor for frequencies (default: 1.0). not yet operational
    * `-srH float`: reference chemical shielding value for 1H (usually TMS. Default:  31.76)
    * `-srC float`: reference chemical shielding value for 13C (usually TMS. Default: 191.81)
    * `-srN float`: reference chemical shielding value for 15N (usually liquid ammonia. Default: 242.8)
    * `-srO float`: reference chemical shielding value for 17O (usually liquid water. Default: 280.2)
    * `-nbo integer`: print all Second Order PTA that involve atom_number. Default: 0
    * `-t float`: threshold to print only NBO interactions > float_number (Default:0.0)
    * `-pf`: prints all vibration frequencies
    * `-S`: save freq/NPA/TDDFT/NMR/last_xyz data, if available, in the log file (Default = false)
    the energy is save in the title section, as well as ZPE, H° and G° if a frequency calculation is available in the same file
    * `-atd`: save all tddft calculations, if available, in the log file (series of TDDFT calc. in a single run, for example after a MD simulation. Default = false). Sets -S flag
    * `-h`: print this help section
* **Current version**: 20250709

### ✅ **cpG** (bash)
Copy of the necessary input files (`.com`, `.qsub`) of a Gaussian calculation as a new one. The job name in the `file.qsub` is also automatically changed (edit the script to rename `file.qsub` to your own submission command)
* **Command:** <span style="color:darkcyan; font-family:monospace; font-weight:bold;">cpG SOURCE_FILES DESTINATION_FILES [R]</span>
* **R [optional]:** Reuse MOs or last geometry from the `.chk` file.
* **Current version**: 20230304

### ✅ **GScan_Analyzis** (python)
A Python-based post-processing tool for Gaussian Potential Energy Surface (PES) scans. It identifies optimized scan points, including checking optimization status and "energy valley" detection.

* **Command:** <span style="color:darkcyan; font-family:monospace; font-weight:bold;">GScan_Analyzis log_file`</span>: plots the optimal energy profile, saves the optimized geometries for each scan point, and exports the structure with the lowest energy. It remains fully functional even if the scan is interrupted or incomplete.
* **Current version**: 20260211

---

## 🤝 How to contribute?

The <span style="color:darkcyan; font-weight:bold;">Quantum Chemistry Corner</span> is a collaborative space. If you have developed a script or a tool (Python, Bash, C or Fortran) that could benefit the computational chemistry community, we welcome your contributions!

To maintain the stability and portability of the pyphyschemtools package, all integrations are supervised by the maintainer (Romuald Poteau, romuald.poteau@utoulouse.fr). Here is the workflow and the technical requirements for submission.

### 📋 Technical Requirements

Each tool must be self-contained and adhere to the following standards:

- **Python Scripts (.py)**:
  - Must include the shebang: `#!/usr/bin/env python3`.
  - Code must be wrapped inside a `main()` function to be compatible with Python entry points.
  - Any external dependencies (e.g., `numpy`, `ase`, `cclib`) must be clearly listed.
- **Bash Scripts**:
    - Must include the shebang: `#!/bin/bash`.
    - No absolute paths: Do not use hardcoded paths (e.g., /home/user/bin/). Only use commands available in the standard $PATH.
- **Compiled Tools (Fortran / C)**:
    - You must provide the source code (`.f90`, `.c`, etc.) for long-term maintenance.
    - Binaries should be compiled statically where possible to ensure they run on different Linux distributions.

### 📖 Documentation Requirements

To be accepted, every new tool must come with a short documentation snippet (in Markdown) to be included in `QuantumChemCorner.md`. Please provide the following details:
- **Tool Name & Type**: (e.g., `MyTool` - Bash script).
- **Short Description**: One or two sentences explaining what the tool does.
- **Command Syntax**: A clear example of how to run it in the terminal.
- **Arguments/Flags**: Explanation of any required or optional parameters.
- **Example Case**: A brief description of the expected output.

### 🚀 Integration Procedure

If your tool meets the requirements:

- **Prepare a bundle**: Include your script/source, a brief description of what it does, and a usage example.
- **Submit to the Maintainer**: I will handle the backend integration, which includes:
    - Placing files in the correct directories (`bash_scripts/`, `bin/`, or the corner root).
    - Creating the Python Wrapper in `wrappers.py` so the tool is accessible globally.
    - Registering the official command in `pyproject.toml`.
    - Validation: A local installation in editable mode (`pip install -e .`) will be performed to ensure the new command works perfectly in the terminal.

### 💡 Why the "Wrapper" approach?

We don't just "drop" scripts into the folder. By using Python wrappers, we ensure:
- **Encoding Safety**: Binary files (like Fortran executables) don't break the PyPI upload process.
- **Global Access**: Once `pyphyschemtools` is installed, your tool is available as a standalone command (e.g., just type `MyNewTool` in the terminal).
- **Argument Handling**: Python's subprocess module safely handles the hand-off of flags and filenames from the shell to your script.