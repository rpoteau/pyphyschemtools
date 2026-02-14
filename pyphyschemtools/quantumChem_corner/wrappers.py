import subprocess
import os
import sys

def run_script(subfolder, filename):
    """Fonction générique pour lancer un script Bash ou un binaire Fortran."""
    # Localise le fichier à l'intérieur du package installé
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, subfolder, filename)
    
    if not os.path.exists(file_path):
        print(f"Erreur : {filename} introuvable dans {file_path}")
        sys.exit(1)
        
    # Exécute le fichier en passant tous les arguments reçus (sys.argv)
    try:
        # Pour les scripts Bash, on force l'usage de bash au cas où le bit d'exécution serait perdu
        if subfolder == "bash_scripts":
            cmd = ["bash", file_path] + sys.argv[1:]
        else:
            cmd = [file_path] + sys.argv[1:]
            
        subprocess.run(cmd)
    except KeyboardInterrupt:
        sys.exit(0)

# Wrappers pour les scripts Bash
def vasp_cv(): run_script("bash_scripts", "VASPcv")
def restart_vasp(): run_script("bash_scripts", "RestartVASP")
def cp_g(): run_script("bash_scripts", "cpG")
def cp_vasp(): run_script("bash_scripts", "cpVASP")
def clean_vasp(): run_script("bash_scripts", "cleanVASPf")
def g_parser(): run_script("bash_scripts", "GParser")
def h_vasp(): run_script("bash_scripts", "hVASP")
def gp2bw(): run_script("bash_scripts", "GP2bw")

# Wrappers pour les binaires Fortran
def manip_cell(): run_script("bin", "ManipCell")
def thermo_vasp(): run_script("bin", "ThermoWithVASP")
def vib_vasp(): run_script("bin", "vibVASP")
def sel4vib_vasp(): run_script("bin", "sel4vibVASP")
def select_lobster(): run_script("bin", "selectLOBSTER")
