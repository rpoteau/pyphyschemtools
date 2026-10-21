# tools4pyPhysChem/__init__.py
"""
The pyphyschemtools library provides a comprehensive suite of utilities for Physical Chemistry, ranging from spectroscopic unit management to kinetic modeling and cheminformatics.
"""

__version__ = "0.12.2"
__last_update__ = "2026-10-21"

import importlib
import importlib.util
from pathlib import Path
from importlib import resources

# 1. FAST IMPORTS
from .visualID_Eng import fg, hl, bg, color, init, apply_css_style, chrono_start, chrono_stop, chrono_show, end
from .core import centerTitle, centertxt, crop_images, save_fig, save_data, get_qc_examples

# Explicitly define what is already imported so that __getattr__ does not interfere.
_EXPLICIT_EXPORTS = {
    "fg", "hl", "bg", "color", "init", "apply_css_style", 
    "chrono_start", "chrono_stop", "end", "centerTitle", 
    "centertxt", "crop_images", "save_fig", "save_data", "get_qc_examples"
}

# 2. AUTOMATIC LAZY LOADING
def __getattr__(name):
    # Si l'attribut est dans les imports explicites, on ne devrait pas être ici,
    # mais au cas où, on le gère.
    if name in _EXPLICIT_EXPORTS:
        # On le récupère dans le namespace local
        return globals()[name]

    modules_to_search = [
        ".AO", ".ML", ".PeriodicTable", ".Chem3D",  
        ".aithermo", ".cheminformatics", ".colab_utils", ".huckel", ".hydrogen",
        ".kinetics", 
        ".misc", ".nano", ".orgachem",
        ".spectro", ".survey", ".symmetry",
        ".sympyUtilities", ".tools4AS", ".units"
    ]

    import_errors = []
    
    for mod_name in modules_to_search:
        try:
            # On tente l'import relatif
            module = importlib.import_module(mod_name, __package__)

            # Effet de bord documenté de Python : importer un sous-module le lie
            # automatiquement comme attribut du package parent, sous son propre
            # nom (ex: pyphyschemtools.PeriodicTable = <module ...>). Si un module
            # définit un objet (classe, fonction) portant EXACTEMENT le même nom
            # que lui-même (ex: PeriodicTable.py / class PeriodicTable), cet effet
            # de bord écrase silencieusement l'entrée déjà posée par ce lazy
            # loader avec le module, au lieu de la classe — et ce dès le tout
            # premier import, quel que soit l'attribut initialement demandé. On
            # restaure donc systématiquement la bonne valeur ici.
            bare_mod_name = mod_name.lstrip(".")
            if (
                bare_mod_name != name
                and hasattr(module, bare_mod_name)
                and getattr(module, bare_mod_name) is not module
            ):
                globals()[bare_mod_name] = getattr(module, bare_mod_name)

            if hasattr(module, name):
                value = getattr(module, name)
                globals()[name] = value
                return value
        except (ImportError, AttributeError) as e:
            # Track errors to help the user diagnose missing dependencies
            import_errors.append(f"  - In {mod_name}: {e}")
            continue

    # Final error message if the attribute is not found
    error_msg = f"module {__name__} has no attribute '{name}'."
    if import_errors:
        error_msg += "\nPossible causes (errors encountered during scan):"
        for err in import_errors:
            error_msg += f"\n{err}"

    raise AttributeError(error_msg)

def get_ppct_data(file: str, main_folder: str="data_examples") -> Path:
    """
    Retrieves the absolute path to a pyphyschemtools resource.
    Compatible with Python 3.11+ using importlib.resources.
    Works across all platforms (Windows, Linux, MacOS) and Google Colab.
    
    Args:
        file (str): The path and name of the file (e.g., "Molecules/betaCD-closed.xyz").
        main_folder (str): The resource subfolder inside the package (defaults to "data_examples").
    
    Returns:
        Path: A pathlib.Path object pointing to the absolute location of the file.
    """
    # Access the data_examples directory inside the installed pyphyschemtools package
    # .files() returns a Traversable object (behaving much like a Path object)
    data_dir = resources.files("pyphyschemtools") / main_folder
    
    # Construct the full internal path
    file_path = data_dir.joinpath(file)
    
    # Verify if the file actually exists
    if not file_path.exists():
        raise FileNotFoundError(
            f"❌ Resource not found: {file}\n"
            f"Expected internal location: pyphyschemtools/{main_folder}/{file}"
        )
        
    # Convert to a standard pathlib.Path object for the user
    return Path(str(file_path))
