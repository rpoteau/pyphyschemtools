# tools4pyPhysChem/__init__.py
__version__ = "0.3.1"
__last_update__ = "2026-02-02"

import importlib
import importlib.util

# 1. FAST IMPORTS
from .visualID_Eng import fg, hl, bg, color, init, apply_css_style, chrono_start, chrono_stop, end
from .core import centerTitle, centertxt, crop_images

# On définit explicitement ce qui est déjà importé pour que __getattr__ ne s'en mêle pas
_EXPLICIT_EXPORTS = {
    "fg", "hl", "bg", "color", "init", "apply_css_style", 
    "chrono_start", "chrono_stop", "end", "centerTitle", 
    "centertxt", "crop_images"
}

# 2. AUTOMATIC LAZY LOADING
def __getattr__(name):
    # Si l'attribut est dans les imports explicites, on ne devrait pas être ici,
    # mais au cas où, on le gère.
    if name in _EXPLICIT_EXPORTS:
        # On le récupère dans le namespace local
        return globals()[name]

    modules_to_search = [
        ".ML", ".PeriodicTable", ".Chem3D",  
        ".aithermo", ".cheminformatics", ".kinetics", 
        ".spectra", ".survey",  
        ".sympyUtilities", ".tools4AS"
    ]
    
    for mod_name in modules_to_search:
        try:
            # On tente l'import relatif
            module = importlib.import_module(mod_name, __package__)
            if hasattr(module, name):
                return getattr(module, name)
        except (ImportError, AttributeError):
            continue

    raise AttributeError(f"module {__name__} has no attribute {name}")
