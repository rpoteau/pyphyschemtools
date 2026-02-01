# tools4pyPhysChem/__init__.py
import importlib
import importlib.util

# 1. FAST IMPORTS
from .visualID_Eng import fg, hl, bg, color, init, apply_css_style, chrono_start, chrono_stop, end
from .core import centerTitle, centertxt, crop_images

# 2. AUTOMATIC LAZY LOADING
def __getattr__(name):
    modules_to_search = [
        ".ML", ".PeriodicTable", ".Chem3D", 
        ".aithermo", ".cheminformatics", ".kinetics"
        ".spectra", ".survey", 
        ".sympyUtilities", ".tools4AS"
    ]
    
    for mod_name in modules_to_search:
        # Get the full path (e.g., tools4pyPhysChem.ML)
        full_mod_path = importlib.util.resolve_name(mod_name, __package__)
        spec = importlib.util.find_spec(full_mod_path)
        
        if spec is not None:
            # We import the module to check its contents
            module = importlib.import_module(mod_name, __package__)
            if hasattr(module, name):
                return getattr(module, name)

    raise AttributeError(f"module {__name__} has no attribute {name}")

