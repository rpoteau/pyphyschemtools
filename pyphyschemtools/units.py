import re
import numpy as np
import scipy.constants as const
import pandas as pd
from IPython.display import display, HTML
from .core import centerTitle, centertxt
from .visualID_Eng import color, fg, hl, bg

class Energy:
    """
    An advanced utility class for Energy management in Physical Chemistry.

    This class handles the conversion between standard energy units (J, eV), 
    thermal energy (K), and spectroscopic units (wavelengths in m/nm/Å or 
    wavenumbers in cm-1).

    Features:
    - Supports scalar values and NumPy arrays.
    - Automatic handling of SI prefixes (milli, micro, kilo, etc.).
    - Seamless conversion between energy (E) and wavelength (lambda) using E = hc/lambda.
    - Support for reciprocal space units (unit-1) and molar units (unit/mol).
    - Integrated error guidance for users

    Accessing numerical values:
    - Use float(obj) to get the scalar value.
    - Use obj.value to get the NumPy array.
    """

    _BASE_TO_JOULE = {
        'J': 1.0, 
        'eV': const.e, 
        'hartree': const.value("Hartree energy"),
        'K': const.k,
        'cal': const.calorie, # Added to support kcal/mol via the same logic
        'm': None,   
        'Å': None,   
    }

    _PREFIX_DATA = {
        # Symbol: (Factor, Prefixe_SI, Name_FR, Name_EN)
        'q': (1e-30, 'quecto', 'quintillionième', 'nonillionth'),
        'r': (1e-27, 'ronto',  'quadrilliardième', 'octillionth'),
        'y': (1e-24, 'yocto',  'quadrillionième', 'septillionth'),
        'z': (1e-21, 'zepto',  'trilliardième', 'sextillionth'),
        'a': (1e-18, 'atto',   'trillionième', 'quintillionth'),
        'f': (1e-15, 'femto',  'billiardième', 'quadrillionth'),
        'p': (1e-12, 'pico',   'billionième', 'trillionth'),
        'n': (1e-9,  'nano',   'milliardième', 'billionth'),
        'µ': (1e-6,  'micro',  'millionième', 'millionth'),
        'm': (1e-3,  'milli',  'millième', 'thousandth'),
        'c': (1e-2,  'centi',  'centième', 'hundredth'),
        'd': (1e-1,  'deci',   'dixième', 'tenth'),
        'da':(1e1,   'deca',   'dizaine', 'ten'),
        'h': (1e2,   'hecto',  'centaine', 'hundred'),
        'k': (1e3,   'kilo',   'mille', 'thousand'),
        'M': (1e6,   'mega',   'million', 'million'),
        'G': (1e9,   'giga',   'milliard', 'billion'),
        'T': (1e12,  'tera',   'billion', 'trillion'),
        'P': (1e15,  'peta',   'billiard', 'quadrillion'),
        'E': (1e18,  'exa',    'trillion', 'quintillion'),
        'Z': (1e21,  'zetta',  'trilliard', 'sextillion'),
        'Y': (1e24,  'yotta',  'quadrillion', 'septillion'),
        'R': (1e27,  'ronna',  'quadrilliard', 'octillion'),
        'Q': (1e30,  'quetta', 'quintillion', 'nonillion'),
    }

    _PREFIXES = {k: v[0] for k, v in _PREFIX_DATA.items()}

    @classmethod
    def show_constants_metadata(cls):
        """
        Displays a styled table of the physical constants used for calculations,
        including their current CODATA values, units, and uncertainties.
        """
        # Map of constants used in the class to their official Scipy/CODATA keys
        constants_to_show = {
            "h": "Planck constant",
            "c": "speed of light in vacuum",
            "e": "elementary charge",
            "k": "Boltzmann constant",
            "N_A": "Avogadro constant",
            "E_h": "Hartree energy",
            "cal": "thermochemical calorie"
        }

        data = []
        for symbol, name in constants_to_show.items():
            try:
                # Scipy returns (value, unit, uncertainty)
                val, unit, unc = const.physical_constants[name]
                data.append({
                    "Symbol": symbol,
                    "CODATA Description": name,
                    "Value": val,
                    "Uncertainty": unc,
                    "Unit": unit
                })
            except KeyError:
                # Handle constants like 'cal' which aren't in physical_constants
                if symbol == "cal":
                    data.append({
                        "Symbol": "cal",
                        "CODATA Description": "Thermochemical calorie",
                        "Value": const.calorie,
                        "Uncertainty": 0.0,
                        "Unit": "J"
                    })

        df = pd.DataFrame(data)
        
        centerTitle("Physical Constants Metadata fro SciPy (CODATA)")
        # Use scientific notation for small/large numbers
        styled_df = df.style.format({
            "Value": "{:.8e}",
            "Uncertainty": "{:.2e}"
        }).hide(axis='index')
        
        display(styled_df)
        
    def __init__(self, value, unit='J'):
        """
        Initialize the Energy object.
        
        Args:
            value (float or array-like): Numerical value(s).
            unit (str): Unit string (e.g., 'J', 'eV', 'nm', 'cm-1', 'dam').
        """
        self.unit = unit
        raw_data = np.asanyarray(value)
        # If it's a single value (0-dimensional), extract it as a scalar
        if raw_data.ndim == 0:
            self.value = raw_data.item()
        else:
            self.value = raw_data
        self._joules = self._calculate_joules(self.value, self.unit)

    def _calculate_joules(self, val, unit_str):
        """
        Internal dispatcher to convert any supported unit to Joules.
        Handles reciprocal units (unit-1) and wavelength units (m, nm, etc.).
        """
        # 1. Special case: Angstrom (Å) and Reciprocal Angstrom (Å-1)
        if unit_str in ['Å', 'A', 'angstrom']:
            return (const.h * const.c) / (val * 1e-10)
        if unit_str in ['Å-1', 'A-1']:
            return val * (const.h * const.c / 1e-10)
        
        # 2. Reciprocal units (e.g., cm-1, nm-1, dam-1)
        if unit_str.endswith('-1'):
            base_len = unit_str[:-2]
            factor = self._get_conversion_factor(base_len)
            return val * (const.h * const.c / factor)

        # 3. Wavelength units (e.g., m, nm, km, dam)
        # Check if it is a pure length or prefixed length
        is_length = False
        if unit_str == 'm':
            is_length = True
        elif unit_str.endswith('m'):
            prefix = unit_str[:-1]
            if prefix in self._PREFIXES or prefix == 'da':
                is_length = True

        if is_length:
             factor = self._get_conversion_factor(unit_str)
             return (const.h * const.c) / (val * factor)

        # 4. Direct Energy units (J, eV, K, kJ/mol...)
        return val * self._get_conversion_factor(unit_str)

    def __float__(self):
        """Explicit scalar conversion for the object itself."""
        if isinstance(self.value, (float, int, np.float64)):
            return float(self.value)
        raise TypeError("This Energy object contains an array and cannot be converted to a single float.")

    def __len__(self):
        """Returns the number of elements in the energy object."""
        if isinstance(self.value, np.ndarray):
            return len(self.value)
        return 1 # A scalar has length 1
        
    def _get_conversion_factor(self, unit_str):
        # 1. Direct hit
        if unit_str in self._BASE_TO_JOULE and self._BASE_TO_JOULE[unit_str] is not None:
            return self._BASE_TO_JOULE[unit_str]

        # 2. Dynamic Molar Units
        if unit_str.endswith('/mol'):
            energy_part = unit_str.replace('/mol', '')
            return self._get_conversion_factor(energy_part) / const.Avogadro

        # 3. Meter logic
        if unit_str == 'm': return 1.0

        # 4. Standard Prefix handling
        prefix, base = None, None
        if unit_str.startswith('da') and len(unit_str) > 2:
            prefix, base = 'da', unit_str[2:]
        # Handle 1-character prefixes (k, M, m, n, etc.)
        elif len(unit_str) > 1 and not unit_str.startswith('da'):
            prefix, base = unit_str[0], unit_str[1:]

        if prefix in self._PREFIXES and base:
            # Case A: Base is the meter (e.g., 'dam', 'km', 'nm')
            if base == 'm': 
                return self._PREFIXES[prefix]
            # Case B: Base is a known energy unit (e.g., 'kJ', 'meV', 'kcal')
            if base in self._BASE_TO_JOULE and self._BASE_TO_JOULE[base] is not None:
                return self._PREFIXES[prefix] * self._BASE_TO_JOULE[base]
        
        # Check if the unit is JUST a prefix (like 'da', 'k', 'M')
        if unit_str in self._PREFIXES or unit_str == 'da':
            suggestion = f"{unit_str}m (for length) or {unit_str}J (for energy)"
            print(f"{color.RED}{hl.BOLD}\n[Error] '{unit_str}' is a prefix, not a unit. Did you mean '{suggestion}'?{color.OFF}")
        else:
            print(f"{color.RED}{hl.BOLD}\n[Error] '{unit_str}' is not a recognized unit.{color.OFF}")

        # Display the valid bases to guide the student
        valid_bases = [u for u, v in self._BASE_TO_JOULE.items() if v is not None] + ['m']
        units_df = pd.DataFrame({'Valid Base Units': sorted(valid_bases)})
        display(units_df.T)
        
        raise ValueError(f"Unknown unit: {unit_str}")

    def to(self, target):
        """
        Converts the current Energy instance to a target unit.

        Args:
            target (str): Target unit string.
                Examples:
                - Any SI prefix + 'J', 'eV', 'cal', or 'm' (e.g., 'fJ', 'PeV', 'ym', 'dam')
                - Molar versions (e.g., 'MJ/mol', 'ueV/mol', 'ncal/mol')
                - Reciprocal versions (e.g., 'cm-1', 'pm-1', 'am-1')
                - Special units: 'K', 'hartree', 'Å', 'Å-1'

        Example:
            >>> e = Energy(500, 'nm')
            >>> e.to('eV')
            2.4798 eV
        """
        if target == 'K':
            res = self._joules / const.k
        elif target in ['Å', 'A', 'angstrom']:
            res = (const.h * const.c / self._joules) / 1e-10
        elif target in ['Å-1', 'A-1']:
            res = self._joules / (const.h * const.c / 1e-10)
        elif target.endswith('-1'):
            base_len = target[:-2]
            factor = self._get_conversion_factor(base_len)
            res = self._joules / (const.h * const.c / factor)
        elif target == 'm' or (target.endswith('m') and (target[:-1] in self._PREFIXES or target[:-2] == 'da')):
            factor = self._get_conversion_factor(target)
            res = (const.h * const.c / self._joules) / factor
        else:
            res = self._joules / self._get_conversion_factor(target)
        return Energy(res, target)

    @classmethod
    def list_units(cls):
        return sorted(list(cls._BASE_TO_JOULE.keys()))

    @classmethod
    def list_prefixes(cls):
        return dict(sorted(cls._PREFIXES.items(), key=lambda item: item[1]))

    @classmethod
    def show_available_tools(cls):
        """Displays available units and prefixes as styled DataFrames."""
        units_df = pd.DataFrame({'Base Unit': cls.list_units()})
        units_df.index.name = 'ID'
        
        prefix_list = []
        for symbol, data in cls._PREFIX_DATA.items():
            factor, si_prefix, name_fr, name_en = data
            prefix_list.append({
                    'SI Prefix': si_prefix.capitalize(),
                    'Symbol': symbol,
                    'Power': f"10^{int(np.round(np.log10(factor)))}",
                    'Name (FR)': name_fr.capitalize(),
                    'Name (EN)': name_en.capitalize(),
                    '_sort': factor
            })
        df = pd.DataFrame(prefix_list).sort_values('_sort')
        styled_df = df[['SI Prefix', 'Symbol', 'Power', 'Name (FR)', 'Name (EN)']].style.hide(axis='index')
                  
        centerTitle("Available Base Units")
        display(units_df.T)
        centerTitle("Available SI Prefixes")
        display(styled_df.hide(axis='index'))

    @classmethod
    def parse(cls, query):
        """Parses strings like '13.6 eV' or '500 nm'."""
        pattern = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(.*)"
        match = re.match(pattern, query.strip())
        
        if not match:
            raise ValueError(f"Could not parse energy string: {query}")
            
        val = float(match.group(1))
        unit = match.group(2).strip() or 'J'
        return cls(val, unit)

    def __add__(self, other):
        if not isinstance(other, Energy):
            raise TypeError("Addition only supported between Energy objects.")
        new_joules = self._joules + other._joules
        
        # Back to the original unit
        if self.unit == 'nm':
            res_val = (const.h * const.c / new_joules) * 1e9
        else:
            res_val = new_joules / self._get_conversion_factor(self.unit)
        return Energy(res_val, self.unit)

    def __repr__(self):
        # Check if value is a NumPy array or a scalar (float/int)
        if isinstance(self.value, np.ndarray):
            return f"Energy Array ({self.unit}, shape={self.value.shape})"
        
        # If it's not an array, it's a scalar (float)
        return f"{float(self.value):.4f} {self.unit}"