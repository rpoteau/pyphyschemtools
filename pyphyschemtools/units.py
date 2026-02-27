import re
import numpy as np
import scipy.constants as const
import pandas as pd
from IPython.display import display, HTML
from .core import centerTitle, centertxt
from .visualID_Eng import color, fg, hl, bg

# =============================================================================
# GLOBAL DATA & CONSTANTS
# =============================================================================

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

# =============================================================================
# BASE CLASS
# =============================================================================

class PhysicalQuantity:

    def __init__(self, value, unit, conversion_map):
        """
        Initialize a physical quantity.
        
        Base class for physical quantities providing common utilities for 
        SI prefix handling, array management, and string parsing.

        Args:
            value (float or array-like): Numerical magnitude.
            unit (str): Unit symbol.
            conversion_map (dict): Mapping from base units to SI.
        """
        self.unit = unit
        self._base_map = conversion_map
        
        # Handle input data as numpy array or scalar
        raw_data = np.asanyarray(value)
        if raw_data.ndim == 0:
            self.value = raw_data.item()
        else:
            self.value = raw_data

    def _get_factor(self, unit_str):
        """
        Resolves the numerical conversion factor for a given unit string,
        including SI prefixes.

        Args:
            unit_str (str): The unit string to resolve (e.g., 'kJ', 'MPa').

        Returns:
            float: The multiplier to reach the SI base unit.
        """
        if unit_str in self._base_map and self._base_map[unit_str] is not None:
            return self._base_map[unit_str]

        if unit_str.endswith('/mol'):
            return self._get_factor(unit_str.replace('/mol', '')) / const.Avogadro

        prefix, base = None, None
        if unit_str.startswith('da') and len(unit_str) > 2:
            prefix, base = 'da', unit_str[2:]
        elif len(unit_str) > 1 and not unit_str.startswith('da'):
            prefix, base = unit_str[0], unit_str[1:]

        if prefix in _PREFIXES and base in self._base_map:
            base_f = self._base_map[base] if self._base_map[base] is not None else 1.0
            return _PREFIXES[prefix] * base_f

        # Error handling with visual guidance
        self._raise_unit_error(unit_str)

    def _raise_unit_error(self, unit_str):
        """Displays error message and valid units table."""
        if unit_str in _PREFIXES or unit_str == 'da':
            msg = f"'{unit_str}' is a prefix, not a unit. Did you mean '{unit_str}J' or '{unit_str}Pa'?"
        else:
            msg = f"'{unit_str}' is not a recognized unit."
        
        print(f"{color.RED}{hl.BOLD}\n[Error] {msg}{color.OFF}")
        valid_bases = [u for u, v in self._base_map.items() if v is not None]
        if 'm' in self._base_map: valid_bases.append('m')
        display(pd.DataFrame({'Valid Base Units': sorted(valid_bases)}).T)
        raise ValueError(f"Unknown unit: {unit_str}")

    @classmethod
    def parse(cls, query):
        """Parses strings like '13.6 eV' or '1013 hPa'."""
        pattern = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(.*)"
        match = re.match(pattern, query.strip())
        if not match:
            raise ValueError(f"Could not parse string: {query}")
        val = float(match.group(1))
        unit = match.group(2).strip() or list(cls._base_map.keys())[0]
        return cls(val, unit)

    def __float__(self):
        """Explicit scalar conversion."""
        if isinstance(self.value, (float, int, np.float64)):
            return float(self.value)
        raise TypeError(f"{self.__class__.__name__} contains an array and cannot be cast to float.")

    def __len__(self):
        """Returns element count."""
        return len(self.value) if isinstance(self.value, np.ndarray) else 1

    def __repr__(self):
        """
        Smart string representation. 
        Uses scientific notation for values < 1e-4 or > 1e6.
        """
        if isinstance(self.value, np.ndarray):
            return f"{self.__class__.__name__} Array ({self.unit}, shape={self.value.shape})"
        
        val = float(self.value)
        # Use scientific notation if the absolute value is too small or too large
        if 0 < abs(val) < 1e-4 or abs(val) > 1e6:
            return f"{val:.4e} {self.unit}"
        else:
            return f"{val:.4f} {self.unit}"

    @staticmethod
    def show_constants_metadata():
        """
        Displays a styled table of the physical constants used for calculations,
        including CODATA values, units, and uncertainties.
        """
        constants_to_show = {
            "h": "Planck constant",
            "c": "speed of light in vacuum",
            "e": "elementary charge",
            "k": "Boltzmann constant",
            "N_A": "Avogadro constant",
            "E_h": "Hartree energy",
            "cal": "thermochemical calorie",
        }

        data = []
        for symbol, name in constants_to_show.items():
            try:
                val, unit, unc = const.physical_constants[name]
                data.append({
                    "Symbol": symbol, "CODATA Description": name,
                    "Value": val, "Uncertainty": unc, "Unit": unit
                })
            except KeyError:
                if symbol == "cal":
                    data.append({
                        "Symbol": "cal", "CODATA Description": "Thermochemical calorie",
                        "Value": const.calorie, "Uncertainty": 0.0, "Unit": "J"
                    })

        df = pd.DataFrame(data)
        centerTitle("Physical Constants Metadata from SciPy (CODATA)")
        styled_df = df.style.format({"Value": "{:.8e}", "Uncertainty": "{:.2e}"}).hide(axis='index')
        display(styled_df)

    @classmethod
    def show_available_units(cls):
        """
        Displays available base units for this class as styled DataFrames.
        """
        # 1. Show Base Units for the specific class (Energy, Pressure, etc.)
        valid_bases = [u for u, v in cls._base_map.items() if v is not None]
        # Add 'm' for Energy if it exists in the map as None (wavelength logic)
        if 'm' in cls._base_map:
            valid_bases.append('m')
            
        units_df = pd.DataFrame({'Available Base Units': sorted(list(set(valid_bases)))})
        
        # 3. Display both only if list_prefixes is True
        centerTitle(f"Available Units for {cls.__name__}")
        display(units_df.T)
            
    @classmethod
    def show_available_units(cls):
        """
        Displays available base units, underlining the SI base unit (factor 1.0).
        """
        # 1. Identify valid units
        valid_bases = [u for u, v in cls._base_map.items() if v is not None]
        if 'm' in cls._base_map:
            valid_bases.append('m')
            
        # 2. Identify which unit is the SI base (factor == 1.0)
        # Handle cases where Temperature doesn't have a _base_map
        si_units = []
        if hasattr(cls, '_base_map'):
            si_units = [u for u, v in cls._base_map.items() if v == 1.0]
        elif cls.__name__ == 'Temperature':
            si_units = ['K']

        units_df = pd.DataFrame({'Available Base Units': sorted(list(set(valid_bases)))})
        
        # 3. Apply Styling
        def underline_si(val):
            return 'color: blue; font-weight: bold;' if val in si_units else ''

        # Apply the style to the data cells
        styled_df = units_df.T.style.map(underline_si)

        # 4. Display
        centerTitle(f"Available Units for {cls.__name__}")
        display(styled_df)

    @property
    def _si_value(self):
        """
        Abstract property. Must be implemented by subclasses to return 
        the magnitude in absolute SI units (e.g., Joules, Pascals, Grams).
        """
        raise NotImplementedError("Subclasses must define _si_value to enable arithmetic.")

    def __add__(self, other):
        """
        Addition operator (+).
        Supports:
        - Quantity + Quantity: Adds absolute SI values and returns result in self.unit.
        - Quantity + Scalar: Assumes the scalar has the same unit as the object.
        """
        # 1. Operation between two objects of the same class (e.g., Mass + Mass)
        if isinstance(other, self.__class__):
            new_si = self._si_value + other._si_value
            # Convert the sum back to the original unit of the first operand
            return self.__class__(new_si / self._get_factor(self.unit), self.unit)
        
        # 2. Operation with a scalar or array (e.g., Mass + 2.5)
        if isinstance(other, (int, float, np.ndarray)):
            # Assumption: The scalar is expressed in the same unit as the object
            return self.__class__(self.value + other, self.unit)
        
        raise TypeError(f"Addition not supported between {self.__class__.__name__} and {type(other)}")

    def __sub__(self, other):
        """
        Subtraction operator (-).
        Supports:
        - Quantity - Quantity: Subtracts SI values and returns result in self.unit.
        - Quantity - Scalar: Assumes the scalar has the same unit as the object.
        """
        if isinstance(other, self.__class__):
            new_si = self._si_value - other._si_value
            return self.__class__(new_si / self._get_factor(self.unit), self.unit)
        
        if isinstance(other, (int, float, np.ndarray)):
            return self.__class__(self.value - other, self.unit)
        
        raise TypeError(f"Subtraction not supported between {self.__class__.__name__} and {type(other)}")

    # Reflection methods to handle Scalar + Quantity (e.g., 2 + Pressure)
    def __radd__(self, other): 
        """Handles commutative addition (Scalar + Quantity)."""
        return self.__add__(other)

    def __rsub__(self, other): 
        """
        Handles non-commutative subtraction (Scalar - Quantity).
        Calculates: (Scalar - self.value) in self.unit.
        """
        if isinstance(other, (int, float, np.ndarray)):
            return self.__class__(other - self.value, self.unit)
        return NotImplemented

    # --- MULTIPLICATION & DIVISION (Scalar Only) ---

    def __mul__(self, other):
        """Quantity * Scalar. Multiplication between two quantities is prohibited."""
        if isinstance(other, (int, float, np.ndarray)):
            return self.__class__(self.value * other, self.unit)
        
        # Explicit error message if trying to multiply two objects
        if isinstance(other, PhysicalQuantity):
            raise TypeError("Multiplication between two physical quantities is not supported to prevent dimension changes.")
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        """Quantity / Scalar. Returns a new object of the same class."""
        if isinstance(other, (int, float, np.ndarray)):
            return self.__class__(self.value / other, self.unit)
        return NotImplemented

    def __rtruediv__(self, other):
        """Scalar / Quantity is prohibited to prevent unit inversion."""
        return NotImplemented
        
# =============================================================================
# ENERGY CLASS
# =============================================================================
    
class Energy(PhysicalQuantity):

    _base_map = {
        'J': 1.0, 
        'eV': const.e, 
        'hartree': const.value("Hartree energy"),
        'K': const.k,
        'cal': const.calorie, # Added to support kcal/mol via the same logic
        'm': None,   
        'Å': None,
        'erg': 1e-7,                # Manual definition cgs
        'BTU': const.Btu,
    }

    def __init__(self, value, unit='J'):
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
        super().__init__(value, unit, self._base_map)
        self._joules = self._calculate_joules(self.value, self.unit)

    @property
    def _si_value(self):
        return self._joules

    def _calculate_joules(self, val, unit_str):
        if unit_str in ['Å', 'A', 'angstrom']: return (const.h * const.c) / (val * 1e-10)
        if unit_str in ['Å-1', 'A-1']: return val * (const.h * const.c / 1e-10)
        if unit_str.endswith('-1'): return val * (const.h * const.c / self._get_factor(unit_str[:-2]))
        if unit_str == 'm' or (unit_str.endswith('m') and not unit_str == 'atm'):
            return (const.h * const.c) / (val * self._get_factor(unit_str))
        return val * self._get_factor(unit_str)

    def to(self, target):
        if target == 'K': res = self._joules / const.k
        elif target in ['Å', 'A', 'angstrom']: res = (const.h * const.c / self._joules) / 1e-10
        elif target in ['Å-1', 'A-1']: res = self._joules / (const.h * const.c / 1e-10)
        elif target.endswith('-1'): res = self._joules / (const.h * const.c / self._get_factor(target[:-2]))
        elif target == 'm' or (target.endswith('m') and not target == 'atm'):
            res = (const.h * const.c / self._joules) / self._get_factor(target)
        else: res = self._joules / self._get_factor(target)
        return Energy(res, target)

    def __add__(self, other):
        if not isinstance(other, Energy):
            raise TypeError("Addition only supported between Energy objects.")
        new_j = self._joules + other._joules
        # Convert back to self.unit, handle wavelength inversion
        if self.unit == 'm' or self.unit.endswith('m') and self.unit != 'atm':
            res_val = (const.h * const.c / new_j) / self._get_factor(self.unit)
        else:
            res_val = new_j / self._get_factor(self.unit)
        return Energy(res_val, self.unit)

    def __sub__(self, other):
        """
        Subtracts another Energy object.
        E_res = E_self - E_other
        """
        if not isinstance(other, Energy):
            raise TypeError("Subtraction only supported between Energy objects.")
        
        new_j = self._joules - other._joules
        
        # Check for non-physical result (negative energy)
        if new_j <= 0 and not isinstance(self.value, np.ndarray):
             print(f"{color.RED}{hl.BOLD}[Warning] Resulting energy is zero or negative.{color.OFF}")

        # Conversion logic back to self.unit
        if self.unit == 'm' or (self.unit.endswith('m') and self.unit != 'atm'):
            # Wavelength: lambda = hc / E
            res_val = (const.h * const.c / new_j) / self._get_factor(self.unit)
        else:
            # Linear units (J, eV, cal, K, hartree)
            res_val = new_j / self._get_factor(self.unit)
            
        return Energy(res_val, self.unit)

# =============================================================================
# PRESSURE CLASS
# =============================================================================

class Pressure(PhysicalQuantity):
    """Pressure management class (Pa, bar, atm, torr, psi)."""
    _base_map = {'Pa': 1.0,
                 'bar': 1e5,
                 'atm': const.atm,
                 'torr': const.mmHg,
                 'psi': const.psi}

    def __init__(self, value, unit='Pa'):
        super().__init__(value, unit, self._base_map)
        self._si_val = self.value * self._get_factor(self.unit)

    @property
    def _si_value(self): 
        return self._si_val
    
    def to(self, target):
        # Using self.__class__ automatically uses 'Pressure', 'Mass', etc.
        return self.__class__(self._si_value / self._get_factor(target), target)

# =============================================================================
# LENGTH CLASS
# =============================================================================

class Length(PhysicalQuantity):
    """
    Distance and Length management.
    Supports SI (m) and Imperial units (inch, foot, yard, mile, nautical mile).
    """
    _base_map = {
        'm': 1.0,
        'Å': 1e-10,
        'A': 1e-10,
        'angstrom': 1e-10,
        'inch': const.inch,
        'ft': const.foot,
        'foot': const.foot,
        'yd': const.yard,
        'yard': const.yard,
        'mi': const.mile,
        'mile': const.mile,
        'nmi': const.nautical_mile
    }

    def __init__(self, value, unit='m'):
        super().__init__(value, unit, self._base_map)
        self._si_val = self.value * self._get_factor(self.unit)

    @property
    def _si_value(self): 
        return self._si_val
    
    def to(self, target):
        # Using self.__class__ automatically uses 'Pressure', 'Mass', etc.
        return self.__class__(self._si_value / self._get_factor(target), target)

# =============================================================================
# AREA CLASS
# =============================================================================

class Area(PhysicalQuantity):
    """
    Area management.
    Supports SI and Si-related units (m2, hectare) and Imperial units (sq inch, sq ft, acre).
    """
    _base_map = {
        'm2': 1.0,
        'cm2': 1e-4,
        'mm2': 1e-6,
        'Å2': 1e-20,
        'A2': 1e-20,
        'ha': const.hectare,       # hectare
        'are': 100.0,
        'in2': const.inch**2,
        'sq_inch': const.inch**2,
        'ft2': const.foot**2,
        'sq_ft': const.foot**2,
        'ac': const.acre,  # acre (international)
        'acre': const.acre,
        'sq_mi': const.mile**2,
    }

    def __init__(self, value, unit='m2'):
        super().__init__(value, unit, self._base_map)
        self._si_val = self.value * self._get_factor(self.unit)

    @property
    def _si_value(self): 
        return self._si_val
    
    def to(self, target):
        # Using self.__class__ automatically uses 'Pressure', 'Mass', etc.
        return self.__class__(self._si_value / self._get_factor(target), target)

        
# =============================================================================
# VOLUME CLASS
# =============================================================================

class Volume(PhysicalQuantity):
    """
    Volume management.
    Supports SI and SI-related units (m3, L) and Imperial units (gallon, quart, pint, fluid ounce).
    Note: Uses US liquid measures from SciPy.
    """
    _base_map = {
        'm3': 1.0,
        'cm3': 1e-6,
        'L': 1e-3,
        'gal': const.gallon,
        'gallon': const.gallon,
         # SciPy doesn't always expose 'quart' or 'pint' directly in all versions
         # 1 gallon = 4 quarts = 8 pints = 128 fluid ounces
        'qt': const.gallon / 4,
        'quart': const.gallon / 4,
        'pt': const.gallon / 8,
        'pint': const.gallon / 8,
        'cup': const.gallon / 16,
        'tsp': const.gallon / 768, # Teaspoon
        'tbsp': const.gallon / 256, # Tablespoon
        'oz': const.fluid_ounce,
        'fluid_ounce': const.fluid_ounce,
        'bbl': const.barrel,
        'barrel': const.barrel,
        'grain': const.grain * 1000, # kg -> g
        'carat': 0.2,                # manual definition (200mg)

    }

    def __init__(self, value, unit='m3'):
        super().__init__(value, unit, self._base_map)
        self._si_val = self.value * self._get_factor(self.unit)

    @property
    def _si_value(self): 
        return self._si_val
    
    def to(self, target):
        # Using self.__class__ automatically uses 'Pressure', 'Mass', etc.
        return self.__class__(self._si_value / self._get_factor(target), target)

# =============================================================================
# MASS CLASS
# =============================================================================

class Mass(PhysicalQuantity):
    """
    Mass management.
    """
    _base_map = {
        'kg': 1.0,           # SI Base Unit
        'g': 1e-3,           # 1 gram = 0.001 kg
        'mg': 1e-6,
        'u': const.u,        # SciPy returns atomic mass in kg already
        'amu': const.u,
        'lb': const.pound,   # SciPy returns lb in kg
        'oz': const.oz,      # SciPy returns oz in kg
        't': 1000.0,         # 1 metric ton = 1000 kg
        'grain': const.grain,
        'carat': 0.0002      # 200mg = 0.0002 kg
    }
    def __init__(self, value, unit='kg'):
        super().__init__(value, unit, self._base_map)
        self._si_val = self.value * self._get_factor(self.unit)

    @property
    def _si_value(self): 
        return self._si_val
    
    def to(self, target):
        # Using self.__class__ automatically uses 'Pressure', 'Mass', etc.
        return self.__class__(self._si_value / self._get_factor(target), target)


# =============================================================================
# MOLAR MASS CLASS
# =============================================================================

class MolarMass(PhysicalQuantity):
    """
    Molar Mass management for chemical species.
    Converts between g/mol, kg/mol, and Da (Daltons/amu).
    """
    _base_map = {
        'kg/mol': 1.0,      # SI Base Unit
        'g/mol': 1e-3,      # 1 g/mol = 0.001 kg/mol
        'Da': 1e-3,         # 1 Dalton = 1 g/mol
        'Dalton': 1e-3,
        'u': 1e-3           # Unified atomic mass unit
    }

    def __init__(self, value, unit='kg/mol'):
        super().__init__(value, unit, self._base_map)
        self._si_val = self.value * self._get_factor(self.unit)

    @property
    def _si_value(self): 
        return self._si_val
    
    def to(self, target):
        # Using self.__class__ automatically uses 'Pressure', 'Mass', etc.
        return self.__class__(self._si_value / self._get_factor(target), target)

# =============================================================================
# DENSITY CLASS
# =============================================================================

class Density(PhysicalQuantity):
    """
    Density (Mass per Volume) management.
    Supports SI (kg/m3) and common lab units (g/cm3, g/mL, lb/ft3).
    """
    _base_map = {
        'kg/m3': 1.0,           # SI Base Unit
        'g/L': 1.0,             # Equivalent to kg/m3
        'g/cm3': 1000.0,        # 1 g/cm3 = 1000 kg/m3
        'g/mL': 1000.0,         # Equivalent to g/cm3
        'kg/L': 1000.0,
        'lb/ft3': const.pound / (const.foot**3), # Now correct for kg base
    }

    def __init__(self, value, unit='kg/cm3'):
        super().__init__(value, unit, self._base_map)
        self._si_val = self.value * self._get_factor(self.unit)

    @property
    def _si_value(self): 
        return self._si_val
    
    def to(self, target):
        # Using self.__class__ automatically uses 'Pressure', 'Mass', etc.
        return self.__class__(self._si_value / self._get_factor(target), target)

# =============================================================================
# TEMPERATURE CLASS
# =============================================================================

class Temperature(PhysicalQuantity):
    def __init__(self, value, unit='K'):
        # 1. Initialize the base class
        super().__init__(value, unit, None) 
        
        # 2. Store the "Ground Truth" (Kelvin) for internal consistency
        self._si_val = self._calculate_kelvin(self.value, self.unit)
        if self._si_val < 0:
            raise ValueError(f"Physically impossible: {value} {unit} calculates to {self._si_val:.2f} K.\n"
                             f"It cannot be below the absolute 0, i.e. 0 K = -459.6700 °F = -273.15 °C.")

    @classmethod
    def show_available_units(cls):
        """Displays available temperature units as a styled DataFrame, highlighting Kelvin in blue."""
        # 1. Define supported units
        valid_bases = ['K', '°C', 'celsius', '°F', 'fahrenheit', '°R', 'rankine']
        units_df = pd.DataFrame({'Available Base Units': sorted(list(set(valid_bases)))})
        
        # 2. Define the styling function specifically for Kelvin
        def color_si(val):
            return 'color: blue; font-weight: bold;' if val == 'K' else ''
        
        # 3. Apply the style to the transposed DataFrame
        styled_df = units_df.T.style.map(color_si)
        
        # 4. Display using your helpers
        centerTitle(f"Available Units for {cls.__name__}")
        display(styled_df)

    @property
    def _si_value(self):
        """Internal SI value (Kelvin)."""
        return self._si_val

    # --- Scientific Guardrails: Math is Forbidden ---
    def _error_msg(self, op):
        return (f"Operation '{op}' is physically invalid. ")

    def __add__(self, other): raise TypeError(self._error_msg("addition"))
    def __sub__(self, other): raise TypeError(self._error_msg("subtraction"))
    def __mul__(self, other): raise TypeError(self._error_msg("multiplication"))
    def __truediv__(self, other): raise TypeError(self._error_msg("division"))

    # --- Core Logic ---
    def _calculate_kelvin(self, val, unit):
        u = unit.strip()
        if u in ['°C', 'celsius']: 
            return val + const.zero_Celsius 
        
        if u in ['°F', 'fahrenheit']: 
            # (F - 32) * 5/9 THEN + 273.15
            return (val - 32) * 5/9 + const.zero_Celsius
            
        if u in ['R', '°R', 'rankine']: 
            return val * 5/9
        return val

    def to(self, target):
        """Converts the absolute temperature to a target scale."""
        k = self._si_value
        t = target.strip()
        if t in ['°C', 'celsius']: 
            res = k - const.zero_Celsius
        elif t in ['°F', 'fahrenheit']: 
            res = (k - const.zero_Celsius) * 9/5 + 32
        elif t in ['°R', 'rankine']: 
            res = k * 9/5
        else: 
            res = k
        return Temperature(res, t)