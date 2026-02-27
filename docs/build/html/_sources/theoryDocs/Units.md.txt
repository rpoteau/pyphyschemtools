# Fundamental Units

**`units.py` module**

The `units` module provides a robust framework for handling physical quantities with automatic SI prefix resolution and CODATA-precise conversions.

The units converters are defined as classes: `PhysicalQuantity`, `Energy`, `Pressure`, `Length`, `Area`, `Volume`, `Mass`, `MolarMass`, `Density`. 

## Quick Start

<div class="intro">
    
### Initialization

Users must first import the appropriate class from `tools4pyPhysChem`, e.g.:

```python
from pyphyschemtools import Pressure
```
<br>
There are two ways to initialize quantities: you can create objects using standard initialization or the human-readable `parse` method.

* **Manual Initialization:** Pass a value and a unit string.
* **String Parsing:** Use `.parse()` for scientific strings like `"13.6 eV"` or `"1.672e2 nm"`.

```python
# Standard way: (value, unit)
p1 = Pressure(1, 'atm')

# Scientific string parsing:
p2 = Pressure.parse("760 torr")

# They are equivalent:
print(p1.to('Pa') == p2.to('Pa'))  # True
```

### Smart Arithmetic

#### Basic operations

Addition and subtraction operations are performed in absolute SI space, but the result is automatically returned in the unit of the first operand to maintain consistency in your calculations

```python
e1 = Energy(3, 'eV')
e2 = Energy(1.602177e-19, 'J')

# Result will be in 'eV' because e1 is the first variable
ep = e1 + e2  
print(ep)  # 4 eV
em = e1 + e2  
print(em)  # 2 eV
```

#### Arithmetic Rules

The `units` module enforces strict physical laws to ensure your calculations remain dimensionally sound:
- **Additive Operations (+, -)**: These are allowed only between compatible quantities (e.g., Mass + Mass). The result is always expressed in the unit of the first operand.
- **Scalar Scaling (*, /)**: You can multiply or divide any physical quantity by a scalar (integer, float, or NumPy array). This is useful for scaling measurements or applying coefficients.
- **Dimensional Protection**: Multiplying or dividing two quantities (e.g., Pressure * Pressure) is forbidden. This prevents the accidental creation of complex derived units (like Pa<sup>2</sup>) that the specific classes are not designed to handle

In summary, arithmetic is allowed between compatible quantities or with scalars, while operations that would result in complex derived units are restricted.

| Operation | Syntax | Allowed? | Resulting Type | Note |
| :--- | :--- | :---: | :--- | :--- |
| **Addition** | `Mass + Mass` | ✅ | `Mass` | Performed in SI, returned in the unit of the first operand. |
| **Scalar Addition** | `Mass + 5.0` | ✅ | `Mass` | The scalar is assumed to share the object's unit. |
| **Scaling** | `Mass * 2.0` | ✅ | `Mass` | Doubling or halving the magnitude. |
| **Division** | `Mass / 2.0` | ✅ | `Mass` | Reducing the magnitude. |
| **Dimensional Error**| `Mass * Mass` | ❌ | `TypeError` | Forbidden to prevent uncontrolled unit exponentiation ($g^2$). |
| **Inversion Error** | `1.0 / Mass` | ❌ | `TypeError` | Forbidden to prevent unit inversion ($g^{-1}$). |
| **Incompatibility** | `Mass + Energy`| ❌ | `TypeError` | Strictly forbidden (cannot add apples to oranges). |

### Discovering Supported Units of each class

Each class knows its own supported base units. You can inspect them alongside global SI prefixes, with the command:

```python
Volume.show_available_units()
```

### CODATA Traceability & Metadata

Transparency is key in Physical Chemistry and in numerical calculations. You can inspect the exact physical constants (Planck, Boltzmann, etc.) used for internal calculations, sourced from the latest CODATA values via `scipy.constants`. It can be done either from a class associated to a physical quantity (`Energy`, etc...) of from the base `PhysicalQuantity` class.

```python
from pyphyschemtools import Energy
Energy.show_constants_metadata()
```
or

```python
from pyphyschemtools import PhysicalQuantity
PhysicalQuantity.show_constants_metadata()
```

### Universal Prefix Support

The recursive prefix manager allows for extreme physical scales. It treats prefixes like **da** (**deca**) or **y** (**yocto**) as dynamic multipliers applicable to (almost) any base unit. Supported SI prefixes range from **yocto**, $10^{-30}$, to **quetta**, $10^{30}$.

```python
print(Energy.parse("4000 ym-1").to('GeV'))
print(Energy.parse("4000 ym-1").to('ZJ/mol'))
```

The list of prefixes can also be obtained either from a class associated to a physical quantity (`Energy`, etc...) of from the base `PhysicalQuantity` class:

```python
from pyphyschemtools import Energy
Energy.show_available_prefixes()
```
or

```python
from pyphyschemtools import PhysicalQuantity
PhysicalQuantity.show_available_prefixes()
```

### Accessing Numerical Data

The primary way to extract numerical values from a `unit` object, for example when you need to perform external calculations or pass the data to a plotting function (like `Matplotlib`) while keeping the original unit's scale is to address it with :

```python
from pyphyschemtools import Pressure
p = Pressure.parse("10.45 MPa").to('bar')
print(p) # 104.5000 bar
print(type(p)) # <class 'pyphyschemtools.units.Pressure'>
print(p.value) # 104.5
print(type(p.value)) # <class 'float'>
print(2*p) # 
print(2*p.value) # 
```

### Batch Processing (Vectorization)

All classes are fully compatible with `NumPy`. You can pass a list or array of values to perform batch conversions. Let's consider the energy levels of the Hydrogen atom:

```python
K = 13.59844
Etab_eV = [-K, -K/4, -K/9, -K/16, -K/25]

# Convert the entire list from eV to hartree
Etab_hartree = Energy(Etab_eV, 'eV').to('hartree')

print(Etab_hartree) # Formatted output: Energy Array (hartree, shape=(5,))
print(Etab_hartree.value) # Formatted output: [-0.49973345 -0.12493336 -0.05552594 -0.03123334 -0.01998934]
```


</div>

## Energy

<div class="intro">

### Mathematical relationships

The `Energy` class is a high-level physical chemistry utility designed to handle the complex conversions between energy, wavelength, temperature, and molar quantities. It features a recursive prefix manager and native support for different energy contexts:

- Photon Energy: Relates energy to wavelength ($\lambda$) or frequency ($\nu$):

$$E = h\nu = \frac{hc}{\lambda}$$

- Wavenumbers: Handles reciprocal length units ($\text{cm}^{-1}$):

$$E = hc\bar{\nu}$$

- Thermal Energy: Relates temperature ($T$) to energy via the Boltzmann constant ($k_B$):

$$E = k_B T$$

- Molar Quantities: Bridges single-particle energy ($E_{particle}$) and macroscopic thermodynamic properties using the Avogadro constant ($N_A$):

$$E_\mathrm{molar} = E_\mathrm{particle} \cdot N_A$$

To use the energy unit conversion engine, import the `Energy` class as follows:

```python
from pyphyschemtools import Energy
```

### Spectroscopic & Molar Conversions

The class automatically detects whether a unit is an energy, a length (wavelength), or a reciprocal length (wavenumber). It also handles the conversion from single-particle energy to molar energy.

```python
print(Energy.parse("4000 cm-1").to('eV'))      # Wavenumber to Energy
print(Energy.parse("1 MeV").to('kJ/mol'))      # Nuclear Energy to Molar Energy
print(Energy.parse("656 nm").to('eV'))         # Wavelength to Energy
print(Energy.parse("1 kcal/mol").to('kJ/mol')) # Thermochemical conversion
```

### Supported units

| Symbol | Name of the Unit |
| :--- | :--- |
| `J` | Joule (SI Base Unit) |
| `eV` | Electronvolt |
| `hartree` | Hartree |
| `cal` | Calorie (thermochemical) |
| `erg` | Erg |
| `BTU` | British Thermal Unit |
| `K` | Kelvin |
| `m` | Meter |
| `Å`, `A` | Ångström |
| `cm-1` | Reciprocal Centimeter |
| `unit-1` | Any reciprocal `unit` length |
| `unit/mol` | Molar units |

</div>

## Pressure

<div class="intro">

### Standard & Industrial Units

The `Pressure` class manages conversions between the SI unit (Pascal), common laboratory units (atm, torr), and industrial standards (bar, psi). It uses exact CODATA values from `scipy.constants` to ensure that even small pressure differences are captured accurately.

### Supported Units

The following units are handled natively:

| Symbol | Name of the Unit |
| :--- | :--- |
| `Pa` | Pascal (SI Base Unit) |
| `bar` | Bar |
| `atm` | Standard Atmosphere |
| `torr` | Torr (mmHg) |
| `psi` | Pound per Square Inch |

### Common Conversions

Whether you are working with vacuum systems (mTorr) or high-pressure reactors (MPa), the class resolves prefixes automatically.

```python
from pyphyschemtools import Pressure

# Vacuum science: Convert mTorr to Pascal
p_vac = Pressure.parse("10 mtorr").to('Pa')
print(p_vac) 

# Industrial monitoring: Convert MPa to bar
p_gauge = Pressure.parse("2.5 MPa").to('bar')
print(p_gauge) 

# Diving/Geology: Convert psi to atm
p_deep = Pressure(3000, 'psi').to('atm')
print(p_deep)
```
</div>

## Length

<div class="intro">

### From Atomic to Macroscopic Scales

The `Length` class handles spatial dimensions across a vast range of magnitudes. It is specifically optimized for physical chemists who frequently switch between spectroscopic units (Ångströms) and standard SI or Imperial units.

### Supported Units

The following units are handled natively:

| Symbol | Name of the Unit |
| :--- | :--- |
| `m` | Meter (SI Base Unit) |
| `Å`, `A`, `angstrom` | Ångström |
| `inch` | Inch |
| `ft`, `foot` | Foot |
| `yd`, `yard` | Yard |
| `mi`, `mile` | Mile |
| `nmi` | Nautical Mile |

### Common Conversions

Whether you are defining a bond length or the dimensions of a chromatography column, the `Length` class ensures dimensional accuracy.

```python
from pyphyschemtools import Length

# Structural Chemistry: Convert Ångströms to nm
bond_len = Length(1.54, 'Å').to('nm')
print(bond_len)

# Lab Equipment: Convert inches to centimeters
tube_dia = Length.parse("0.25 inch").to('cm')
print(tube_dia)

# Geographical scale: Convert kilometers to miles
dist = Length(10, 'km').to('mi')
print(dist)
```

</div>

## Area

<div class="intro">

### Surface and Interface Dimensions

The `Area` class handles two-dimensional spatial measurements. In physical chemistry, this is critical for calculating cross-sectional areas of molecular beams, the active surface of electrodes, or the footprint of Langmuir monolayers.

### Supported Units

The following units are handled natively:

| Symbol | Name of the Unit |
| :--- | :--- |
| `m2` | Square Meter (SI Base Unit) |
| `cm2` | Square Centimeter |
| `mm2` | Square Millimeter |
| `A2, Å2` | Square Ångström |
| `ha` | Hectare |
| `are` | Are |
| `in2`, `sq_inch` | Square Inch |
| `ft2`, `sq_ft` | Square Foot |
| `ac`, `acre` | Acre |
| `sq_mi` | Square Mile |

### Common Conversions

Whether you are calculating the area of a microscope slide or the cross-section of a capillary, the `Area` class simplifies the math.

```python
from pyphyschemtools import Area

# Microscopy: Convert square inches to square centimeters
slide_area = Area(1, 'sq_inch').to('cm2')
print(slide_area)

# Surface Science: Convert Å² (atomic scale) to nm²
site_area = Area(20, 'Å2').to('nm2')
print(site_area)

# Agriculture/Environment: Convert hectares to acres
field = Area.parse("2.5 ha").to('acre')
print(field)
```

</div>

## Volume

<div class="intro">

### Three-Dimensional Capacity

The `Volume` class manages three-dimensional space measurements. It is essential for stoichiometry, titrations, and chemical engineering, bridging the gap between metric laboratory glassware (L, mL) and industrial containers (gallons, barrels).

### Supported Units

The following units are handled natively:

| Symbol | Name of the Unit |
| :--- | :--- |
| `m3` | Cubic Meter (SI Base Unit) |
| `cm3` | Cubic Centimeter (mL) |
| `L` | Liter |
| `gal`, `gallon` | US Liquid Gallon |
| `qt`, `quart` | US Liquid Quart |
| `pt`, `pint` | US Liquid Pint |
| `cup` | US Cup |
| `tsp` | Teaspoon |
| `tbsp` | Tablespoon |
| `oz`, `fluid_ounce` | US Fluid Ounce |
| `bbl`, `barrel` | Oil Barrel |

### Common Conversions

Whether you are preparing a buffer or scaling a pilot plant reactor, the `Volume` class ensures precise fluid management.

```python
from pyphyschemtools import Volume

# Lab Preparation: Convert Liters to Milliliters
v_flask = Volume(0.25, 'L').to('mL')
print(v_flask)

# US Standards: Convert fluid ounces to mL
v_sample = Volume.parse("8 oz").to('mL')
print(v_sample)

# Bulk Chemistry: Convert Gallons to Liters
v_tank = Volume(55, 'gal').to('L')
print(v_tank)
```

</div>

## Mass

<div class="intro">

### From Subatomic to Industrial Scales

The `Mass` class manages the physical quantity of matter. It is designed to handle the extreme scales encountered in physical chemistry, from the mass of a single proton (atomic mass units) to the mass of large-scale chemical reactors (metric tons).

### Supported Units

The following units are handled natively:

| Symbol | Name of the Unit |
| :--- | :--- |
| `kg` | Kilogram (SI Base Unit) |
| `g` | Gram |
| `t` | Metric Ton |
| `u`, `amu` | Unified Atomic Mass Unit |
| `carat` | Carat |
| `lb` | Pound |
| `oz` | Ounce |
| `gr` | Grain |

### Common Conversions

Whether you are weighing a catalyst on a microbalance or ordering bulk solvents, the `Mass` class ensures CODATA precision.

```python
from pyphyschemtools import Mass

# Analytical Chemistry: Convert grams to milligrams
reagent = Mass(0.005, 'g').to('mg')
print(reagent)

# Particle Physics: Convert atomic mass units to grams
proton_mass = Mass(1.007, 'u').to('g')
print(proton_mass)

# Logistics: Convert kilograms to pounds
shipping_wt = Mass(25, 'kg').to('lb')
print(shipping_wt)
```

</div>

## Molar Mass

<div class="intro">

### The Stoichiometric Bridge

The `MolarMass` class (or Molecular Weight) is the fundamental scaling factor in chemical calculations. It relates the mass of a substance to the amount of substance (moles), bridging the gap between single molecules and bulk materials using the Avogadro constant ($N_A$).

### Supported Units

The following units are handled natively:

| Symbol | Name of the Unit |
| :--- | :--- |
| `kg/mol` | Kilogram per mole (SI Base Unit) |
| `g/mol` | Gram per mole |
| `Da`, `Dalton` | Dalton |
| `u` | Unified atomic mass unit |

### Common Conversions

Whether you are calculating the molecular weight of a small organic molecule or a massive protein, the `MolarMass` class ensures consistent dimensionality.

```python
from pyphyschemtools import MolarMass

# Organic Synthesis: Convert g/mol to kg/mol for industrial scaling
mw_glucose = MolarMass(180.16, 'g/mol').to('kg/mol')
print(mw_glucose)

# Biochemistry: Working with Daltons for proteins
bsa_mw = MolarMass(66463, 'Da').to('g/mol')
print(bsa_mw)

# High-Precision Physics: Convert unified atomic mass units
c12_mass = MolarMass(12.000, 'u').to('g/mol')
print(c12_mass)
```

</div>

## Density

<div class="intro">

### Mass per Volume Relationships

The `Density` class manages the concentration of matter within a specific space ($\rho = m/V$). It is a vital property for solvent preparation, determining the concentration of solutions, and identifying unknown substances in the laboratory.

### Supported Units

The following units are handled natively:

| Symbol | Name of the Unit |
| :--- | :--- |
| `kg/m3` | Kilogram per Cubic Meter (SI Base Unit) |
| `g/cm3` | Gram per Cubic Centimeter |
| `g/mL` | Gram per Milliliter |
| `g/L` | Gram per Liter |
| `lb/ft3` | Pound per Cubic Foot |

### Common Conversions

Whether you are converting from high-density materials (gold) to industrial fluids, the `Density` class ensures accuracy.

```python
from pyphyschemtools import Density

# Materials Characterization: Convert g/cm3 to SI units
rho_au = Density(19.3, 'g/cm3').to('kg/m3')
print(rho_au)

# Industrial Standards: Convert pounds per cubic foot to g/cm3
rho_gasoline = Density.parse("45 lb/ft3").to('g/cm3')
print(rho_gasoline)

# Solution Prep: Convert g/L (common in biology) to g/mL
rho_sol = Density(150, 'g/L').to('g/mL')
print(rho_sol)
```

</div>

## Temperature

<div class="intro">

### Absolute and Relative Scales

The `Temperature` class handles conversions between thermodynamic temperature (Kelvin) and common relative scales (Celsius, Fahrenheit). Unlike other physical quantities, temperature conversions involve both a scaling factor and a numerical offset.

### Supported Units

The following units are handled natively by the `Temperature` class:

| Symbol | Name of the Unit |
| :--- | :--- |
| `K` | Kelvin (SI Base Unit) |
| `°C`, `celsius` | Degree Celsius |
| `°F`, `fahrenheit` | Degree Fahrenheit |
| `°R`, `rankine` | Degree Rankine |

### Common Conversions

The `Temperature` class ensures that the delicate offsets (like the 273.15 constant for Celsius) are applied with scientific precision.

```python
from pyphyschemtools import Temperature

# Laboratory Standard: Convert Celsius to Kelvin
t_room = Temperature(25, '°C').to('K')
print(t_room)

# US Engineering: Convert Fahrenheit to Celsius
t_body = Temperature(98.6, '°F').to('°C')
print(t_body)

# Cryogenics: Convert Kelvin to Fahrenheit
t_liquid_N2 = Temperature(77, 'K').to('°C')
print(t_liquid_N2)
```

</div>