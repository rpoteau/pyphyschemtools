# units

So far, it only contains the `Energy` class

## Energy

<div class="intro">

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

### Exploring Capabilities

To inspect the internal logic and supported units, you can display the built-in reference tables. This provides a live view of all supported SI prefixes (from **yocto** $10^{-30}$ to **quetta** $10^{30}$) and recognized base units.

```python
display(Energy.show_available_tools())
```

### Traceability of Constants
The class ensures scientific rigor by using the latest CODATA values via `scipy.constants`. You can query the metadata to see the exact values, units, and uncertainties for the constants used in the background.

```python
Energy.show_constants_metadata()
```

### Creating Energy Objects
You can instantiate an `Energy` object using two different methods depending on your data source.

**String Parsing**. 
Useful for user inputs or reading from text files.

```python
print(Energy.parse("13.59844 eV").to('hartree'))
```

**Direct Instantiation**. Recommended for programmatic usage and mathematical variables.

```python
print(Energy(13.59844, 'eV').to('hartree'))
```

### Spectroscopic & Molar Conversions

The class automatically detects whether a unit is an energy, a length (wavelength), or a reciprocal length (wavenumber). It also handles the conversion from single-particle energy to molar energy.

```python
print(Energy.parse("4000 cm-1").to('eV'))      # Wavenumber to Energy
print(Energy.parse("1 MeV").to('kJ/mol'))      # Nuclear Energy to Molar Energy
print(Energy.parse("656 nm").to('eV'))         # Wavelength to Energy
print(Energy.parse("1 kcal/mol").to('kJ/mol')) # Thermochemical conversion
```

### Universal Prefix Support

The recursive prefix manager allows for extreme physical scales. It treats prefixes like **da** (**deca**) or **y** (**yocto**) as dynamic multipliers applicable to any base unit.

```python
print(Energy.parse("4000 ym-1").to('GeV'))
print(Energy.parse("4000 ym-1").to('ZJ/mol'))
```

### Integration with Variables

Using Python f-strings, you can easily inject variables into the parsing logic for dynamic workflows.

```python
e = Energy.parse("1 kcal/mol").to('kJ/mol')
print(e)        # Formatted Output: 4.1840 kJ/mol
print(e.value)  # Numerical Output: 4.184
```

### Batch Processing (Vectorization)

The class is fully compatible with `NumPy`. You can pass a list or array of values (such as the energy levels of the Hydrogen atom) to perform batch conversions.

```python
K = 13.59844
Etab_eV = [-K, -K/4, -K/9, -K/16, -K/25]

# Convert the entire list from eV to hartree
Etab_hartree = Energy(Etab_eV, 'eV').to('hartree')

print(Etab_hartree) # Formatted output: Energy Array (hartree, shape=(5,))
print(Etab_hartree.value) # Formatted output: [-0.49973345 -0.12493336 -0.05552594 -0.03123334 -0.01998934]
```

`