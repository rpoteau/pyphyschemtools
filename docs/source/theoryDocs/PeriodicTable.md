# Periodic Table

**`PeriodicTable.py` module**

An interactive wrapper around the [`mendeleev`](https://mendeleev.readthedocs.io/)
library: element properties, isotope tables, mononuclidic elements, and a
`bokeh`-based periodic table you can hover over in a notebook.

Two classes are provided, sharing the same underlying data and methods:

- **`TableauPeriodique`** — the original, French-language class: element names,
  family names, and all printed output are in French.
- **`PeriodicTable`** — an English-language subclass of `TableauPeriodique`. It
  reuses everything (data, corrections, extra columns) but skips the French
  localization step, so element/family names and printed output are in
  English instead.

> **Note:** both classes build on `mendeleev`'s own element and isotope data
> (fetched once, at instantiation) and add a few corrections/extensions on
> top — a group-12 "not a transition metal" classification fix, a working
> first-ionization-energy column, and pedagogical warnings around
> mononuclidic/radiogenic elements and IUPAC/CIAAW atomic-weight ranges (see
> below).

## Quick Start

```python
from pyphyschemtools import TableauPeriodique, PeriodicTable

# French
tp = TableauPeriodique()
tp.prop('O')            # detailed properties of oxygen, in French
tp.isotopes('Cl')        # isotope table + pedagogical warnings
tp.mononucleides()       # all mononuclidic elements
tp.afficher()            # interactive periodic table (bokeh)

# English
pt = PeriodicTable()
pt.prop('O')
pt.isotopes('Cl')
pt.mononucleides()
pt.afficher()
```

<u>ou bien</u>

```python
import pyphyschemtools.PeriodicTable as PT
tp = PT.TableauPeriodique()
tp.prop(8)
```

## Element properties

`prop(elt_id)` prints a detailed summary of one element — symbol, atomic
number, electron configuration, group/period/block, family, molar mass,
natural isotopes, Pauling electronegativity, first ionization energy,
electron affinity, atomic radius, and a short textual description/sources/uses
(all from `mendeleev`). `elt_id` can be a symbol (`'O'`) or an atomic number
(`8`).

## Isotopes

`isotopes(elt_id)` displays a table of an element's isotopes (mass number,
isotopic mass, natural abundance, half-life), sorted by increasing mass
number, followed by its molar mass. An isotope with no abundance listed is
simply not naturally occurring (e.g. the radioactive isotopes of a
radioelement).

Two pedagogical warnings are printed automatically when they apply:

- **Radiogenic reclassification.** A handful of elements that look
  "mononuclidic" at first glance (because `mendeleev` lists more than one
  isotope with nonzero natural abundance) hold a non-primordial isotope
  produced by the decay of a longer-lived parent. Thorium is the IUPAC/CIAAW
  reference case: trace amounts of $^{230}\text{Th}$, a decay product of
  $^{238}\text{U}$, led the Commission to reclassify Th as binuclidic in 2013.
- **IUPAC/CIAAW atomic-weight ranges.** For 12 elements (H, Li, B, C, N, O,
  Mg, Si, S, Cl, Br, Tl), natural isotopic composition varies enough — by
  geological origin and, for some, industrial isotope separation — that
  IUPAC/CIAAW publishes a *range* `[min, max]` rather than a single standard
  atomic weight. `mendeleev`'s single `atomic_weight` value is only a
  conventional choice within that range, and is **not** guaranteed to equal
  $\sum (\text{abundance} \times \text{isotopic mass})$ computed from
  `mendeleev`'s own isotope table (lithium is a striking example of this).

## Mononuclidic elements

`mononucleides(exclude_radiogenic=True)` (French: `exclure_radiogeniques`)
lists every element with a single naturally-occurring isotope — symbol, mass
number, isotopic mass, molar mass, half-life ("stable" when applicable). By
default, elements like thorium — which hold only a dominant primordial
isotope plus a radiogenic trace isotope — are excluded, matching the strict
IUPAC/CIAAW definition. Pass `exclude_radiogenic=False` to include them
instead; they are then marked with a trailing `*` and footnoted.

Implementation note: both classes fetch the whole `isotopes` and `elements`
tables in two bulk queries and group/merge them with `pandas`, rather than
querying `mendeleev` one element at a time — this brings the running time
down from several seconds to well under a tenth of a second.

## Interactive periodic table

`afficher(self)` opens an interactive `bokeh` periodic table in the current
Jupyter notebook: each tile shows the symbol, atomic number, name and molar
mass, colored by family, with a hover tooltip giving name, family, atomic
number, molar mass, atomic radius, first ionization energy, electron
affinity, Pauling electronegativity and electron configuration — in French
for `TableauPeriodique`, in English for `PeriodicTable`.

## API reference

```{eval-rst}
.. automodule:: pyphyschemtools.PeriodicTable
   :members:
   :undoc-members:
   :show-inheritance:
```
