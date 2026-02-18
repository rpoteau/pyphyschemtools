.. pyphyschemtools documentation master file, created by
   sphinx-quickstart on Mon Feb  2 15:36:12 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pyphyschemtools documentation
=============================

.. image:: _static/tools4pyPC_banner.svg
   :alt: pyPCtools banner
   :align: center
   :width: 1000px

.. raw:: html

    <br>

Welcome to **pyphyschemtools**. This documentation combines computational tools with scientific theoretical reviews to support your chemical and physical chemistry workflows.

The package provides a versatile ecosystem for researchers:

* **Python "API of APIs"**: High-level wrappers that unify various libraries (like ``mendeleev``, ``scipy``, and ``rdkit``) into a simplified, consistent interface for Python scripts and Jupyter Notebooks. See the **Docs and Theory** section below, as well as the **Interactive Examples** and the **API Reference**.
* **CLI Tools**: A robust set of command-line utilities for rapid post-processing and job management in VASP and Gaussian environments. For more details on computational workflows and tool usage, please see the :doc:`theoryDocs/QuantumChemCorner`.

.. toctree::
   :maxdepth: 2
   :caption: 📚 Docs and Theory

   theoryDocs/Cheminformatics
   theoryDocs/Chem3D
   theoryDocs/Kinetics
   theoryDocs/Nano
   theoryDocs/PeriodicTable
   theoryDocs/Spectra
   theoryDocs/Units
   theoryDocs/Misc

.. toctree::
   :maxdepth: 2
   :caption: 🚀 Interactive Examples

   Examples (Google Colab) <https://colab.research.google.com/github/rpoteau/pyphyschemtools/blob/main/Examples.ipynb> 

.. toctree::
   :maxdepth: 2
   :caption: ⚛️ Quantum Chemistry Corner

   theoryDocs/QuantumChemCorner

.. toctree::
   :maxdepth: 2
   :caption: 🛠️ API Reference (Technical)

   modules
