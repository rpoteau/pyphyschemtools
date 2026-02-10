# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pyphyschemtools'
version = '0.6'
copyright = '2026, Romuald Poteau'
author = 'Romuald Poteau'
release = '0.6.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'myst_parser', # Markdown
    'sphinx.ext.autodoc', # Docstriogs extraction
    'sphinx_copybutton',
    'sphinx.ext.napoleon', # Support Google/NumPy docstrings format
    "sphinx.ext.githubpages",
    'nbsphinx', # Notebooks
    'sphinx.ext.mathjax', # LaTeX
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

# Configuration MyST to authorize titlew with anchors
myst_enable_extensions = ["dollarmath", "amsmath"]
myst_heading_anchors = 3

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_baseurl = "https://pyphyschemtools.readthedocs.io/"
html_extra_path = []

def setup(app):
    app.add_css_file('visualID.css') 

import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

# Optionnel : permettre à Sphinx de chercher des fichiers .md et .ipynb
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# Prevents notebook execution if they are already saved with their outputs.
# (Useful for long calculations or complex widgets)
nbsphinx_execute = 'never'

mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"

mathjax3_config = {
    'chtml': {
        'fontURL': 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/output/chtml/fonts/woff-v2',
        'scale': 1.15,
        'mtextInheritFont': True,  # Text inside math will look like Calibri
        'merrorInheritFont': True,
        # REMOVED: fontContext: 'inherit' (this was making symbols look weird)
    },
    'tex': {
        'macros': {
            'RR': '{\\bf R}',
        }
    },
    'options': {
        'renderActions': {
            'addMenu': []
        }
    },
}
