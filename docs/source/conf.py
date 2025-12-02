# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Magneton'
copyright = '2025, zhen'
author = 'zhen'
release = '1.0'

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx_rtd_theme",   # ReadTheDocs Theme
    "myst_parser",        # Markdown Support
    "nbsphinx",           # Jupyter Notebook Support
    "sphinx.ext.mathjax", # LaTeX Support
    "sphinx.ext.autodoc", # Automatic API Documentation
    "sphinx.ext.viewcode" # Show source code link
]

templates_path = ['_templates']
exclude_patterns = []

# Support .rst, .md, .ipynb
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
    '.ipynb': 'nbsphinx',
}

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']

html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 3,
    "style_nav_header_background": "#2a4d69",
}
