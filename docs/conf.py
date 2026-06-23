"""Sphinx configuration for escape-fel documentation."""
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "escape-fel"
author = "Henrik Lemke"
copyright = "2024, Henrik Lemke"
release = "0.0.1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx_copybutton",
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_static_path = ["_static"]

html_theme_options = {
    "sidebar_hide_name": False,
    "light_css_variables": {
        "color-brand-primary": "#0071bc",
        "color-brand-content": "#0071bc",
    },
}

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "special-members": "__init__, __getitem__, __len__",
}

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "dask": ("https://docs.dask.org/en/stable", None),
    "h5py": ("https://docs.h5py.org/en/stable", None),
}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
]
myst_heading_anchors = 3

nbsphinx_execute = "never"

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
