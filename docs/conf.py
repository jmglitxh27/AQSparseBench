"""Sphinx configuration for AQSparseBench."""

from __future__ import annotations

import os
import sys
from datetime import date

sys.path.insert(0, os.path.abspath(".."))

project = "AQSparseBench"
author = "AQSparseBench contributors"
copyright = f"{date.today().year}, {author}"
release = version = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_autodoc_typehints",
    "rinoh.frontend.sphinx",
]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

nitpicky = False
default_role = "py:obj"

pygments_style = "sphinx"
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autosummary_generate = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

html_theme = "furo"
html_title = f"{project} {release}"
html_static_path: list[str] = []

latex_documents = [
    (
        "index",
        "aqsparsebench.tex",
        f"{project} Documentation",
        author,
        "manual",
    ),
]

rinoh_documents = [
    {
        "doc": "index",
        "target": "aqsparsebench",
        "title": f"{project} documentation",
        "author": author,
        "toctree_only": False,
        "template": "book",
    }
]
