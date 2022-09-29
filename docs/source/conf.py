# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
# -- Path setup --------------------------------------------------------------
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath('../../src/image_segmentation'))
# sys.path.insert(0, os.path.abspath("../../src/data"))
# sys.path.insert(0, os.path.abspath("../../src/models"))
# sys.path.insert(0, os.path.abspath("../../src/tools"))
# sys.path.insert(0, os.path.abspath("../../src/visualization"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'image_segmentation'
copyright = '2022, Guillaume Barree'
author = 'Guillaume Barree'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc']

templates_path = ['_templates']
exclude_patterns: list[str] = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
