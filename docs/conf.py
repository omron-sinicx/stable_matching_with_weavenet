# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'WeaveNet'
copyright = '2023, Atsushi Hashimoto'
author = 'Atsushi Hashimoto'
release = '0.1.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
import sys, os

sys.path.append(os.path.abspath('../src'))

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon']

# Automatically extract typehints when specified and place them in
# descriptions of the relevant function/method.
autodoc_typehints = "description"
autodoc_default_options = {
#    'members': 'var1, var2',
    'member-order': 'bysource',
#    'special-members': '__init__',
#    'undoc-members': True,
#    'exclude-members': '__weakref__'
}

# Don't show class signature with the class' name.
autodoc_class_signature = "separated"

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sizzle'
html_static_path = ['_static']
