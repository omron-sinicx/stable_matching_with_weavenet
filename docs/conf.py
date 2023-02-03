# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import inspect
import pkg_resources

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

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon','sphinx_multiversion','autodocsumm','sphinx.ext.linkcode','enum_tools.autoenum']

# Automatically extract typehints when specified and place them in
# descriptions of the relevant function/method.
autodoc_typehints = "description"
autodoc_default_options = {
#    'members': 'var1, var2',
    'member-order': 'bysource',
#    'special-members': '__init__',
    'undoc-members': False,    
    'exclude-members': 'training',#__weakref__'    
}

# Don't show class signature with the class' name.
#autodoc_class_signature = "separated"

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'sizzle'
#html_theme = 'sphinx_rtd_theme'

#extensions.append("sphinxjp.themes.basicstrap")
#html_theme = 'basicstrap'
#html_theme = 'groundwork'
html_theme = 'renku'
#html_theme = 'python_docs_theme'

html_static_path = ['_static']


# a better implementation with version: https://gist.github.com/nlgranger/55ff2e7ff10c280731348a16d569cb73
linkcode_url = "https://github.com/omron-sinicx/weavenet-egal/tree/main/{filepath}#L{linestart}-L{linestop}"
# Current git reference. Uses branch/tag name if found, otherwise uses commit hash

#linkcode_url = "https://github.com/omron-sinicx/weavenet-egal/blob/" \
#               + linkcode_revision + "/{filepath}#L{linestart}-L{linestop}"

def linkcode_resolve(domain, info):
    if domain != 'py' or not info['module']:
        return None

    modname = info['module']
    topmodulename = modname.split('.')[0]
    fullname = info['fullname']

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split('.'):
        try:
            obj = getattr(obj, part)
        except Exception:
            return None

    '''
    try:
        modpath = pkg_resources.require(topmodulename)[0].location
        filepath = os.path.relpath(inspect.getsourcefile(obj), modpath)
        if filepath is None:
            return
    except Exception:
        return None
    '''
    filepath = os.path.relpath(inspect.getsourcefile(obj), './')
    print(filepath)
    try:
        source, lineno = inspect.getsourcelines(obj)
    except OSError:
        return None
    else:
        linestart, linestop = lineno, lineno + len(source) - 1

    return linkcode_url.format(
        filepath=filepath, linestart=linestart, linestop=linestop)

