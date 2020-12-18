import configparser
from datetime import date
import os
import subprocess

import graphviz


config = configparser.ConfigParser()
config.read(os.path.join('..', 'setup.cfg'))

# Project -----------------------------------------------------------------
author = config['metadata']['author']
copyright = f'2019-{date.today().year} audEERING GmbH'
project = config['metadata']['name']
# The x.y.z version read from tags
try:
    version = subprocess.check_output(
        ['git', 'describe', '--tags', '--always']
    )
    version = version.decode().strip()
except Exception:
    version = '<unknown>'
title = f'{project} Documentation'


# General -----------------------------------------------------------------
master_doc = 'index'
extensions = []
source_suffix = '.rst'
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']
pygments_style = None
extensions = [
    'sphinx.ext.graphviz',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # support for Google-style docstrings
    'sphinx_autodoc_typehints',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx_copybutton',
    'sphinxcontrib.katex',  # has to be before jupyter_sphinx
    'jupyter_sphinx',
]

napoleon_use_ivar = True  # List of class attributes
autodoc_inherit_docstrings = False  # disable docstring inheritance
intersphinx_mapping = {
    'audeer': ('https://audeering.github.io/audeer/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'python': ('https://docs.python.org/3/', None),
}
# Ignore package dependencies during building the docs
autodoc_mock_imports = [
    'tqdm',
]

# Reference with :ref:`data-header:Database`
autosectionlabel_prefix_document = True
autosectionlabel_maxdepth = 2

# Do not copy prompot output
copybutton_prompt_text = r'>>> |\.\.\. |$ '
copybutton_prompt_is_regexp = True

# Mapping to external documentation
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
}

linkcheck_ignore = [
    'https://www.isca-speech.org',
]


# HTML --------------------------------------------------------------------
html_theme = 'sphinx_audeering_theme'
html_theme_options = {
    'display_version': True,
    'logo_only': False,
    'footer_links': False,
    'wide_pages': ['data-example'],
}
html_context = {
    'display_github': True,
}
html_title = title


# Graphviz figures --------------------------------------------------------
dot_files = [
    './pics/tables.dot',
    './pics/audformat.dot',
    './pics/workflow.dot',
]
for dot_file in dot_files:
    graphviz.render('dot', 'svg', dot_file)
