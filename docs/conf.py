import configparser
from datetime import date
import os
import subprocess

import graphviz


config = configparser.ConfigParser()
config.read(os.path.join('..', 'setup.cfg'))

# Project -----------------------------------------------------------------
author = config['metadata']['author']
copyright = f'2020-{date.today().year} audEERING GmbH'
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
linkcheck_ignore = [
    'https://gitlab.audeering.com',
]
# Ignore package dependencies during building the docs
autodoc_mock_imports = [
    'tqdm',
]

# Reference with :ref:`data-header:Database`
autosectionlabel_prefix_document = True
autosectionlabel_maxdepth = 2

# Do not copy prompot output
copybutton_prompt_text = r'>>> |\.\.\. '
copybutton_prompt_is_regexp = True

# Mapping to external documentation
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
}

# Disable Gitlab as we need to sign in
linkcheck_ignore = [
    'https://gitlab.audeering.com',
]

# Disable auto-build of Jupyter notebooks
nbsphinx_execute = 'never'
# This is processed by Jinja2 and inserted before each Jupyter notebook
nbsphinx_prolog = r"""
{% set docname = env.doc2path(env.docname, base='docs') %}
{% set base_url = "https://gitlab.audeering.com/tools/audfoo/raw" %}

.. role:: raw-html(raw)
    :format: html

:raw-html:`<div class="notebook"><a href="{{ base_url }}/{{ env.config.version }}/{{ docname }}?inline=false"> Download notebook: {{ docname }}</a></div>`
"""  # noqa: E501
nbsphinx_timeout = 3600

# HTML --------------------------------------------------------------------
html_theme = 'sphinx_audeering_theme'
html_theme_options = {
    'display_version': True,
    'logo_only': False,
    'wide_pages': ['data-example'],
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
