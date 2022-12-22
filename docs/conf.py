import configparser
from datetime import date
import os
import shutil

import audeer


config = configparser.ConfigParser()
config.read(os.path.join('..', 'setup.cfg'))


# Project -----------------------------------------------------------------
author = config['metadata']['author']
copyright = f'2019-{date.today().year} audEERING GmbH'
project = config['metadata']['name']
version = audeer.git_repo_version()
title = 'Documentation'


# General -----------------------------------------------------------------
master_doc = 'index'
extensions = []
source_suffix = '.rst'
exclude_patterns = [
    'api-src',
    'build',
    'tests',
    'Thumbs.db',
    '.DS_Store',
    '**.ipynb_checkpoints',
    'emodb-src',
    '__pycache__',
]
templates_path = ['_templates']
pygments_style = None
extensions = [
    'sphinx.ext.graphviz',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # support for Google-style docstrings
    'sphinx.ext.autosummary',
    'sphinx_autodoc_typehints',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx_copybutton',
    'sphinxcontrib.katex',  # has to be before jupyter_sphinx
    'jupyter_sphinx',
]

napoleon_use_ivar = True  # List of class attributes
autodoc_inherit_docstrings = True  # disable docstring inheritance
# autosummary_generate_overwrite = True

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

# Do not copy prompt output
copybutton_prompt_text = r'>>> |\.\.\. |$ '
copybutton_prompt_is_regexp = True

linkcheck_ignore = [
    'https://www.isca-speech.org',
]

# Graphviz figures
graphviz_output_format = 'svg'

# Disable auto-generation of TOC entries in the API
# https://github.com/sphinx-doc/sphinx/issues/6316
toc_object_entries = False


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


# Copy API (sub-)module RST files to docs/api/ folder ---------------------
audeer.rmdir('api')
audeer.mkdir('api')
api_src_files = audeer.list_file_names('api-src')
api_dst_files = [
    audeer.path('api', os.path.basename(src_file))
    for src_file in api_src_files
]
for src_file, dst_file in zip(api_src_files, api_dst_files):
    shutil.copyfile(src_file, dst_file)
