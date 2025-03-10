from datetime import date

import toml

import audeer


config = toml.load(audeer.path("..", "pyproject.toml"))


# Project -----------------------------------------------------------------
project = config["project"]["name"]
copyright = f"2019-{date.today().year} audEERING GmbH"
author = ", ".join(author["name"] for author in config["project"]["authors"])
version = audeer.git_repo_version()
title = "Documentation"


# General -----------------------------------------------------------------
master_doc = "index"
extensions = []
source_suffix = ".rst"
exclude_patterns = [
    "api-src",
    "build",
    "tests",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
    "emodb-src",
    "__pycache__",
]
pygments_style = None
extensions = [
    "sphinx.ext.graphviz",
    "sphinx.ext.napoleon",  # support for Google-style docstrings
    "sphinx_autodoc_typehints",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinxcontrib.katex",  # has to be before jupyter_sphinx
    "sphinx_apipages",
]

napoleon_use_ivar = True  # List of class attributes
autodoc_inherit_docstrings = True  # disable docstring inheritance
# autosummary_generate_overwrite = True

intersphinx_mapping = {
    "audeer": ("https://audeering.github.io/audeer/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}
# Ignore package dependencies during building the docs
autodoc_mock_imports = [
    "tqdm",
]

# Reference with :ref:`data-header:Database`
autosectionlabel_prefix_document = True
autosectionlabel_maxdepth = 2

# Do not copy prompt output
copybutton_prompt_text = r">>> |\.\.\. |$ "
copybutton_prompt_is_regexp = True

linkcheck_ignore = [
    "https://www.isca-speech.org",
    "http://emodb.bilderbar.info",
]

# Graphviz figures
graphviz_output_format = "svg"

apipages_hidden_methods = [
    "__add__",
    "__call__",
    "__contains__",
    "__eq__",
    "__getitem__",
    "__iter__",
    "__len__",
    "__setitem__",
]

# HTML --------------------------------------------------------------------
html_theme = "sphinx_audeering_theme"
html_theme_options = {
    "display_version": True,
    "logo_only": False,
    "footer_links": False,
    "wide_pages": ["data-example"],
}
html_context = {
    "display_github": True,
}
html_title = title
