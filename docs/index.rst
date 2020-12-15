.. documentation master file

.. include:: ../README.rst
    :end-line: 6

.. toctree::
    :caption: Getting started
    :hidden:

    install
    usage

.. toctree::
    :caption: Format specification
    :hidden:

    data-introduction
    data-format
    data-conventions
    data-header
    data-tables
    data-example

.. Warning: then usage of genindex is a hack to get a TOC entry, see
.. https://stackoverflow.com/a/42310803. This might break the usage of sphinx if
.. you want to create something different than HTML output.
.. toctree::
    :caption: API Documentation
    :hidden:

    api
    api-define
    api-errors
    api-testing
    api-utils
    genindex

.. toctree::
    :caption: Development
    :hidden:

    contributing
    changelog
