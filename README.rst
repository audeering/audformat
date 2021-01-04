=========
audformat
=========

|tests| |coverage| |docs| |python-versions| |license|

Specification and reference implementation of **audformat**.

audformat stores media data,
such as audio or video,
together with corresponding annotations
in a pre-defined way.
This makes it easy to combine or replace databases
in machine learning projects.

An audformat database is a folder
that contains media files
together with a header YAML file
and one or several files storing the annotations.
The database is represented as an ``audformat.Database`` object
and can be loaded with ``audformat.Database.load()``
or written to disk with ``audformat.Database.save()``.

Have a look at the installation_ and usage_ instructions
and the `format specifications`_ as a starting point.


.. _installation: https://audeering.github.io/audformat/install.html
.. _usage: https://audeering.github.io/audformat/create-database.html
.. _format specifications: https://audeering.github.io/audformat/data-introduction.html


.. badges images and links:
.. |tests| image:: https://github.com/audeering/audformat/workflows/Test/badge.svg
    :target: https://github.com/audeering/audformat/actions?query=workflow%3ATest
    :alt: Test status
.. |coverage| image:: https://codecov.io/gh/audeering/audformat/branch/master/graph/badge.svg?token=1FEG9P5XS0
    :target: https://codecov.io/gh/audeering/audformat/
    :alt: code coverage
.. |docs| image:: https://img.shields.io/pypi/v/audformat?label=docs
    :target: https://audeering.github.io/audformat/
    :alt: audformat's documentation
.. |license| image:: https://img.shields.io/badge/license-MIT-green.svg
    :target: https://github.com/audeering/audformat/blob/master/LICENSE
    :alt: audformat's MIT license
.. |python-versions| image:: https://img.shields.io/pypi/pyversions/audformat.svg
    :target: https://pypi.org/project/audformat/
    :alt: audformats's supported Python versions
