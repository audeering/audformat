=========
audformat
=========

Specification and reference implementation of **audformat**.

audformat is intended to store media data,
such as audio or video,
together with corresponding annotations
in a pre-defined way.
An audformat database is a folder that contains
media files together with a header YAML file
and one or several files storing the annotations.

An audformat database is represented as a
``audformat.Database`` object and can be loaded with
``audformat.Database.load()``
or written to disk with
``audformat.Database.save()``.

Have a look at the installation_ and usage_ instructions
and the `format specifications`_ as a starting point.

.. _installation: https://audeering.github.io/audformat/installation.html
.. _usage: https://audeering.github.io/audformat/create-database.html
.. _format specifications: https://audeering.github.io/audformat/data-introduction.html
