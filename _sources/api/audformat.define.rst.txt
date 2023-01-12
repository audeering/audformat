audformat.define
================

.. automodule:: audformat.define

The :mod:`audformat.define` module
provides definitions of string values
that are part of the
:ref:`audformat specifications <data-format:Database>`
and should not be changed by a user,
namely:

.. autosummary::
    :toctree:
    :nosignatures:

    DataType
    IndexField
    IndexType
    MediaType
    RaterType
    SplitType
    TableStorageFormat
    Usage

In addition,
it provides definitions of string values
commonly used inside a database,
like licenses or scheme labels.
Those are not part of the
:ref:`audformat specifications <data-format:Database>`
and can be changed by a user,
namely:

.. autosummary::
    :toctree:
    :nosignatures:

    Gender
    License
