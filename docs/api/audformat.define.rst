audformat.define
================

.. automodule:: audformat.define

The :mod:`audformat.define` module
provides definitions of string values
that are part of the
:ref:`audformat specifications <data-format:Database>`
and should not be changed by a user,
namely those inside

* :class:`audformat.define.DataType`
* :class:`audformat.define.IndexField`
* :class:`audformat.define.IndexType`
* :class:`audformat.define.MediaType`
* :class:`audformat.define.RaterType`
* :class:`audformat.define.SplitType`
* :class:`audformat.define.TableStorageFormat`
* :class:`audformat.define.Usage`

In addition,
it provides definitions of string values
commonly used inside a database,
like licenses or scheme labels.
Those are not part of the
:ref:`audformat specifications <data-format:Database>`
and can be changed by a user,
namely those inside

* :class:`audformat.define.Gender`
* :class:`audformat.define.License`

.. rubric:: Classes

.. autosummary::
    :toctree:
    :nosignatures:

    DataType
    Gender
    IndexField
    IndexType
    License
    MediaType
    RaterType
    SplitType
    TableStorageFormat
    Usage
