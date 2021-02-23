audformat
=========

.. automodule:: audformat


assert_index
------------

.. autofunction:: assert_index

Column
------

.. autoclass:: Column
    :members:

filewise_index
--------------

.. autofunction:: filewise_index

segmented_index
---------------

.. autofunction:: segmented_index

Database
--------

.. autoclass:: Database
    :members:
    :special-members:

index_type
----------

.. autofunction:: index_type

Media
---------

.. autoclass:: Media
    :members:

Rater
-----

.. autoclass:: Rater
    :members:

Scheme
------

.. autoclass:: Scheme
    :members:

Split
-----

Tables can be classified by splits.
Usually one of :class:`define.SplitType`.

.. autoclass:: Split
    :members:

Table
-----

Annotation data is organized in tables,
which consist of file names
and columns that assign labels
or numerical values to the files.

There are two types of tables:

* :class:`define.TableType.FILEWISE` tables annotate whole files
* :class:`define.TableType.SEGMENTED` tables annotate file segments

.. autoclass:: Table
    :members:
    :special-members:
