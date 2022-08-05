audformat
=========

.. automodule:: audformat


assert_index
------------

.. autofunction:: assert_index

assert_no_duplicates
--------------------

.. autofunction:: assert_no_duplicates

Column
------

.. autoclass:: Column
    :members:
    :inherited-members:

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
    :inherited-members:
    :special-members:

index_type
----------

.. autofunction:: index_type

is_filewise_index
-----------------

.. autofunction:: is_filewise_index

is_segmented_index
------------------

.. autofunction:: is_segmented_index

Media
---------

.. autoclass:: Media
    :members:
    :inherited-members:

MiscTable
---------

.. autoclass:: MiscTable
    :members:
    :inherited-members:
    :special-members: __add__, __getitem__, __setitem__

Rater
-----

.. autoclass:: Rater
    :members:
    :inherited-members:

Scheme
------

.. autoclass:: Scheme
    :members:
    :inherited-members:

Split
-----

Tables can be classified by splits.
Usually one of :class:`define.SplitType`.

.. autoclass:: Split
    :members:
    :inherited-members:

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
    :inherited-members:
    :special-members: __add__, __getitem__, __setitem__
