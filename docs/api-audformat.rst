audata
======

.. automodule:: audata

AudioInfo
---------

.. autoclass:: audata.AudioInfo
    :members:

BadIdError
----------

.. autoclass:: audata.BadIdError
    :members:

BadTypeError
------------

.. autoclass:: audata.BadTypeError
    :members:

BadValueError
-------------

.. autoclass:: audata.BadValueError
    :members:

CannotCreateSegmentedTable
--------------------------

.. autoclass:: audata.CannotCreateSegmentedTable
    :members:

Column
------

.. autoclass:: audata.Column
    :members:

ColumnFilter
------------

.. autoclass:: audata.ColumnFilter
    :members:
    :special-members:

ColumnNotAssignedToTableError
-----------------------------

.. autoclass:: audata.ColumnNotAssignedToTableError
    :members:

Database
--------

.. autoclass:: audata.Database
    :members:
    :special-members:

DatabaseFilter
--------------

.. autoclass:: audata.DatabaseFilter
    :members:
    :special-members:

Filter
------

.. autoclass:: audata.Filter
    :members:
    :special-members:

MediaInfo
---------

.. autoclass:: audata.MediaInfo
    :members:

Rater
-----

.. autoclass:: audata.Rater
    :members:

RedundantArgumentError
----------------------

.. autoclass:: audata.RedundantArgumentError
    :members:

Scheme
------

.. autoclass:: audata.Scheme
    :members:

Split
-----

Tables can be classified by splits.
Usually one of :class:`audata.define.SplitType`.

.. autoclass:: audata.Split
    :members:

Table
-----

Annotation data is organized in tables,
which consist of file names
and columns that assign labels
or numerical values to the files.

There are two types of tables:

* :class:`audata.define.TableType.FILEWISE` tables annotate whole files
* :class:`audata.define.TableType.SEGMENTED` tables annotate file segments

.. autoclass:: audata.Table
    :members:

TableFilter
-----------

.. autoclass:: audata.TableFilter
    :members:
    :special-members:

VideoInfo
---------

.. autoclass:: audata.VideoInfo
    :members:
