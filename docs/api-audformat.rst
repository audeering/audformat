audformat
=========

.. automodule:: audformat

AudioInfo
---------

.. autoclass:: audformat.AudioInfo
    :members:

Column
------

.. autoclass:: audformat.Column
    :members:

Database
--------

.. autoclass:: audformat.Database
    :members:
    :special-members:

index
-----

.. autofunction:: audformat.index

index_type
----------

.. autofunction:: audformat.index_type

MediaInfo
---------

.. autoclass:: audformat.MediaInfo
    :members:

Rater
-----

.. autoclass:: audformat.Rater
    :members:

Scheme
------

.. autoclass:: audformat.Scheme
    :members:

Split
-----

Tables can be classified by splits.
Usually one of :class:`audformat.define.SplitType`.

.. autoclass:: audformat.Split
    :members:

Table
-----

Annotation data is organized in tables,
which consist of file names
and columns that assign labels
or numerical values to the files.

There are two types of tables:

* :class:`audformat.define.TableType.FILEWISE` tables annotate whole files
* :class:`audformat.define.TableType.SEGMENTED` tables annotate file segments

.. autoclass:: audformat.Table
    :members:

VideoInfo
---------

.. autoclass:: audformat.VideoInfo
    :members:
