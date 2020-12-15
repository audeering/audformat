Database
========

audformat is implemented in the :class:`audformat.Database`.
Written to hard disk it is converted to a YAML file (*Header*),
which contains information about the raters,
annotation schemes and meta information.
Actual annotations are stored
across (possibly) multiple CSV files (*Tables*).
Each table column is linked to a scheme and/or to a rater.
Each table row is linked to a media file,
or a specific segment in a media file.

.. table:: Parts of a database stored in the audformat on the hard disk.

    =======================  ==========================================
    File                     Content
    =======================  ==========================================
    ``db.yaml``              Meta information, schemes, list of raters
    ``db.<table_id>.csv``    Table with files or file segments as index
                             and columns holding annotations,
                             such as speaker id,
                             emotion ratings, ...
    ``<folder(s)/file(s)>``  Audio/Video files referenced in the tables
    =======================  ==========================================

The connection between the header and the tables
is highlighted in the following sketch:

.. figure:: pics/audformat.dot.svg
    :alt: audformat
    :align: center

    Connection between header definitions and table entries.

The annotations stored in the tables
can be accessed as :class:`pandas.DataFrame`.
The following sketch shows an example instance of a database:

.. figure:: pics/tables.dot.svg
    :alt: Header and Tables
    :align: center

    Example content of tables and there connection to the header.
