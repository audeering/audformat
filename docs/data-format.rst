Database
========

An audformat database consists of a **header**,
several **tables**,
and **media** files.
On hard disk all of them are stored inside a single folder.
The header is stored as a YAML file,
the tables contain labels stored in (possibly) multiple CSV or PARQUET files,
and the media files are usually stored in sub-folders.
Media files are not restricted to a particular file type.
Usually, they consist of audio, video, or text files.
Each table column is linked to a scheme and/or to a rater.
Each table row is linked to a media file,
or,
if applicable,
a specific segment in a media file.
If no links to media files are given,
the table is called miscellaneous table,
or short **misc table**.
The database is implemented as :class:`audformat.Database`.

.. table:: Parts of a database stored in audformat on the hard disk.

    ====================================  ==========================================
    File                                  Content
    ====================================  ==========================================
    ``db.yaml``                           Meta information, schemes, list of raters
    ``db.<table_id>.[csv|parquet]``       Table with files or file segments as index
                                          and columns holding annotations
    ``db.<misc_table_id>.[csv|parquet]``  Misc table with unspecified index
                                          and columns holding annotations
    ``<folder(s)/file(s)>``               Media files referenced in the tables
    ====================================  ==========================================

The connection between the header, media files and a table
is highlighted in the following sketch:

.. graphviz:: pics/audformat-table.dot
    :alt: audformat table
    :align: center
    :caption: Connection between header definitions and table entries.

The connection between the header and a misc table
is highlighted in the following sketch:

.. graphviz:: pics/audformat-misc-table.dot
    :alt: audformat misc table
    :align: center
    :caption: Connection between header definitions and misc table entries.

The annotations stored in the tables
can be accessed as :class:`pandas.DataFrame`.
The following sketch shows an example instance of a database:

.. graphviz:: pics/audformat-database.dot
    :alt: Header and Tables
    :align: center
    :caption: Example content of tables and there connection to the header.
