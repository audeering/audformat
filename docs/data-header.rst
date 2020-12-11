Header
======

The header of a database is organized in different parts:

* Core information about the database
* Information about media
* Information about raters
* Information about schemes
* Information about splits
* Tables and their columns

Each part is characterized by a set of fields (see below).
Note that fields marked as mandatory do not have a default value
and have to be set explicitly.
Optionally, every object can have an arbitrary number of ``meta`` fields.
These can be used to store additional information about the database.
Fields set to ``None`` can be omitted.


Database
--------

This part of the header is represented by :class:`audformat.Database`.

==============  =========  ====================================================
Field           Mandatory  Description
==============  =========  ====================================================
name            yes        Database name
source          yes        Original source,
                           e.g. link to webpage where files are hosted
usage           yes        What the database can be used for,
                           see :class:`audformat.define.Usage`
description                Description of the database
expires                    Until when we are allowed to use the data
languages                  List of languages that appear in the media files
media                      Dictionary of media objects (see below)
raters                     Dictionary of rater objects (see below)
schemes                    Dictionary of scheme objects (see below)
splits                     Dictionary of rater objects (see below)
tables                     Dictionary of tables (see below)
*meta-key-1*               1st optional meta field
...                        ...
*meta-key-N*               Nth optional meta field
==============  =========  ====================================================

Minimal example
^^^^^^^^^^^^^^^

.. code-block:: yaml

    name: databasename
    source: URL
    usage: commercial

audformat implementation
^^^^^^^^^^^^^^^^^^^^^^^^

.. jupyter-execute::

    import audformat


    # Create Database
    db = audformat.Database(
        name='databasename',
        source='https://gitlab.audeering.com/data/databasename',
        usage=audformat.define.Usage.COMMERCIAL,
    )
    db


Rater
-----

This part of the header is represented by :class:`audformat.Rater`.

==============  =========  ====================================================
Field           Mandatory  Description
==============  =========  ====================================================
id              yes        Unique identifier of rater
type            yes        Rater type, see :class:`audformat.define.RaterType`
description                Description of rater
*meta-key-1*               1st optional meta field
...                        ...
*meta-key-N*               Nth optional meta field
==============  =========  ====================================================

Minimal example
^^^^^^^^^^^^^^^

.. code-block:: yaml

    raters:
        raterid:
            type: human

audformat implementation
^^^^^^^^^^^^^^^^^^^^^^^^

.. jupyter-execute::

    # Create minimal Rater
    rater = audformat.Rater(audformat.define.RaterType.HUMAN)
    # Add Rater to Database
    db.raters['raterid'] = rater
    # Access type of Rater
    db.raters['raterid'].type
    # Access raters
    db.raters


Scheme
------

This part of the header is represented by :class:`audformat.Scheme`.

==============  =========  ====================================================
Field           Mandatory  Description
==============  =========  ====================================================
dtype           yes        Data type, see :class:`audformat.define.DataType`
id              yes        Unique identifier of scheme
description                Description of scheme
labels                     Dictionary or list with labels
                           (elements or keys must fit ``dtype``)
minimum                    Minimum label value (only applied if ``dtype`` is
                           numeric)
maximum                    Maximum label value (only applied if ``dtype`` is
                           numeric)
*meta-key-1*               1st optional meta field
...                        ...
*meta-key-N*               Nth optional meta field
==============  =========  ====================================================

Minimal example
^^^^^^^^^^^^^^^

.. code-block:: yaml

    schemes:
        schemeid:
            dtype: human

audformat implementation
^^^^^^^^^^^^^^^^^^^^^^^^

.. jupyter-execute::

    # Create minimal Scheme
    scheme = audformat.Scheme(audformat.define.DataType.FLOAT)
    # Add Scheme to Database
    db.schemes['schemeid'] = scheme
    # Access dtype of Scheme
    db.schemes['schemeid'].dtype
    # Access schemes
    db.schemes


Split
-----

This part of the header is represented by :class:`audformat.Split`.

==============  =========  ====================================================
Field           Mandatory  Description
==============  =========  ====================================================
id              yes        Unique identifier of split
type            yes        Split type,
                           typically one of :class:`audformat.define.SplitType`
description                Description of split
*meta-key-1*               1st optional meta field
...                        ...
*meta-key-N*               Nth optional meta field
==============  =========  ====================================================

Minimal example
^^^^^^^^^^^^^^^

.. code-block:: yaml

    splits:
        splitid:
            type: test

audformat implementation
^^^^^^^^^^^^^^^^^^^^^^^^

.. jupyter-execute::

    # Create minimal Split
    split = audformat.Split(audformat.define.SplitType.TEST)
    # Add Split to Database
    db.splits['splitid'] = split
    # Access type of Split
    db.splits['splitid'].type
    # Access splits
    db.splits


Media
-----

This part of the header is represented by :class:`audformat.Media`.

To store media information use:

================  =========  ====================================================
Field             Mandatory  Description
================  =========  ====================================================
id                yes        Unique identifier of media type
type                         Media type, one of (``audio``, ``video``, ``other``)
bit_depth                    Audio bit depth
channels                     Number of audio channels
description                  Description
format                       Media file format (e.g. ``wav`` or ``mp4``)
sampling_rate                Audio sampling rate in Hz
video_fps                    Video rate in frames per seconds
video_resolution             Video resolution in pixels (``width`` x ``height``)
video_channels               Number of channels per pixel (e.g. 3 for RGB)
video_depth                  Number of bits per video channel
*meta-key-1*                 1st optional meta field
...                          ...
*meta-key-N*                 Nth optional meta field
================  =========  ====================================================

Minimal example
^^^^^^^^^^^^^^^

.. code-block:: yaml

    media:
        mediaid:
            type: audio

audformat implementation
^^^^^^^^^^^^^^^^^^^^^^^^

.. jupyter-execute::

    # Create minimal media information
    media = audformat.Media()
    # Add media to Database
    db.media['mediaid'] = media
    # Access type of Media
    db.media['mediaid'].type
    # Access media
    db.media


Table
-----

This part of the header is represented by :class:`audformat.Table`

==============  =========  ====================================================
Field           Mandatory  Description
==============  =========  ====================================================
id              yes        Unique identifier of table
type            yes        Table type, see :class:`audformat.define.TableType`
columns                    Dictionary of columns (see below)
description                Description of table
media_id                   Files in this table are of this media type
split_id                   The split the table belongs to
*meta-key-1*               1st optional meta field
...                        ...
*meta-key-N*               Nth optional meta field
==============  =========  ====================================================

Minimal example
^^^^^^^^^^^^^^^

.. code-block:: yaml

    tables:
        tableid:
            type: filewise

audformat implementation
^^^^^^^^^^^^^^^^^^^^^^^^

.. jupyter-execute::

    # Create minimal Table
    table = audformat.Table(audformat.filewise_index())
    # Add Table to Database
    db.tables['tableid'] = table
    # Access type of Table
    db.tables['tableid'].type
    # Add Table to Database (short notation)
    db['tableid'] = table
    # Access type of Table (short notation)
    db['tableid'].type
    # Access tables
    db.tables


Column
------

This part of the header is represented by :class:`audformat.Column`

==============  =========  ====================================================
Field           Mandatory  Description
==============  =========  ====================================================
id              yes        Unique identifier of column
description                Description of column
scheme_id                  The scheme the values in this column belong to
rater_id                   The rater who assigned the values
*meta-key-1*               1st optional meta field
...                        ...
*meta-key-N*               Nth optional meta field
==============  =========  ====================================================

Minimal example
^^^^^^^^^^^^^^^

.. code-block:: yaml

    tables:
        tableid:
            type: filewise
            columns:
                columnid:

audformat implementation
^^^^^^^^^^^^^^^^^^^^^^^^

.. jupyter-execute::

    # Create minimal Column
    column = audformat.Column()
    # Add Column to Table
    db.tables['tableid'].columns['columnid'] = column
    # Add Column to Table (short notation)
    db['tableid']['columnid'] = column
    # Access columns
    db['tableid'].columns
