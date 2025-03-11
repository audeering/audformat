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
                           one of ``"commercial"``, ``"other"``,
                           ``"research"``, ``"restricted"``, ``"unrestricted"``
author                     Author(s) of the database
description                Description of the database
expires                    Until when we are allowed to use the data
languages                  List of languages that appear in the media files
license                    License of the database
license_url                Link to license statement
attachments                Dictionary of attachment objects (see below)
media                      Dictionary of media objects (see below)
organization               Organization that created the database
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

>>> import audformat
>>> # Create Database
>>> db = audformat.Database(
...     name="databasename",
...     source="https://gitlab.audeering.com/data/databasename",
...     usage="commercial",
... )
>>> db
name: databasename
source: https://gitlab.audeering.com/data/databasename
usage: commercial
languages: []


Attachment
----------

This part of the header is represented by :class:`audformat.Attachment`.

==============  =========  ====================================================
Field           Mandatory  Description
==============  =========  ====================================================
path            yes        Relative path to attached file/folder
description                Description of rater
*meta-key-1*               1st optional meta field
...                        ...
*meta-key-N*               Nth optional meta field
==============  =========  ====================================================

Minimal example
^^^^^^^^^^^^^^^

.. code-block:: yaml

    attachments:
        attachmentid:
            path: docs/setup.pdf

audformat implementation
^^^^^^^^^^^^^^^^^^^^^^^^

>>> # Create minimal Attachment
>>> attachment = audformat.Attachment("docs/setup.pdf")
>>> # Add Attachment to Database
>>> db.attachments["attachmentid"] = attachment
>>> # Access path of Attachment
>>> db.attachments["attachmentid"].path
'docs/setup.pdf'
>>> # Access attachments
>>> db.attachments
attachmentid:
  {path: docs/setup.pdf}


Rater
-----

This part of the header is represented by :class:`audformat.Rater`.

==============  =========  ====================================================
Field           Mandatory  Description
==============  =========  ====================================================
id              yes        Unique identifier of rater
type            yes        Rater type, one of ``"human"``, ``"machine"``,
                           ``"other"``, ``"ground truth"``, ``"vote"``
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

>>> # Create minimal Rater
>>> rater = audformat.Rater("human")
>>> # Add Rater to Database
>>> db.raters["raterid"] = rater
>>> # Access type of Rater
>>> db.raters["raterid"].type
'human'
>>> # Access raters
>>> db.raters
raterid:
  {type: human}


Scheme
------

This part of the header is represented by :class:`audformat.Scheme`.

==============  =========  ====================================================
Field           Mandatory  Description
==============  =========  ====================================================
dtype           yes        Data type, one of ``"bool"``,
                           ``"int"``, ``"float"``, ``"object"``,
                           ``"str"``, ``"time"``, ``"date"``
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
            dtype: float

audformat implementation
^^^^^^^^^^^^^^^^^^^^^^^^

>>> # Create minimal Scheme
>>> scheme = audformat.Scheme("float")
>>> # Add Scheme to Database
>>> db.schemes["schemeid"] = scheme
>>> # Access dtype of Scheme
>>> db.schemes["schemeid"].dtype
'float'
>>> # Access schemes
>>> db.schemes
schemeid:
  {dtype: float}


Split
-----

This part of the header is represented by :class:`audformat.Split`.

==============  =========  ====================================================
Field           Mandatory  Description
==============  =========  ====================================================
id              yes        Unique identifier of split
type            yes        Split type, one of ``"train"``, ``"dev"``,
                           ``"other"``, ``"test"``
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

>>> # Create minimal Split
>>> split = audformat.Split("test")
>>> # Add Split to Database
>>> db.splits["splitid"] = split
>>> # Access type of Split
>>> db.splits["splitid"].type
'test'
>>> # Access splits
>>> db.splits
splitid:
  {type: test}


Media
-----

This part of the header is represented by :class:`audformat.Media`.

To store media information use:

================  =========  ====================================================
Field             Mandatory  Description
================  =========  ====================================================
id                yes        Unique identifier of media type
type                         Media type, one of ``"audio"``, ``"video"``,
                             ``"other"``
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

>>> # Create minimal media information
>>> media = audformat.Media("audio")
>>> # Add media to Database
>>> db.media["mediaid"] = media
>>> # Access type of Media
>>> db.media["mediaid"].type
'audio'
>>> # Access media
>>> db.media
mediaid:
  {type: audio}


Table
-----

This part of the header is represented by :class:`audformat.Table`

==============  =========  ====================================================
Field           Mandatory  Description
==============  =========  ====================================================
id              yes        Unique identifier of table
type            yes        Table type, one of ``"filewise"``, ``"segmented"``
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

>>> # Create minimal Table
>>> table = audformat.Table(audformat.filewise_index())
>>> # Add Table to Database
>>> db.tables["tableid"] = table
>>> # Access type of Table
>>> db.tables["tableid"].type
'filewise'
>>> # Add Table to Database (short notation)
>>> db["tableid"] = table
>>> # Access type of Table (short notation)
>>> db["tableid"].type
'filewise'
>>> # Access tables
>>> db.tables
tableid:
  {type: filewise}


Misc Table
----------

This part of the header is represented by :class:`audformat.MiscTable`.

==============  =========  ====================================================
Field           Mandatory  Description
==============  =========  ====================================================
id              yes        Unique identifier of misc table
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

    misc_tables:
        misctableid:
            levels: [idx]

audformat implementation
^^^^^^^^^^^^^^^^^^^^^^^^

>>> # Create minimal Misc Table
>>> import pandas as pd
>>> misc_table = audformat.MiscTable(pd.Index([], name="idx"))
>>> # Add Misc Table to Database
>>> db.misc_tables["misctableid"] = misc_table
>>> # Access dataframe of Misc Table
>>> db.misc_tables["misctableid"].df
Empty DataFrame
Columns: []
Index: []
>>> # Add Misc Table to Database (short notation)
>>> db["misctableid"] = misc_table
>>> # Access dataframe of Misc Table (short notation)
>>> db["misctableid"].df
Empty DataFrame
Columns: []
Index: []
>>> # Access misc tables
>>> db.misc_tables
misctableid:
  levels: {idx: object}


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

>>> # Create minimal Column
>>> column = audformat.Column()
>>> # Add Column to Table
>>> db.tables["tableid"].columns["columnid"] = column
>>> # Add Column to Table (short notation)
>>> db["tableid"]["columnid"] = column
>>> # Access columns
>>> db["tableid"].columns
columnid:
  {}
