.. _create-a-database:

Create a database
=================

The following examples show you
how to create a :class:`audformat.Database`
and store it to the disk.


Use existing audio and CSV files
--------------------------------

Assume you have three audio files
in a sub-folder :file:`audio`::

    audio/utt01.wav
    audio/utt02.wav
    audio/utt03.wav

Let's assume in addition you have
three raters annotated the files
for the emotion anger in the range 0 to 5.

.. code-block:: python

    import io


    # Create dummy CSV data.
    # In a real example this would be a file on your disk
    CSV = io.StringIO("""
    file,R1,R2,R3
    audio/utt01.wav,1,0,1
    audio/utt02.wav,1,3,2
    audio/utt03.wav,,5,4
    """)

From the CSV file you create a :class:`audformat.Database`.
The database will contain one :class:`audformat.Table`
with the ID ``anger``,
split ID ``train``,
and media ID ``microphone``.
The table contains three :class:`audformat.Column` objects,
one per rater.
Each column has the scheme ID ``anger``
for the defined scheme covering the emotion anger.


.. code-block:: python

    import audformat


    df = audformat.utils.read_csv(CSV)

    # Create database
    db = audformat.Database(
        name="foo",
        source="https://github.com/audeering/audformat/",
        usage=audformat.define.Usage.UNRESTRICTED,
    )

    # Add media, split and scheme
    db.media["microphone"] = audformat.Media(
        audformat.define.MediaType.AUDIO,
        format="wav",
    )
    db.splits["train"] = audformat.Split(
        audformat.define.SplitType.TRAIN,
    )
    db.schemes["anger"] = audformat.Scheme(
        audformat.define.DataType.INTEGER,
        minimum=0,
        maximum=5,
    )

    # Create table and fill with data
    db["anger"] = audformat.Table(
        index=df.index,
        media_id="microphone",
        split_id="train",
    )
    for rater in df.columns:
        db.raters[rater] = audformat.Rater()
        db["anger"][rater] = audformat.Column(scheme_id="anger", rater_id=rater)
        db["anger"][rater].set(df[rater])


The resulting :class:`audformat.Database` will then contain:

>>> db
name: foo
source: https://github.com/audeering/audformat/
usage: unrestricted
languages: []
media:
  microphone: {type: audio, format: wav}
raters:
  R1: {type: human}
  R2: {type: human}
  R3: {type: human}
schemes:
  anger: {dtype: int, minimum: 0, maximum: 5}
splits:
  train: {type: train}
tables:
  anger:
    type: filewise
    split_id: train
    media_id: microphone
    columns:
      R1: {scheme_id: anger, rater_id: R1}
      R2: {scheme_id: anger, rater_id: R2}
      R3: {scheme_id: anger, rater_id: R3}

For more information on how to define a database,
have a look at the code examples in the
:ref:`database specification <data-header:Database>`.


Create a test database
----------------------

If you want to write unit tests using a :class:`audformat.Database`,
or you just want to play around with a database
without creating one, you can use :mod:`audformat.testing`.
It provides you with a command to create a database,
containing all possible :ref:`tables types <data-tables:Tables>`:

.. code-block:: python

    import audformat.testing


    db = audformat.testing.create_db()

Which results in the following :class:`audformat.Table` objects:

>>> db.tables
files:
  type: filewise
  split_id: train
  media_id: microphone
  columns:
    bool: {scheme_id: bool, rater_id: gold}
    date: {scheme_id: date, rater_id: gold}
    float: {scheme_id: float, rater_id: gold}
    int: {scheme_id: int, rater_id: gold}
    label: {scheme_id: label, rater_id: gold}
    label_map_int: {scheme_id: label_map_int, rater_id: gold}
    label_map_misc: {scheme_id: label_map_misc, rater_id: gold}
    label_map_str: {scheme_id: label_map_str, rater_id: gold}
    string: {scheme_id: string, rater_id: gold}
    time: {scheme_id: time, rater_id: gold}
    no_scheme: {}
segments:
  type: segmented
  split_id: dev
  media_id: microphone
  columns:
    bool: {scheme_id: bool, rater_id: gold}
    date: {scheme_id: date, rater_id: gold}
    float: {scheme_id: float, rater_id: gold}
    int: {scheme_id: int, rater_id: gold}
    label: {scheme_id: label, rater_id: gold}
    label_map_int: {scheme_id: label_map_int, rater_id: gold}
    label_map_misc: {scheme_id: label_map_misc, rater_id: gold}
    label_map_str: {scheme_id: label_map_str, rater_id: gold}
    string: {scheme_id: string, rater_id: gold}
    time: {scheme_id: time, rater_id: gold}
    no_scheme: {}

Or you can create a database,
containing only the minimum entries,
required by the :ref:`database specification <data-header:Database>`:

>>> audformat.testing.create_db(minimal=True)
name: unittest
source: internal
usage: unrestricted
languages: [deu, eng]
