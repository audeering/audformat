Usage
=====

:mod:`audformat` is the reference implementation
of the :ref:`data-format:audformat`.
So it's main usage is to store data in that format
or to access or change data from the format.
:ref:`data-format:audformat` is represented by
the :class:`audformat.Database` object.
A :class:`audformat.Database` can be generated manually,
or loaded from disk using :meth:`audformat.Database.load()`.
An existing database can written to disk with
:meth:`audformat.Database.save()`.


Create a database from CSV files
--------------------------------

The following example shows
how to create a :class:`audformat.Database`
from ratings stored in a CSV file.
The raters rated the emotion anger in the range 0 to 5.

.. jupyter-execute::

    import io
    import audformat


    # Create dummy CSV data
    CSV = io.StringIO('''
    file,R1,R2,R3
    audio/utt01.wav,1,0,1
    audio/utt02.wav,1,3,2
    audio/utt03.wav,,5,4
    ''')
    df = audformat.utils.read_csv(CSV)

    # Create database
    db = audformat.Database(
        name='foo',
        source='https://gitlab.audeering.com/tools/audformat',
        usage=audformat.define.Usage.COMMERCIAL,
    )

    # Add media, split and scheme
    db.media['microphone'] = audformat.Media(
        audformat.define.MediaType.AUDIO,
        format='wav',
    )
    db.splits['train'] = audformat.Split(
        audformat.define.SplitType.TRAIN,
    )
    db.schemes['anger'] = audformat.Scheme(
        audformat.define.DataType.INTEGER,
        minimum=0,
        maximum=5,
    )

    # Create table and fill with data
    db['anger'] = audformat.Table(
        index=df.index,
        media_id='microphone',
        split_id='train')
    for rater in df.columns:
        db.raters[rater] = audformat.Rater()
        db['anger'][rater] = audformat.Column(scheme_id='anger', rater_id=rater)
        db['anger'][rater].set(df[rater])


The resulting :class:`audformat.Database` will then contain:

.. jupyter-execute::

    db

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

.. jupyter-execute::

    import audformat.testing


    db = audformat.testing.create_db()

Which results in the following :class:`audformat.Table`:

.. jupyter-execute::

    db.tables

Or you can create a database,
containing only the minimum entries,
required by the :ref:`database specification <data-header:Database>`:

.. jupyter-execute::

    db_minimal = audformat.testing.create_db(minimal=True)

Which results in the following :class:`audformat.Database`:

.. jupyter-execute::

    db_minimal


Accessing data in a database
----------------------------

Annotation labels can be accessed
by the :meth:`audformat.Table.get` method:

.. jupyter-execute::

    table = db.tables['files'].get()

Which returns the following :class:`pandas.DataFrame`:

.. jupyter-execute::

    table.iloc[0:2, 0:2]

Or you can directly access a column with :meth:`audformat.Column.get()`:

.. jupyter-execute::

    column = db.tables['files'].columns['string'].get()

Which results in the following :class:`pandas.Series`:

.. jupyter-execute::

    column[0:2]

For more information on how to access or add data
have a look at the code examples in the
:ref:`table specification <data-tables:Tables>`.


Changing database entries
-------------------------

To convert to absolute file paths in all tables, do:

.. code-block:: python

    db.map_files(os.path.abspath)


Combining data from tables
--------------------------

It can happen that labels in your database are stored
in tables of different type as some labels belong to the whole file,
others don't. The following examples highlights this with the labels
for age and likability:

.. jupyter-execute::
    :hide-output:

    db = audformat.testing.create_db(minimal=True)
    db.schemes['age'] = audformat.Scheme(
        audformat.define.DataType.INTEGER,
        minimum=20,
        maximum=50,
    )
    db.schemes['likability'] = audformat.Scheme(
        audformat.define.DataType.FLOAT,
    )
    audformat.testing.add_table(
        db,
        table_id='age',
        index_type=audformat.define.IndexType.FILEWISE,
        columns='age',
        num_files=3,
    )
    audformat.testing.add_table(
        db,
        table_id='likability',
        index_type=audformat.define.IndexType.SEGMENTED,
        columns='likability',
        num_files=4,
    )

Which results in the following two :class:`pandas.DataFrame`:

.. jupyter-execute::

    display(
        db['age'].get(),
        db['likability'].get(),
    )

You can simply combine both tables with:

.. jupyter-execute::

    combined_table = db['likability'] + db['age']

Which results in the following :class:`pandas.DataFrame`:

.. jupyter-execute::

    combined_table.get()

Or, if you just want to have the likability information for all segments,
for which age information is available:

.. jupyter-execute::

    df_likability = db['likability'].get(
        db['age'].files,
    )

Which results in the following :class:`pandas.DataFrame`:

.. jupyter-execute::

    df_likability

Or, if you want to have the age information for segments
in the likeability table:

.. jupyter-execute::

    df_age = db['age'].get(df_likability.index)

Which results in the following :class:`pandas.DataFrame`:

.. jupyter-execute::

    df_age
