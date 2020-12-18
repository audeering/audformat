Combine tables
==============

It can happen that labels in your database are stored
in tables of different type as some labels belong to the whole file,
others don't. The following examples highlights this with the labels
for age and likability:

.. jupyter-execute::
    :hide-output:

    import audformat.testing


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
