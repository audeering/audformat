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

So far we have combined tables using the ``+`` operator.
The result is a table that is no longer attached to a database.
That means that meta information about the media
or referenced schemes is discarded.
If you want to keep this information,
you can use :meth:`audformat.Table.update`,
which also works across databases,
as we will demonstrate with the following example.

First we create a second database
and add a gender scheme:

.. jupyter-execute::

    db2 = audformat.testing.create_db(minimal=True)
    db2.schemes['gender'] = audformat.Scheme(
        labels=['female', 'male'],
    )
    db2.schemes

Next, we add a table and fill in some gender information:

.. jupyter-execute::

    audformat.testing.add_table(
        db2,
        table_id='gender_and_age',
        index_type=audformat.define.IndexType.FILEWISE,
        columns='gender',
        num_files=[2, 3, 4],
    ).get()

Now, we update the table with age values from the other database.

.. jupyter-execute::

    db2['gender_and_age'].update(db['age']).get()

And also copies the according scheme to the database:

.. jupyter-execute::

    db2.schemes
