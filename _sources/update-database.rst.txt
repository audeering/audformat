Update a database
=================

A feature that is especially useful for databases growing over time,
is the possibility to update a :class:`audformat.Database`.
For instance, consider the following database that contains
age labels for a hundred files.

.. jupyter-execute::

    import audformat
    import audformat.testing


    db = audformat.testing.create_db(minimal=True)
    db.schemes['age'] = audformat.Scheme(
        audformat.define.DataType.INTEGER,
        minimum=20,
        maximum=50,
    )
    audformat.testing.add_table(
        db,
        table_id='table',
        index_type=audformat.define.IndexType.FILEWISE,
        columns='age',
        num_files=100,
    )
    db

.. jupyter-execute::

    db['table'].df

Now assume we record more files to add to our original database.
The new files are stored together with annotations in a second database,
that is then added to the original database.

.. jupyter-execute::

    db_update = audformat.testing.create_db(minimal=True)
    db_update.schemes['age'] = db.schemes['age']
    audformat.testing.add_table(
        db_update,
        table_id='table',
        index_type=audformat.define.IndexType.FILEWISE,
        columns='age',
        num_files=range(101, 105),
    )
    db.update(db_update)  # update original database with new data
    db['table'].df

Or we find out that some files in the original database have wrong labels.
To update those, we again start from a fresh database containing only
the critical files, relabel them and then update the original database.

.. jupyter-execute::

    db_update = audformat.testing.create_db(minimal=True)
    db_update.schemes['age'] = db.schemes['age']
    audformat.testing.add_table(
        db_update,
        table_id='table',
        index_type=audformat.define.IndexType.FILEWISE,
        columns='age',
        num_files=10,
    )
    db.update(db_update, overwrite=True)  # overwrite existing labels
    db['table'].df

Finally, we want to add gender information to the database.
Again, it might be easier to start with a fresh database to
collect the new labels and only later merge it into our original database.

.. jupyter-execute::

    db_update = audformat.Database(
        name='update',
        languages=audformat.utils.map_language('french'),
    )
    db_update.schemes['gender'] = audformat.Scheme(
            labels=['female', 'male'],
        )
    audformat.testing.add_table(
        db_update,
        table_id='table',
        index_type=audformat.define.IndexType.FILEWISE,
        columns='gender',
        num_files=len(db.files),
    )
    db.update(db_update)
    db['table'].df

Note that this not only updates the table data,
but also adds the new gender scheme:

.. jupyter-execute::

    db.schemes
