.. _update-a-database:

Update a database
=================

A feature that is especially useful for databases growing over time,
is the possibility to update a :class:`audformat.Database`.
For instance, consider the following database that contains
age labels for a hundred files.

.. code-block:: python

    import audformat
    import audformat.testing


    db = audformat.testing.create_db(minimal=True)
    db.schemes["age"] = audformat.Scheme(
        audformat.define.DataType.INTEGER,
        minimum=20,
        maximum=50,
    )
    audformat.testing.add_table(
        db,
        table_id="table",
        index_type=audformat.define.IndexType.FILEWISE,
        columns="age",
        num_files=100,
    )

>>> db
name: unittest
source: internal
usage: unrestricted
languages: [deu, eng]
schemes:
  age: {dtype: int, minimum: 20, maximum: 50}
tables:
  table:
    type: filewise
    columns:
      age: {scheme_id: age}
>>> db["table"].df
               age
file
audio/001.wav   44
audio/002.wav   36
audio/003.wav   24
audio/004.wav   44
audio/005.wav   37
...            ...
audio/096.wav   23
audio/097.wav   42
audio/098.wav   35
audio/099.wav   27
audio/100.wav   39
<BLANKLINE>
[100 rows x 1 columns]

Now assume we record more files to add to our original database.
The new files are stored together with annotations in a second database,
that is then added to the original database.

.. code-block:: python

    db_update = audformat.testing.create_db(minimal=True)
    db_update.schemes["age"] = db.schemes["age"]
    audformat.testing.add_table(
        db_update,
        table_id="table",
        index_type=audformat.define.IndexType.FILEWISE,
        columns="age",
        num_files=range(101, 105),
    )
    db.update(db_update)  # update original database with new data

>>> db["table"].df
               age
file
audio/001.wav   44
audio/002.wav   36
audio/003.wav   24
audio/004.wav   44
audio/005.wav   37
...            ...
audio/100.wav   39
audio/101.wav   46
audio/102.wav   43
audio/103.wav   21
audio/104.wav   45
<BLANKLINE>
[104 rows x 1 columns]

Or we find out that some files in the original database have wrong labels.
To update those, we again start from a fresh database containing only
the critical files, relabel them and then update the original database.

.. code-block:: python

    db_update = audformat.testing.create_db(minimal=True)
    db_update.schemes["age"] = db.schemes["age"]
    audformat.testing.add_table(
        db_update,
        table_id="table",
        index_type=audformat.define.IndexType.FILEWISE,
        columns="age",
        num_files=10,
    )
    db.update(db_update, overwrite=True)  # overwrite existing labels

>>> db["table"].df
               age
file
audio/001.wav   48
audio/002.wav   45
audio/003.wav   28
audio/004.wav   35
audio/005.wav   37
...            ...
audio/100.wav   39
audio/101.wav   46
audio/102.wav   43
audio/103.wav   21
audio/104.wav   45
<BLANKLINE>
[104 rows x 1 columns]

Finally, we want to add gender information to the database.
Again, it might be easier to start with a fresh database to
collect the new labels and only later merge it into our original database.

.. code-block:: python

    db_update = audformat.Database(
        name="update",
        languages=audformat.utils.map_language("french"),
    )
    db_update.schemes["gender"] = audformat.Scheme(
            labels=["female", "male"],
        )
    audformat.testing.add_table(
        db_update,
        table_id="table",
        index_type=audformat.define.IndexType.FILEWISE,
        columns="gender",
        num_files=len(db.files),
    )
    db.update(db_update)

>>> db["table"].df
               age  gender
file
audio/001.wav   48    male
audio/002.wav   45    male
audio/003.wav   28    male
audio/004.wav   35    male
audio/005.wav   37  female
...            ...     ...
audio/100.wav   39    male
audio/101.wav   46  female
audio/102.wav   43    male
audio/103.wav   21    male
audio/104.wav   45    male
<BLANKLINE>
[104 rows x 2 columns]

Note that this not only updates the table data,
but also adds the new gender scheme:

>>> db.schemes
age:
  {dtype: int, minimum: 20, maximum: 50}
gender:
  dtype: str
  labels: [female, male]
