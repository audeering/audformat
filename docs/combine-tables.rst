.. _combine-tables:

Combine tables
==============

It can happen that labels in your database are stored
in tables of different type as some labels belong to the whole file,
others don't. The following examples highlights this with the labels
for age and likability:

.. code-block:: python

    import audformat.testing

    db = audformat.testing.create_db(minimal=True)
    db.schemes["age"] = audformat.Scheme(
        audformat.define.DataType.INTEGER,
        minimum=20,
        maximum=50,
    )
    db.schemes["likability"] = audformat.Scheme(
        audformat.define.DataType.FLOAT,
    )
    audformat.testing.add_table(
        db,
        table_id="age",
        index_type=audformat.define.IndexType.FILEWISE,
        columns="age",
        num_files=3,
    )
    audformat.testing.add_table(
        db,
        table_id="likability",
        index_type=audformat.define.IndexType.SEGMENTED,
        columns="likability",
        num_files=4,
    )

Which results in the following two :class:`pandas.DataFrame`:

>>> db["age"].get()
               age
file
audio/001.wav   40
audio/002.wav   33
audio/003.wav   38

>>> db["likability"].get()
                                                                   likability
file          start                     end
audio/001.wav 0 days 00:00:00.063... 0 days 00:00:00.670...    0.310583
              0 days 00:00:02.176... 0 days 00:00:02.570...    0.047821
              0 days 00:00:03.254... 0 days 00:00:03.748...    0.288177
              0 days 00:00:03.959... 0 days 00:00:04.110...    0.848677
              0 days 00:00:04.541... 0 days 00:00:04.962...    0.787870
audio/002.wav 0 days 00:00:00.003... 0 days 00:00:00.037...    0.122311
              0 days 00:00:00.796... 0 days 00:00:01.075...    0.646814
              0 days 00:00:01.110... 0 days 00:00:01.439...    0.837227
              0 days 00:00:02.194... 0 days 00:00:03.675...    0.430492
              0 days 00:00:04.187... 0 days 00:00:04.738...    0.944455
audio/003.wav 0 days 00:00:00.574... 0 days 00:00:00.929...    0.422895
              0 days 00:00:01.450... 0 days 00:00:01.953...    0.559592
              0 days 00:00:02.808... 0 days 00:00:03.099...    0.586085
              0 days 00:00:03.225... 0 days 00:00:04.007...    0.701001
              0 days 00:00:04.015... 0 days 00:00:04.430...    0.492587
audio/004.wav 0 days 00:00:00.276... 0 days 00:00:00.625...    0.717525
              0 days 00:00:01.335... 0 days 00:00:01.920...    0.705011
              0 days 00:00:02.320... 0 days 00:00:02.419...    0.607638
              0 days 00:00:02.798... 0 days 00:00:02.892...    0.918130
              0 days 00:00:03.617... 0 days 00:00:03.943...    0.414640


You can simply combine both tables with:

>>> combined_table = db["likability"] + db["age"]
>>> combined_table.get()
                                                                   likability   age
file          start                     end
audio/001.wav 0 days 00:00:00.063... 0 days 00:00:00.670...    0.310583  <NA>
              0 days 00:00:02.176... 0 days 00:00:02.570...    0.047821  <NA>
              0 days 00:00:03.254... 0 days 00:00:03.748...    0.288177  <NA>
              0 days 00:00:03.959... 0 days 00:00:04.110...    0.848677  <NA>
              0 days 00:00:04.541... 0 days 00:00:04.962...    0.787870  <NA>
audio/002.wav 0 days 00:00:00.003... 0 days 00:00:00.037...    0.122311  <NA>
              0 days 00:00:00.796... 0 days 00:00:01.075...    0.646814  <NA>
              0 days 00:00:01.110... 0 days 00:00:01.439...    0.837227  <NA>
              0 days 00:00:02.194... 0 days 00:00:03.675...    0.430492  <NA>
              0 days 00:00:04.187... 0 days 00:00:04.738...    0.944455  <NA>
audio/003.wav 0 days 00:00:00.574... 0 days 00:00:00.929...    0.422895  <NA>
              0 days 00:00:01.450... 0 days 00:00:01.953...    0.559592  <NA>
              0 days 00:00:02.808... 0 days 00:00:03.099...    0.586085  <NA>
              0 days 00:00:03.225... 0 days 00:00:04.007...    0.701001  <NA>
              0 days 00:00:04.015... 0 days 00:00:04.430...    0.492587  <NA>
audio/004.wav 0 days 00:00:00.276... 0 days 00:00:00.625...    0.717525  <NA>
              0 days 00:00:01.335... 0 days 00:00:01.920...    0.705011  <NA>
              0 days 00:00:02.320... 0 days 00:00:02.419...    0.607638  <NA>
              0 days 00:00:02.798... 0 days 00:00:02.892...    0.918130  <NA>
              0 days 00:00:03.617... 0 days 00:00:03.943...    0.414640  <NA>
audio/001.wav 0 days 00:00:00           NaT                               NaN    40
audio/002.wav 0 days 00:00:00           NaT                               NaN    33
audio/003.wav 0 days 00:00:00           NaT                               NaN    38

Or, if you just want to have the likability information for all segments,
for which age information is available:

>>> df_likability = db["likability"].get(db["age"].files)
>>> df_likability
                                                                   likability
file          start                     end
audio/001.wav 0 days 00:00:00.063... 0 days 00:00:00.670...    0.310583
              0 days 00:00:02.176... 0 days 00:00:02.570...    0.047821
              0 days 00:00:03.254... 0 days 00:00:03.748...    0.288177
              0 days 00:00:03.959... 0 days 00:00:04.110...    0.848677
              0 days 00:00:04.541... 0 days 00:00:04.962...    0.787870
audio/002.wav 0 days 00:00:00.003... 0 days 00:00:00.037...    0.122311
              0 days 00:00:00.796... 0 days 00:00:01.075...    0.646814
              0 days 00:00:01.110... 0 days 00:00:01.439...    0.837227
              0 days 00:00:02.194... 0 days 00:00:03.675...    0.430492
              0 days 00:00:04.187... 0 days 00:00:04.738...    0.944455
audio/003.wav 0 days 00:00:00.574... 0 days 00:00:00.929...    0.422895
              0 days 00:00:01.450... 0 days 00:00:01.953...    0.559592
              0 days 00:00:02.808... 0 days 00:00:03.099...    0.586085
              0 days 00:00:03.225... 0 days 00:00:04.007...    0.701001
              0 days 00:00:04.015... 0 days 00:00:04.430...    0.492587

Or, if you want to have the age information for segments
in the likeability table:

>>> db["age"].get(df_likability.index)
                                                                   age
file          start                     end
audio/001.wav 0 days 00:00:00.063... 0 days 00:00:00.670...   40
              0 days 00:00:02.176... 0 days 00:00:02.570...   40
              0 days 00:00:03.254... 0 days 00:00:03.748...   40
              0 days 00:00:03.959... 0 days 00:00:04.110...   40
              0 days 00:00:04.541... 0 days 00:00:04.962...   40
audio/002.wav 0 days 00:00:00.003... 0 days 00:00:00.037...   33
              0 days 00:00:00.796... 0 days 00:00:01.075...   33
              0 days 00:00:01.110... 0 days 00:00:01.439...   33
              0 days 00:00:02.194... 0 days 00:00:03.675...   33
              0 days 00:00:04.187... 0 days 00:00:04.738...   33
audio/003.wav 0 days 00:00:00.574... 0 days 00:00:00.929...   38
              0 days 00:00:01.450... 0 days 00:00:01.953...   38
              0 days 00:00:02.808... 0 days 00:00:03.099...   38
              0 days 00:00:03.225... 0 days 00:00:04.007...   38
              0 days 00:00:04.015... 0 days 00:00:04.430...   38

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

>>> db2 = audformat.testing.create_db(minimal=True)
>>> db2.schemes["gender"] = audformat.Scheme(labels=["female", "male"])
>>> db2.schemes
gender:
  dtype: str
  labels: [female, male]

Next, we add a table and fill in some gender information:

>>> audformat.testing.add_table(
...     db2,
...     table_id="gender_and_age",
...     index_type=audformat.define.IndexType.FILEWISE,
...     columns="gender",
...     num_files=[2, 3, 4],
... ).get()
              gender
file
audio/002.wav   male
audio/003.wav   male
audio/004.wav   male

Now, we update the table with age values from the other database.

>>> db2["gender_and_age"].update(db["age"]).get()
              gender   age
file
audio/002.wav   male    33
audio/003.wav   male    38
audio/004.wav   male  <NA>
audio/001.wav    NaN    40

And also copies the according scheme to the database:

>>> db2.schemes
age:
  {dtype: int, minimum: 20, maximum: 50}
gender:
  dtype: str
  labels: [female, male]
