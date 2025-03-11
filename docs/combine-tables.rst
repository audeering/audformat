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
audio/001.wav 0 days 00:00:00.063022023 0 days 00:00:00.670806417    0.310583
              0 days 00:00:02.176912810 0 days 00:00:02.570855787    0.047821
              0 days 00:00:03.254253920 0 days 00:00:03.748281529    0.288177
              0 days 00:00:03.959315192 0 days 00:00:04.110064420    0.848677
              0 days 00:00:04.541308146 0 days 00:00:04.962982764    0.787870
audio/002.wav 0 days 00:00:00.003443562 0 days 00:00:00.037705840    0.122311
              0 days 00:00:00.796582868 0 days 00:00:01.075538128    0.646814
              0 days 00:00:01.110602019 0 days 00:00:01.439743769    0.837227
              0 days 00:00:02.194859538 0 days 00:00:03.675142916    0.430492
              0 days 00:00:04.187563347 0 days 00:00:04.738401164    0.944455
audio/003.wav 0 days 00:00:00.574500931 0 days 00:00:00.929483764    0.422895
              0 days 00:00:01.450303730 0 days 00:00:01.953211769    0.559592
              0 days 00:00:02.808842477 0 days 00:00:03.099035455    0.586085
              0 days 00:00:03.225926686 0 days 00:00:04.007331084    0.701001
              0 days 00:00:04.015322811 0 days 00:00:04.430922264    0.492587
audio/004.wav 0 days 00:00:00.276361409 0 days 00:00:00.625730112    0.717525
              0 days 00:00:01.335888842 0 days 00:00:01.920178620    0.705011
              0 days 00:00:02.320023510 0 days 00:00:02.419307606    0.607638
              0 days 00:00:02.798096347 0 days 00:00:02.892323558    0.918130
              0 days 00:00:03.617486299 0 days 00:00:03.943024074    0.414640


You can simply combine both tables with:

>>> combined_table = db["likability"] + db["age"]
>>> combined_table.get()
                                                                   likability   age
file          start                     end
audio/001.wav 0 days 00:00:00.063022023 0 days 00:00:00.670806417    0.310583  <NA>
              0 days 00:00:02.176912810 0 days 00:00:02.570855787    0.047821  <NA>
              0 days 00:00:03.254253920 0 days 00:00:03.748281529    0.288177  <NA>
              0 days 00:00:03.959315192 0 days 00:00:04.110064420    0.848677  <NA>
              0 days 00:00:04.541308146 0 days 00:00:04.962982764    0.787870  <NA>
audio/002.wav 0 days 00:00:00.003443562 0 days 00:00:00.037705840    0.122311  <NA>
              0 days 00:00:00.796582868 0 days 00:00:01.075538128    0.646814  <NA>
              0 days 00:00:01.110602019 0 days 00:00:01.439743769    0.837227  <NA>
              0 days 00:00:02.194859538 0 days 00:00:03.675142916    0.430492  <NA>
              0 days 00:00:04.187563347 0 days 00:00:04.738401164    0.944455  <NA>
audio/003.wav 0 days 00:00:00.574500931 0 days 00:00:00.929483764    0.422895  <NA>
              0 days 00:00:01.450303730 0 days 00:00:01.953211769    0.559592  <NA>
              0 days 00:00:02.808842477 0 days 00:00:03.099035455    0.586085  <NA>
              0 days 00:00:03.225926686 0 days 00:00:04.007331084    0.701001  <NA>
              0 days 00:00:04.015322811 0 days 00:00:04.430922264    0.492587  <NA>
audio/004.wav 0 days 00:00:00.276361409 0 days 00:00:00.625730112    0.717525  <NA>
              0 days 00:00:01.335888842 0 days 00:00:01.920178620    0.705011  <NA>
              0 days 00:00:02.320023510 0 days 00:00:02.419307606    0.607638  <NA>
              0 days 00:00:02.798096347 0 days 00:00:02.892323558    0.918130  <NA>
              0 days 00:00:03.617486299 0 days 00:00:03.943024074    0.414640  <NA>
audio/001.wav 0 days 00:00:00           NaT                               NaN    40
audio/002.wav 0 days 00:00:00           NaT                               NaN    33
audio/003.wav 0 days 00:00:00           NaT                               NaN    38

Or, if you just want to have the likability information for all segments,
for which age information is available:

>>> df_likability = db["likability"].get(db["age"].files)
>>> df_likability
                                                                   likability
file          start                     end
audio/001.wav 0 days 00:00:00.063022023 0 days 00:00:00.670806417    0.310583
              0 days 00:00:02.176912810 0 days 00:00:02.570855787    0.047821
              0 days 00:00:03.254253920 0 days 00:00:03.748281529    0.288177
              0 days 00:00:03.959315192 0 days 00:00:04.110064420    0.848677
              0 days 00:00:04.541308146 0 days 00:00:04.962982764    0.787870
audio/002.wav 0 days 00:00:00.003443562 0 days 00:00:00.037705840    0.122311
              0 days 00:00:00.796582868 0 days 00:00:01.075538128    0.646814
              0 days 00:00:01.110602019 0 days 00:00:01.439743769    0.837227
              0 days 00:00:02.194859538 0 days 00:00:03.675142916    0.430492
              0 days 00:00:04.187563347 0 days 00:00:04.738401164    0.944455
audio/003.wav 0 days 00:00:00.574500931 0 days 00:00:00.929483764    0.422895
              0 days 00:00:01.450303730 0 days 00:00:01.953211769    0.559592
              0 days 00:00:02.808842477 0 days 00:00:03.099035455    0.586085
              0 days 00:00:03.225926686 0 days 00:00:04.007331084    0.701001
              0 days 00:00:04.015322811 0 days 00:00:04.430922264    0.492587

Or, if you want to have the age information for segments
in the likeability table:

>>> db["age"].get(df_likability.index)
                                                                   age
file          start                     end
audio/001.wav 0 days 00:00:00.063022023 0 days 00:00:00.670806417   40
              0 days 00:00:02.176912810 0 days 00:00:02.570855787   40
              0 days 00:00:03.254253920 0 days 00:00:03.748281529   40
              0 days 00:00:03.959315192 0 days 00:00:04.110064420   40
              0 days 00:00:04.541308146 0 days 00:00:04.962982764   40
audio/002.wav 0 days 00:00:00.003443562 0 days 00:00:00.037705840   33
              0 days 00:00:00.796582868 0 days 00:00:01.075538128   33
              0 days 00:00:01.110602019 0 days 00:00:01.439743769   33
              0 days 00:00:02.194859538 0 days 00:00:03.675142916   33
              0 days 00:00:04.187563347 0 days 00:00:04.738401164   33
audio/003.wav 0 days 00:00:00.574500931 0 days 00:00:00.929483764   38
              0 days 00:00:01.450303730 0 days 00:00:01.953211769   38
              0 days 00:00:02.808842477 0 days 00:00:03.099035455   38
              0 days 00:00:03.225926686 0 days 00:00:04.007331084   38
              0 days 00:00:04.015322811 0 days 00:00:04.430922264   38

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
