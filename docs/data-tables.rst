Tables
======

A table links labels to media files.
It consists of one or three index columns
followed by an arbitrary number of label columns.
Labels can either refer to whole files or part of files.
An empty label means that no label has been assigned (yet).

There are two types of tables:

* **Filewise**: labels refer to whole files
* **Segmented**: labels refer to specific parts of files (segments)

Each type comes with a characteristic index.


Filewise
--------

==============  ====================================================
Index columns   Description
==============  ====================================================
file            Path to media file
==============  ====================================================

audformat implementation
^^^^^^^^^^^^^^^^^^^^^^^^

Create a filewise index:

>>> import audformat
>>> filewise_index = audformat.filewise_index(["f1", "f2", "f3"])
>>> filewise_index
Index(['f1', 'f2', 'f3'], dtype='string', name='file')

Create database and add table with a filewise index:

>>> db = audformat.Database("mydb")
>>> db["filewise"] = audformat.Table(filewise_index)
>>> db["filewise"]["values"] = audformat.Column()
>>> db.tables["filewise"]
type: filewise
columns:
  values: {}

Assign labels to a table:

>>> values_list = [1, 2, 3]
>>> values_dict = {"values": values_list}
>>> db["filewise"].set(values_dict)

Access labels as :class:`pandas.DataFrame`:

>>> db["filewise"].get()
     values
file
f1        1
f2        2
f3        3

Assign labels to a column:

>>> db["filewise"]["values"].set(values_list)

Access labels as :class:`pandas.Series`

>>> db["filewise"]["values"].get()
file
f1    1
f2    2
f3    3
Name: values, dtype: object

Access labels and convert index to a segmented index:

>>> db["filewise"]["values"].get(as_segmented=True)
file  start   end
f1    0 days  NaT    1
f2    0 days  NaT    2
f3    0 days  NaT    3
Name: values, dtype: object

Access labels from a filewise table with a segmented index:

>>> segmented_index = audformat.segmented_index(
...     files=["f1", "f1", "f1", "f2"],
...     starts=["0s", "1s", "2s", "0s"],
...     ends=["1s", "2s", "3s", None],
... )
>>> db["filewise"].get(segmented_index)
                                     values
file start           end
f1   0 days 00:00:00 0 days 00:00:01      1
     0 days 00:00:01 0 days 00:00:02      1
     0 days 00:00:02 0 days 00:00:03      1
f2   0 days 00:00:00 NaT                  2

Access labels from a filewise column with a segmented index:

>>> db["filewise"]["values"].get(segmented_index)
file  start            end
f1   0 days 00:00:00 0 days 00:00:01      1
     0 days 00:00:01 0 days 00:00:02      1
     0 days 00:00:02 0 days 00:00:03      1
f2   0 days 00:00:00 NaT                  2
Name: values, dtype: object

Segmented
---------

==============  ====================================================
Index columns   Description
==============  ====================================================
file            Path to media file
start           Start time of the segment
                (relative to the beginning of the file)
end             End time of the segment
                (relative to the beginning of the file)
==============  ====================================================

audformat implementation
^^^^^^^^^^^^^^^^^^^^^^^^

Create a segmented index:

>>> segmented_index = audformat.segmented_index(
...     files=["f1", "f1", "f1", "f2", "f3"],
...     starts=["0s", "1s", "2s", "0s", "1m"],
...     ends=["1s", "2s", "3s", None, "1h"],
... )
>>> segmented_index
MultiIndex([('f1', '0 days 00:00:00', '0 days 00:00:01'),
            ('f1', '0 days 00:00:01', '0 days 00:00:02'),
            ('f1', '0 days 00:00:02', '0 days 00:00:03'),
            ('f2', '0 days 00:00:00',               NaT),
            ('f3', '0 days 00:01:00', '0 days 01:00:00')],
           names=['file', 'start', 'end'])

Add table with a segmented index:

>>> db["segmented"] = audformat.Table(segmented_index)
>>> db["segmented"]["values"] = audformat.Column()
>>> db.tables["segmented"]
type: segmented
columns:
  values: {}

Assign labels to the whole table:

>>> values_list = [1, 2, 3, 4, 5]
>>> values_dict = {"values": values_list}
>>> db["segmented"].set(values_dict)

Access all labels as :class:`pandas.DataFrame`:

>>> db["segmented"].get()
                                     values
file start           end
f1   0 days 00:00:00 0 days 00:00:01      1
     0 days 00:00:01 0 days 00:00:02      2
     0 days 00:00:02 0 days 00:00:03      3
f2   0 days 00:00:00 NaT                  4
f3   0 days 00:01:00 0 days 01:00:00      5

Assign labels to a column:

>>> db["segmented"]["values"].set(values_list)

Access labels from a column as :class:`pandas.Series`:

>>> db["segmented"]["values"].get()
file  start            end
f1    0 days 00:00:00  0 days 00:00:01    1
      0 days 00:00:01  0 days 00:00:02    2
      0 days 00:00:02  0 days 00:00:03    3
f2    0 days 00:00:00  NaT                4
f3    0 days 00:01:00  0 days 01:00:00    5
Name: values, dtype: object

Access labels from a segmented table with a filewise index:

>>> filewise_index = audformat.filewise_index(["f1", "f2"])
>>> db["segmented"].get(filewise_index)
                                     values
file start           end
f1   0 days 00:00:00 0 days 00:00:01      1
     0 days 00:00:01 0 days 00:00:02      2
     0 days 00:00:02 0 days 00:00:03      3
f2   0 days 00:00:00 NaT                  4

Access labels from a segmented column with a filewise index:

>>> db["segmented"]["values"].get(filewise_index)
file  start            end
f1    0 days 00:00:00  0 days 00:00:01    1
      0 days 00:00:01  0 days 00:00:02    2
      0 days 00:00:02  0 days 00:00:03    3
f2    0 days 00:00:00  NaT                4
Name: values, dtype: object
