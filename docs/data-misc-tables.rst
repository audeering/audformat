Misc Tables
===========

A miscellaneous (misc) table links labels to an index.
The index has no restrictions
and can contain an arbitrary number of columns (called levels),
but should not contain duplicated entries.
In the CSV file of the misc table,
the index columns are stored in front of the label columns.
An empty label means that no label has been assigned (yet).


audformat implementation
------------------------

Create an index with the levels ``"file"`` and ``"other"``:

.. code-block:: python

    import audformat
    import audformat.testing
    import pandas as pd


    index = pd.MultiIndex.from_tuples(
        [
            ("f1", "f2"),
            ("f1", "f3"),
            ("f2", "f3"),
        ],
        names=["file", "other"],
    )

>>> index
MultiIndex([('f1', 'f2'),
            ('f1', 'f3'),
            ('f2', 'f3')],
           names=['file', 'other'])

Create database and add misc table with the index:

>>> db = audformat.testing.create_db(minimal=True)
>>> db["misc"] = audformat.MiscTable(index)
>>> db["misc"]["values"] = audformat.Column()
>>> db.misc_tables["misc"]
levels: {file: object, other: object}
columns:
  values: {}

Assign labels to a table:

>>> values_list = [0, 1, 0]
>>> values_dict = {"values": values_list}
>>> db["misc"].set(values_dict)
>>> db["misc"].get()
           values
file other
f1   f2         0
     f3         1
f2   f3         0

Assign labels to a column:

>>> db["misc"]["values"].set(values_list)
>>> db["misc"]["values"].get()
file  other
f1    f2       0
      f3       1
f2    f3       0
Name: values, dtype: object

Access labels from a misc table with an index:

>>> db["misc"].get(index[:2])
           values
file other
f1   f2         0
     f3         1
