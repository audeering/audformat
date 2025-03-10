Working with a database
=======================


Accessing data
--------------

Annotation labels can be accessed
by the :meth:`audformat.Table.get` method:

.. code-block:: python

    import audformat.testing
    import random


    random.seed(1)


    db = audformat.testing.create_db()
    table = db["files"].get()
    # Short for:
    # table = db.tables["files"].get()

Which returns the following :class:`pandas.DataFrame`:

>>> table.iloc[0:2, 0:2]
               bool                    date
file
audio/001.wav  <NA>                     NaT
audio/002.wav  True 1970-01-01 00:00:00.170


Or you can directly access a column with :meth:`audformat.Column.get()`:

.. code-block:: python

    column = db["files"]["string"].get()
    # Short for:
    # column = db.tables["files"].columns["string"].get()

Which results in the following :class:`pandas.Series`:

>>> column[0:2]
file
audio/001.wav    vv7xWJcOxL
audio/002.wav    eF5Of2ICvG
Name: string, dtype: string


For more information on how to access or add data
have a look at the code examples in the
:ref:`table specification <data-tables:Tables>`.


Changing referenced files
-------------------------

To convert to absolute file paths in all tables, do:

.. code-block:: python

    import os


    db.map_files(os.path.abspath)
