Working with a database
=======================


.. Enforce HTML output for pd.Series
.. jupyter-execute::
    :hide-code:
    :hide-output:

    import audformat


    audformat.core.common.format_series_as_html()


Accessing data
--------------

Annotation labels can be accessed
by the :meth:`audformat.Table.get` method:

.. jupyter-execute::

    import audformat.testing


    db = audformat.testing.create_db()
    table = db['files'].get()
    # Short for:
    # table = db.tables['files'].get()

Which returns the following :class:`pandas.DataFrame`:

.. jupyter-execute::

    table.iloc[0:2, 0:2]

Or you can directly access a column with :meth:`audformat.Column.get()`:

.. jupyter-execute::

    column = db['files']['string'].get()
    # Short for:
    # column = db.tables['files'].columns['string'].get()

Which results in the following :class:`pandas.Series`:

.. jupyter-execute::

    column[0:2]

For more information on how to access or add data
have a look at the code examples in the
:ref:`table specification <data-tables:Tables>`.


Changing referenced files
-------------------------

To convert to absolute file paths in all tables, do:

.. jupyter-execute::

    import os


    db.map_files(os.path.abspath)
