Misc Tables
===========

.. Enforce HTML output for pd.Series
.. jupyter-execute::
    :hide-code:
    :hide-output:

    import audformat


    audformat.core.common.format_series_as_html()


A misc table or miscellaneous table links labels to an index.
The index has no restrictions,
but should not contain duplicated entries.
The arbitrary number of index columns of a misc table
is followed by an arbitrary number of label columns.
An empty label means that no label has been assigned (yet).


audformat implementation
------------------------

Create an index with the levels `file` and `other`:

.. jupyter-execute::

    import audformat
    import audformat.testing
    import pandas as pd


    index = pd.MultiIndex.from_tuples(
        [
            ('f1', 'f2'),
            ('f1', 'f3'),
            ('f2', 'f3'),
        ],
        names=['file', 'other'],
    )
    index

Create database and add misc table with the index:

.. jupyter-execute::

    db = audformat.testing.create_db(minimal=True)
    db['misc'] = audformat.MiscTable(index)
    db['misc']['values'] = audformat.Column()
    db.misc_tables['misc']

Assign labels to a table:

.. jupyter-execute::

    values_list = [0, 1, 0]
    values_dict = {'values': values_list}
    db['misc'].set(values_dict)

Access labels as :class:`pandas.DataFrame`:

.. jupyter-execute::

    db['misc'].get()

Assign labels to a column:

.. jupyter-execute::

    db['misc']['values'].set(values_list)

Access labels as :class:`pandas.Series`

.. jupyter-execute::

    db['misc']['values'].get()

Access labels from a misc table with an index:

.. jupyter-execute::

    db['misc'].get(index[:2])
