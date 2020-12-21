Tables
======

.. Enforce HTML output for pd.Series
.. jupyter-execute::
    :hide-code:
    :hide-output:

    import audformat


    audformat.core.common.format_series_as_html()


A table links labels to media files.
It consists of one or more index columns
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

.. jupyter-execute::

    import audformat
    import audformat.testing


    filewise_index = audformat.filewise_index(
        ['f1', 'f2', 'f3'],
    )
    filewise_index

Create database and add table with a filewise index:

.. jupyter-execute::

    db = audformat.testing.create_db(minimal=True)
    db['filewise'] = audformat.Table(filewise_index)
    db['filewise']['values'] = audformat.Column()
    db.tables['filewise']

Assign labels to a table:

.. jupyter-execute::

    values_list = [1, 2, 3]
    values_dict = {'values': values_list}
    db['filewise'].set(values_dict)

Access labels as :class:`pandas.DataFrame`:

.. jupyter-execute::

    db['filewise'].get()

Assign labels to a column:

.. jupyter-execute::

    db['filewise']['values'].set(values_list)

Access labels as :class:`pandas.Series`

.. jupyter-execute::

    db['filewise']['values'].get()

Create a segmented index:

.. jupyter-execute::

    import pandas as pd


    segmented_index = audformat.segmented_index(
        files=['f1', 'f1', 'f1', 'f2'],
        starts=['0s', '1s', '2s', '0s'],
        ends=['1s', '2s', '3s', pd.NaT],
    )
    segmented_index

Access labels from a filewise table with a segmented index:

.. jupyter-execute::

    db['filewise'].get(segmented_index)

Access labels from a filewise column with a segmented index:

.. jupyter-execute::

    db['filewise']['values'].get(segmented_index)
    

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

.. jupyter-execute::

    segmented_index = audformat.segmented_index(
        files=['f1', 'f1', 'f1', 'f2', 'f3'],
        starts=['0s', '1s', '2s', '0s', '1m'],
        ends=['1s', '2s', '3s', pd.NaT, '1h'],
    )
    segmented_index

Add table with a segmented index:

.. jupyter-execute::

    db['segmented'] = audformat.Table(segmented_index)
    db['segmented']['values'] = audformat.Column()
    db.tables['segmented']

Assign labels to the whole table:

.. jupyter-execute::

    values_list = [1, 2, 3, 4, 5]
    values_dict = {'values': values_list}
    db['segmented'].set(
        values_dict,
    )

Access all labels as :class:`pandas.DataFrame`:

.. jupyter-execute::

    db['segmented'].get()

Assign labels to a column:

.. jupyter-execute::

    db['segmented']['values'].set(values_list)

Access labels from a column as :class:`pandas.Series`:

.. jupyter-execute::

    db['segmented']['values'].get()

Create a filewise index:

.. jupyter-execute::

    filewise_index = audformat.filewise_index(
        ['f1', 'f2'],
    )
    filewise_index

Access labels from a segmented table with a filewise index:

.. jupyter-execute::

    db['segmented'].get(filewise_index)

Access labels from a segmented column with a filewise index:

.. jupyter-execute::

    db['segmented']['values'].get(filewise_index)
