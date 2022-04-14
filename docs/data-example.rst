Example
=======

Header (YAML):

.. jupyter-execute::

    import audformat.testing

    db = audformat.testing.create_db()
    db

Filewise table as :class:`pd.DataFrame`:

.. jupyter-execute::

    db['files'].get()

and as CSV:

.. jupyter-execute::
    :hide-code:

    df = db['files'].get()
    # Since pandas 1.4.0 DataFrame.to_csv()
    # no longer works for categories with dtype Int64
    # we have to convert column to plain Int64
    df['label_map_int'] = df['label_map_int'].astype('Int64')
    print(df.to_csv())

Segmented table as :class:`pd.DataFrame`:

.. jupyter-execute::

    db['segments'].get()

and as CSV:

.. jupyter-execute::
    :hide-code:

    df = db['segments'].get()
    # Since pandas 1.4.0 DataFrame.to_csv()
    # no longer works for categories with dtype Int64
    # we have to convert column to plain Int64
    df['label_map_int'] = df['label_map_int'].astype('Int64')
    print(df.to_csv())
