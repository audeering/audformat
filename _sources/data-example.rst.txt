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

    print(db['files'].get().to_csv())

Segmented table as :class:`pd.DataFrame`:

.. jupyter-execute::

    db['segments'].get()

and as CSV:

.. jupyter-execute::
    :hide-code:

    print(db['segments'].get().to_csv())
