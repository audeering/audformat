.. _map-scheme-labels:

Map scheme labels
=================

.. Enforce HTML output for pd.Series
.. jupyter-execute::
    :hide-code:
    :hide-output:

    import audformat


    audformat.core.common.format_series_as_html()


The ``labels`` attribute of schemes can be used to
encode additional information about the table data.
In the following example we have a scheme
``'transcription'`` that maps IDs to words.
And a scheme ``'speaker'`` that holds gender and age
information about the speakers in the database.

.. jupyter-execute::
    :hide-output:

    import audformat.testing

    db = audformat.testing.create_db(minimal=True)
    db.schemes['transcription'] = audformat.Scheme(
        labels={
            0: 'hello',
            1: 'goodbye',
        }
    )
    db.schemes['speaker'] = audformat.Scheme(
        labels={
            'spk1': {
                'gender': 'male',
                'age': 33,
            },
            'spk2': {
                'gender': 'female',
                'age': 30,
            },
            'spk3': {
                'gender': 'male',
                'age': 37,
            },
        }
    )
    audformat.testing.add_table(
        db,
        'files',
        audformat.define.IndexType.FILEWISE,
    )

If we request the ``transcription`` column,
we get a :class:`pandas.Series` with the word IDs:

.. jupyter-execute::

    db['files']['transcription'].get()

But if we are interested in the actual transcribed words,
we can use the ``map`` argument to request them.

.. jupyter-execute::

    db['files']['transcription'].get(map='transcription')

Note that we can pass any string to ``map``.
It will be used as the name of
the returned :class:`pandas.Series`.

.. jupyter-execute::

    db['files']['transcription'].get(map='word')

Likewise, if we request the speaker column,
a list of names is returned:

.. jupyter-execute::

    db['files']['speaker'].get()

If we are interested in the the age of the speakers, we can do:

.. jupyter-execute::

    db['files']['speaker'].get(map='age')

This also works for tables.
Here we pass a dictionary with column names
as keys and scheme fields as values.

.. jupyter-execute::

    map = {
        'speaker': 'age',
    }
    db['files'].get(map=map)

It is possible to map several columns at once
and to map the same column to multiple fields.

.. jupyter-execute::

    map = {
        'transcription': 'words',
        'speaker': ['age', 'gender'],
    }
    db['files'].get(map=map)

To keep the original columns values,
we can include the column name in the list.

.. jupyter-execute::

    map = {
        'transcription': ['transcription', 'words'],
        'speaker': ['speaker', 'age', 'gender'],
    }
    db['files'].get(map=map)
