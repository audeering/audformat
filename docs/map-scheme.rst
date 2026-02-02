.. _map-scheme-labels:

Map scheme labels
=================

The ``labels`` attribute of schemes can be used to
encode additional information about the table data.
In the following example we have a scheme
``"transcription"`` that maps IDs to words.
And a scheme ``"speaker"`` that holds gender and age
information about the speakers in the database.

.. code-block:: python

    import audformat.testing

    db = audformat.testing.create_db(minimal=True)
    db.schemes["transcription"] = audformat.Scheme(
        labels={
            0: "hello",
            1: "goodbye",
        }
    )
    db.schemes["speaker"] = audformat.Scheme(
        labels={
            "spk1": {
                "gender": "male",
                "age": 33,
            },
            "spk2": {
                "gender": "female",
                "age": 30,
            },
            "spk3": {
                "gender": "male",
                "age": 37,
            },
        }
    )
    audformat.testing.add_table(
        db,
        "files",
        audformat.define.IndexType.FILEWISE,
    )

If we request the ``transcription`` column,
we get a :class:`pandas.Series` with the word IDs:

>>> db["files"]["transcription"].get()
file
audio/001.wav    0
audio/002.wav    1
audio/003.wav    1                                                                                                                                                        audio/004.wav    0
audio/005.wav    0
Name: transcription, dtype: category
Categories (2, int64): [0, 1]

But if we are interested in the actual transcribed words,
we can use the ``map`` argument to request them.

>>> db["files"]["transcription"].get(map="transcription")
file
audio/001.wav      hello
audio/002.wav    goodbye
audio/003.wav    goodbye
audio/004.wav      hello
audio/005.wav      hello
Name: transcription, dtype: string

Note that we can pass any string to ``map``.
It will be used as the name of
the returned :class:`pandas.Series`.

>>> db["files"]["transcription"].get(map="word")
file
audio/001.wav      hello
audio/002.wav    goodbye
audio/003.wav    goodbye
audio/004.wav      hello
audio/005.wav      hello
Name: word, dtype: string

Likewise, if we request the speaker column,
a list of names is returned:

>>> db["files"]["speaker"].get()
file
audio/001.wav    spk2
audio/002.wav    spk1
audio/003.wav    spk1
audio/004.wav    spk1
audio/005.wav    spk3
Name: speaker, dtype: category
Categories (3, object): ['spk1', 'spk2', 'spk3']

If we are interested in the age of the speakers, we can do:

>>> db["files"]["speaker"].get(map="age")
file
audio/001.wav    30
audio/002.wav    33
audio/003.wav    33
audio/004.wav    33
audio/005.wav    37
Name: age, dtype: Int64

This also works for tables.
Here we pass a dictionary with column names
as keys and scheme fields as values.

>>> map = {"speaker": "age"}
>>> db["files"].get(map=map)
              transcription  age
file
audio/001.wav             0   30
audio/002.wav             1   33
audio/003.wav             1   33
audio/004.wav             0   33
audio/005.wav             0   37

It is possible to map several columns at once
and to map the same column to multiple fields.

>>> map = {"transcription": "words", "speaker": ["age", "gender"]}
>>> db["files"].get(map=map)
                 words  age  gender
file
audio/001.wav    hello   30  female
audio/002.wav  goodbye   33    male
audio/003.wav  goodbye   33    male
audio/004.wav    hello   33    male
audio/005.wav    hello   37    male

To keep the original columns values,
we can include the column name in the list.

>>> map = {
...     "transcription": ["transcription", "words"],
...     "speaker": ["speaker", "age", "gender"],
... }
>>> db["files"].get(map=map)
              speaker transcription    words  age  gender
file
audio/001.wav    spk2             0    hello   30  female
audio/002.wav    spk1             1  goodbye   33    male
audio/003.wav    spk1             1  goodbye   33    male
audio/004.wav    spk1             0    hello   33    male
audio/005.wav    spk3             0    hello   37    male
