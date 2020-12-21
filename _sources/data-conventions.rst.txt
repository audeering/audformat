Conventions
===========

.. jupyter-execute::
    :hide-code:
    :hide-output:

    import audformat


Database name
-------------

The name of a database should be lowercase
and must not contain blanks or
special characters.
If you have different versions,
or very long names you can use ``-``
to increase readability.

.. jupyter-execute::

    audformat.Database(name='librispeech-mfa-cseg-pho')


Table and scheme names
----------------------

Use lower case for table and scheme names.

Here's a list of common scheme names:

=============================  =============================================
Name                           Content
=============================  =============================================
emotion                        emotion categories
arousal / valence / dominance  emotion dimensions
speaker                        unique speaker id
role                           the role a speaker has, e.g. agent vs. client
duration                       file duration
transcription                  transcriptions on word or phonetic level
                               accurate enough to be used for ASR
text                           transcriptions that are not accurate enough
                               for ASR
=============================  =============================================

In case you have several schemes of the same type,
append ``-xxx``.
E.g. use ``transcription-word``
and ``transcription-phoneme``
if a database offers word
*and* phoneme transcriptions.


Tables, schemes, and raters
---------------------------

Consider one table per scheme
with the name of the scheme.
Use lower case for table and scheme names.
If you have multiple raters,
name each column after the name of the rater.

.. jupyter-execute::

    db = audformat.Database('mydata')

    db.schemes['arousal'] = audformat.Scheme(audformat.define.DataType.FLOAT)
    db.schemes['valence'] = audformat.Scheme(audformat.define.DataType.FLOAT)

    db.raters['rater1'] = audformat.Rater()
    db.raters['rater2'] = audformat.Rater()

    for scheme_id in db.schemes:
        db[scheme_id] = audformat.Table(audformat.filewise_index())
        for rater_id in db.raters:
            db[scheme_id][rater_id] = audformat.Column(
                scheme_id=scheme_id,
                rater_id=rater_id,
            )

    db


Database splits
---------------

If an official split into training,
development
and test set
consists,
consider one table per split,
named ``scheme_id.split``.

.. jupyter-execute::

    db = audformat.Database('mydata')

    db.schemes['arousal'] = audformat.Scheme(audformat.define.DataType.FLOAT)

    db.splits['train'] = audformat.Split(audformat.define.SplitType.TRAIN)
    db.splits['dev'] = audformat.Split(audformat.define.SplitType.DEVELOP)
    db.splits['test'] = audformat.Split(audformat.define.SplitType.TEST)

    for scheme_id in db.schemes:
        for split_id in db.splits:
            table_id = f'{scheme_id}.{split_id}'
            db[table_id] = audformat.Table(
                index=audformat.filewise_index(),
                split_id=split_id,
            )

    db
        

Gold standard annotation
------------------------

Annotations by several raters
belonging to the same scheme
should be stored in a single table,
but **not** aggregated,
e.g. by adding a column with mean or some other metric.
Instead a new table with the postfix ``.gold_standard``
should be created
to store the average of all rater.
In addition,
a rater with the id ``'gold_standard'``
and the type ``audformat.define.RaterType.VOTE``
should be created
and associated with the column
holding the gold standard values.

.. jupyter-execute::

    db = audformat.Database('mydata')

    db.schemes['arousal'] = audformat.Scheme(audformat.define.DataType.FLOAT)

    db.raters['rater1'] = audformat.Rater()
    db.raters['rater2'] = audformat.Rater()
    db.raters['gold_standard'] = audformat.Rater(audformat.define.RaterType.VOTE)

    for scheme_id in db.schemes:
        db[scheme_id] = audformat.Table(audformat.filewise_index())
        for rater_id in ['rater1', 'rater2']:
            db[scheme_id][rater_id] = audformat.Column(
                scheme_id=scheme_id,
                rater_id=rater_id,
            )
        gold_id = f'{scheme_id}.gold_standard'
        db[gold_id] = audformat.Table(audformat.filewise_index())
        db[gold_id][scheme_id] = audformat.Column(
            scheme_id=scheme_id,
            rater_id='gold_standard',
        )

    db


Confidence values
-----------------

Assume you have an annotation
that does not only provide a value,
but also a confidence of that value.
In this case you create
two schemes,
one for the value,
and one for the confidence
using the same scheme ID,
but followed by ``.confidence``.

The confidence values should be stored in a separate table.
Or it can be stored within the same table as a different column,
which might be worth considering when storing the gold standard.

.. jupyter-execute::

    db = audformat.Database('mydata')

    db.schemes['arousal'] = audformat.Scheme(audformat.define.DataType.FLOAT)
    db.schemes['arousal.confidence'] = audformat.Scheme(
        audformat.define.DataType.FLOAT,
        minimum=0,
        maximum=1,
    )

    db.raters['gold_standard'] = audformat.Rater(audformat.define.RaterType.VOTE)

    db['arousal'] = audformat.Table(audformat.filewise_index())
    for scheme_id in db.schemes:
        db['arousal'][scheme_id] = audformat.Column(
            scheme_id=scheme_id,
            rater_id='gold_standard',
        )

    db


File and speaker information
----------------------------

Meta information like speaker ID
or file duration
should be collected in a table ``files``.
If you have metadata
belonging only to segments,
collect it in a table ``segments``.

Additional meta information,
that is bound to another information
like age of speaker,
should be collected in the header
as it can be later automatically mapped.

.. jupyter-execute::

    db = audformat.Database('mydata')

    M = audformat.define.Gender.MALE
    F = audformat.define.Gender.FEMALE
    speaker = {
        'speaker1': {'gender': F, 'age': 31},
        'speaker2': {'gender': M, 'age': 85},
    }

    db.schemes['speaker'] = audformat.Scheme(labels=speaker)
    db['files'] = audformat.Table(
        index=audformat.filewise_index(['a.wav', 'b.wav'])
    )
    db['files']['speaker'] = audformat.Column(scheme_id='speaker')
    db['files']['speaker'].set(['speaker1', 'speaker2'])

    db


.. jupyter-execute::

    db['files'].get()

You can access the additional information with the ``map`` argument
of :meth:`audformat.Table.get`,
see :ref:`map-scheme-labels`
for an extended documentation.

.. jupyter-execute::

    db['files'].get(map={'speaker': 'gender'})


File duration and temporal data
-------------------------------

It is recommended to store file durations
for every database
in a table ``files``.
This information is in principle redundant
as you can calculate the duration always on the fly,
but if you have thousands of files
this might take some time.

Every temporal data
like file durations
should be stored as :class:`pandas.Timedelta`
or :class:`datetime.datetime`.

.. jupyter-execute::
    :hide-code:
    :hide-output:

    import audiofile as af
    import numpy as np


    signal = np.ones([0, 1000])


.. jupyter-execute::

    import audeer
    import audiofile as af
    import numpy as np
    import pandas as pd


    # Create dummy WAV files
    sampling_rate = 1000
    af.write('a.wav', np.ones([1, 1000]), sampling_rate)
    af.write('b.wav', np.ones([1, 500]), sampling_rate)

    db = audformat.Database('mydata')

    db.schemes['duration'] = audformat.Scheme(dtype=audformat.define.DataType.TIME)
    db['files'] = audformat.Table(
        index=audformat.filewise_index(['a.wav', 'b.wav'])
    )
    db['files']['duration'] = audformat.Column(scheme_id='duration')
    durations = audeer.run_tasks(
        task_func=lambda x: pd.to_timedelta(af.duration(x), unit='s'),
        params=[([f], {}) for f in db.files],
        num_workers=12,
        progress_bar=False,
    )
    db['files']['duration'].set(durations)

    db

.. jupyter-execute::

    db['files'].get()


.. Clean up
.. jupyter-execute::
    :hide-code:
    :hide-output:

    import os

    os.remove('a.wav')
    os.remove('b.wav')
