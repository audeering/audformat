Conventions
===========

Database name
-------------

The name of a database should be lowercase
and must not contain blanks or
special characters.
If you have different versions,
or very long names you can use ``-``
to increase readability.

.. code-block:: python

    import audformat

    audformat.Database(name="librispeech-mfa-cseg-pho")


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

.. code-block:: python

    db = audformat.Database("mydata")

    db.schemes["arousal"] = audformat.Scheme(audformat.define.DataType.FLOAT)
    db.schemes["valence"] = audformat.Scheme(audformat.define.DataType.FLOAT)

    db.raters["rater1"] = audformat.Rater()
    db.raters["rater2"] = audformat.Rater()

    for scheme_id in db.schemes:
        db[scheme_id] = audformat.Table(audformat.filewise_index())
        for rater_id in db.raters:
            db[scheme_id][rater_id] = audformat.Column(
                scheme_id=scheme_id,
                rater_id=rater_id,
            )

>>> db
name: mydata
source: ''
usage: unrestricted
languages: []
raters:
  rater1: {type: human}
  rater2: {type: human}
schemes:
  arousal: {dtype: float}
  valence: {dtype: float}
tables:
  arousal:
    type: filewise
    columns:
      rater1: {scheme_id: arousal, rater_id: rater1}
      rater2: {scheme_id: arousal, rater_id: rater2}
  valence:
    type: filewise
    columns:
      rater1: {scheme_id: valence, rater_id: rater1}
      rater2: {scheme_id: valence, rater_id: rater2}


Database splits
---------------

If an official split into training,
development
and test set
consists,
consider one table per split,
named ``scheme_id.split``.

.. code-block:: python

    db = audformat.Database("mydata")

    db.schemes["arousal"] = audformat.Scheme(audformat.define.DataType.FLOAT)

    db.splits["train"] = audformat.Split(audformat.define.SplitType.TRAIN)
    db.splits["dev"] = audformat.Split(audformat.define.SplitType.DEVELOP)
    db.splits["test"] = audformat.Split(audformat.define.SplitType.TEST)

    for scheme_id in db.schemes:
        for split_id in db.splits:
            table_id = f"{scheme_id}.{split_id}"
            db[table_id] = audformat.Table(
                index=audformat.filewise_index(),
                split_id=split_id,
            )

>>> db
name: mydata
source: ''
usage: unrestricted
languages: []
schemes:
  arousal: {dtype: float}
splits:
  dev: {type: dev}
  test: {type: test}
  train: {type: train}
tables:
  arousal.dev: {type: filewise, split_id: dev}
  arousal.test: {type: filewise, split_id: test}
  arousal.train: {type: filewise, split_id: train}


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
a rater with the id ``"gold_standard"``
and the type ``audformat.define.RaterType.VOTE``
should be created
and associated with the column
holding the gold standard values.

.. code-block:: python

    db = audformat.Database("mydata")

    db.schemes["arousal"] = audformat.Scheme(audformat.define.DataType.FLOAT)

    db.raters["rater1"] = audformat.Rater()
    db.raters["rater2"] = audformat.Rater()
    db.raters["gold_standard"] = audformat.Rater(audformat.define.RaterType.VOTE)

    for scheme_id in db.schemes:
        db[scheme_id] = audformat.Table(audformat.filewise_index())
        for rater_id in ["rater1", "rater2"]:
            db[scheme_id][rater_id] = audformat.Column(
                scheme_id=scheme_id,
                rater_id=rater_id,
            )
        gold_id = f"{scheme_id}.gold_standard"
        db[gold_id] = audformat.Table(audformat.filewise_index())
        db[gold_id][scheme_id] = audformat.Column(
            scheme_id=scheme_id,
            rater_id="gold_standard",
        )

>>> db
name: mydata
source: ''
usage: unrestricted
languages: []
raters:
  gold_standard: {type: vote}
  rater1: {type: human}
  rater2: {type: human}
schemes:
  arousal: {dtype: float}
tables:
  arousal:
    type: filewise
    columns:
      rater1: {scheme_id: arousal, rater_id: rater1}
      rater2: {scheme_id: arousal, rater_id: rater2}
  arousal.gold_standard:
    type: filewise
    columns:
      arousal: {scheme_id: arousal, rater_id: gold_standard}


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

.. code-block:: python

    db = audformat.Database("mydata")

    db.schemes["arousal"] = audformat.Scheme(audformat.define.DataType.FLOAT)
    db.schemes["arousal.confidence"] = audformat.Scheme(
        audformat.define.DataType.FLOAT,
        minimum=0,
        maximum=1,
    )

    db.raters["gold_standard"] = audformat.Rater(audformat.define.RaterType.VOTE)

    db["arousal"] = audformat.Table(audformat.filewise_index())
    for scheme_id in db.schemes:
        db["arousal"][scheme_id] = audformat.Column(
            scheme_id=scheme_id,
            rater_id="gold_standard",
        )

>>> db
name: mydata
source: ''
usage: unrestricted
languages: []
raters:
  gold_standard: {type: vote}
schemes:
  arousal: {dtype: float}
  arousal.confidence: {dtype: float, minimum: 0, maximum: 1}
tables:
  arousal:
    type: filewise
    columns:
      arousal: {scheme_id: arousal, rater_id: gold_standard}
      arousal.confidence: {scheme_id: arousal.confidence, rater_id: gold_standard}


File and speaker information
----------------------------

Meta information like speaker ID
that is not included in another table
should be collected in a table ``files``.
If you have metadata
belonging only to segments,
collect it in a table ``segments``.

Additional meta information,
that is bound to another information
like age of speaker,
should be collected in the header
as it can be later automatically mapped.

.. code-block:: python

    db = audformat.Database("mydata")

    M = audformat.define.Gender.MALE
    F = audformat.define.Gender.FEMALE
    speaker = {
        "speaker1": {"gender": F, "age": 31},
        "speaker2": {"gender": M, "age": 85},
    }

    db.schemes["speaker"] = audformat.Scheme(labels=speaker)
    db["files"] = audformat.Table(
        index=audformat.filewise_index(["a.wav", "b.wav"])
    )
    db["files"]["speaker"] = audformat.Column(scheme_id="speaker")
    db["files"]["speaker"].set(["speaker1", "speaker2"])

>>> db
name: mydata
source: ''
usage: unrestricted
languages: []
schemes:
  speaker:
    dtype: str
    labels:
      speaker1: {gender: female, age: 31}
      speaker2: {gender: male, age: 85}
tables:
  files:
    type: filewise
    columns:
      speaker: {scheme_id: speaker}
>>> db["files"].get()
        speaker
file
a.wav  speaker1
b.wav  speaker2

You can access the additional information with the ``map`` argument
of :meth:`audformat.Table.get`,
see :ref:`map-scheme-labels`
for an extended documentation.

>>> db["files"].get(map={"speaker": "gender"})
       gender
file
a.wav  female
b.wav    male


Temporal data
-------------

Temporal duration data
like response time of a rater
should be stored as :class:`pd.Timedelta`.
Temporal dates
like time of rating
should be stored as :class:`datetime.datetime`.

.. code-block:: python

    import pandas as pd


    times = [2.1, 0.1]  # in seconds

    db = audformat.Database("mydata")

    db.schemes["time"] = audformat.Scheme(audformat.define.DataType.TIME)
    db.raters["rater"] = audformat.Rater()

    db["files"] = audformat.Table(
        index=audformat.filewise_index(["a.wav", "b.wav"])
    )
    db["files"]["time"] = audformat.Column(
        scheme_id="time",
        rater_id="rater",
    )
    db["files"]["time"].set(pd.to_timedelta(times, unit="s"))

>>> db["files"].get()
                        time
file
a.wav 0 days 00:00:02.100000
b.wav 0 days 00:00:00.100000
