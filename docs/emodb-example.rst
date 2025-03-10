.. _emodb-example:

Emodb example
=============

In this example we download the small emodb_ database,
that contains sentences
spoken with different emotions
by different actors.
The audio is stored as WAV files.


Get source database
-------------------

First we download the source emodb_ database
to the folder :file:`emodb-src`.

.. code-block:: python

    import os
    import urllib.request

    import audeer


    # Get database source
    source = "http://emodb.bilderbar.info/download/download.zip"
    src_dir = "emodb-src"
    if not os.path.exists(src_dir):
        urllib.request.urlretrieve(source, "emodb.zip")
        audeer.extract_archive("emodb.zip", src_dir)

>>> sorted(os.listdir(src_dir))
['erkennung.txt', 'erklaerung.txt', 'lablaut', 'labsilb', 'silb', 'wav']


Gather metadata and annotations
-------------------------------

Afterwards we collect all metadata
and annotations
that we would like to store in the audformat version
of the database.

First, have a look at the file names.

>>> sorted(os.listdir(os.path.join(src_dir, "wav")))[:3]
['03a01Fa.wav', '03a01Nc.wav', '03a01Wa.wav']

As described in the `emodb documentation`_
the encoding is the following.

======== ===============
Position Encoding
======== ===============
0..1     speaker
2..4     spoken sentence
5        emotion
6        repetition
======== ===============

For speaker further information is provided.

========== ====== ===
Speaker ID Gender Age
========== ====== ===
03         male   31
08         female 34
09         female 21
10         male   32
11         male   26
12         male   30
13         female 32
14         female 35
15         male   25
16         female 31
========== ====== ===

For the sentences we have transcriptions.

==== ==================================================================================
Code Transcription
==== ==================================================================================
a01  Der Lappen liegt auf dem Eisschrank.
a02  Das will sie am Mittwoch abgeben.
a04  Heute Abend könnte ich es ihm sagen.
a05  Das schwarze Stück Papier befindet sich da oben neben dem Holzstück.
a07  In sieben Stunden wird es soweit sein.
b01  Was sind denn das für Tüten, die da unter dem Tisch stehen?
b02  Sie haben es gerade hoch getragen und jetzt gehen sie wieder runter.
b03  An den Wochenenden bin ich jetzt immer nach Hause gefahren und habe Agnes besucht.
b09  Ich will das eben wegbringen und dann mit Karl was trinken gehen.
b10  Die wird auf dem Platz sein, wo wir sie immer hinlegen.
==== ==================================================================================

The emotion codes belong to the following emotions.

==== =========
Code Emotion
==== =========
W    anger
L    boredom
E    disgust
A    fear
F    happiness
T    sadness
N    neutral
==== =========

As stated in the `emodb paper`_,
the acted emotions were further evaluated
by 20 participants
that had to assign emotion labels
to the audio presentations.
Their agreement of the rating is stored
as the ``erkannt`` column
in the file :file:`erkennung.txt`.
We will read in this file
and use the annotations to add a confidence column
to the emotion table.

.. code-block:: python

    import audformat
    import pandas as pd

    # Prepare functions for getting information from file names
    def parse_names(names, from_i, to_i, is_number=False, mapping=None):
        for name in names:
            key = name[from_i:to_i]
            if is_number:
                key = int(key)
            yield mapping[key] if mapping else key


    description = (
       "Berlin Database of Emotional Speech. "
       "A German database of emotional utterances "
       "spoken by actors "
       "recorded as a part of the DFG funded research project "
       "SE462/3-1 in 1997 and 1999. "
       "Recordings took place in the anechoic chamber "
       "of the Technical University Berlin, "
       "department of Technical Acoustics. "
       "It contains about 500 utterances "
       "from ten different actors "
       "expressing basic six emotions and neutral."
    )

    files = sorted(
        [f"wav/{f}" for f in os.listdir(os.path.join(src_dir, "wav"))]
    )
    names = [audeer.basename_wo_ext(f) for f in files]

    emotion_mapping = {
        "W": "anger",
        "L": "boredom",
        "E": "disgust",
        "A": "fear",
        "F": "happiness",
        "T": "sadness",
        "N": "neutral",
    }
    emotions = list(parse_names(names, from_i=5, to_i=6, mapping=emotion_mapping))

    y = pd.read_csv(
        os.path.join(src_dir, "erkennung.txt"),
        usecols=["Satz", "erkannt"],
        index_col="Satz",
        sep=r"\t",
        encoding="Latin-1",
        decimal=",",
        engine="python",
    ).squeeze("columns")
    y.index = "wav/" + y.index
    y = y.loc[files]
    y = y.replace(to_replace=u"\xa0", value="", regex=True)
    y = y.replace(to_replace=",", value=".", regex=True)
    confidences = y.astype("float").values

    male = audformat.define.Gender.MALE
    female = audformat.define.Gender.FEMALE
    de = audformat.utils.map_language("de")
    df_speaker = pd.DataFrame(
        index=pd.Index([3, 8, 9, 10, 11, 12, 13, 14, 15, 16], name="speaker"),
        columns=["age", "gender", "language"],
        data = [
            [31, male, de],
            [34, female, de],
            [21, female, de],
            [32, male, de],
            [26, male, de],
            [30, male, de],
            [32, female, de],
            [35, female, de],
            [25, male, de],
            [31, female, de],
       ],
    )
    speakers = list(parse_names(names, from_i=0, to_i=2, is_number=True))

    transcription_mapping = {
        "a01": "Der Lappen liegt auf dem Eisschrank.",
        "a02": "Das will sie am Mittwoch abgeben.",
        "a04": "Heute abend könnte ich es ihm sagen.",
        "a05": "Das schwarze Stück Papier befindet sich da oben neben dem "
               "Holzstück.",
        "a07": "In sieben Stunden wird es soweit sein.",
        "b01": "Was sind denn das für Tüten, die da unter dem Tisch "
               "stehen.",
        "b02": "Sie haben es gerade hochgetragen und jetzt gehen sie "
               "wieder runter.",
        "b03": "An den Wochenenden bin ich jetzt immer nach Hause "
               "gefahren und habe Agnes besucht.",
        "b09": "Ich will das eben wegbringen und dann mit Karl was "
               "trinken gehen.",
        "b10": "Die wird auf dem Platz sein, wo wir sie immer hinlegen.",
    }
    transcriptions = list(parse_names(names, from_i=2, to_i=5))


Create audformat database
-------------------------

Now we create the database object
and assign the information to it.

.. code-block:: python

    db = audformat.Database(
        name="emodb",
        source=source,
        usage=audformat.define.Usage.UNRESTRICTED,
        languages=[de],
        description=description,
        meta={
            "pdf": (
                "http://citeseerx.ist.psu.edu/viewdoc/"
                "download?doi=10.1.1.130.8506&rep=rep1&type=pdf"
            ),
        },
    )

    # Media
    db.media["microphone"] = audformat.Media(
        format="wav",
        sampling_rate=16000,
        channels=1,
    )

    # Raters
    db.raters["gold"] = audformat.Rater()

    # Schemes
    db.schemes["emotion"] = audformat.Scheme(
        labels=[str(x) for x in emotion_mapping.values()],
        description="Six basic emotions and neutral.",
    )
    db.schemes["confidence"] = audformat.Scheme(
        "float",
        minimum=0,
        maximum=1,
        description="Confidence of emotion ratings.",
    )
    db.schemes["age"] = audformat.Scheme(
        "int",
        minimum=0,
        description="Age of speaker",
    )
    db.schemes["gender"] = audformat.Scheme(
        labels=["female", "male"],
        description="Gender of speaker",
    )
    db.schemes["language"] = audformat.Scheme(
        "str",
        description="Language of speaker",
    )
    db.schemes["transcription"] = audformat.Scheme(
        labels=transcription_mapping,
        description="Sentence produced by actor.",
    )

    # MiscTable
    db["speaker"] = audformat.MiscTable(df_speaker.index)
    db["speaker"]["age"] = audformat.Column(scheme_id="age")
    db["speaker"]["gender"] = audformat.Column(scheme_id="gender")
    db["speaker"]["language"] = audformat.Column(scheme_id="language")
    db["speaker"].set(df_speaker.to_dict(orient="list"))

    # MiscTable as Scheme
    db.schemes["speaker"] = audformat.Scheme(
        labels="speaker",
        dtype="int",
        description=(
            "The actors could produce each sentence as often as "
            "they liked and were asked to remember a real "
            "situation from their past when they had felt this "
            "emotion."
        ),
    )

    # Tables
    index = audformat.filewise_index(files)
    db["files"] = audformat.Table(index)

    db["files"]["speaker"] = audformat.Column(scheme_id="speaker")
    db["files"]["speaker"].set(speakers)

    db["files"]["transcription"] = audformat.Column(scheme_id="transcription")
    db["files"]["transcription"].set(transcriptions)

    db["emotion"] = audformat.Table(index)
    db["emotion"]["emotion"] = audformat.Column(
        scheme_id="emotion",
        rater_id="gold",
    )
    db["emotion"]["emotion"].set(emotions)
    db["emotion"]["emotion.confidence"] = audformat.Column(
        scheme_id="confidence",
        rater_id="gold",
    )
    db["emotion"]["emotion.confidence"].set(confidences / 100.0)


Inspect database header
-----------------------

Before storing the database,
we can inspect its header.

>>> db
name: emodb
description: Berlin Database of Emotional Speech. A German database of emotional utterances
  spoken by actors recorded as a part of the DFG funded research project SE462/3-1
  in 1997 and 1999. Recordings took place in the anechoic chamber of the Technical
  University Berlin, department of Technical Acoustics. It contains about 500 utterances
  from ten different actors expressing basic six emotions and neutral.
source: http://emodb.bilderbar.info/download/download.zip
usage: unrestricted
languages: [deu]
media:
  microphone: {type: other, format: wav, channels: 1, sampling_rate: 16000}
raters:
  gold: {type: human}
schemes:
  age: {description: Age of speaker, dtype: int, minimum: 0}
  confidence: {description: Confidence of emotion ratings., dtype: float, minimum: 0,
    maximum: 1}
  emotion:
    description: Six basic emotions and neutral.
    dtype: str
    labels: [anger, boredom, disgust, fear, happiness, sadness, neutral]
  gender:
    description: Gender of speaker
    dtype: str
    labels: [female, male]
  language: {description: Language of speaker, dtype: str}
  speaker: {description: The actors could produce each sentence as often as they liked
      and were asked to remember a real situation from their past when they had felt
      this emotion., dtype: int, labels: speaker}
  transcription:
    description: Sentence produced by actor.
    dtype: str
    labels: {a01: Der Lappen liegt auf dem Eisschrank., a02: Das will sie am Mittwoch
        abgeben., a04: Heute abend könnte ich es ihm sagen., a05: Das schwarze Stück
        Papier befindet sich da oben neben dem Holzstück., a07: In sieben Stunden
        wird es soweit sein., b01: 'Was sind denn das für Tüten, die da unter dem
        Tisch stehen.', b02: Sie haben es gerade hochgetragen und jetzt gehen sie
        wieder runter., b03: An den Wochenenden bin ich jetzt immer nach Hause gefahren
        und habe Agnes besucht., b09: Ich will das eben wegbringen und dann mit Karl
        was trinken gehen., b10: 'Die wird auf dem Platz sein, wo wir sie immer hinlegen.'}
tables:
  emotion:
    type: filewise
    columns:
      emotion: {scheme_id: emotion, rater_id: gold}
      emotion.confidence: {scheme_id: confidence, rater_id: gold}
  files:
    type: filewise
    columns:
      speaker: {scheme_id: speaker}
      transcription: {scheme_id: transcription}
misc_tables:
  speaker:
    levels: {speaker: int}
    columns:
      age: {scheme_id: age}
      gender: {scheme_id: gender}
      language: {scheme_id: language}
pdf: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.130.8506&rep=rep1&type=pdf


Inspect database tables
-----------------------

First check which tables are available.

>>> list(db)
['emotion', 'files', 'speaker']

Then list the first 10 entries of every table.

>>> db["files"].get()[:10]
                speaker transcription
file
wav/03a01Fa.wav       3           a01
wav/03a01Nc.wav       3           a01
wav/03a01Wa.wav       3           a01
wav/03a02Fc.wav       3           a02
wav/03a02Nc.wav       3           a02
wav/03a02Ta.wav       3           a02
wav/03a02Wb.wav       3           a02
wav/03a02Wc.wav       3           a02
wav/03a04Ad.wav       3           a04
wav/03a04Fd.wav       3           a04
>>> db["emotion"].get()[:10]
                   emotion  emotion.confidence
file
wav/03a01Fa.wav  happiness                0.90
wav/03a01Nc.wav    neutral                1.00
wav/03a01Wa.wav      anger                0.95
wav/03a02Fc.wav  happiness                0.85
wav/03a02Nc.wav    neutral                1.00
wav/03a02Ta.wav    sadness                0.90
wav/03a02Wb.wav      anger                1.00
wav/03a02Wc.wav      anger                1.00
wav/03a04Ad.wav       fear                0.90
wav/03a04Fd.wav  happiness                1.00
>>> db["speaker"].get()[:10]
         age  gender language
speaker
3         31    male      deu
8         34  female      deu
9         21  female      deu
10        32    male      deu
11        26    male      deu
12        30    male      deu
13        32  female      deu
14        35  female      deu
15        25    male      deu
16        31  female      deu

Columns might contain labels,
that provide additional mappings.
You can access this additional information
with the ``map`` argument of :meth:`audformat.Table.get`,
see :ref:`map-scheme-labels`
for an extended documentation.

>>> db["files"].get(map={"speaker": ["speaker", "age", "gender"]})[:10]
                speaker transcription  age gender
file
wav/03a01Fa.wav       3           a01   31   male
wav/03a01Nc.wav       3           a01   31   male
wav/03a01Wa.wav       3           a01   31   male
wav/03a02Fc.wav       3           a02   31   male
wav/03a02Nc.wav       3           a02   31   male
wav/03a02Ta.wav       3           a02   31   male
wav/03a02Wb.wav       3           a02   31   male
wav/03a02Wc.wav       3           a02   31   male
wav/03a04Ad.wav       3           a04   31   male
wav/03a04Fd.wav       3           a04   31   male


Store database to disk
----------------------

Now we store the database in the folder ``emodb``.
Note, that we have to make sure
that the media files are located at the correct position ourselves.

.. code-block:: python

    import shutil


    db_dir = audeer.mkdir(tmpdir, "emodb")
    shutil.copytree(
        os.path.join(src_dir, "wav"),
        os.path.join(db_dir, "wav"),
    )
    db.save(db_dir)

>>> sorted(os.listdir(db_dir))
['db.emotion.parquet', 'db.files.parquet', 'db.speaker.parquet', 'db.yaml', 'wav']

You can read the database from disk as well.

>>> db = audformat.Database.load(db_dir)
>>> db.name
'emodb'

.. _emodb: http://emodb.bilderbar.info
.. _emodb documentation: http://emodb.bilderbar.info/index-1280.html
.. _emodb paper: https://www.isca-speech.org/archive/archive_papers/interspeech_2005/i05_1517.pdf
