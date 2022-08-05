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

.. jupyter-execute::

    import os
    import urllib.request

    import audeer


    # Get database source
    source = 'http://emodb.bilderbar.info/download/download.zip'
    src_dir = 'emodb-src'
    if not os.path.exists(src_dir):
        urllib.request.urlretrieve(source, 'emodb.zip')
        audeer.extract_archive('emodb.zip', src_dir)

    os.listdir(src_dir)


Gather metadata and annotations
-------------------------------

Afterwards we collect all metadata
and annotations
that we would like to store in the audformat version
of the database.

First, have a look at the file names.

.. jupyter-execute::

    os.listdir(os.path.join(src_dir, 'wav'))[:3]

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

.. jupyter-execute::

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
       'Berlin Database of Emotional Speech. '
       'A German database of emotional utterances '
       'spoken by actors '
       'recorded as a part of the DFG funded research project '
       'SE462/3-1 in 1997 and 1999. '
       'Recordings took place in the anechoic chamber '
       'of the Technical University Berlin, '
       'department of Technical Acoustics. '
       'It contains about 500 utterances '
       'from ten different actors '
       'expressing basic six emotions and neutral.'
    )

    files = sorted(
        [os.path.join('wav', f) for f in os.listdir(os.path.join(src_dir, 'wav'))]
    )
    names = [audeer.basename_wo_ext(f) for f in files]

    emotion_mapping = {
        'W': 'anger',
        'L': 'boredom',
        'E': 'disgust',
        'A': 'fear',
        'F': 'happiness',
        'T': 'sadness',
        'N': 'neutral',
    }
    emotions = list(parse_names(names, from_i=5, to_i=6, mapping=emotion_mapping))

    y = pd.read_csv(
        os.path.join(src_dir, 'erkennung.txt'),
        usecols=['Satz', 'erkannt'],
        index_col='Satz',
        delim_whitespace=True,
        encoding='Latin-1',
        decimal=',',
        converters={'Satz': lambda x: os.path.join('wav', x)},
    ).squeeze('columns')
    y = y.loc[files]
    y = y.replace(to_replace=u'\xa0', value='', regex=True)
    y = y.replace(to_replace=',', value='.', regex=True)
    confidences = y.astype('float').values

    male = audformat.define.Gender.MALE
    female = audformat.define.Gender.FEMALE
    de = audformat.utils.map_language('de')
    df_speaker = pd.DataFrame(
        index=pd.Index([3, 8, 9, 10, 11, 12, 13, 14, 15, 16], name='speaker'),
        columns=['age', 'gender', 'language'],
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
        'a01': 'Der Lappen liegt auf dem Eisschrank.',
        'a02': 'Das will sie am Mittwoch abgeben.',
        'a04': 'Heute abend könnte ich es ihm sagen.',
        'a05': 'Das schwarze Stück Papier befindet sich da oben neben dem '
               'Holzstück.',
        'a07': 'In sieben Stunden wird es soweit sein.',
        'b01': 'Was sind denn das für Tüten, die da unter dem Tisch '
               'stehen.',
        'b02': 'Sie haben es gerade hochgetragen und jetzt gehen sie '
               'wieder runter.',
        'b03': 'An den Wochenenden bin ich jetzt immer nach Hause '
               'gefahren und habe Agnes besucht.',
        'b09': 'Ich will das eben wegbringen und dann mit Karl was '
               'trinken gehen.',
        'b10': 'Die wird auf dem Platz sein, wo wir sie immer hinlegen.',
    }
    transcriptions = list(parse_names(names, from_i=2, to_i=5))


Create audformat database
-------------------------

Now we create the database object
and assign the information to it.

.. jupyter-execute::

    db = audformat.Database(
        name='emodb',
        source=source,
        usage=audformat.define.Usage.UNRESTRICTED,
        languages=[de],
        description=description,
        meta={
            'pdf': (
                'http://citeseerx.ist.psu.edu/viewdoc/'
                'download?doi=10.1.1.130.8506&rep=rep1&type=pdf'
            ),
        },
    )

    # Media
    db.media['microphone'] = audformat.Media(
        format='wav',
        sampling_rate=16000,
        channels=1,
    )

    # Raters
    db.raters['gold'] = audformat.Rater()

    # Schemes
    db.schemes['emotion'] = audformat.Scheme(
        labels=[str(x) for x in emotion_mapping.values()],
        description='Six basic emotions and neutral.',
    )
    db.schemes['confidence'] = audformat.Scheme(
        'float',
        minimum=0,
        maximum=1,
        description='Confidence of emotion ratings.',
    )
    db.schemes['age'] = audformat.Scheme(
        'int',
        minimum=0,
        description='Age of speaker',
    )
    db.schemes['gender'] = audformat.Scheme(
        labels=['female', 'male'],
        description='Gender of speaker',
    )
    db.schemes['language'] = audformat.Scheme(
        'str',
        description='Language of speaker',
    )
    db.schemes['transcription'] = audformat.Scheme(
        labels=transcription_mapping,
        description='Sentence produced by actor.',
    )

    # MiscTable
    db['speaker'] = audformat.MiscTable(df_speaker.index)
    db['speaker']['age'] = audformat.Column(scheme_id='age')
    db['speaker']['gender'] = audformat.Column(scheme_id='gender')
    db['speaker']['language'] = audformat.Column(scheme_id='language')
    db['speaker'].set(df_speaker.to_dict(orient='list'))

    # MiscTable as Scheme
    db.schemes['speaker'] = audformat.Scheme(
        labels='speaker',
        dtype='int',
        description=(
            'The actors could produce each sentence as often as '
            'they liked and were asked to remember a real '
            'situation from their past when they had felt this '
            'emotion.'
        ),
    )

    # Tables
    index = audformat.filewise_index(files)
    db['files'] = audformat.Table(index)

    db['files']['speaker'] = audformat.Column(scheme_id='speaker')
    db['files']['speaker'].set(speakers)

    db['files']['transcription'] = audformat.Column(scheme_id='transcription')
    db['files']['transcription'].set(transcriptions)

    db['emotion'] = audformat.Table(index)
    db['emotion']['emotion'] = audformat.Column(
        scheme_id='emotion',
        rater_id='gold',
    )
    db['emotion']['emotion'].set(emotions)
    db['emotion']['emotion.confidence'] = audformat.Column(
        scheme_id='confidence',
        rater_id='gold',
    )
    db['emotion']['emotion.confidence'].set(confidences / 100.0)


Inspect database header
-----------------------

Before storing the database,
we can inspect its header.

.. jupyter-execute::

    db


Inspect database tables
-----------------------

First check which tables are available.

.. jupyter-execute::

    list(db)

Then list the first 10 entries of every table.

.. jupyter-execute::

    db['files'].get()[:10]

.. jupyter-execute::

    db['emotion'].get()[:10]

.. jupyter-execute::

    db['speaker'].get()[:10]

Columns might contain labels,
that provide additional mappings.
You can access this additional information
with the ``map`` argument of :meth:`audformat.Table.get`,
see :ref:`map-scheme-labels`
for an extended documentation.

.. jupyter-execute::

    db['files'].get(map={'speaker': ['speaker', 'age', 'gender']})[:10]


Store database to disk
----------------------

Now we store the database in the folder ``emodb``.
Note, that we have to make sure
that the media files are located at the correct position ourselves.

.. jupyter-execute::

    import shutil


    db_dir = audeer.mkdir('emodb')
    shutil.copytree(
        os.path.join(src_dir, 'wav'),
        os.path.join(db_dir, 'wav'),
    )
    db.save(db_dir)

    os.listdir(db_dir)


You can read the database from disk as well.

.. jupyter-execute::

    db = audformat.Database.load(db_dir)
    db.name


.. Clean up
.. jupyter-execute::
    :hide-code:
    :hide-output:

    shutil.rmtree(db_dir)


.. _emodb: http://emodb.bilderbar.info
.. _emodb documentation: http://emodb.bilderbar.info/index-1280.html
.. _emodb paper: https://www.isca-speech.org/archive/archive_papers/interspeech_2005/i05_1517.pdf
