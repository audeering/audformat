Conventions
===========

* The name of a database should be lowercase and must not contain blanks or
  special characters,
  e.g. ``My Nice-Database!`` becomes ``mynicedatabase``.
  If you have different versions, or very long names you can use ``-``
  to increase readability, e.g. ``librispeech-mfa-cseg-pho``
* Consider one table per scheme with the name of the scheme.
  Example: a database ``db`` with schemes ``arousal`` and ``valence``
  may consist of tables ``db['arousal']`` and ``db['valence']``.
  If you have multiple raters,
  name each column after the name of the rater.
* If an official split into training, development and test set consists,
  consider one table per split.
  To continue the previous example, we end up with six tables:
  ``db['arousal.train']``,
  ``db['arousal.dev']``,
  ``db['arousal.test']``,
  ``db['valence.train']``,
  ``db['valence.dev']``, and
  ``db['valence.test']``.
* Annotations by several raters belonging to
  the same scheme should be stored in a single table,
  but **not** aggregated,
  e.g. by adding a column with mean or some other metric.
  Instead a new table with the postfix ``.gold_standard`` should be created
  to store the average of all rater,
  e.g. ``db['arousal.gold_standard']``.
  In addition, a rater with the id ``'gold_standard'``
  and the type ``audformat.define.RaterType.VOTE``
  should be created and associated with the column holding the gold standard values.
* Do not mix annotations and meta information.
* Meta information
  (e.g. speaker id or file duration)
  that applies to whole files,
  can be collected in a table ``files``
  (or ``segments`` if it applies to segments).
* Use Pandas' ``Timedelta`` or ``Date`` class to store temporal information.
  Example: file duration as ``Timedelta``.
* Use lower case for table names and schemes.

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
E.g. use ``transcription-wrd`` and ``transcription-pho``
if a database offers word *and* phoneme transcriptions.
