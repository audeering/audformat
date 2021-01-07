Introduction
============

:ref:`audformat <data-format:Database>` was created
to handle all databases inside audEERING
in an unified way to make them easily accessible.

The main challenge in specifying the format was to
**make the format universal enough
to be applicable to many use cases as possible,
but simple enough to be understood and parsed easily**.

:ref:`audformat <data-format:Database>` can store the following information:

* Raters (humans, aggregated, machine)
* Labels (categorical, numeric, etc.)
* Annotations (file or segment, rater, confidence, multi-label)
* Metadata (recording date/location, source, speaker age/gender, etc.
  and may apply to whole corpus or individual files / segments)

The format further allows to

* link audio files to meta data and annotations
* use generic tools to access the data,
  create statistics,
  merge annotations
* search / filter information in databases
* use machine learning tools to easily access data
  and run experiments across databases
