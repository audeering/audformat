Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog`_,
and this project adheres to `Semantic Versioning`_.


Version 1.3.2 (2025-05-27)
--------------------------

* Added: support for Python 3.13
* Added: support for Python 3.12
* Fixed: ``audformat.Database.update()`` for misc tables
* Fixed: ``audformat.Database.get()``
  if ``scheme`` is stored in a segmented table,
  and ``additional_schemes`` are stored in filewise tables.
  Before,
  values of the additional schemes were set to ``<NA>``
* Fixed: if ``audformat.Column.set()`` assigns values
  to a column with scheme labels
  given by a misc table,
  the misc table is only converted once to a dictionary
  instead of once for each value


Version 1.3.1 (2024-09-16)
--------------------------

* Changed: replace unmaintained ``iso-639`` dependency
  with ``iso639-lang``
* Fixed: ensure ``poetry`` can manage ``audformat``


Version 1.3.0 (2024-07-18)
--------------------------

* Added: ``strict`` argument
  to ``audformat.utils.hash()``.
  If set to ``True``,
  the order of the data,
  and its level/column names
  are taken into account
  when calculating the hash
* Changed: store tables per default as parquet files,
  by changing the default value of ``storage_format``
  to ``"parquet"``
  in ``audformat.Table.save()``
  and ``audformat.Database.save()``
* Fixed: load csv tables with ``pandas.read_csv()``,
  if ``pyarrow.csv.read_csv()`` fails


Version 1.2.0 (2024-06-25)
--------------------------

* Added: expand format specifications
  to allow parquet files
  as table files
* Added: support for storing tables as parquet files
  by adding ``"parquet"``
  (``audformat.define.TableStorageFormat.PARQUET``)
  as an option
  for the ``storage_format`` argument
  of ``audformat.Table.save()``
  and ``audformat.Database.save()``
* Added: support for ``numpy>=2.0``
* Added: mention text files
  as potential media files
  in the documentation
* Added: mention in the documentation of ``audformat.utils.hash()``
  that column/level names do not influence its hash value
* Added: warn in the documentation of ``audformat.utils.hash()``
  that the hash of a dataframe or series,
  containing ``"Int64"`` as data type,
  changes with ``pandas>=2.2.0``
* Fixed: ensure ``"boolean"`` data type
  is always used
  in indices of misc tables
  that store boolean values


Version 1.1.4 (2024-05-15)
--------------------------

* Fixed: ``audformat.Database.get()``,
  if its argument ``additional_schemes``
  contains a non-existent scheme


Version 1.1.3 (2024-04-26)
--------------------------

* Added: ``as_dataframe`` argument
  to ``audformat.utils.read_csv()``
* Fixed: ``audformat.utils.read_csv()``
  now treats float/integer values
  in ``start``, ``end`` columns
  as seconds


Version 1.1.2 (2024-02-02)
--------------------------

* Fixed: ``audformat.Database.load()``
  when loading databases
  with a misc table
  that has an assigned split


Version 1.1.1 (2024-01-25)
--------------------------

* Changed: depend on ``audeer>=2.0.0``
* Fixed: ``pandas`` deprecation warnings


Version 1.1.0 (2023-11-30)
--------------------------

* Added: ``audformat.Database.get()`` method
  to retrieve labels based on their schemes
  and independent of the tables
  in which they are stored
* Added: ``aggregate_function``
  and ``aggregate_strategy``
  arguments to ``audformat.utils.concat()``
  to support overlapping values
  in the objects
  that should be concatenated
* Changed: ``audformat.Column.get(map=...)``
  now returns dtype of labels
* Changed: ``audformat.Column.get(map=...)``
  does no longer raise an error
  if some of the mapped values
  are not available
  when stored in a dictionary
  as scheme labels
* Fixed: avoid deprecation warning
  by replacing
  ``pkg_resources``
  internally with
  ``importlib.metadata``


Version 1.0.3 (2023-10-11)
--------------------------

* Fixed: ``audformat.utils.hash()`` for ``pandas>=2.1.0``
* Fixed: remove upper limit of ``pandas`` dependency


Version 1.0.2 (2023-10-09)
--------------------------

* Fixed: require ``pandas<2.1.0``
  as ``pandas>=2.1.0`` introduced a bug
  in calculating the hash of an index
* Removed: deprecated ``root`` argument
  from ``audformat.testing.create_audio_files()``


Version 1.0.1 (2023-05-08)
--------------------------

* Fixed: ensure ``audformat.utils.to_segmented_index()``
  and ``audformat.Table.get()``
  with ``as_segmented=True``
  uses same precision for ``end`` values
  as ``audformat.segmented_index()``


Version 1.0.0 (2023-04-27)
--------------------------

* Added: ``audformat.Scheme.labels_as_list`` property
  to list all scheme labels
* Added: example to the documentation of
  ``audformat.utils.to_filewise_index()``
* Changed: convert dates to UTC timezone
  in ``audformat.Column.set()``
  when using a scheme of type ``"date"``
* Fixed: support ``pandas>=2.0.0``
* Fixed: mention ``author``,
  ``license``,
  ``license_url``,
  ``organization``
  in the specification documentation
  of the database header
* Fixed: missing ``Raises`` section
  in the documentation of ``audformat.Database.load()``
  and ``audformat.Database.attachments``
* Fixed: when the ``root`` argument
  of ``audformat.utils.expand_file_path()``
  is a relative path
  it is no longer expanded to an absolute path


Version 0.16.1 (2023-03-29)
---------------------------

* Added: ``copy_attachments`` argument
  to ``audformat.Database.update()``
* Changed: preserve ``dtypes``
  when ``audformat.Table.get()``
  is called with an index
* Changed: speed up ``audformat.utils.union()``
* Changed: allow to save a database
  with missing attachments


Version 0.16.0 (2023-01-12)
---------------------------

* Added: ``audformat.Attachment`` to store
  any kind of files/folders as part of the database
* Added: support for Python 3.10
* Added: support for Python 3.11
* Changed: require ``audeer>=1.19.0``
* Changed: split API documentation into sub-pages
  for each function
* Fixed: support ``"meta"`` as key in meta dictionaries
  like the one passed as ``meta`` argument
  to ``audformat.Database``


Version 0.15.4 (2022-11-01)
---------------------------

* Fixed: avoid ``FutureWarning``
  when setting values in place for a series
  in ``audformat.Column.set()``
* Fixed: improve sketches
  in the specifications section
  of the documentation


Version 0.15.3 (2022-09-19)
---------------------------

* Changed: ``audformat.Column.set()``
  now lists values
  not matching
  the scheme of the column
  in the corresponding error message
* Fixed: ``audformat.Column.set()``
  checking of values
  for a scheme with minimum and/or maximum
  when input values are given
  as ``np.array``
  and contain ``NaN``
  or ``None``
* Fixed: ``audformat.Column.set()``
  checking of values
  for a scheme with minimum and/or maximum
  when minimum or maximum is 0


Version 0.15.2 (2022-08-17)
---------------------------

* Added: ``audformat.Table.map_files()``
* Fixed: ``audformat.Database.load()``
  for databases that contain a scheme
  with labels stored in a misc table
  that is using schemes for its columns.
  Before it could fail
  if the schemes were not loaded in the correct order
* Fixed: ``audformat.Table.drop_index()``
  and ``audformat.MiscTable.drop_index()``
  when the provided index to drop
  contains entries
  not present in the index of the table.
  Before it was extending the table
  by those entries
  besides dropping overlapping indices


Version 0.15.1 (2022-08-11)
---------------------------

* Added: ``audformat.Scheme.uses_table``
  to indicate if the scheme uses a misc table
  to store its labels
* Added: usage example to docstring of
  ``audfromat.utils.to_segmented_index()``
* Changed: forbid nesting of misc tables as scheme labels
* Fixed: support for ``pd.Index``
  and ``pd.Series``
  in ``audformat.utils.to_filewise_index()``
* Fixed: description of ``audformat.Schemes.labels``
  in API documentation


Version 0.15.0 (2022-08-05)
---------------------------

* Added: ``audformat.MiscTable``
  which can store data
  not associated with media files
* Added: store scheme labels in a misc table
* Added: dictionary ``audformat.Database.misc_tables``
  holding misc tables of a database
* Added: ``audformat.utils.difference()``
  for finding index entries
  that are only part of a single index
  for a given sequence of indices
* Added: ``audformat.utils.is_index_alike()``
  for checking if a sequence of indices
  has the same number of levels,
  level names,
  and matching dtypes
* Added: ``audformat.define.DataType.OBJECT``
* Added: ``audformat.utils.set_index_dtypes()``
  to change dtypes of an index
* Added: ``audformat.testing.add_misc_table()``
* Added: ``audformat.Database.__iter__``
  iterates through all (misc) tables,
  e.g. a user can do ``list(db)``
  to get a list of all (misc) tables
* Changed: ``audformat.Database.update()``
  can now join schemes
  with different labels
* Changed: ``audformat.utils.union()``,
  ``audformat.utils.intersect()``,
  and ``audformat.utils.concat()``
  now support any kind of index
* Changed: ``audformat.utils.intersect()``
  no longer removes segments
  from a segmented index
  that are contained
  in a filewise index
* Changed: require ``pandas>=1.4.1``
* Changed: use ``pandas`` dtype ``"string"``
  instead of ``"object"``
  for storing ``audformat`` dtype ``"str"`` entries
* Changed: use a misc table
  to store the ``"speaker"`` scheme labels
  in the emodb example
  in the documentation
* Changed: ``audformat.utils.join_labels()``
  raises ``ValueError``
  if labels are of different dtype
* Fixed: ensure column IDs are different from index level names
* Fixed: make sure
  ``audformat.Column.set()``
  converts data to dtype of scheme
  before checking if values are in min-max-range
  of scheme
* Fixed: links to ``pandas`` API in the documentation
* Fixed: include methods
  ``to_dict()``,
  ``from_dict()``,
  ``dump()``,
  and attributes
  ``description``,
  ``meta``
  in the documentation for the classes
  ``audformat.Column``,
  ``audformat.Database``,
  ``audformat.Media``,
  ``audformat.Rater``,
  ``audformat.Scheme``,
  ``audformat.Split``,
  ``audformat.Table``
* Fixed: type hint of argument ``dtype``
  in the documentation of ``audformat.Scheme``
* Removed: support for Python 3.7


Version 0.14.3 (2022-06-01)
---------------------------

* Added: ``audformat.utils.map_country()``
* Changed: improve speed of ``audformat.Table.drop_files()``
  for segmented tables


Version 0.14.2 (2022-04-29)
---------------------------

* Added: ``audformat.utils.index_has_overlap()``
* Added: ``audformat.utils.iter_index_by_file()``
* Changed: store categories with integers as ``int64`` instead of ``Int64``
* Changed: require ``audeer>=1.18.0``
* Changed: support ``pandas>=1.4.0``


Version 0.14.1 (2022-03-03)
---------------------------

* Added: ``audformat.utils.map_file_path()``


Version 0.14.0 (2022-02-24)
---------------------------

* Changed: ensure ``audformat.testing.create_database()``
  uses Unix path separators
* Changed: don't allow ``\`` path entries
  in a portable database
* Changed: mark deprecated ``root`` argument
  of ``audformat.testing.create_audio_files()``
  to be removed in version 1.0.0


Version 0.13.3 (2022-02-07)
---------------------------

* Fixed: conversion of pickle protocol 5 files
  to pickle protocol 4 in cache


Version 0.13.2 (2022-01-27)
---------------------------

* Fixed: reintroduce sorting the output of
  ``audformat.Database.files`` and
  ``audformat.Database.segments``


Version 0.13.1 (2022-01-26)
---------------------------

* Fixed: changelog for 0.13.0


Version 0.13.0 (2022-01-26)
---------------------------

* Changed: ``audformat.utils.union()`` no longer sorts levels
* Changed: ``audformat.Table.save()`` forces pickle format 4
* Changed: clean up test requirements
* Changed: require ``pandas < 1.4.0``


Version 0.12.4 (2022-01-12)
---------------------------

* Changed: the API documentation on the ``language`` argument
  of ``audformat.Database`` is more verbose now
* Changed: the difference between
  ``audformat.define.DataType.TIME``
  and ``audformat.define.DataType.DATE``
  is now discussed in the API documentation
* Fixed: saving a not loaded table to CSV
  when a PKL file is present
* Fixed: ``pandas`` deprecation warnings


Version 0.12.3 (2022-01-03)
---------------------------

* Removed: Python 3.6 support


Version 0.12.2 (2021-11-18)
---------------------------

* Added: ``audformat.assert_no_duplicates()``
* Changed: ``audformat.assert_index()`` no longer checks for duplicates


Version 0.12.1 (2021-11-17)
---------------------------

* Added: ``audformat.utils.hash()``
* Added: ``audformat.utils.expand_file_path()``
* Added: ``audformat.utils.replace_file_extension()``
* Changed: use ``yaml.CLoader`` for faster header reading


Version 0.12.0 (2021-11-10)
---------------------------

* Added: ``as_segmented``, ``allow_nat``, ``root``, ``num_workers``
  arguments to ``audformat.Table.get()``
* Added: ``as_segmented``, ``allow_nat``, ``root``, ``num_workers``
  arguments to ``audformat.Column.get()``
* Added: ``files_duration`` argument
  to ``audformat.utils.to_segmented_index()``
* Added: ``audformat.Database.files_duration()``
* Changed: changed default value of ``load_data`` argument
  in ``audformat.Database.load()`` to ``False``
* Changed: speed up ``audformat.Database.files``
  and ``audformat.Database.segments``
* Fixed: re-add support for ``pandas>=1.3``


Version 0.11.6 (2021-08-20)
---------------------------

* Added: support for Python 3.9
* Fixed: speed up ``audformat.utils.union()``
* Fixed: ``audformat.Column.set()`` with ``pd.Series``
  and ``np.array`` for a scheme with fixed labels
  and containing ``NaN`` values


Version 0.11.5 (2021-08-09)
---------------------------

* Removed: duration scheme and column
  from conventions
  and emodb example


Version 0.11.4 (2021-08-05)
---------------------------

* Added: custom ``BadKeyError`` when key is not found
* Changed: limit to ``pandas <1.3``
  until it works again for newer ``pandas`` versions
* Changed: remove the ``<1.0.0`` limit for ``audiofile``
  as a stable release is available and the API has not changed


Version 0.11.3 (2021-06-10)
---------------------------

* Added: ``audformat.utils.duration``
* Fixed: description of ``audformat.Database.is_portable``
  in documentation


Version 0.11.2 (2021-05-12)
---------------------------

* Added: ``audformat.utils.join_schemes``


Version 0.11.1 (2021-05-11)
---------------------------

* Added: ``Database.is_portable``
* Added: ``copy_media`` argument to ``Database.update()``
* Changed: remove ``root`` argument from ``testing.create_audio_files()`` and instead use ``Database.root``
* Fixed: ``utils.concat()`` converts to nullable dtype
* Fixed: ``utils.concat()`` returns ``DataFrame`` if input contains at least one ``DataFrame``


Version 0.11.0 (2021-05-06)
---------------------------

Note: tables stored from this version upwards cannot be loaded with older versions

* Added: ``Database.root``
* Added: ``utils.join_labels()``
* Added: ``Scheme.replace_labels()``
* Changed: set dependency to ``pandas>=1.1.5``
* Changed: do not compress pickled table files


Version 0.10.2 (2021-04-22)
---------------------------

* Changed: ``allow_nat`` argument to ``utils.to_segmented_index()``


Version 0.10.1 (2021-03-31)
---------------------------

* Fixed: ``audformat.assert_index()`` checks for correct dtypes


Version 0.10.0 (2021-03-18)
---------------------------

* Added: ``audformat.Database.update()``
* Added: ``audformat.Table.update()``
* Added: ``overwrite`` argument to ``audformat.utils.concat()``
* Changed: result of ``audformat.Table.__add__()`` is no longer assigned to a ``audformat.Database``


Version 0.9.8 (2021-02-23)
--------------------------

* Added: ``audformat.Database.license``
* Added: ``audformat.Database.license_url``
* Added: ``audformat.Database.author``
* Added: ``audformat.Database.organization``
* Added: ``audformat.utils.intersect()`` for index objects
* Added: ``audformat.utils.union()`` for index objects
* Changed: ``Database.load()`` raises error if table file missing
* Changed: forbid duplicates in ``audformat`` conform indices
* Fixed: ``audformat.Table.__add__()`` returned wrong values
  for some index combinations


Version 0.9.7 (2021-02-01)
--------------------------

* Added: ``update_other_formats`` argument to ``audformat.Table.save()``
  to make sure existing files in other formats are updated as well
* Changed: use ``round_trip`` argument when loading CSV files
  to ensure dataframes are equal after storing and loading again


Version 0.9.6 (2021-01-28)
--------------------------

* Fixed: implemented ``audformat.Database.__eq__`` and return ``True``
  for identical databases


Version 0.9.5 (2021-01-14)
--------------------------

* Changed: use nullable Pandas' type ``"boolean"`` for ``bool`` schemes
* Fixed: ``Scheme.draw()`` generates boolean values if scheme is ``bool``


Version 0.9.4 (2021-01-11)
--------------------------

* Changed: add arguments ``num_workers`` and ``verbose`` to
  ``audformat.Database.load()``


Version 0.9.3 (2021-01-07)
--------------------------

* Fixed: avoid sphinx syntax in CHANGELOG


Version 0.9.2 (2021-01-07)
--------------------------

* Changed: add arguments ``num_workers`` and ``verbose`` to
  ``audformat.Database.drop_files()``,
  ``audformat.Database.map_files()``,
  ``audformat.Database.pick_files()``,
  ``audformat.Database.save()``
* Changed: ``audformat.segmented_index()``
  support ``int`` and ``float``, which will be interpreted as seconds
* Fixed: ``audformat.utils.to_segmented_index()``
  returns correct index type for ``NaT``


Version 0.9.1 (2020-12-21)
--------------------------

* Fixed: add column name to HTML Series output in docs
* Fixed: removed mentioning of
  ``NotConformToUnifiedFormat`` error
  and ``RedundantArgumentError`` error
* Fixed: add missing errors to docstring
  of ``audformat.Table.set()``
  and ``audformat.Column.set()``


Version 0.9.0 (2020-12-18)
--------------------------

* Added: initial release public release


.. _Keep a Changelog:
    https://keepachangelog.com/en/1.0.0/
.. _Semantic Versioning:
    https://semver.org/spec/v2.0.0.html
