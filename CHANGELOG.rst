Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog`_,
and this project adheres to `Semantic Versioning`_.


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
