Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog`_,
and this project adheres to `Semantic Versioning`_.


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

* Changed: use nullable Pandas' type ``'boolean'`` for ``bool`` schemes
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
