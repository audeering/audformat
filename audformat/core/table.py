from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence
import copy
import os
import pickle
import typing

import pandas as pd
import pyarrow as pa
import pyarrow.csv as csv
import pyarrow.parquet as parquet

import audeer

from audformat.core import define
from audformat.core import utils
from audformat.core.column import Column
from audformat.core.common import HeaderBase
from audformat.core.common import HeaderDict
from audformat.core.common import to_pandas_dtype
from audformat.core.errors import BadIdError
from audformat.core.index import filewise_index
from audformat.core.index import index_type
from audformat.core.index import is_filewise_index
from audformat.core.index import is_segmented_index
from audformat.core.media import Media
from audformat.core.split import Split
from audformat.core.typing import Values


class Base(HeaderBase):
    r"""Table base class."""

    def __init__(
        self,
        index: pd.Index = None,
        *,
        split_id: str = None,
        media_id: str = None,
        description: str = None,
        meta: dict = None,
    ):
        super().__init__(description=description, meta=meta)

        self.split_id = split_id
        r"""Split ID"""
        self.media_id = media_id
        r"""Media ID"""
        self.columns = HeaderDict(
            sort_by_key=False,
            value_type=Column,
            set_callback=self._set_column,
        )
        r"""Table columns"""

        self._df = pd.DataFrame(index=index)
        self._db = None
        self._id = None

    def __add__(self, other: typing.Self) -> typing.Self:
        r"""Create new table by combining two tables.

        The new :ref:`combined table <combine-tables>`
        contains index and columns of both tables.
        Missing values will be set to ``NaN``.

        If table is conform to
        :ref:`table specifications <data-tables:Tables>`
        and at least one table is segmented,
        the output has a segmented index.

        Columns with the same identifier are combined to a single column.
        This requires that:

        1. both columns have the same dtype
        2. in places where the indices overlap the values of both columns
           match or one column contains ``NaN``

        Media and split information,
        as well as,
        references to schemes and raters are discarded.
        If you intend to keep them,
        use ``update()``.

        Args:
            other: the other table

        Raises:
            ValueError: if columns with the same name have different dtypes
            ValueError: if values in the same position do not match
            ValueError: if level and dtypes of indices do not match

        """
        df = utils.concat([self.df, other.df])

        table = self.__new__(type(self))
        table.__init__(df.index)
        for column_id in df:
            table[column_id] = Column()
        table._df = df

        return table

    def __getitem__(self, column_id: str) -> Column:
        r"""Return view to a column.

        Args:
            column_id: column identifier

        """
        return self.columns[column_id]

    def __eq__(
        self,
        other: Base,
    ) -> bool:
        r"""Compare if table equals other table."""
        if self.dump() != other.dump():
            return False
        return self.df.equals(other.df)

    def __len__(self) -> int:
        r"""Number of rows in table."""
        return len(self.df)

    def __setitem__(self, column_id: str, column: Column) -> Column:
        r"""Add new column to table.

        Args:
            column_id: column identifier
            column: column

        Raises:
            BadIdError: if a column with a ``scheme_id`` or ``rater_id`` is
                added that does not exist
            ValueError: if column ID is not different from level names
            ValueError: if the column is linked to a scheme
                that is using labels from a misc table,
                but the misc table the column is assigned to
                is already used by the same or another scheme

        """
        if (
            column.scheme_id is not None
            and self.db is not None
            and column.scheme_id in self.db.schemes
        ):
            # check if scheme uses
            # labels from a table
            scheme = self.db.schemes[column.scheme_id]
            if scheme.uses_table:
                # check if scheme uses
                # labels from this table
                if self._id == scheme.labels:
                    raise ValueError(
                        f"Scheme "
                        f"'{column.scheme_id}' "
                        f"uses misc table "
                        f"'{self._id}' "
                        f"as labels and cannot be used "
                        f"with columns of the same table."
                    )

                # check if this table
                # is already used with a scheme
                for scheme_id in self.db.schemes:
                    if self._id == self.db.schemes[scheme_id].labels:
                        raise ValueError(
                            f"Since the misc table "
                            f"'{self._id}' "
                            f"is used as labels in scheme "
                            f"'{scheme_id}' "
                            f"its columns cannot be used with a scheme "
                            f"that also uses labels from a misc table."
                        )

        self.columns[column_id] = column
        return column

    @property
    def db(self):
        r"""Database object.

        Returns:
            database object or ``None`` if not assigned yet

        """
        return self._db

    @property
    def df(self) -> pd.DataFrame:
        r"""Table data.

        Returns:
            data

        """
        if self._df is None:
            # if database was loaded with 'load_data=False'
            # we have to load the table data now
            path = os.path.join(self.db.root, f"{self.db._name}.{self._id}")
            self.load(path)
        return self._df

    @property
    def index(self) -> pd.Index:
        r"""Table index.

        Returns:
            index

        """
        return self.df.index

    @property
    def media(self) -> Media | None:
        r"""Media object.

        Returns:
            media object or ``None`` if not available

        """
        if self.media_id is not None and self.db is not None:
            return self.db.media[self.media_id]

    @property
    def split(self) -> Split | None:
        r"""Split object.

        Returns:
            split object or ``None`` if not available

        """
        if self.split_id is not None and self.db is not None:
            return self.db.splits[self.split_id]

    def copy(self) -> typing.Self:
        r"""Copy table.

        Return:
            new table object

        """
        table = self.__class__(
            self.df.index,
            media_id=self.media_id,
            split_id=self.split_id,
        )
        table._db = self.db
        for column_id, column in self.columns.items():
            table.columns[column_id] = Column(
                scheme_id=column.scheme_id,
                rater_id=column.rater_id,
                description=column.description,
                meta=column.meta.copy(),
            )
        table._df = self.df.copy()
        return table

    def drop_columns(
        self,
        column_ids: str | Sequence[str],
        *,
        inplace: bool = False,
    ) -> typing.Self:
        r"""Drop columns by ID.

        Args:
            column_ids: column IDs
            inplace: drop columns in place

        Returns:
            new object if ``inplace=False``, otherwise ``self``

        """
        if not inplace:
            return self.copy().drop_columns(column_ids, inplace=True)

        if isinstance(column_ids, str):
            column_ids = [column_ids]
        column_ids_ = set()
        for column_id in column_ids:
            column_ids_.add(column_id)
        self.df.drop(column_ids_, inplace=True, axis="columns")
        for column_id in column_ids_:
            self.columns.pop(column_id)

        return self

    def drop_index(
        self,
        index: pd.Index,
        *,
        inplace: bool = False,
    ) -> typing.Self:
        r"""Drop rows from index.

        Args:
            index: index object
            inplace: drop index in place

        Returns:
            new object if ``inplace=False``, otherwise ``self``

        Raises:
            ValueError: if level and dtypes of index does not match table index

        """
        table = self if inplace else self.copy()

        index = _maybe_convert_dtype_to_string(index)
        _assert_table_index(table, index, "drop rows from")

        index = utils.intersect([table.index, index])
        new_index = utils.difference([table.index, index])
        table._df = table.df.reindex(new_index)

        if inplace:
            _maybe_update_scheme(table)

        return table

    def extend_index(
        self,
        index: pd.Index,
        *,
        fill_values: object | dict[str, object] = None,
        inplace: bool = False,
    ) -> typing.Self:
        r"""Extend table with new rows.

        Args:
            index: index object
            fill_values: replace NaN with these values (either a scalar
                applied to all columns or a dictionary with column name as
                key)
            inplace: extend index in place

        Returns:
            new object if ``inplace=False``, otherwise ``self``

        Raises:
            ValueError: if level and dtypes of index does not match table index

        """
        table = self if inplace else self.copy()

        index = _maybe_convert_dtype_to_string(index)
        _assert_table_index(table, index, "extend")

        new_index = utils.union([table.index, index])
        table._df = table.df.reindex(new_index)
        if fill_values is not None:
            if isinstance(fill_values, dict):
                for key, value in fill_values.items():
                    table.df.fillna({key: value}, inplace=True)
            else:
                table.df.fillna(fill_values, inplace=True)

        if inplace:
            _maybe_update_scheme(table)

        return table

    def get(
        self,
        index: pd.Index = None,
        *,
        map: dict[str, str | Sequence[str]] = None,
        copy: bool = True,
    ) -> pd.DataFrame:
        r"""Get labels.

        By default, all labels of the table are returned,
        use ``index`` to get a subset.

        Examples are provided with the
        :ref:`table specifications <data-tables:Tables>`,
        and for ``map`` in :ref:`map-scheme-labels`.

        Args:
            index: index
            copy: return a copy of the labels
            map: map scheme or scheme fields to column values.
                For example if your table holds a column ``speaker`` with
                speaker IDs, which is assigned to a scheme that contains a
                dict mapping speaker IDs to age and gender entries,
                ``map={'speaker': ['age', 'gender']}``
                will replace the column with two new columns that map ID
                values to age and gender, respectively.
                To also keep the original column with speaker IDS, you can do
                ``map={'speaker': ['speaker', 'age', 'gender']}``

        Returns:
            labels

        Raises:
            FileNotFoundError: if file is not found
            RuntimeError: if table is not assign to a database
            ValueError: if trying to map without a scheme
            ValueError: if trying to map from a scheme that has no labels
            ValueError: if trying to map to a non-existing field

        """
        result_is_copy = False

        if index is None:
            result = self.df
        else:
            result = self._get_by_index(index)

        if map is not None:
            if self.db is None:
                raise RuntimeError(
                    "Cannot map schemes, " "table is not assigned to a database."
                )

            if not result_is_copy:
                result = result.copy()
                result_is_copy = True  # to avoid another copy

            for column, mapped_columns in map.items():
                mapped_columns = audeer.to_list(mapped_columns)
                if len(mapped_columns) == 1:
                    result[mapped_columns[0]] = self.columns[column].get(
                        index,
                        map=mapped_columns[0],
                    )
                else:
                    for mapped_column in mapped_columns:
                        if mapped_column != column:
                            result[mapped_column] = self.columns[column].get(
                                index,
                                map=mapped_column,
                            )
                if column not in mapped_columns:
                    result.drop(columns=column, inplace=True)

        return result.copy() if (copy and not result_is_copy) else result

    def load(
        self,
        path: str,
    ):
        r"""Load table data from disk.

        Tables are stored as CSV, PARQUET and/or PKL files to disk.
        If the PKL file exists,
        it will load the PKL file
        as long as its modification date is the newest,
        otherwise it will raise an error
        and ask to delete one of the files.

        Args:
            path: file path without extension

        Raises:
            RuntimeError: if table file(s) are missing
            RuntimeError: if CSV or PARQUET file is newer than PKL file

        """
        path = audeer.path(path)
        csv_file = f"{path}.{define.TableStorageFormat.CSV}"
        parquet_file = f"{path}.{define.TableStorageFormat.PARQUET}"
        pkl_file = f"{path}.{define.TableStorageFormat.PICKLE}"

        if (
            not os.path.exists(pkl_file)
            and not os.path.exists(csv_file)
            and not os.path.exists(parquet_file)
        ):
            raise RuntimeError(
                f"No file found for table with path '{path}.{{csv|parquet|pkl}}'"
            )

        # Load from PKL if file exists
        # and is newer than CSV or PARQUET file.
        # If files are written by Database.save()
        # this is always the case
        # as it stores first the PKL file
        pickled = False
        if os.path.exists(pkl_file):
            for file in [parquet_file, csv_file]:
                if os.path.exists(file) and os.path.getmtime(file) > os.path.getmtime(
                    pkl_file
                ):
                    ext = audeer.file_extension(file).upper()
                    raise RuntimeError(
                        f"The table {ext} file '{file}' is newer "
                        f"than the table PKL file '{pkl_file}'. "
                        f"If you want to load from the {ext} file, "
                        "please delete the PKL file. "
                        "If you want to load from the PKL file, "
                        f"please delete the {ext} file."
                    )
            pickled = True

        if pickled:
            try:
                self._load_pickled(pkl_file)
            except (AttributeError, ValueError, EOFError) as ex:
                # If exception is raised
                # (e.g. unsupported pickle protocol)
                # try to load from PARQUET or CSV
                # and save it again
                # otherwise raise error
                if os.path.exists(parquet_file):
                    self._load_parquet(parquet_file)
                    self._save_pickled(pkl_file)
                elif os.path.exists(csv_file):
                    self._load_csv(csv_file)
                    self._save_pickled(pkl_file)
                else:
                    raise ex
        elif os.path.exists(parquet_file):
            self._load_parquet(parquet_file)
        else:
            self._load_csv(csv_file)

    def pick_columns(
        self,
        column_ids: str | Sequence[str],
        *,
        inplace: bool = False,
    ) -> typing.Self:
        r"""Pick columns by ID.

        All other columns will be dropped.

        Args:
            column_ids: column IDs
            inplace: pick columns in place

        Returns:
            new object if ``inplace=False``, otherwise ``self``

        """
        if isinstance(column_ids, str):
            column_ids = [column_ids]
        drop_ids = set()
        for column_id in list(self.columns):
            if column_id not in column_ids:
                drop_ids.add(column_id)
        return self.drop_columns(list(drop_ids), inplace=inplace)

    def pick_index(
        self,
        index: pd.Index,
        *,
        inplace: bool = False,
    ) -> typing.Self:
        r"""Pick rows from index.

        Args:
            index: index object
            inplace: pick index in place

        Returns:
            new object if ``inplace=False``, otherwise ``self``

        Raises:
            ValueError: if level and dtypes of index does not match table index

        """
        table = self if inplace else self.copy()

        index = _maybe_convert_dtype_to_string(index)
        _assert_table_index(table, index, "pick rows from")

        new_index = utils.intersect([table.index, index])
        table._df = table.df.reindex(new_index)

        if inplace:
            _maybe_update_scheme(table)

        return table

    def save(
        self,
        path: str,
        *,
        storage_format: str = define.TableStorageFormat.PARQUET,
        update_other_formats: bool = True,
    ):
        r"""Save table data to disk.

        Existing files will be overwritten.

        When using ``"parquet"`` as ``storage_format``
        a hash,
        based on the content of the table,
        is stored under the key ``b"hash"``
        in the metadata of the schema of the parquet file.
        This provides a deterministic hash for the file,
        as md5 sums of parquet files,
        containing identical information,
        often differ.
        Reasons include factors like the library
        that wrote the parquet file,
        the chosen compression codec
        and metadata written by the library.

        The hash can be accessed with ``pyarrow`` by::

            pyarrow.parquet.read_schema(f"{path}.parquet").metadata[b"hash"].decode()

        The hash is used by :mod:`audb`
        when publishing a database
        to track changes of database files.

        Args:
            path: file path without extension
            storage_format: storage format of table.
                See :class:`audformat.define.TableStorageFormat`
                for available formats
            update_other_formats: if ``True`` it will not only save
                to the given ``storage_format``,
                but update all files stored in other storage formats as well

        """
        path = audeer.path(path)
        define.TableStorageFormat._assert_has_attribute_value(storage_format)

        csv_file = f"{path}.{define.TableStorageFormat.CSV}"
        parquet_file = f"{path}.{define.TableStorageFormat.PARQUET}"
        pickle_file = f"{path}.{define.TableStorageFormat.PICKLE}"

        # Ensure the following storage order:
        # 1. PARQUET file
        # 2. CSV file
        # 3. PKL file
        # The PKl is expected to be the oldest by load(),
        # the order of PARQUET and CSV file
        # is only a convention for now.
        if storage_format == define.TableStorageFormat.PICKLE:
            if update_other_formats and os.path.exists(parquet_file):
                self._save_parquet(parquet_file)
            if update_other_formats and os.path.exists(csv_file):
                self._save_csv(csv_file)
            self._save_pickled(pickle_file)

        if storage_format == define.TableStorageFormat.PARQUET:
            self._save_parquet(parquet_file)
            if update_other_formats and os.path.exists(csv_file):
                self._save_csv(csv_file)
            if update_other_formats and os.path.exists(pickle_file):
                self._save_pickled(pickle_file)

        if storage_format == define.TableStorageFormat.CSV:
            if update_other_formats and os.path.exists(parquet_file):
                self._save_parquet(parquet_file)
            self._save_csv(csv_file)
            if update_other_formats and os.path.exists(pickle_file):
                self._save_pickled(pickle_file)

    def set(
        self,
        values: (dict[str, Values] | pd.DataFrame),
        *,
        index: pd.Index = None,
    ):
        r"""Set labels.

        By default, all labels of the table are replaced,
        use ``index`` to select a subset.
        If a column is assigned to a :class:`Scheme`
        values will be automatically converted
        to match its dtype.

        Examples are provided with the
        :ref:`table specifications <data-tables:Tables>`.

        Args:
            values: dictionary of values with ``column_id`` as key
            index: index

        Raises:
            ValueError: if values cannot be converted
                to match the schemes dtype

        """
        for idx, data in values.items():
            self.columns[idx].set(data, index=index)

    def update(
        self,
        others: typing.Self | Sequence[typing.Self],
        *,
        overwrite: bool = False,
    ) -> typing.Self:
        r"""Update table with other table(s).

        Table which calls ``update()``
        to :ref:`combine tables <combine-tables>`
        must be assigned to a database.
        For all tables media and split must match.

        Columns that are not yet part of the table will be added and
        referenced schemes or raters are copied.
        For overlapping columns, schemes and raters must match.

        Columns with the same identifier are combined to a single column.
        This requires that both columns have the same dtype
        and if ``overwrite`` is set to ``False``,
        values in places where the indices overlap have to match
        or one column contains ``NaN``.
        If ``overwrite`` is set to ``True``,
        the value of the last table in the list is kept.

        The index type of the table must not change.

        Args:
            others: table object(s)
            overwrite: overwrite values where indices overlap

        Returns:
            the updated table

        Raises:
            RuntimeError: if table is not assign to a database
            ValueError: if split or media does not match
            ValueError: if overlapping columns reference different schemes
                or raters
            ValueError: if a missing scheme or rater cannot be copied
                because a different object with the same ID exists
            ValueError: if values in same position overlap
            ValueError: if level and dtypes of table indices do not match

        """
        if self.db is None:
            raise RuntimeError("Table is not assigned to a database.")

        others = audeer.to_list(others)

        for other in others:
            _assert_table_index(self, other.index, "update")

        def raise_error(
            msg,
            left: HeaderDict | None,
            right: HeaderDict | None,
        ):
            raise ValueError(f"{msg}:\n" f"{left}\n" "!=\n" f"{right}")

        def assert_equal(
            msg: str,
            left: HeaderDict | None,
            right: HeaderDict | None,
        ):
            equal = True
            if left and right:
                equal = left == right
            elif left or right:
                equal = False
            if not equal:
                raise_error(msg, left, right)

        missing_schemes = {}
        missing_raters = {}

        for other in others:
            assert_equal(
                "Media of table " f"'{other._id}' " "does not match",
                self.media,
                other.media,
            )

            assert_equal(
                "Split of table " f"'{other._id}' " "does not match",
                self.split,
                other.split,
            )

            # assert schemes match for overlapping columns and
            # look for missing schemes in new columns,
            # raise an error if a different scheme with same ID exists
            for column_id, column in other.columns.items():
                if column_id in self.columns:
                    assert_equal(
                        "Scheme of common column "
                        f"'{other._id}.{column_id}' "
                        "does not match",
                        self.columns[column_id].scheme,
                        column.scheme,
                    )
                else:
                    if column.scheme is not None:
                        if column.scheme_id in self.db.schemes:
                            assert_equal(
                                "Cannot copy scheme of column "
                                f"'{other._id}.{column_id}' "
                                "as a different scheme with ID "
                                f"'{column.scheme_id}' "
                                "exists",
                                self.db.schemes[column.scheme_id],
                                column.scheme,
                            )
                        else:
                            missing_schemes[column.scheme_id] = column.scheme

            # assert raters match for overlapping columns and
            # look for missing raters in new columns,
            # raise an error if a different rater with same ID exists
            for column_id, column in other.columns.items():
                if column_id in self.columns:
                    assert_equal(
                        f"self['{self._id}']['{column_id}'].rater "
                        "does not match "
                        f"other['{other._id}']['{column_id}'].rater",
                        self.columns[column_id].rater,
                        column.rater,
                    )
                else:
                    if column.rater is not None:
                        if column.rater_id in self.db.raters:
                            assert_equal(
                                f"db1.raters['{column.scheme_id}'] "
                                "does not match "
                                f"db2.raters['{column.scheme_id}']",
                                self.db.raters[column.rater_id],
                                column.rater,
                            )
                        else:
                            missing_raters[column.rater_id] = column.rater

        # concatenate table data
        df = utils.concat(
            [self.df] + [other.df for other in others],
            overwrite=overwrite,
        )

        # insert missing schemes and raters
        for scheme_id, scheme in missing_schemes.items():
            self.db.schemes[scheme_id] = copy.copy(scheme)
        for rater_id, rater in missing_raters.items():
            self.db.raters[rater_id] = copy.copy(rater)

        # insert new columns
        for other in others:
            for column_id, column in other.columns.items():
                if column_id not in self.columns:
                    self.columns[column_id] = copy.copy(column)

        # update table data
        self._df = df

        return self

    def _get_by_index(
        self,
        index: pd.Index,
    ) -> (pd.DataFrame, bool):  # pragma: no cover
        # Executed when calling `self.get(index=index)`.
        # Returns `df, df_is_copy`
        raise NotImplementedError()

    @property
    def _levels_and_dtypes(self) -> dict[str, str]:
        r"""Levels and dtypes of index columns.

        Returns:
            dictionary with index levels (column names)
            and associated audformat data type

        """
        # The returned dictionary is used
        # to infer index column names and dtypes
        # when reading CSV files.
        raise NotImplementedError()  # pragma: no cover

    def _load_csv(self, path: str):
        r"""Load table from CSV file.

        The loaded table is stored under ``self._df``.

        Loading a CSV file with :func:`pandas.read_csv()` is slower
        than the method applied here.
        We first load the CSV file as a :class:`pyarrow.Table`
        and convert it to a dataframe afterwards.
        If this fails,
        we fall back to :func:`pandas.read_csv()`.

        Args:
            path: path to table, including file extension

        """
        levels = list(self._levels_and_dtypes.keys())
        columns = list(self.columns.keys())
        try:
            table = csv.read_csv(
                path,
                read_options=csv.ReadOptions(
                    column_names=levels + columns,
                    skip_rows=1,
                ),
                convert_options=csv.ConvertOptions(
                    column_types=self._pyarrow_csv_schema(),
                    strings_can_be_null=True,
                ),
            )
            df = self._pyarrow_table_to_dataframe(table, from_csv=True)
        except pa.lib.ArrowInvalid:
            # If pyarrow fails to parse the CSV file
            # https://github.com/audeering/audformat/issues/449

            # Collect csv file columns and data types.
            # index
            columns_and_dtypes = self._levels_and_dtypes
            # columns
            for column_id, column in self.columns.items():
                if column.scheme_id is not None:
                    columns_and_dtypes[column_id] = self.db.schemes[
                        column.scheme_id
                    ].dtype
                else:
                    columns_and_dtypes[column_id] = define.DataType.OBJECT

            # Replace data type with converter for dates or timestamps
            converters = {}
            dtypes_wo_converters = {}
            for column, dtype in columns_and_dtypes.items():
                if dtype == define.DataType.DATE:
                    converters[column] = lambda x: pd.to_datetime(x)
                elif dtype == define.DataType.TIME:
                    converters[column] = lambda x: pd.to_timedelta(x)
                else:
                    dtypes_wo_converters[column] = to_pandas_dtype(dtype)

            df = pd.read_csv(
                path,
                usecols=list(columns_and_dtypes.keys()),
                dtype=dtypes_wo_converters,
                index_col=levels,
                converters=converters,
                float_precision="round_trip",
            )

        self._df = df

    def _load_parquet(self, path: str):
        r"""Load table from PARQUET file.

        The loaded table is stored under ``self._df``.

        Args:
            path: path to table, including file extension

        """
        # Read PARQUET file
        table = parquet.read_table(path)
        df = self._pyarrow_table_to_dataframe(table)

        self._df = df

    def _load_pickled(self, path: str):
        r"""Load table from PKL file.

        The loaded table is stored under ``self._df``.

        Args:
            path: path to table, including file extension

        """
        # Older versions of audformat used xz compression
        # which produced smaller files,
        # but was slower.
        # The try-except statement allows backward compatibility
        try:
            df = pd.read_pickle(path)
        except pickle.UnpicklingError:
            df = pd.read_pickle(path, compression="xz")

        # Older versions of audformat stored columns
        # assigned to a string scheme as 'object',
        # so we need to convert those to 'string'
        for column_id, column in self.columns.items():
            if (
                column.scheme_id is not None
                and (self.db.schemes[column.scheme_id].dtype == define.DataType.STRING)
                and df[column_id].dtype == "object"
            ):
                df[column_id] = df[column_id].astype("string", copy=False)
        # Fix index entries as well
        df.index = _maybe_convert_dtype_to_string(df.index)

        self._df = df

    def _pyarrow_convert_dtypes(
        self,
        df: pd.DataFrame,
        *,
        convert_all: bool = False,
    ) -> pd.DataFrame:
        r"""Convert dtypes that are not handled by pyarrow.

        This adjusts dtypes in a dataframe,
        that could not be set correctly
        when converting to the dataframe
        from pyarrow.

        Args:
            df: dataframe,
            convert_all: if ``False``,
                converts all columns with
                ``"object"`` audformat dtype,
                and all columns with a scheme with labels.
                If ``"True"``,
                it converts additionally all columns with
                ``"bool"``, ``"int"``, and ``"time"`` audformat dtypes

        Returns:
            dataframe with converted dtypes

        """
        # Collect columns with dtypes,
        # that cannot directly be converted
        # from pyarrow to pandas
        bool_columns = []
        int_columns = []
        time_columns = []
        object_columns = []

        # Collect columns
        # with scheme labels
        labeled_columns = []

        # Collect columns,
        # belonging to the table index
        # (not the index of the provided dataframe)
        index_columns = []

        # --- Index ---
        index_columns += list(self._levels_and_dtypes.keys())
        for level, dtype in self._levels_and_dtypes.items():
            if dtype == define.DataType.BOOL:
                bool_columns.append(level)
            elif dtype == define.DataType.INTEGER:
                int_columns.append(level)
            elif dtype == define.DataType.TIME:
                time_columns.append(level)
            elif dtype == define.DataType.OBJECT:
                object_columns.append(level)

        # --- Columns ---
        for column_id, column in self.columns.items():
            if column.scheme_id is not None:
                scheme = self.db.schemes[column.scheme_id]
                if scheme.labels is not None:
                    labeled_columns.append(column_id)
                elif scheme.dtype == define.DataType.BOOL:
                    bool_columns.append(column_id)
                elif scheme.dtype == define.DataType.INTEGER:
                    int_columns.append(column_id)
                elif scheme.dtype == define.DataType.TIME:
                    time_columns.append(column_id)
                elif scheme.dtype == define.DataType.OBJECT:
                    object_columns.append(column_id)
            else:
                # No scheme defaults to `object` dtype
                object_columns.append(column_id)

        if convert_all:
            for column in bool_columns:
                df[column] = df[column].astype("boolean")
            for column in int_columns:
                df[column] = df[column].astype("Int64")
            for column in time_columns:
                df[column] = df[column].astype("timedelta64[ns]")
        for column in object_columns:
            df[column] = df[column].astype("object")
            df[column] = df[column].replace(pd.NA, None)
        for column in labeled_columns:
            scheme = self.db.schemes[self.columns[column].scheme_id]
            labels = scheme._labels_to_list()
            if len(labels) > 0 and isinstance(labels[0], int):
                # allow nullable
                labels = pd.array(labels, dtype="int64")
            dtype = pd.api.types.CategoricalDtype(
                categories=labels,
                ordered=False,
            )
            df[column] = df[column].astype(dtype)
        return df

    def _pyarrow_csv_schema(self) -> pa.Schema:
        r"""Data type mapping for reading CSV file with pyarrow.

        This provides a schema,
        defining pyarrow dtypes
        for the columns of a CSV file.

        The dtypes are extracted from the audformat schemes,
        and converted to the pyarrow dtypes.

        Returns:
            pyarrow schema for reading a CSV file

        """
        # Mapping from audformat to pyarrow dtypes
        to_pyarrow_dtype = {
            define.DataType.BOOL: pa.bool_(),
            define.DataType.DATE: pa.timestamp("ns"),
            define.DataType.FLOAT: pa.float64(),
            define.DataType.INTEGER: pa.int64(),
            define.DataType.STRING: pa.string(),
            # A better fitting type would be `pa.duration("ns")`,
            # but this is not yet supported
            # when reading CSV files
            define.DataType.TIME: pa.string(),
        }

        # Collect pyarrow dtypes
        # of all columns,
        # including index columns.
        # The dtypes are stored as a tuple
        # ``(column, dtype)``,
        # and are used to create
        # the pyarrow.Schema
        # used when reading the CSV file
        pyarrow_dtypes = []
        # Index
        for level, dtype in self._levels_and_dtypes.items():
            if dtype in to_pyarrow_dtype:
                pyarrow_dtypes.append((level, to_pyarrow_dtype[dtype]))
        # Columns
        for column_id, column in self.columns.items():
            if column.scheme_id is not None:
                dtype = self.db.schemes[column.scheme_id].dtype
                if dtype in to_pyarrow_dtype:
                    pyarrow_dtypes.append((column_id, to_pyarrow_dtype[dtype]))

        return pa.schema(pyarrow_dtypes)

    def _pyarrow_table_to_dataframe(
        self,
        table: pa.Table,
        *,
        from_csv: bool = False,
    ) -> pd.DataFrame:
        r"""Convert pyarrow table to pandas dataframe.

        Args:
            table: pyarrow table
            from_csv: if ``True`` it assumes,
                that ``table`` was created by reading a CSV file,
                and it will convert all needed dtypes

        Returns:
            dataframe

        """
        df = table.to_pandas(
            deduplicate_objects=False,
            types_mapper={
                pa.string(): pd.StringDtype(),
            }.get,  # we have to provide a callable, not a dict
        )
        # Adjust dtypes and set index
        df = self._pyarrow_convert_dtypes(df, convert_all=from_csv)
        index_columns = list(self._levels_and_dtypes.keys())
        df = self._set_index(df, index_columns)
        return df

    def _save_csv(self, path: str):
        # Load table before opening CSV file
        # to avoid creating a CSV file
        # that is newer than the PKL file
        df = self.df  # loads table
        with open(path, "w") as fp:
            df.to_csv(fp, encoding="utf-8")

    def _save_parquet(self, path: str):
        r"""Save table as PARQUET file.

        A PARQUET file is written in a non-deterministic way,
        and we cannot track changes by its MD5 sum.
        To make changes trackable,
        we store a hash in its metadata.

        The hash is calculated from the pyarrow schema
        (to track column names and data types)
        and the pandas dataframe
        (to track values and order or rows),
        from which the PARQUET file is generated.

        The hash of the PARQUET file can then be read by::

            pyarrow.parquet.read_schema(path).metadata[b"hash"].decode()

        Args:
            path: path, including file extension

        """
        table = pa.Table.from_pandas(self.df.reset_index(), preserve_index=False)

        # Create hash of table
        table_hash = utils.hash(self.df, strict=True)

        # Store in metadata of file,
        # see https://stackoverflow.com/a/58978449
        metadata = {"hash": table_hash}
        table = table.replace_schema_metadata({**metadata, **table.schema.metadata})

        parquet.write_table(table, path, compression="snappy")

    def _save_pickled(self, path: str):
        self.df.to_pickle(
            path,
            protocol=4,  # supported by Python >= 3.4
        )

    def _set_column(self, column_id: str, column: Column) -> Column:
        levels = (
            self.index.names
            if isinstance(self.index, pd.MultiIndex)
            else [self.index.name]
        )
        if column_id in levels:
            raise ValueError(
                f"Cannot add column with ID "
                f"'{column_id}' "
                f"when there is an "
                f"index level with same name. "
                f"Level names are: "
                f"{levels}."
            )

        if column.scheme_id is not None and column.scheme_id not in self.db.schemes:
            raise BadIdError("column", column.scheme_id, self.db.schemes)

        if column.rater_id is not None and column.rater_id not in self.db.raters:
            raise BadIdError("rater", column.rater_id, self.db.raters)

        if column.scheme_id is not None:
            dtype = self.db.schemes[column.scheme_id].to_pandas_dtype()
        else:
            dtype = object

        self.df[column_id] = pd.Series(dtype=dtype)

        column._id = column_id
        column._table = self

        return column

    def _set_index(self, df: pd.DataFrame, columns: Sequence) -> pd.DataFrame:
        r"""Set columns as index.

        Setting of index columns is performed inplace!

        Args:
            df: dataframe
            columns: columns to be set as index of dataframe

        Returns:
            updated dataframe

        """
        # When assigning more than one column,
        # a MultiIndex is assigned.
        # Setting a MultiIndex does not always preserve pandas dtypes,
        # so we need to set them manually.
        #
        if len(columns) > 1:
            dtypes = {column: df[column].dtype for column in columns}
        df.set_index(columns, inplace=True)
        if len(columns) > 1:
            df.index = utils.set_index_dtypes(df.index, dtypes)
        return df


class MiscTable(Base):
    r"""Miscellaneous table.

    .. note:: Intended for use with tables
        that have an index that is not conform to
        :ref:`table specifications <data-tables:Tables>`.
        Otherwise, use :class:`audformat.Table`.

    To fill a table with labels,
    add one or more :class:`audformat.Column`
    and use :meth:`audformat.MiscTable.set` to set the values.
    When adding a column,
    the column ID must be different
    from the index level names.
    When initialized with a single-level
    :class:`pandas.MultiIndex`,
    the index will be converted to a
    :class:`pandas.Index`.

    Args:
        index: table index with non-empty and unique level names
        split_id: split identifier (must exist)
        media_id: media identifier (must exist)
        description: database description
        meta: additional meta fields

    Raises:
        ValueError: if level names of index are empty or not unique

    Examples:
        >>> index = pd.MultiIndex.from_tuples(
        ...     [
        ...         ("f1", "f2"),
        ...         ("f1", "f3"),
        ...         ("f2", "f3"),
        ...     ],
        ...     names=["file", "other"],
        ... )
        >>> index = utils.set_index_dtypes(index, "string")
        >>> table = MiscTable(
        ...     index,
        ...     split_id=define.SplitType.TEST,
        ... )
        >>> table["match"] = Column()
        >>> table
        levels: {file: str, other: str}
        split_id: test
        columns:
          match: {}
        >>> table.get()
                   match
        file other
        f1   f2      NaN
             f3      NaN
        f2   f3      NaN
        >>> table.set({"match": [True, False, True]})
        >>> table.get()
                   match
        file other
        f1   f2     True
             f3    False
        f2   f3     True
        >>> table.get(index[:2])
                   match
        file other
        f1   f2     True
             f3    False
        >>> index_new = pd.MultiIndex.from_tuples(
        ...     [
        ...         ("f4", "f1"),
        ...     ],
        ...     names=["file", "other"],
        ... )
        >>> index_new = utils.set_index_dtypes(index_new, "string")
        >>> table_ex = table.extend_index(
        ...     index_new,
        ...     inplace=False,
        ... )
        >>> table_ex.get()
                    match
        file other
        f1   f2      True
             f3     False
        f2   f3      True
        f4   f1       NaN
        >>> table_ex.set(
        ...     {"match": True},
        ...     index=index_new,
        ... )
        >>> table_ex.get()
                    match
        file other
        f1   f2      True
             f3     False
        f2   f3      True
        f4   f1      True
        >>> table_str = MiscTable(index)
        >>> table_str["strings"] = Column()
        >>> table_str.set({"strings": ["a", "b", "c"]})
        >>> (table + table_str).get()
                    match strings
        file other
        f1   f2      True       a
             f3     False       b
        f2   f3      True       c
        >>> (table_ex + table_str).get()
                    match strings
        file other
        f1   f2      True       a
             f3     False       b
        f2   f3      True       c
        f4   f1      True     NaN

    """

    def __init__(
        self,
        index: pd.Index,
        *,
        split_id: str = None,
        media_id: str = None,
        description: str = None,
        meta: dict = None,
    ):
        self.levels = None
        r"""Index levels."""

        if index is not None:
            # convert single-level pd.MultiIndex to pd.Index
            if isinstance(index, pd.MultiIndex) and index.nlevels == 1:
                index = index.get_level_values(0)

            # Ensure integers are stored as Int64,
            # and bool values as boolean,
            # compare audformat.core.common.to_pandas_dtype()
            index = utils._maybe_convert_pandas_dtype(index)

            levels = utils._levels(index)
            if not all(levels) or len(levels) > len(set(levels)):
                raise ValueError(
                    f"Got index with levels "
                    f"{levels}, "
                    f"but names must be non-empty and unique."
                )
            dtypes = utils._audformat_dtypes(index)
            self.levels = {level: dtype for level, dtype in zip(levels, dtypes)}

        super().__init__(
            index,
            split_id=split_id,
            media_id=media_id,
            description=description,
            meta=meta,
        )

    def _get_by_index(self, index: pd.Index) -> pd.DataFrame:
        return self.df.loc[index]

    @property
    def _levels_and_dtypes(self) -> dict[str, str]:
        r"""Levels and dtypes of index columns.

        Returns:
            dictionary with index levels (column names)
            and associated audformat data type

        """
        return self.levels


class Table(Base):
    r"""Table conform to :ref:`table specifications <data-tables:Tables>`.

    Consists of a list of file names to which it assigns
    numerical values or labels.
    To fill a table with labels,
    add one or more :class:`audformat.Column`
    and use :meth:`audformat.Table.set` to set the values.
    When adding a column,
    the column ID must be different
    from the index level names,
    which are ``'file'``
    in case of a ``filewise`` table
    and ``'file'``, ``'start'`` and ``'end'``
    in case of ``segmented`` table.

    Args:
        index: index conform to
            :ref:`table specifications <data-tables:Tables>`.
            If ``None`` creates an empty filewise table
        split_id: split identifier (must exist)
        media_id: media identifier (must exist)
        description: database description
        meta: additional meta fields

    Raises:
        ValueError: if index not conform to
            :ref:`table specifications <data-tables:Tables>`

    Examples:
        >>> index = filewise_index(["f1", "f2", "f3"])
        >>> table = Table(
        ...     index,
        ...     split_id=define.SplitType.TEST,
        ... )
        >>> table["values"] = Column()
        >>> table
        type: filewise
        split_id: test
        columns:
          values: {}
        >>> table.get()
             values
        file
        f1      NaN
        f2      NaN
        f3      NaN
        >>> table.set({"values": [0, 1, 2]})
        >>> table.get()
             values
        file
        f1        0
        f2        1
        f3        2
        >>> table.get(index[:2])
             values
        file
        f1        0
        f2        1
        >>> table.get(as_segmented=True)
                        values
        file start  end
        f1   0 days NaT      0
        f2   0 days NaT      1
        f3   0 days NaT      2
        >>> index_new = filewise_index("f4")
        >>> table_ex = table.extend_index(
        ...     index_new,
        ...     inplace=False,
        ... )
        >>> table_ex.get()
             values
        file
        f1        0
        f2        1
        f3        2
        f4      NaN
        >>> table_ex.set(
        ...     {"values": 3},
        ...     index=index_new,
        ... )
        >>> table_ex.get()
             values
        file
        f1        0
        f2        1
        f3        2
        f4        3
        >>> table_str = Table(index)
        >>> table_str["strings"] = Column()
        >>> table_str.set({"strings": ["a", "b", "c"]})
        >>> (table + table_str).get()
             values strings
        file
        f1        0       a
        f2        1       b
        f3        2       c
        >>> (table_ex + table_str).get()
             values strings
        file
        f1        0       a
        f2        1       b
        f3        2       c
        f4        3     NaN

    """

    def __init__(
        self,
        index: pd.Index = None,
        *,
        split_id: str = None,
        media_id: str = None,
        description: str = None,
        meta: dict = None,
    ):
        if index is None:
            index = filewise_index()

        index = _maybe_convert_dtype_to_string(index)

        self.type = index_type(index)
        r"""Table type

        See :class:`audformat.define.IndexType`
        for possible values.

        """

        super().__init__(
            index,
            split_id=split_id,
            media_id=media_id,
            description=description,
            meta=meta,
        )

    @property
    def ends(self) -> pd.Index:
        r"""Segment end times.

        Returns:
            timestamps

        """
        if self.is_segmented:
            return self.df.index.get_level_values(define.IndexField.END)
        else:
            return utils.to_segmented_index(self.df.index).get_level_values(
                define.IndexField.END
            )

    @property
    def files(self) -> pd.Index:
        r"""Files referenced in the table.

        Returns:
            files

        """
        # We use len() here as self.df.index.empty takes a very long time
        if len(self.df.index) == 0:
            return filewise_index()
        else:
            index = self.df.index.get_level_values(define.IndexField.FILE)
            index.name = define.IndexField.FILE
            return index

    @property
    def is_filewise(self) -> bool:
        r"""Check if filewise table.

        Returns:
            ``True`` if filewise table.

        """
        return self.type == define.IndexType.FILEWISE

    @property
    def is_segmented(self) -> bool:
        r"""Check if segmented table.

        Returns:
            ``True`` if segmented table.

        """
        return self.type == define.IndexType.SEGMENTED

    @property
    def starts(self) -> pd.Index:
        r"""Segment start times.

        Returns:
            timestamps

        """
        if self.is_segmented:
            return self.df.index.get_level_values(define.IndexField.START)
        else:
            return utils.to_segmented_index(self.df.index).get_level_values(
                define.IndexField.START
            )

    def drop_files(
        self,
        files: (str | Sequence[str] | Callable[[str], bool]),
        *,
        inplace: bool = False,
    ) -> Table:
        r"""Drop files.

        Remove rows with a reference to listed or matching files.

        Args:
            files: list of files or condition function
            inplace: drop files in place

        Returns:
            new object if ``inplace=False``, otherwise ``self``

        """
        if not inplace:
            return self.copy().drop_files(files, inplace=True)

        if isinstance(files, str):
            files = [files]
        if callable(files):
            sel = self.files.to_series().apply(files)
            self._df = self.df[~sel.values]
        else:
            index = self.files.intersection(files)
            index.name = define.IndexField.FILE
            if self.is_segmented:
                level = "file"
            else:
                level = None
            self.df.drop(index, inplace=True, level=level)

        return self

    def get(
        self,
        index: pd.Index = None,
        *,
        map: dict[str, str | Sequence[str]] = None,
        copy: bool = True,
        as_segmented: bool = False,
        allow_nat: bool = True,
        root: str = None,
        num_workers: int | None = 1,
        verbose: bool = False,
    ) -> pd.DataFrame:
        r"""Get labels.

        By default, all labels of the table are returned,
        use ``index`` to get a subset.

        Examples are provided with the
        :ref:`table specifications <data-tables:Tables>`.

        Args:
            index: index conform to
                :ref:`table specifications <data-tables:Tables>`
            copy: return a copy of the labels
            map: :ref:`map scheme or scheme fields to column values
                <map-scheme-labels>`.
                For example if your table holds a column ``speaker`` with
                speaker IDs, which is assigned to a scheme that contains a
                dict mapping speaker IDs to age and gender entries,
                ``map={'speaker': ['age', 'gender']}``
                will replace the column with two new columns that map ID
                values to age and gender, respectively.
                To also keep the original column with speaker IDS, you can do
                ``map={'speaker': ['speaker', 'age', 'gender']}``
            as_segmented: if set to ``True``
                and table has a filewise index,
                the index of the returned table
                will be converted to a segmented index.
                ``start`` will be set to ``0`` and
                ``end`` to ``NaT`` or to the file duration
                if ``allow_nat`` is set to ``False``
            allow_nat: if set to ``False``,
                ``end=NaT`` is replaced with file duration
            root: root directory under which the files are stored.
                Provide if file names are relative and
                database was not saved or loaded from disk.
                If ``None`` :attr:`audformat.Database.root` is used.
                Only relevant if ``allow_nat`` is set to ``False``
            num_workers: number of parallel jobs.
                If ``None`` will be set to the number of processors
                on the machine multiplied by 5
            verbose: show progress bar

        Returns:
            labels

        Raises:
            FileNotFoundError: if file is not found
            RuntimeError: if table is not assign to a database
            ValueError: if trying to map without a scheme
            ValueError: if trying to map from a scheme that has no labels
            ValueError: if trying to map to a non-existing field

        """
        result = super().get(index, map=map, copy=copy)

        # if necessary, convert to segmented index and replace NaT
        is_segmented = is_segmented_index(result.index)
        if (not is_segmented and as_segmented) or (is_segmented and not allow_nat):
            files_duration = None
            if self.db is not None:
                files_duration = self.db._files_duration
                root = root or self.db.root
            new_index = utils.to_segmented_index(
                result.index,
                allow_nat=allow_nat,
                files_duration=files_duration,
                root=root,
                num_workers=num_workers,
                verbose=verbose,
            )
            result = result.set_axis(new_index)

        return result

    def map_files(
        self,
        func: Callable[[str], str],
    ):
        r"""Apply function to file names in table.

        If speed is crucial,
        see :func:`audformat.utils.map_file_path`
        for further hints how to optimize your code.

        Args:
            func: map function

        """
        self.df.index = utils.map_file_path(self.df.index, func)

    def pick_files(
        self,
        files: (str | Sequence[str] | Callable[[str], bool]),
        *,
        inplace: bool = False,
    ) -> Table:
        r"""Pick files.

        Keep only rows with a reference to listed files or matching files.

        Args:
            files: list of files or condition function
            inplace: pick files in place

        Returns:
            new object if ``inplace=False``, otherwise ``self``

        """
        if not inplace:
            return self.copy().pick_files(files, inplace=True)

        if isinstance(files, str):
            files = [files]
        if callable(files):
            sel = self.files.to_series().apply(files)
            self._df = self.df[sel.values]
        else:
            index = self.files.intersection(files)
            index.name = define.IndexField.FILE
            self._df = self.get(index, copy=False)

        return self

    def _get_by_index(
        self,
        index: pd.Index,
    ) -> pd.DataFrame:
        if index_type(self.index) == index_type(index):
            result = self.df.loc[index]
        else:
            files = index.get_level_values(define.IndexField.FILE)
            if self.is_filewise:  # index is segmented
                result = self.df.loc[files]
                result.index = index
            else:  # index is filewise
                files = list(dict.fromkeys(files))  # remove duplicates
                result = self.df.loc[files]

        return result

    @property
    def _levels_and_dtypes(self) -> dict[str, str]:
        r"""Levels and dtypes of index columns.

        Returns:
            dictionary with index levels (column names)
            and associated audformat data type

        """
        levels_and_dtypes = {}
        levels_and_dtypes[define.IndexField.FILE] = define.DataType.STRING
        if self.type == define.IndexType.SEGMENTED:
            levels_and_dtypes[define.IndexField.START] = define.DataType.TIME
            levels_and_dtypes[define.IndexField.END] = define.DataType.TIME
        return levels_and_dtypes


def _assert_table_index(
    table: Base,
    index: pd.Index,
    operation: str,
):
    r"""Raise error if index does not match table."""
    if isinstance(table, Table):
        input_type = index_type(index)
        if table.type != input_type:
            raise ValueError(
                f"Cannot "
                f"{operation} "
                f"a "
                f"{table.type} "
                f"table with a "
                f"{input_type} "
                f"index."
            )
    elif not utils.is_index_alike([table.index, index]):
        want = (
            index.dtypes
            if isinstance(index, pd.MultiIndex)
            else pd.Series(index.dtype, pd.Index([index.name]))
        )
        want = "\n\t".join(want.to_string().split("\n"))

        got = (
            table.index.dtypes
            if isinstance(table.index, pd.MultiIndex)
            else pd.Series(table.index.dtype, pd.Index([table.index.name]))
        )
        got = "\n\t".join(got.to_string().split("\n"))

        raise ValueError(
            f"Cannot "
            f"{operation} "
            f"table if input index and table index are not alike.\n"
            f"Expected index:\n"
            f"\t{want}"
            f"\nbut yours is:\n"
            f"\t{got}"
        )


def _maybe_convert_dtype_to_string(
    index: pd.Index,
) -> pd.Index:
    r"""Possibly set dtype of file level to 'string'."""
    if (is_filewise_index(index) and index.dtype == "object") or (
        is_segmented_index(index) and index.dtypes[define.IndexField.FILE] == "object"
    ):
        index = utils.set_index_dtypes(
            index,
            {define.IndexField.FILE: "string"},
        )
    return index


def _maybe_update_scheme(
    table: Base,
):
    r"""Replace labels if table is used in a scheme."""
    if table.db is not None and isinstance(table, MiscTable):
        for scheme in table.db.schemes.values():
            if table._id == scheme.labels:
                scheme.replace_labels(table._id)
