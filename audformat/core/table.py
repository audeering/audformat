from collections.abc import Iterable
import os
import typing

import numpy as np
import pandas as pd

import audeer

from audformat.core import define
from audformat.core import utils
from audformat.core.column import Column
from audformat.core.common import (
    HeaderBase,
    HeaderDict,
)
from audformat.core.errors import (
    BadIdError,
)
from audformat.core.index import (
    filewise_index,
    segmented_index,
    index_type,
)
from audformat.core.typing import (
    Values,
)


def index_to_dict(index: pd.Index) -> dict:
    r"""Convert :class:`pandas.Index` to a dictionary.

    Returns a dictionary with keys files, starts, and ends.

    """
    d = dict(
        [
            (define.IndexField.FILE + 's', None),
            (define.IndexField.START + 's', None),
            (define.IndexField.END + 's', None),
        ]
    )

    d[define.IndexField.FILE + 's'] = index.get_level_values(
        define.IndexField.FILE).values
    if index_type(index) == define.IndexType.SEGMENTED:
        d[define.IndexField.START + 's'] = index.get_level_values(
            define.IndexField.START).values.astype(np.timedelta64)
        d[define.IndexField.END + 's'] = index.get_level_values(
            define.IndexField.END).values.astype(np.timedelta64)

    return d


class Table(HeaderBase):
    r"""Table with annotation data.

    Consists of a list of file names to which it assigns
    numerical values or labels.
    To fill a table with labels,
    add one ore more :class:`audformat.Column`
    and use :meth:`audformat.Table.set` to set the values.

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

    Example:
        >>> index = filewise_index(['f1', 'f2', 'f3'])
        >>> table = Table(
        ...     index,
        ...     split_id=define.SplitType.TEST,
        ... )
        >>> table['values'] = Column()
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
        >>> table.set({'values': [0, 1, 2]})
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
        >>> index_ex = filewise_index('f4')
        >>> table_ex = table.extend_index(
        ...     index_ex,
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
        ...     {'values': 3},
        ...     index=index_ex,
        ... )
        >>> table_ex.get()
             values
        file
        f1        0
        f2        1
        f3        2
        f4        3
        >>> table_str = Table(index)
        >>> table_str['strings'] = Column()
        >>> table_str.set({'strings': ['a', 'b', 'c']})
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
    TYPE = 'type'

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

        if index is None:
            index = filewise_index()

        self.type = index_type(index)
        r"""Table type"""
        self.split_id = split_id
        r"""Split ID"""
        self.media_id = media_id
        r"""Media ID"""
        self.columns = HeaderDict(
            sorted_iter=False,
            value_type=Column,
            set_callback=self._set_column,
        )
        r"""Table columns"""

        self._df = pd.DataFrame(index=index)
        self._db = None
        self._id = None

    @property
    def df(self) -> pd.DataFrame:
        r"""Table data.

        Returns:
            data

        """
        return self._df

    @property
    def ends(self) -> pd.Index:
        r"""Segment end times.

        Returns:
            timestamps

        """
        if self.is_segmented:
            return self.df.index.get_level_values(
                define.IndexField.END
            )
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
        if self.df.index.empty:
            return filewise_index()
        else:
            index = self.df.index.get_level_values(define.IndexField.FILE)
            index.name = define.IndexField.FILE
            return index

    @property
    def index(self) -> pd.Index:
        r"""Table index.

        Returns:
            index

        """
        return self._df.index

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
            return self.df.index.get_level_values(
                define.IndexField.START
            )
        else:
            return utils.to_segmented_index(self.df.index).get_level_values(
                define.IndexField.START
            )

    def copy(self) -> 'Table':
        r"""Copy table.

        Return:
            new ``Table`` object

        """
        table = Table(
            self._df.index,
            media_id=self.media_id,
            split_id=self.split_id,
        )
        table._db = self._db
        for column_id, column in self.columns.items():
            table.columns[column_id] = Column(
                scheme_id=column.scheme_id,
                rater_id=column.rater_id,
                description=column.description,
                meta=column.meta.copy()
            )
        table._df = self._df.copy()
        return table

    def drop_columns(
            self,
            column_ids: typing.Union[str, typing.Sequence[str]],
            *,
            inplace: bool = False,
    ) -> 'Table':
        r"""Drop columns by ID.

        Args:
            column_ids: column IDs
            inplace: drop columns in place

        Returns:
            new ``Table`` if ``inplace=False``, otherwise ``self``

        """
        if not inplace:
            return self.copy().drop_columns(column_ids, inplace=True)

        if isinstance(column_ids, str):
            column_ids = [column_ids]
        column_ids_ = set()
        for column_id in column_ids:
            column_ids_.add(column_id)
        self.df.drop(column_ids_, inplace=True, axis='columns')
        for column_id in column_ids_:
            self.columns.pop(column_id)

        return self

    def drop_files(
            self,
            files: typing.Union[
                str,
                typing.Sequence[str],
                typing.Callable[[str], bool],
            ],
            *,
            inplace: bool = False,
    ) -> 'Table':
        r"""Drop files.

        Remove rows with a reference to listed or matching files.

        Args:
            files: list of files or condition function
            inplace: drop files in place

        Returns:
            new ``Table`` if ``inplace=False``, otherwise ``self``

        """
        if not inplace:
            return self.copy().drop_files(files, inplace=True)

        if isinstance(files, str):
            files = [files]
        if callable(files):
            sel = self.files.to_series().apply(files)
            self._df = self._df[~sel.values]
        else:
            index = self.files.intersection(files)
            index.name = define.IndexField.FILE
            self._df.drop(index, inplace=True)

        return self

    def drop_index(
            self,
            index: pd.Index,
            *,
            inplace: bool = False,
    ) -> 'Table':
        r"""Drop rows from index.

        Args:
            index: index conform to
                :ref:`table specifications <data-tables:Tables>`
            inplace: drop index in place

        Returns:
            new ``Table`` if ``inplace=False``, otherwise ``self``

        Raises:
            ValueError: if table type is not matched

        """
        if not inplace:
            return self.copy().drop_index(index, inplace=True)

        input_type = index_type(index)
        if self.type != input_type:
            raise ValueError(
                'It is not possible to drop a '
                f'{input_type} index '
                'from a '
                f'{self.type} '
                'index'
            )
        new_index = self._df.index.difference(index)
        self._df = self._df.reindex(new_index)

        return self

    def extend_index(
            self,
            index: pd.Index,
            *,
            fill_values: typing.Union[
                typing.Any, typing.Dict[str, typing.Any]
            ] = None,
            inplace: bool = False,
    ) -> 'Table':
        r"""Extend table by new rows.

        Args:
            index: index conform to
                :ref:`table specifications <data-tables:Tables>`
            fill_values: replace NaN with these values (either a scalar
                applied to all columns or a dictionary with column name as
                key)
            inplace: extend index in place

        Returns:
            new ``Table`` if ``inplace=False``, otherwise ``self``

        Raises:
            ValueError: if index type is not matched

        """
        if not inplace:
            return self.copy().extend_index(
                index, fill_values=fill_values, inplace=True,
            )

        input_type = index_type(index)
        if self.type != input_type:
            raise ValueError(
                f'Cannot extend a '
                f'{self.type} '
                f'table with a '
                f'{input_type} '
                f'index.'
            )
        new_index = self._df.index.union(index)
        self._df = self._df.reindex(new_index)
        if fill_values is not None:
            if isinstance(fill_values, dict):
                for key, value in fill_values.items():
                    self._df[key].fillna(value, inplace=True)
            else:
                self._df.fillna(fill_values, inplace=True)

        return self

    def get(
            self,
            index: pd.Index = None,
            *,
            map: typing.Dict[
                str, typing.Union[str, typing.Sequence[str]]
            ] = None,
            copy: bool = True,
    ) -> pd.DataFrame:
        r"""Get labels.

        By default all labels of the table are returned,
        use ``index`` to get a subset.

        Examples are provided with the
        :ref:`table specifications <data-tables:Tables>`.

        Args:
            index: index conform to
                :ref:`table specifications <data-tables:Tables>`
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
            RuntimeError: if table is not assign to a database
            ValueError: if trying to map without a scheme
            ValueError: if trying to map from a scheme that has no labels
            ValueError: if trying to map to a non-existing field

        """
        result_is_copy = False

        if index is None:
            result = self._df
        else:
            if index_type(self.index) == index_type(index):
                result = self._df.loc[index]
            else:
                files = index.get_level_values(define.IndexField.FILE)
                if self.is_filewise:  # index is segmented
                    result = pd.DataFrame(
                        self._df.loc[files].values,
                        index,
                        columns=self.columns
                    )
                    result_is_copy = True  # to avoid another copy
                else:  # index is filewise
                    files = list(dict.fromkeys(files))  # remove duplicates
                    result = self._df.loc[files]

        if map is not None:

            if self._db is None:
                raise RuntimeError(
                    'Table is not assigned to a database.'
                )

            if not result_is_copy:
                result = result.copy()
                result_is_copy = True  # to avoid another copy

            for column, mapped_columns in map.items():
                mapped_columns = audeer.to_list(mapped_columns)
                if len(mapped_columns) == 1:
                    result[mapped_columns[0]] = self.columns[column].get(
                        index, map=mapped_columns[0],
                    )
                else:
                    for mapped_column in mapped_columns:
                        if mapped_column != column:
                            result[mapped_column] = self.columns[column].get(
                                index, map=mapped_column,
                            )
                if column not in mapped_columns:
                    result.drop(columns=column, inplace=True)

        return result.copy() if (copy and not result_is_copy) else result

    def load(
            self,
            path: str,
    ):
        r"""Load table data from disk.

        Args:
            path: file path without extension

        """
        path = audeer.safe_path(path)
        pickled = os.path.exists(path + f'.{define.TableStorageFormat.PICKLE}')
        if pickled:
            self._load_pickled(path + f'.{define.TableStorageFormat.PICKLE}')
        else:
            self._load_csv(path + f'.{define.TableStorageFormat.CSV}')

    def pick_columns(
            self,
            column_ids: typing.Union[str, typing.Sequence[str]],
            *,
            inplace: bool = False,
    ) -> 'Table':
        r"""Pick columns by ID.

        All other columns will be dropped.

        Args:
            column_ids: column IDs
            inplace: pick columns in place

        Returns:
            new ``Table`` if ``inplace=False``, otherwise ``self``

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
    ) -> 'Table':
        r"""Pick rows from index.

        Args:
            index: index conform to
                :ref:`table specifications <data-tables:Tables>`
            inplace: pick index in place

        Returns:
            new ``Table`` if ``inplace=False``, otherwise ``self``

        Raises:
            ValueError: if table type is not matched

        """
        if not inplace:
            return self.copy().pick_index(index, inplace=True)

        input_type = index_type(index)
        if self.type != input_type:
            raise ValueError(
                'It is not possible to pick a '
                f'{input_type} '
                'index from a '
                f'{self.type} '
                f'index'
            )
        new_index = self._df.index.intersection(index)
        self._df = self._df.reindex(new_index)

        return self

    def pick_files(
            self,
            files: typing.Union[
                str,
                typing.Sequence[str],
                typing.Callable[[str], bool],
            ],
            *,
            inplace: bool = False,
    ) -> 'Table':
        r"""Pick files.

        Keep only rows with a reference to listed files or matching files.

        Args:
            files: list of files or condition function
            inplace: pick files in place

        Returns:
            new ``Table`` if ``inplace=False``, otherwise ``self``

        """
        if not inplace:
            return self.copy().pick_files(files, inplace=True)

        if isinstance(files, str):
            files = [files]
        if callable(files):
            sel = self.files.to_series().apply(files)
            self._df = self._df[sel.values]
        else:
            index = self.files.intersection(files)
            index.name = define.IndexField.FILE
            self._df = self.get(index, copy=False)

        return self

    def save(
            self,
            path: str,
            *,
            storage_format: define.TableStorageFormat = (
                define.TableStorageFormat.CSV
            ),
    ):
        r"""Save table data to disk.

        Args:
            path: file path without extension
            storage_format: storage format of tables.
                See :class:`audformat.define.TableStorageFormat`
                for available formats

        """
        path = audeer.safe_path(path)
        define.TableStorageFormat.assert_has_value(storage_format)
        if storage_format == define.TableStorageFormat.PICKLE:
            self._df.to_pickle(
                path + f'.{define.TableStorageFormat.PICKLE}',
                compression='xz',
            )
        else:
            with open(path + f'.{define.TableStorageFormat.CSV}', 'w') as fp:
                self.df.to_csv(fp, encoding='utf-8')

    def set(
            self,
            values: typing.Union[
                typing.Dict[str, Values],
                pd.DataFrame,
            ],
            *,
            index: pd.Index = None,
    ):
        r"""Set labels.

        By default all labels of the table are replaced,
        use ``index`` to select a subset.
        If a column is assigned to a :class:`Scheme`
        values have to match its ``dtype``.

        Examples are provided with the
        :ref:`table specifications <data-tables:Tables>`.

        Args:
            values: dictionary of values with ``column_id`` as key
            index: index conform to
                :ref:`table specifications <data-tables:Tables>`

        Raises:
            ValueError: if values do not match scheme

        """
        for idx, data in values.items():
            self.columns[idx].set(data, index=index)

    def __add__(self, other: 'Table') -> 'Table':
        r"""Create new table by combining two tables.

        If the tables are of the same type, the created tables contain index
        and columns of both tables.
        If the tables are not of the same type and one of the tables is a
        file-wise table, the created table has the index of segmented table
        and columns of both tables.
        Missing values will be set to ``NaN``.
        References to schemes and raters are always preserved.
        Media and split information only when they match.

        .. warning::
            Columns with the same identifier are combined to a single column.
            This requires that either the indices of the tables do not
            overlap or at least one table contains a ``NaN`` in places where
            the indices overlap.

        Args:
            other: the other table

        Raises:
            ValueError: if overlapping values are detected that are not ``NaN``

        """
        if self == other:
            return self

        df_self = self._df
        df_other = other._df
        df_other_copy = False

        if other.type != self.type:

            def add_files_to_dict(d, additional_files, table_type):
                if not additional_files.empty:
                    d['files'] = np.r_[d['files'], additional_files.values]
                    if table_type == define.IndexType.SEGMENTED:
                        d_append = index_to_dict(
                            utils.to_segmented_index(additional_files)
                        )
                        d['starts'] = np.r_[d['starts'], d_append['starts']]
                        d['ends'] = np.r_[d['ends'], d_append['ends']]

            if other.type == define.IndexType.FILEWISE:
                missing_files = other.files.unique().difference(
                    self.files.unique()
                )
                d = index_to_dict(self._df.index)
                add_files_to_dict(d, missing_files, self.type)
                df_other = other._df.reindex(d['files'])
                df_other.set_index(segmented_index(**d), inplace=True)
            elif self.type == define.IndexType.FILEWISE:
                missing_files = self.files.unique().difference(
                    other.files.unique()
                )
                d = index_to_dict(other._df.index)
                add_files_to_dict(d, missing_files, other.type)
                df_self = self._df.reindex(d['files'])
                df_self.set_index(segmented_index(**d), inplace=True)

        # figure out column names, schemes and raters
        df_columns = []
        scheme_ids = {}
        rater_ids = {}
        for column_id, column in self.columns.items():
            df_columns.append(column_id)
            scheme_ids[column_id] = column.scheme_id
            rater_ids[column_id] = column.rater_id
        for column_id, column in other.columns.items():
            if column_id in df_self.columns:
                if not df_self.index.intersection(df_other.index).empty:
                    if not df_other_copy:
                        df_other = df_other.copy()
                        df_other_copy = True
                    df_other.update(df_self, errors='raise')
            else:
                df_columns.append(column_id)
                scheme_ids[column_id] = column.scheme_id
                rater_ids[column_id] = column.rater_id

        # concatenate frames
        df = utils.concat([df_self, df_other])
        df.columns = df_columns

        # create table
        media_id = self.media_id if self.media_id == other.media_id else None
        split_id = self.split_id if self.split_id == other.split_id else None
        table = Table(df.index, media_id=media_id, split_id=split_id)
        table._db = self._db
        for column_id in df_columns:
            table[column_id] = Column(
                scheme_id=scheme_ids[column_id],
                rater_id=rater_ids[column_id],
            )
        table._df = df

        return table

    def __getitem__(self, column_id: str) -> Column:
        r"""Return view to a column.

        Args:
            column_id: column identifier

        """
        return self.columns[column_id]

    def __len__(self) -> int:
        return len(self._df)

    def __setitem__(self, column_id: str, column: Column) -> Column:
        r"""Add new column to table.

        Args:
            column_id: column identifier
            column: column

        Raises:
            BadIdError: if a column with a ``scheme_id`` or ``rater_id`` is
                added that does not exist

        """
        self.columns[column_id] = column
        return column

    def _load_csv(self, path: str):

        usecols = []
        dtypes = {}
        converters = {}
        schemes = self._db.schemes

        # index columns

        if self.type == define.IndexType.SEGMENTED:
            converters[define.IndexField.START] = \
                lambda x: pd.to_timedelta(x)
            converters[define.IndexField.END] = \
                lambda x: pd.to_timedelta(x)
            index_col = [define.IndexField.FILE,
                         define.IndexField.START,
                         define.IndexField.END]
        else:
            index_col = [define.IndexField.FILE]

        usecols.extend(index_col)

        # other columns

        for column_id, column in self.columns.items():
            usecols.append(column_id)
            if column.scheme_id is not None:
                dtype = schemes[column.scheme_id].to_pandas_dtype()
                # if column contains timestamps
                # we have to convert them later
                if dtype == 'timedelta64[ns]':
                    converters[column_id] = lambda x: pd.to_timedelta(x)
                elif dtype == 'datetime64[ns]':
                    converters[column_id] = lambda x: pd.to_datetime(x)
                else:
                    dtypes[column_id] = dtype
            else:
                dtypes[column_id] = 'str'

        # read csv

        df = pd.read_csv(
            path,
            usecols=usecols,
            dtype=dtypes,
            index_col=index_col,
            converters=converters,
        )

        self._df = df

    def _load_pickled(self, path: str):

        df = pd.read_pickle(path, compression='xz')
        for column_id in df:
            # Categories of type Int64 are somehow converted to int64.
            # We have to change back to Int64 to make column nullable.
            if isinstance(df[column_id].dtype,
                          pd.core.dtypes.dtypes.CategoricalDtype):
                if isinstance(df[column_id].dtype.categories,
                              pd.core.indexes.numeric.Int64Index):
                    labels = df[column_id].dtype.categories.values
                    labels = pd.array(labels, dtype=pd.Int64Dtype())
                    dtype = pd.api.types.CategoricalDtype(
                        categories=labels,
                        ordered=False)
                    df[column_id] = df[column_id].astype(dtype)

        self._df = df

    def _set_column(self, column_id: str, column: Column) -> Column:
        if column.scheme_id is not None and \
                column.scheme_id not in self._db.schemes:
            raise BadIdError('column', column.scheme_id, self._db.schemes)
        if column.rater_id is not None and \
                column.rater_id not in self._db.raters:
            raise BadIdError('rater', column.rater_id, self._db.raters)

        if column.scheme_id is not None:
            dtype = self._db.schemes[column.scheme_id].to_pandas_dtype()
        else:
            dtype = object
        self._df[column_id] = pd.Series(dtype=dtype)

        #  if table is empty we need to fix index names
        if self._df.empty:
            if self.is_filewise:
                self._df.index.name = define.IndexField.FILE
            elif self.is_segmented:
                self._df.index.rename(
                    [
                        define.IndexField.FILE,
                        define.IndexField.START,
                        define.IndexField.END
                    ],
                    inplace=True,
                )

        column._id = column_id
        column._table = self
        return column
