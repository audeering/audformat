import os
import typing

import numpy as np
import pandas as pd

from audformat.core import define
from audformat.core import utils
from audformat.core.index import index_type
from audformat.core.index import index as create_index
from audformat.core.column import Column
from audformat.core.common import (
    HeaderBase,
    HeaderDict,
)
from audformat.core.errors import (
    BadIdError,
    BadIndexTypeError,
)


class Table(HeaderBase):
    r"""Table with annotation data.

    Consists of a list of file names to which it assigns
    numerical values or labels.
    To add data to the table,
    add a column first (see :class:`audformat.Column`) and afterwards
    use `set()` and `set_conf()` to set the values.

    Args:
        index: index conform to
            :ref:`table specifications <data-tables:Tables>`.
            If ``None`` creates an empty filewise table
        split_id: split identifier (must exist)
        media_id: media identifier (must exist)
        description: database description
        meta: additional meta fields

    Raises:
        NotConformToUnifiedFormat: if index is not conform to Unified Format

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
            index = create_index([])

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
            return utils.to_segmented(self.df.index).get_level_values(
                define.IndexField.END
            )

    @property
    def files(self) -> pd.Index:
        r"""Files referenced in the table.

        Returns:
            files

        """
        if self.df.index.empty:
            return create_index([])
        else:
            return self.df.index.get_level_values(define.IndexField.FILE)

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
            return utils.to_segmented(self.df.index).get_level_values(
                define.IndexField.START
            )

    def drop(
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
            table

        """
        if not inplace:
            return self._copy().drop(column_ids, inplace=True)

        if isinstance(column_ids, str):
            column_ids = [column_ids]
        column_ids_ = set()
        for column_id in column_ids:
            column_ids_.add(column_id)
        self.df.drop(column_ids_, inplace=True, axis='columns')
        for column_id in column_ids_:
            self.columns.pop(column_id)

        return self

    def extend(
            self,
            index: typing.Union[pd.Index, pd.Series, pd.DataFrame],
            *,
            fill_values: typing.Union[
                typing.Any, typing.Dict[str, typing.Any]
            ] = None,
    ):
        r"""Extend table by new rows.

        Args:
            index: index conform to
                :ref:`table specifications <data-tables:Tables>`
            fill_values: replace NaN with these values (either a scalar
                applied to all columns or a dictionary with column name as
                key)

        Raises:
            ValueError: if table type is not matched

        """
        table_type = index_type(index)
        if self.type != table_type:
            raise ValueError(
                f'Cannot extend a '
                f'{self.type} '
                f'table with a '
                f'{table_type} '
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

    def from_frame(
            self,
            frame: typing.Union[pd.Series, pd.DataFrame],
            *,
            scheme_ids: typing.Union[str, typing.Dict[str, str]] = None,
            rater_ids: typing.Union[str, typing.Dict[str, str]] = None,
    ):
        r"""Adds columns of a :class:`pandas.DataFrame` to the table.

        Column names are used as identifiers. If the index of ``frame``
        follows the :ref:`table specifications <data-tables:Tables>`,
        it will be used as index.

        Args:
            frame: table data
            scheme_ids: a dictionary mapping a specific column to a scheme
                identifier. If a single scheme identifier is passed, all
                columns will be assigned to it
            rater_ids: a dictionary mapping a specific column to a rater
                identifier. If a single rater identifier is passed, all
                columns will be assigned to it

        """
        if isinstance(frame, pd.Series):
            frame = frame.to_frame()

        if isinstance(scheme_ids, str):
            scheme_ids = {name: scheme_ids for name in frame.columns}
        if isinstance(rater_ids, str):
            rater_ids = {name: rater_ids for name in frame.columns}

        scheme_ids = scheme_ids or {}
        rater_ids = rater_ids or {}

        for column_id, column in frame.items():
            scheme_id = scheme_ids[column_id] \
                if column_id in scheme_ids else None
            rater_id = rater_ids[column_id] \
                if column_id in rater_ids else None
            self[column_id] = Column(scheme_id=scheme_id,
                                     rater_id=rater_id)
            self[column_id].set(column.values, index=frame.index)

    def get(
            self,
            index: typing.Union[pd.Index, pd.Series, pd.DataFrame] = None,
    ) -> pd.DataFrame:
        r"""Get a copy of the labels in this table.

        Examples are provided with the
        :ref:`table specifications <data-tables:Tables>`.

        Args:
            index: index conform to
                :ref:`table specifications <data-tables:Tables>`

        Returns:
            table data

        Raises:
            RedundantArgumentError: for not allowed combinations
                of input arguments

        """
        return self._get(index).copy()

    def load(
            self,
            root: str,
            name: str,
    ):
        r"""Load table data from disk.

        Args:
            root: root directory
            name: file name

        """
        compressed = os.path.exists(os.path.join(root, name + '.pkl'))

        if compressed:
            self._load_compressed(os.path.join(root, name + '.pkl'))
        else:
            self._load_csv(os.path.join(root, name + '.csv'))

    def pick(
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
            table

        """
        if isinstance(column_ids, str):
            column_ids = [column_ids]
        drop_ids = set()
        for column_id in list(self.columns):
            if column_id not in column_ids:
                drop_ids.add(column_id)
        return self.drop(list(drop_ids), inplace=inplace)

    def save(
            self,
            root: str,
            name: str,
            compressed: bool = False,
    ):
        r"""Save table data to disk.

        Args:
            root: root directory
            name: file name, if ``None`` set to ``<id>_<version>``.
            compressed: store tables in compressed format instead of CSV

        """
        if compressed:
            self._df.to_pickle(
                os.path.join(root, name + '.pkl'), compression='xz')
        else:
            with open(os.path.join(root, name + '.csv'), 'w') as fp:
                self.df.to_csv(fp, encoding='utf-8')

    def set(
            self,
            values: define.Typing.VALUES,
            *,
            index: pd.Index = None,
    ):
        r"""Set labels in this table.

        Examples are provided with the
        :ref:`table specifications <data-tables:Tables>`.

        Args:
            values: dictionary of values with ``column_id`` as key
            index: index conform to Unified Format

        Raises:
            RedundantArgumentError: for not allowed combinations
                of input arguments

        """
        self._set(values, index)

    def __add__(self, other: 'Table') -> 'Table':
        r""" Creates a new table by adding two tables.

        If the tables are of the same type, the created tables contains index
        and columns of both tables. If the tables are not of the same type
        and one of the tables is a file-wise table, the created table has the
        index of segmented table and columns of both tables.
        References to schemes and raters are always preserved. Media and split
        information only when they match.

        .. warning::
            * Columns with the same identifier are combined to a single column.
              This requires that either the indices of the tables do not
              overlap or at least one table contains a NaN in places where the
              indices overlap.

        Args:
            other: the other table

        Raises:
            ValueError: Overlapping non-NA data is detected

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
                        d_append = utils.index_to_dict(
                            utils.to_segmented(additional_files)
                        )
                        d['starts'] = np.r_[d['starts'], d_append['starts']]
                        d['ends'] = np.r_[d['ends'], d_append['ends']]

            if other.type == define.IndexType.FILEWISE:
                missing_files = other.files.unique().difference(
                    self.files.unique()
                )
                d = utils.index_to_dict(self._df.index)
                add_files_to_dict(d, missing_files, self.type)
                df_other = other.get(create_index(**d))
            elif self.type == define.IndexType.FILEWISE:
                missing_files = self.files.unique().difference(
                    other.files.unique()
                )
                d = utils.index_to_dict(other._df.index)
                add_files_to_dict(d, missing_files, other.type)
                df_self = self.get(create_index(**d))

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
        table.from_frame(df, scheme_ids=scheme_ids, rater_ids=rater_ids)

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

    def _copy(self) -> 'Table':
        table = Table(
            self._df.index,
            media_id=self.media_id,
            split_id=self.split_id,
        )
        table._db = self._db
        table.from_frame(self._df,
                         scheme_ids={k: v.scheme_id
                                     for k, v in self.columns.items()},
                         rater_ids={k: v.rater_id
                                    for k, v in self.columns.items()})
        return table

    def _get(
            self,
            index: typing.Optional[pd.Index],
    ) -> pd.DataFrame:

        if index is None:
            return self._df

        input_type = index_type(index)
        table_type = index_type(self.index)

        if table_type == input_type:
            return self._df.loc[index]
        else:
            files = index.get_level_values(define.IndexField.FILE)
            if self.is_filewise:
                reindex = self._df.reindex(files)
                return pd.DataFrame(
                    reindex.values, index, columns=self.columns,
                )
            elif type == define.IndexType.FILEWISE:
                return self._df.loc[files]

        raise BadIndexTypeError(
            str(input_type), [define.IndexType.FILEWISE, str(table_type)],
        )

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

    def _load_compressed(self, path: str):

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

    def _set(self, values, index):
        # if isinstance(values, pd.DataFrame):
        #     if index_type(values) is not None:
        #         utils.check_redundant_arguments(files=files, starts=starts,
        #                                         ends=ends)
        #         # Get values, files, starts, ends from pd.DataFrame
        #         return self._set(**utils.frame_to_dict(values))
        for idx, data in values.items():
            self.columns[idx].set(data, index=index)

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
                self._df.index.rename([define.IndexField.FILE,
                                       define.IndexField.START,
                                       define.IndexField.END],
                                      inplace=True)

        column._id = column_id
        column._table = self
        return column
