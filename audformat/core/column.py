import typing

import pandas as pd

from audformat.core import define
from audformat.core import utils
from audformat.core.index import index_type
from audformat.core.common import HeaderBase
from audformat.core.errors import ColumnNotAssignedToTableError


class Column(HeaderBase):
    r"""Represents a table column (see :class:`audformat.Table`) and
    optionally links it to a scheme (see :class:`audformat.Scheme`) and
    a rater (see :class:`audformat.Rater`).

    Args:
        scheme_id: scheme identifier (must exist)
        rater_id: rater identifier (must exist)
        description: table description
        meta: additional meta fields

    """
    def __init__(
            self,
            *,
            scheme_id: str = None,
            rater_id: str = None,
            description: str = None,
            meta: dict = None,
    ):
        super().__init__(description=description, meta=meta)

        self.scheme_id = scheme_id
        r"""Scheme identifier"""
        self.rater_id = rater_id
        r"""Rater identifier"""
        self._table = None
        self._id = None

    def get(
            self,
            index: typing.Union[pd.Index, pd.Series, pd.DataFrame] = None,
    ) -> pd.Series:
        r"""Get a copy of the labels in this column.

        Examples are provided with the
        :ref:`table specifications <data-tables:Tables>`.

        You can use ``index`` or a any allowed combinations of ``files``,
        ``starts``, and ``ends`` as arguments.

        Args:
            index: index conform to
                :ref:`table specifications <data-tables:Tables>`

        Raises:
            RedundantArgumentError: for not allowed combinations
                of input arguments
            ColumnNotAssignToTable: if column is not assign to a table

        """
        if self._table is None:
            raise ColumnNotAssignedToTableError()

        return self._table._get(index)[self._id].copy()

    def set(
            self,
            values: define.Typing.VALUES,
            *,
            index: pd.Index = None,
    ):
        r"""Set labels in this column.

        Examples are provided with the
        :ref:`table specifications <data-tables:Tables>`.

        You can use ``starts`` and ``ends`` as arguments.

        Args:
            values: a list of values
                matching the ``dtype`` of the corresponding :class:`Scheme`
            index: index conform to Unified Format

        Raises:
            RedundantArgumentError: for not allowed combinations
                of input arguments
            ColumnNotAssignToTable: if column is not assign to a table

        """
        if self._table is None:
            raise ColumnNotAssignedToTableError()

        self._set(self._table.df, self._id, values, index)

    def _set(
            self,
            df: pd.DataFrame,
            column_id: str,
            values: define.Typing.VALUES,
            index: pd.Index,
    ):
        if utils.is_scalar(values):
            if index is None:
                values = [values] * len(self._table)
            else:
                values = [values] * len(index)
        values = utils.to_array(values)
        if index is None:
            index = df.index
        df.loc[index, column_id] = pd.Series(
            values,
            index=index,
            dtype=df[column_id].dtype,
        )

        # if isinstance(values, pd.Series):
        #     if index_type(values) is not None:
        #         utils.check_redundant_arguments(
        #             files=files, starts=starts, ends=ends,
        #         )
        #         # Get values, files, starts, ends from pd.Series
        #         return self._set(
        #             df, column_id, **utils.series_to_dict(values)
        #         )
        #
        # if not utils.is_scalar(values):
        #     values = utils.to_array(values)
        # files = utils.to_array(files)
        # starts = utils.to_array(starts)
        # ends = utils.to_array(ends)
        #
        # if files is None:
        #     if utils.is_scalar(values):
        #         values = [values] * len(self._table)
        #     df.loc[:, column_id] = pd.Series(
        #         values,
        #         index=df.index,
        #         dtype=df[column_id].dtype,
        #     )
        # else:
        #     if self._table.is_segmented \
        #             and starts is not None \
        #             and ends is not None:
        #         idx = pd.MultiIndex.from_arrays(
        #             [files, starts, ends],
        #             names=[define.IndexField.FILE,
        #                    define.IndexField.START,
        #                    define.IndexField.END])
        #         index = df.loc[idx, column_id].index
        #         if utils.is_scalar(values):
        #             values = [values] * len(index)
        #         df.loc[idx, column_id] = pd.Series(
        #             values,
        #             index=index,
        #             dtype=df[column_id].dtype,
        #         )
        #     else:
        #         files = utils.remove_duplicates(files)
        #         index = df.loc[files].index
        #         if utils.is_scalar(values):
        #             values = [values] * len(index)
        #         df.loc[files, column_id] = pd.Series(
        #             values,
        #             index=index,
        #             dtype=df[column_id].dtype,
        #         )
