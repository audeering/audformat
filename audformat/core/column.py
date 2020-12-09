import typing

import pandas as pd

from audformat.core import define
from audformat.core.index import (
    index_type,
    is_scalar,
    to_array,
)
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
            index: pd.Index = None,
            *,
            copy: bool = True,
    ) -> pd.Series:
        r"""Get a copy of the labels in this column.

        Examples are provided with the
        :ref:`table specifications <data-tables:Tables>`.

        You can use ``index`` or a any allowed combinations of ``files``,
        ``starts``, and ``ends`` as arguments.

        Args:
            index: index conform to
                :ref:`table specifications <data-tables:Tables>`
            copy: return a new object

        Raises:
            ColumnNotAssignToTable: if column is not assign to a table

        """
        if self._table is None:
            raise ColumnNotAssignedToTableError()

        result = self._table.get(index, copy=False)[self._id]
        return result.copy() if copy else result

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
            ColumnNotAssignToTable: if column is not assign to a table

        """
        if self._table is None:
            raise ColumnNotAssignedToTableError()

        column_id = self._id
        df = self._table.df

        if index is None:
            index = df.index

        if index_type(df.index) == index_type(index):
            if is_scalar(values):
                values = [values] * len(index)
            values = to_array(values)
            df.loc[index, column_id] = pd.Series(
                values,
                index=index,
                dtype=df[column_id].dtype,
            )
        else:
            if not self._table.is_filewise:
                files = index.get_level_values(define.IndexField.FILE)
                index = df.loc[files].index
                return self.set(values, index=index)
            else:
                raise ValueError(
                    'Cannot set value to a filewise column '
                    'using a segmented index.'
                )
