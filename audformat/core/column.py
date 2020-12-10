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
from audformat.core.typing import Values


class Column(HeaderBase):
    r"""Table column.

    Represents a table column (see :class:`audformat.Table`) and
    optionally links it to a scheme (see :class:`audformat.Scheme`) and
    a rater (see :class:`audformat.Rater`).

    Args:
        scheme_id: scheme identifier (must exist)
        rater_id: rater identifier (must exist)
        description: table description
        meta: additional meta fields

    Example:
        >>> Column(scheme_id='emotion')
        {scheme_id: emotion}

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
        r"""Get labels.

        By default all labels of the column are returned,
        use ``index`` to get a subset.

        Examples are provided with the
        :ref:`table specifications <data-tables:Tables>`.

        Args:
            index: index conform to
                :ref:`table specifications <data-tables:Tables>`
            copy: return a copy of the labels

        Returns:
            labels

        Raises:
            ColumnNotAssignToTable: if column is not assign to a table


        """
        if self._table is None:
            raise ColumnNotAssignedToTableError()

        result = self._table.get(index, copy=False)[self._id]
        return result.copy() if copy else result

    def set(
            self,
            values: Values,
            *,
            index: pd.Index = None,
    ):
        r"""Set labels.

        By default all labels of the column are replaced,
        use ``index`` to set a subset.
        If columns is assigned to a :class:`Scheme`
        values have to match its ``dtype``.

        Examples are provided with the
        :ref:`table specifications <data-tables:Tables>`.

        Args:
            values: list of values
            index: index conform to
                :ref:`table specifications <data-tables:Tables>`

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
                    'Cannot set values of a filewise column '
                    'using a segmented index.'
                )
