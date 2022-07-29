from __future__ import annotations  # allow typing without string

import typing

import numpy as np
import pandas as pd

from audformat.core import define
from audformat.core.common import HeaderBase
from audformat.core.index import (
    index_type,
    is_scalar,
    to_array,
)
from audformat.core.rater import Rater
from audformat.core.typing import Values

if typing.TYPE_CHECKING:
    # Fix to make mypy work without circular imports,
    # compare
    # https://adamj.eu/tech/2021/05/13/python-type-hints-how-to-fix-circular-imports/
    from audformat.core.scheme import Scheme  # pragma: nocover


def assert_values(
        values: Values,
        scheme: Scheme,
):
    r"""Raise error if values do not match scheme."""
    ok = True

    if scheme.labels is not None or \
            scheme.minimum is not None or \
            scheme.maximum is not None:

        if is_scalar(values):
            ok = values in scheme
        else:
            if isinstance(values, pd.Series):
                values = values.values
            if isinstance(values, np.ndarray):
                if scheme.is_numeric:
                    values = [
                        np.min(values),
                        np.max(values),
                    ]
            values = list(set(values))
            ok = all([value in scheme for value in values])

    if not ok:
        raise ValueError(
            f'Some value(s) do not match scheme:\n{scheme}.'
        )


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

    @property
    def rater(self) -> typing.Optional[Rater]:
        r"""Rater object.

        Returns:
            rater object or ``None`` if not available

        """
        if (
                self.rater_id is not None
        ) and (
                self.table is not None
        ) and (
                self.table.db is not None
        ):
            return self.table.db.raters[self.rater_id]

    @property
    def scheme(self) -> typing.Optional[Scheme]:
        r"""Scheme object.

        Returns:
            scheme object or ``None`` if not available

        """
        if (
                self.scheme_id is not None
        ) and (
                self.table is not None
        ) and (
                self.table.db is not None
        ):
            return self.table.db.schemes[self.scheme_id]

    @property
    def table(self):
        r"""Table object.

        Returns:
            table object or ``None`` if not assigned yet

        """
        return self._table

    def get(
            self,
            index: pd.Index = None,
            *,
            map: str = None,
            copy: bool = True,
            as_segmented: bool = False,
            allow_nat: bool = True,
            root: str = None,
            num_workers: typing.Optional[int] = 1,
            verbose: bool = False,
    ) -> pd.Series:
        r"""Get labels.

        By default, all labels of the column are returned,
        use ``index`` to get a subset.

        Examples are provided with the
        :ref:`table specifications <data-tables:Tables>`.

        Args:
            index: index conform to
                :ref:`table specifications <data-tables:Tables>`
            copy: return a copy of the labels
            map: :ref:`map scheme or scheme field to column values
                <map-scheme-labels>`.
                For example if your column holds speaker IDs and is
                assigned to a scheme that contains a dict mapping
                speaker IDs to age entries, ``map='age'``
                will replace the ID values with the age of the speaker
            as_segmented: if set to ``True``
                and column has a filewise index,
                the index of the returned column
                will be converted to a segmented index.
                ``start`` will be set to ``0`` and
                ``end`` to ``NaT`` or to the file duration
                if ``allow_nat`` is set to ``False``.
                If column belongs to a miscellaneous table,
                this and the following arguments have no effect
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
            RuntimeError: if column is not assigned to a table
            ValueError: if trying to map without a scheme,
                or from a scheme that has no labels,
                or from a scheme that has only a list of labels,
                or to a non-existing field

        """
        if self._table is None:
            raise RuntimeError(
                'Column is not assigned to a table.'
            )

        if hasattr(self._table, 'type'):
            result = self._table.get(
                index,
                copy=False,
                as_segmented=as_segmented,
                allow_nat=allow_nat,
                root=root,
                num_workers=num_workers,
                verbose=verbose,
            )
        else:
            result = self._table.get(
                index,
                copy=False,
            )
        result = result[self._id]

        if map is not None:

            copy = False  # to avoid another copy

            if self.scheme_id is None:
                raise ValueError(
                    f"Column '{self._id}' is not assigned to a scheme."
                )

            scheme = self._table._db.schemes[self.scheme_id]
            labels = scheme._labels_to_dict()

            if labels is None:
                raise ValueError(
                    f"Scheme '{self.scheme_id}' has no labels."
                )

            if not any(labels.values()):
                raise ValueError(
                    f"Scheme '{self.scheme_id}' provides no mapping "
                    "for its labels."
                )

            mapping = {}
            for key, value in labels.items():
                if isinstance(value, dict):
                    if map in value:
                        value = value[map]
                    else:
                        raise ValueError(
                            f"Cannot map "
                            f"'{self._id}' "
                            f"to "
                            f"'{map}'. "
                            f"Expected one of "
                            f"{list(value)}."
                        )
                mapping[key] = value
            result = result.map(mapping)
            result.name = map

        return result.copy() if copy else result

    def set(
            self,
            values: Values,
            *,
            index: pd.Index = None,
    ):
        r"""Set labels.

        By default, all labels of the column are replaced,
        use ``index`` to set a subset.
        If columns is assigned to a :class:`Scheme`
        values will be automatically converted
        to match its dtype.

        Examples are provided with the
        :ref:`table specifications <data-tables:Tables>`.

        Args:
            values: list of values
            index: index conform to
                :ref:`table specifications <data-tables:Tables>`

        Raises:
            RuntimeError: if column is not assign to a table
            ValueError: if trying to set values of a filewise column
                using a segmented index
            ValueError: if values cannot be converted
                to match the schemes dtype

        """
        if self._table is None:
            raise RuntimeError(
                'Column is not assigned to a table.'
            )

        column_id = self._id
        df = self._table.df

        if index is None:
            index = df.index

        if self.scheme_id is not None:
            scheme = self._table._db.schemes[self.scheme_id]
            assert_values(values, scheme)
            dtype = scheme.to_pandas_dtype()
        else:
            dtype = df[column_id].dtype

        if hasattr(self._table, 'type') and \
                self._table.type != index_type(index):
            # special case where a filewise / segmented table
            # is requested with an index of the other type
            if not self._table.is_filewise:
                files = index.get_level_values(define.IndexField.FILE)
                index = df.loc[files].index
                return self.set(values, index=index)
            else:
                raise ValueError(
                    'Cannot set values of a filewise column '
                    'using a segmented index.'
                )
        else:
            if is_scalar(values):
                values = [values] * len(index)
            values = to_array(values)
            df.loc[index, column_id] = pd.Series(
                values,
                index=index,
                dtype=dtype,
            )

    def __eq__(
            self,
            other: 'Column',
    ) -> bool:
        if self.dump() != other.dump():
            return False
        if self._table is not None and other._table is not None:
            return self._table.df[self._id].equals(other._table.df[other._id])
        return self._table is None and other._table is None
