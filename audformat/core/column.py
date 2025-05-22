from __future__ import annotations

import itertools
import typing
import warnings

import numpy as np
import pandas as pd

from audformat.core import define
from audformat.core.common import HeaderBase
from audformat.core.common import to_audformat_dtype
from audformat.core.common import to_pandas_dtype
from audformat.core.index import index_type
from audformat.core.index import is_scalar
from audformat.core.index import to_array
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
    error_msg = ""

    if (
        scheme.labels is not None
        or scheme.minimum is not None
        or scheme.maximum is not None
    ):
        if is_scalar(values):
            values = [values]
        elif isinstance(values, pd.Series):
            values = values.values
        values = [v for v in values if v is not None and not pd.isna(v)]
        if not values:
            return

    if scheme.labels is not None:
        bad_values = set(values) - set(scheme.labels_as_list)
        if len(bad_values) > 0:
            # Convert only `max_display` entries from set to list
            max_display = 10
            show_bad_values = sorted(
                [v for v in itertools.islice(bad_values, max_display)]
            )
            error_msg = str(show_bad_values)[1:-1]
            if len(bad_values) > max_display:
                error_msg += ", ..."
            error_msg += "\n"

    if scheme.is_numeric:
        if scheme.minimum is not None:
            min_value = min(values)
            if float(min_value) < scheme.minimum:
                error_msg += f"minimum {min_value} smaller than scheme minimum\n"
        if scheme.maximum is not None:
            max_value = max(values)
            if float(max_value) > scheme.maximum:
                error_msg += f"maximum {max_value} larger than scheme maximum\n"

    if error_msg:
        raise ValueError(
            f"Some value(s) do not match scheme\n{scheme}\n"
            f"with scheme ID '{scheme._id}':\n"
            f"{error_msg}"
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

    Examples:
        >>> Column(scheme_id="emotion")
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
    def rater(self) -> Rater | None:
        r"""Rater object.

        Returns:
            rater object or ``None`` if not available

        """
        if (
            (self.rater_id is not None)
            and (self.table is not None)
            and (self.table.db is not None)
        ):
            return self.table.db.raters[self.rater_id]

    @property
    def scheme(self) -> Scheme | None:
        r"""Scheme object.

        Returns:
            scheme object or ``None`` if not available

        """
        if (
            (self.scheme_id is not None)
            and (self.table is not None)
            and (self.table.db is not None)
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
        num_workers: int | None = 1,
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
            raise RuntimeError("Column is not assigned to a table.")

        if hasattr(self._table, "type"):
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
                raise ValueError(f"Column '{self._id}' is not assigned to a scheme.")

            scheme = self._table._db.schemes[self.scheme_id]
            labels = scheme._labels_to_dict()

            if labels is None:
                raise ValueError(f"Scheme '{self.scheme_id}' has no labels.")

            if not any(labels.values()):
                raise ValueError(
                    f"Scheme '{self.scheme_id}' provides no mapping " "for its labels."
                )

            # Check that at least one key is available for map
            # if labels are stored as dictionary
            keys = []
            for key, value in labels.items():
                if isinstance(value, dict):
                    keys += list(value.keys())
            keys = sorted(list(set(keys)))
            if len(keys) > 0 and map not in keys:
                raise ValueError(
                    f"Cannot map "
                    f"'{self._id}' "
                    f"to "
                    f"'{map}'. "
                    f"Expected one of "
                    f"{list(keys)}."
                )

            mapping = {}
            for key, value in labels.items():
                if isinstance(value, dict):
                    if map in value:
                        value = value[map]
                    else:
                        value = np.nan
                mapping[key] = value

            result = result.map(mapping)
            result.name = map

            if (
                scheme.uses_table
                and self._table._db[scheme.labels][map].scheme is not None
                #                       ^           ^
                #                   misc table   column
            ):
                # Infer dtype from misc table
                misc_table_id = scheme.labels
                column = self._table._db[misc_table_id][map]
                dtype = column.scheme.to_pandas_dtype()
            else:
                # Infer dtype from actual labels
                dtype = pd.api.types.infer_dtype(list(result.values))
                dtype = to_pandas_dtype(to_audformat_dtype(dtype))

            result = result.astype(dtype)

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
            raise RuntimeError("Column is not assigned to a table.")

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

        if hasattr(self._table, "type") and self._table.type != index_type(index):
            # special case where a filewise / segmented table
            # is requested with an index of the other type
            if not self._table.is_filewise:
                files = index.get_level_values(define.IndexField.FILE)
                index = df.loc[files].index
                return self.set(values, index=index)
            else:
                raise ValueError(
                    "Cannot set values of a filewise column " "using a segmented index."
                )
        else:
            if is_scalar(values):
                values = [values] * len(index)
            values = to_array(values)
            if dtype == "datetime64[ns]":
                # Ensure all date values are timezone unaware,
                # see https://github.com/audeering/audformat/issues/364
                values = [
                    pd.to_datetime(value).tz_localize(None)
                    if value is not None
                    else value
                    for value in values
                ]
            with warnings.catch_warnings():
                # Avoid FutureWarning and DeprecationWarning
                # for pandas 1.5.0 to 1.5.3
                # for setting values in place
                # as introduced at
                # https://pandas.pydata.org/docs/dev/whatsnew/v1.5.0.html#inplace-operation-when-setting-values-with-loc-and-iloc
                # For pandas >=2.0.0 values are always set in place
                for warning in [FutureWarning, DeprecationWarning]:
                    warnings.simplefilter(action="ignore", category=warning)
                df.loc[index, column_id] = pd.Series(
                    values,
                    index=index,
                    dtype=dtype,
                )

    def __eq__(
        self,
        other: Column,
    ) -> bool:
        r"""Compare if column equals another column."""
        if self.dump() != other.dump():
            return False
        if self._table is not None and other._table is not None:
            return self._table.df[self._id].equals(other._table.df[other._id])
        return self._table is None and other._table is None
