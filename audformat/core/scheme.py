from __future__ import annotations

import datetime
import random
import string

import pandas as pd

from audformat.core import common
from audformat.core import define


class Scheme(common.HeaderBase):
    r"""A scheme defines valid values of an annotation.

    Allowed values for ``dtype`` are:
    ``'bool'``,
    ``'int'``,
    ``'float'``,
    ``'object'``,
    ``'str'``,
    ``'time'``,
    and ``'date'``
    (see :class:`audformat.define.DataType`).
    Values can be restricted to a set of labels
    provided by a list,
    dictionary
    or a table ID of a :class:`audformat.MiscTable`,
    where the values of the index are used as labels.
    A continuous range can be limited by a minimum and
    maximum value.

    Args:
        dtype: if ``None`` derived from ``labels``,
            otherwise set to ``'str'``
        labels: list, dictionary or table ID
            of a corresponding :class:`audformat.MiscTable`
            containing labels as index.
            If a table ID is provided,
            ``dtype`` has to be specified
        minimum: minimum value
        maximum: maximum value
        description: scheme description
        meta: additional meta fields

    Raises:
        BadValueError: if an invalid ``dtype`` is passed
        ValueError: if ``labels`` are not passed as string, list, or dictionary
        ValueError: if ``labels`` is a table ID,
            but ``dtype`` is not specified
        ValueError: if ``labels`` are not of same data type
        ValueError: ``dtype`` does not match type of ``labels``
            if ``labels`` is a list or dictionary
        ValueError: when assigning a scheme,
            that contains a table ID as ``labels``,
            to a database,
            but the corresponding misc table is not part of the database,
            or the given table ID is not a misc table,
            or its index is multi-dimensional,
            or its index contains duplicates,
            or ``dtype`` does not match type of labels
            from misc table,
            or ``dtype`` is set to ``bool``,
            or the misc table has a column
            that is already assigned to a scheme
            with labels from another misc table

    Examples:
        >>> Scheme()
        {dtype: str}
        >>> Scheme(labels=["a", "b", "c"])
        dtype: str
        labels: [a, b, c]
        >>> Scheme(define.DataType.INTEGER)
        {dtype: int}
        >>> Scheme("float", minimum=0, maximum=1)
        {dtype: float, minimum: 0, maximum: 1}
        >>> # Use index of misc table as labels
        >>> import audformat
        >>> db = audformat.Database("mydb")
        >>> db["speaker"] = audformat.MiscTable(pd.Index(["spk1", "spk2"], name="speaker"))
        >>> Scheme("str", labels="speaker")
        {dtype: str, labels: speaker}

    """  # noqa: E501

    # Mapping for dtype input argument,
    # e.g. to allow `str` besides `'str'`.
    # This behavior is only for convenience
    # and not mentioned in the docstring
    _dtypes = {
        "bool": define.DataType.BOOL,
        bool: define.DataType.BOOL,
        "str": define.DataType.STRING,
        str: define.DataType.STRING,
        "int": define.DataType.INTEGER,
        int: define.DataType.INTEGER,
        "float": define.DataType.FLOAT,
        float: define.DataType.FLOAT,
        "time": define.DataType.TIME,
        pd.Timedelta: define.DataType.TIME,
        "date": define.DataType.DATE,
        datetime.datetime: define.DataType.DATE,
        pd.Timestamp: define.DataType.DATE,
    }

    def __init__(
        self,
        dtype: str = None,
        *,
        labels: dict | list | str = None,
        minimum: int | float = None,
        maximum: int | float = None,
        description: str = None,
        meta: dict = None,
    ):
        super().__init__(description=description, meta=meta)

        self._db = None
        self._id = None

        if dtype is not None:
            if dtype in self._dtypes:
                dtype = self._dtypes[dtype]
            define.DataType._assert_has_attribute_value(dtype)

        if dtype is None and labels is None:
            dtype = define.DataType.STRING

        if labels is not None:
            self._check_labels(labels)

            if isinstance(labels, str):
                # Labels from misc table
                if dtype is None:
                    raise ValueError(
                        "'dtype' has to be provided "
                        "when using a misc table as labels."
                    )
                if dtype == define.DataType.BOOL:
                    raise ValueError(
                        "'dtype' cannot be 'bool' " "when using a misc table as labels."
                    )
            else:
                # Labels from list or dictionary
                dtype_labels = self._dtype_from_labels(labels)
                if dtype is not None and dtype != dtype_labels:
                    raise ValueError(
                        "Data type is set to "
                        f"'{dtype}', "
                        "but data type of labels is "
                        f"'{dtype_labels}'."
                    )
                dtype = dtype_labels

        self.dtype = dtype
        r"""Data type.

        Possible return values are given by
        :class:`audformat.define.DataType`.

        """
        self.labels = labels
        r"""Labels or ID of misc table holding the labels"""
        self.minimum = minimum if self.is_numeric else None
        r"""Minimum value"""
        self.maximum = maximum if self.is_numeric else None
        r"""Maximum value"""

    @property
    def is_numeric(self) -> bool:
        r"""Data type is numeric.

        Returns:
            ``True`` if data type is numeric

        """
        return self.dtype in (define.DataType.INTEGER, define.DataType.FLOAT)

    @property
    def labels_as_list(self) -> list:
        r"""Scheme labels as list.

        If scheme does not define labels
        an empty list is returned.

        Returns:
            list of labels

        """
        if self.labels is None:
            return []
        else:
            return self._labels_to_list(self.labels)

    @property
    def uses_table(self) -> bool:
        r"""Scheme has labels stored in a misc table.

        If property is ``True``
        the attribute ``labels``
        is set to an ID of a
        :class:`audformat.MiscTable`
        where the actual label
        values are stored.

        Returns:
            ``True`` if scheme has labels stored in a misc table

        """
        return isinstance(self.labels, str)

    def draw(
        self,
        n: int,
        *,
        str_len: int = 10,
        p_none: bool = None,
    ) -> list:
        r"""Randomly draws values from scheme.

        Args:
            n: number of values
            str_len: string length if drawing from a string scheme without
                labels
            p_none: probability for drawing an invalid value

        Returns:
            list with values

        """
        x = None
        if self.labels is None:
            if self.dtype == define.DataType.BOOL:
                x = [random.choice([False, True]) for _ in range(n)]
            elif self.dtype == define.DataType.DATE:
                x = [
                    pd.to_datetime(round(random.random(), 2), unit="s")
                    for _ in range(n)
                ]
            elif self.dtype == define.DataType.INTEGER:
                minimum = self.minimum or 0
                maximum = self.maximum or minimum + 100
                x = [random.randrange(minimum, maximum) for _ in range(n)]
            elif self.dtype == define.DataType.FLOAT:
                minimum = self.minimum or 0.0
                maximum = self.maximum or minimum + 1.0
                x = [random.uniform(minimum, maximum) for _ in range(n)]
            elif self.dtype == define.DataType.TIME:
                x = [
                    pd.to_timedelta(round(random.random(), 2), unit="s")
                    for _ in range(n)
                ]
            else:
                seq = string.ascii_letters + string.digits
                x = [
                    "".join([random.choice(seq) for _ in range(str_len)])
                    for _ in range(n)
                ]
        else:
            labels = self._labels_to_list()
            x = [random.choice(labels) for _ in range(n)]

        if p_none is not None:
            for idx in range(len(x)):
                if random.random() <= p_none:
                    x[idx] = None

        return x

    def to_pandas_dtype(
        self,
    ) -> str | pd.api.types.CategoricalDtype:
        r"""Convert data type to :mod:`pandas` data type.

        If ``labels`` is not ``None``, :class:`pandas.CategoricalDtype` is
        returned. Otherwise the following rules are applied:

        * ``str`` -> ``str``
        * ``int`` -> ``Int64``  (to allow NaN)
        * ``float`` -> ``float``
        * ``time`` -> ``timedelta64[ns]``
        * ``date`` -> ``datetime64[ns]``

        Returns:
            :mod:`pandas` data type

        """
        if self.labels is not None:
            labels = self._labels_to_list()
            if len(labels) > 0 and isinstance(labels[0], int):
                # allow nullable
                labels = pd.array(labels, dtype="int64")
            dtype = pd.api.types.CategoricalDtype(
                categories=labels,
                ordered=False,
            )
        else:
            dtype = common.to_pandas_dtype(self.dtype)
        return dtype

    def replace_labels(
        self,
        labels: dict | list | str,
    ):
        r"""Replace labels.

        If scheme is part of a :class:`audformat.Database`
        the dtype of all :class:`audformat.Column` objects
        that reference the scheme will be updated.
        Removed labels are set to ``NaN``.

        Args:
            labels: new labels

        Raises:
            ValueError: if scheme does not define labels
            ValueError: if dtype of new labels does not match dtype of
                scheme
            ValueError: if ``labels`` is a misc table ID
                and the scheme is already assigned to a database,
                but the corresponding misc table is not part of the database,
                or the given table ID is not a misc table,
                or its index is multi-dimensional,
                or its index contains duplicates,
                or the misc table has a column
                that is already assigned to a scheme
                with labels from another misc table

        Examples:
            >>> speaker = Scheme(
            ...     labels={
            ...         0: {"gender": "female"},
            ...         1: {"gender": "male"},
            ...     }
            ... )
            >>> speaker
            dtype: int
            labels:
              0: {gender: female}
              1: {gender: male}
            >>> speaker.replace_labels(
            ...     {
            ...         1: {"gender": "male", "age": 33},
            ...         2: {"gender": "female", "age": 44},
            ...     }
            ... )
            >>> speaker
            dtype: int
            labels:
              1: {gender: male, age: 33}
              2: {gender: female, age: 44}

        """
        if self.labels is None:
            raise ValueError(
                "Cannot replace labels when " "scheme does not define labels."
            )
        self._check_labels(labels)

        if not isinstance(labels, str) or self._db is not None:
            # Check change of data type
            # for list, dict and assigned misc table
            dtype_labels = self._dtype_from_labels(labels)
            if dtype_labels != self.dtype:
                raise ValueError(
                    "Data type of labels must not change: \n"
                    f"'{self.dtype}' \n"
                    f"!=\n"
                    f"'{dtype_labels}'"
                )

        self.labels = labels

        if self._db is not None and self._id is not None:
            labels = self._labels_to_list(labels)
            for table in list(self._db.tables.values()) + list(
                self._db.misc_tables.values()
            ):
                for column in table.columns.values():
                    if column.scheme_id == self._id:
                        y = column._table.df[column._id]
                        y = y.cat.set_categories(
                            new_categories=labels,
                            ordered=False,
                        )
                        column._table.df[column._id] = y

    def _check_labels(
        self,
        labels: dict | list | str,
    ):
        r"""Raise label related errors."""
        if not isinstance(labels, (dict, list, str)):
            raise ValueError(
                "Labels must be passed " "as a dictionary, list or ID of a misc table."
            )

        if self._db is not None and isinstance(labels, str):
            table_id = labels
            if table_id not in self._db:
                raise ValueError(
                    f"The misc table '{table_id}' used as scheme labels "
                    "needs to be assigned to the database."
                )
            if table_id not in self._db.misc_tables:
                raise ValueError(
                    f"The table '{table_id}' used as scheme labels "
                    "needs to be a misc table."
                )
            for column in self._db.misc_tables[table_id].columns.values():
                if column.scheme_id is not None:
                    scheme = self._db.schemes[column.scheme_id]
                    if scheme.uses_table:
                        raise ValueError(
                            f"The misc table "
                            f"'{table_id}' "
                            f"cannot be used as scheme labels "
                            f"when one of its columns is "
                            f"assigned to a scheme that "
                            f"uses labels from a misc table."
                        )
            if self._db[table_id].index.nlevels > 1:
                raise ValueError(
                    f"Index of misc table '{table_id}' used as scheme labels "
                    "is only allowed to have a single level."
                )
            if sum(self._db[table_id].index.duplicated()) > 0:
                raise ValueError(
                    f"Index of misc table '{table_id}' used as scheme labels "
                    "is not allowed to contain duplicates."
                )
            dtype_labels = self._dtype_from_labels(labels)
            if self.dtype != dtype_labels:
                raise ValueError(
                    "Data type is set to "
                    f"'{self.dtype}', "
                    "but data type of labels in misc table is "
                    f"'{dtype_labels}'."
                )

    def _dtype_from_labels(
        self,
        labels: dict | list | str,
    ) -> str:
        r"""Derive audformat dtype from labels."""
        if isinstance(labels, str):
            # misc table
            # dtype is stored in the levels dictionary
            # as audformat dtypes.
            # In addition, a misc table used as labels
            # is only allowed to have a one dimensional index,
            # so we need to get only the first entry of the dict.
            levels = self._db[labels].levels
            dtype = next(iter(levels.values()))
        else:
            # dict or list
            labels = self._labels_to_list(labels)
            if len(labels) > 0:
                dtype = type(labels[0])
            else:
                dtype = "str"
            if not all(isinstance(x, dtype) for x in labels):
                raise ValueError("All labels must be of the same data type.")
            dtype = common.to_audformat_dtype(dtype)
        define.DataType._assert_has_attribute_value(dtype)

        return dtype

    def _labels_to_dict(
        self,
        labels: dict | list | str = None,
    ) -> dict:
        r"""Return actual labels as dict."""
        if labels is None:
            labels = self.labels
        if isinstance(labels, str):
            if self._db is None or labels not in self._db:
                labels = {}
            else:
                labels = self._db[labels].df.to_dict("index")
        elif isinstance(labels, list):
            labels = {label: {} for label in labels}
        return labels

    def _labels_to_list(
        self,
        labels: dict | list | str = None,
    ) -> list:
        r"""Convert labels to actual labels as list."""
        return list(self._labels_to_dict(labels))

    def __contains__(self, item: object) -> bool:
        r"""Check if scheme contains data type of item.

        ``None``, ``NaT`` and ``NaN`` always match

        Returns:
            ``True`` if item is covered by scheme

        """
        if item is not None and not pd.isna(item):
            if self.labels is not None:
                labels = self._labels_to_dict()
                return item in labels
            if self.is_numeric:
                item = float(item)
                if self.minimum is not None and not item >= self.minimum:
                    return False
                if self.maximum is not None and not item <= self.maximum:
                    return False
        return True
