import datetime
import random
import string
import typing

import pandas as pd

from audformat.core import define
from audformat.core.common import HeaderBase


class Scheme(HeaderBase):
    r"""A scheme defines valid values of an annotation.

    Allowed values for ``dtype`` are:
    ``'bool'``, ``'str'``, ``'int'``, ``'float'``, ``'time'``, and ``'date'``
    (see :class:`audformat.define.DataType`).
    Values can be restricted to a set of labels provided by a
    list or a dictionary.
    A continuous range can be limited by a minimum and
    maximum value.

    Args:
        dtype: if ``None`` derived from ``labels``, otherwise set to ``'str'``
        labels: list or dictionary with valid labels.
        minimum: minimum value
        maximum: maximum value
        description: scheme description
        meta: additional meta fields

    Raises:
        BadValueError: if an invalid ``dtype`` is passed
        ValueError: if ``labels`` are not passed as list or dictionary
        ValueError: if ``labels`` are not of same data type
        ValueError: ``dtype`` does not match type of ``labels``

    Example:
        >>> Scheme()
        {dtype: str}
        >>> Scheme(labels=['a', 'b', 'c'])
        dtype: str
        labels: [a, b, c]
        >>> Scheme(define.DataType.INTEGER)
        {dtype: int}
        >>> Scheme(float, minimum=0, maximum=1)
        {dtype: float, minimum: 0, maximum: 1}

    """
    _dtypes = {
        'bool': define.DataType.BOOL,
        bool: define.DataType.BOOL,
        'str': define.DataType.STRING,
        str: define.DataType.STRING,
        'int': define.DataType.INTEGER,
        int: define.DataType.INTEGER,
        'float': define.DataType.FLOAT,
        float: define.DataType.FLOAT,
        'time': define.DataType.TIME,
        pd.Timedelta: define.DataType.TIME,
        'date': define.DataType.DATE,
        datetime.datetime: define.DataType.DATE,
    }

    def __init__(
            self,
            dtype: typing.Union[typing.Type, define.DataType] = None,
            *,
            labels: typing.Union[dict, list] = None,
            minimum: typing.Union[int, float] = None,
            maximum: typing.Union[int, float] = None,
            description: str = None,
            meta: dict = None,
    ):
        super().__init__(description=description, meta=meta)

        if dtype is not None:
            if dtype in self._dtypes:
                dtype = self._dtypes[dtype]
            define.DataType.assert_has_attribute_value(dtype)

        if dtype is None and labels is None:
            dtype = define.DataType.STRING

        if labels is not None and len(labels) > 0:
            if not isinstance(labels, (dict, list)):
                raise ValueError(
                    'Labels must be passed as a dictionary or a list.'
                )
            derived_dtype = type(list(labels)[0])
            if not all(isinstance(x, derived_dtype) for x in list(labels)):
                raise ValueError(
                    'All labels must be of the same data type.'
                )
            if derived_dtype in self._dtypes:
                derived_dtype = self._dtypes[derived_dtype]
            define.DataType.assert_has_attribute_value(derived_dtype)
            if dtype is not None:
                if dtype != derived_dtype:
                    raise ValueError(
                        "Data type is set to "
                        f"'{dtype}', "
                        "but data type of labels is "
                        f"'{derived_dtype}'."
                    )
            dtype = derived_dtype

        self.dtype = dtype
        r"""Data type"""
        self.labels = labels
        r"""List of labels"""
        self.minimum = minimum if self.is_numeric else None
        r"""Minimum value"""
        self.maximum = maximum if self.is_numeric else None
        r"""Maximum value"""

    @property
    def is_numeric(self) -> bool:
        r"""Check if data type is numeric.

        Returns:
            ``True`` if data type is numeric.

        """
        return self.dtype in (define.DataType.INTEGER, define.DataType.FLOAT)

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
                x = [pd.to_datetime(round(random.random(), 2), unit='s')
                     for _ in range(n)]
            elif self.dtype == define.DataType.INTEGER:
                minimum = self.minimum or 0
                maximum = self.maximum or minimum + 100
                x = [random.randrange(minimum, maximum)
                     for _ in range(n)]
            elif self.dtype == define.DataType.FLOAT:
                minimum = self.minimum or 0.0
                maximum = self.maximum or minimum + 1.0
                x = [random.uniform(minimum, maximum) for _ in range(n)]
            elif self.dtype == define.DataType.TIME:
                x = [pd.to_timedelta(round(random.random(), 2), unit='s')
                     for _ in range(n)]
            else:
                seq = string.ascii_letters + string.digits
                x = [''.join([random.choice(seq) for _ in range(str_len)])
                     for _ in range(n)]
        elif type(self.labels) in (list, dict):
            x = [random.choice(list(self.labels)) for _ in range(n)]

        if p_none is not None:
            for idx in range(len(x)):
                if random.random() <= p_none:
                    x[idx] = None

        return x

    def to_pandas_dtype(self) -> typing.Union[
        str, pd.api.types.CategoricalDtype,
    ]:
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
            labels = list(self.labels)
            if len(labels) > 0 and isinstance(labels[0], int):
                # allow nullable
                labels = pd.array(labels, dtype=pd.Int64Dtype())
            return pd.api.types.CategoricalDtype(
                categories=labels, ordered=False,
            )
        elif self.dtype == define.DataType.BOOL:
            return 'boolean'
        elif self.dtype == define.DataType.DATE:
            return 'datetime64[ns]'
        elif self.dtype == define.DataType.INTEGER:
            return 'Int64'
        elif self.dtype == define.DataType.TIME:
            return 'timedelta64[ns]'
        return self.dtype

    def __contains__(self, item: typing.Any) -> bool:
        r"""Check if scheme contains data type of item.

        ``None``, ``NaT`` and ``NaN`` always match

        Returns:
            ``True`` if item is covered by scheme

        """
        if item is not None and not pd.isna(item):
            if self.labels is not None:
                return item in self.labels
            if self.is_numeric:
                if self.minimum and not item >= self.minimum:
                    return False
                if self.maximum and not item <= self.maximum:
                    return False
        return True
