from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from collections.abc import Sequence
import inspect
import os
import textwrap

import oyaml as yaml
import pandas as pd

from audformat import define
from audformat.core.errors import BadKeyError
from audformat.core.errors import BadTypeError
from audformat.core.errors import BadValueError


class HeaderDict(OrderedDict):
    r"""Custom implementation of a dictionary.

    * by default ``items()`` returns a sorted iterator (see ``sorted_iter``)
    * prints the dictionary in a prettified way:

      .. code-block::

          key-1:
            value-1
          key-2:
            value-2

    Args:
        *args: default arguments
        sort_by_key: if ``True`` maintain a sorted order,
            otherwise order by assignment
        value_type: only accept values of this type
        get_callback: call this function when an item is requested
        set_callback: call this function when an item is added
        **kwargs: default keyword arguments

    Raises:
        BadTypeError: if a value of an invalid type is added

    Examples:
        >>> HeaderDict()

        >>> d = HeaderDict()
        >>> d["b"] = 1
        >>> d["a"] = 0
        >>> d
        a:
          0
        b:
          1
        >>> d = HeaderDict(sort_by_key=False)
        >>> d["b"] = 1
        >>> d["a"] = 0
        >>> d
        b:
          1
        a:
          0
        >>> HeaderDict(
        ...     get_callback=lambda key, value: value + value,
        ...     set_callback=lambda key, value: value.upper(),
        ...     foo="bar",
        ... )
        foo:
          BARBAR

    """

    def __init__(
        self,
        *args,
        sort_by_key: bool = True,
        value_type: type = None,
        get_callback: Callable = None,
        set_callback: Callable = None,
        **kwargs,
    ):
        self.sort_by_key = sort_by_key
        r"""Sort items by key"""
        self.value_type = value_type
        r"""Only accept values of this type"""
        self.get_callback = get_callback
        r"""Callback function when item is requested"""
        self.set_callback = set_callback
        r"""Callback function when item is added"""

        super().__init__(*args, **kwargs)

    def __getitem__(self, key):
        if key not in self:
            raise BadKeyError(key, list(self))
        value = super().__getitem__(key)
        if self.get_callback is not None:
            value = self.get_callback(key, value)
        return value

    def __iter__(self):
        if self.sort_by_key:
            return sorted(super().keys()).__iter__()
        else:
            return super().__iter__()

    def __repr__(self) -> str:
        return self.dump()

    def __reversed__(self):
        if self.sort_by_key:
            return sorted(super().keys()).__reversed__()
        else:
            return super().__reversed__()

    def __setitem__(self, key, value):
        if self.set_callback is not None:
            value = self.set_callback(key, value)
        if self.value_type is not None:
            if not isinstance(value, self.value_type):
                raise BadTypeError(value, self.value_type)
        super().__setitem__(key, value)

    def dump(self) -> str:
        if not self:
            return ""
        else:
            return "\n".join(
                [
                    "{}:\n{}".format(key, textwrap.indent(str(self[key]), "  "))
                    for key in self
                ]
            )

    def items(self):
        if self.sort_by_key:
            return iter(sorted(super().items()))
        else:
            return super().items()

    def keys(self):
        return iter([key for key, _ in self.items()])

    def popitem(self, last: bool = True):
        if len(self) > 0 and self.sort_by_key:
            keys = list(self)
            if last:
                key = keys[-1]
            else:
                key = keys[0]
            value = super().pop(key)
            return key, value
        else:
            return super().popitem(last)

    def values(self):
        return iter([value for _, value in self.items()])


class HeaderBase:
    r"""Base class for header objects.

    Args:
        description: description string
        meta: dictionary with meta fields

    """

    def __init__(
        self,
        *,
        description: str = None,
        meta: dict = None,
    ):
        self.description = description
        r"""Description"""
        self.meta = None
        r"""Dictionary with meta fields"""
        if meta is not None:
            self.meta = HeaderDict(meta)
        else:
            self.meta = HeaderDict()

    @staticmethod
    def _value(value):
        if isinstance(value, list):
            return [HeaderBase._value(v) for v in value]
        elif dict in inspect.getmro(value.__class__):
            d = OrderedDict()
            for key, v in value.items():
                d[key] = HeaderBase._value(v)
            return d
        elif HeaderBase in inspect.getmro(value.__class__):
            return value.to_dict()
        else:
            return value

    @staticmethod
    def _dict_filter(key, value) -> bool:
        if key == "meta":
            return False
        if key.startswith("_"):
            return False
        if value is None:
            return False
        if isinstance(value, HeaderDict) and not value:
            return False
        return True

    def to_dict(self) -> dict:
        r"""Serialize object to dictionary.

        Returns:
            dictionary with attributes

        """
        d = OrderedDict()
        for key, value in self.__dict__.items():
            if self._dict_filter(key, value):
                d[key] = HeaderBase._value(value)
        if self.meta:
            for key, value in self.meta.items():
                d[key] = HeaderBase._value(value)
        return d

    def from_dict(
        self,
        d: dict,
        ignore_keys: Sequence[str] = None,
    ):
        r"""Deserialize object from dictionary.

        Args:
            d: dictionary of class variables to assign
            ignore_keys: variables listed here will be ignored

        """
        for key, value in d.items():
            if ignore_keys and key in ignore_keys:
                continue
            if key in self.__dict__ and key != "meta":
                if self.__dict__[key] is not None:
                    assert isinstance(value, type(self.__dict__[key]))
                self.__dict__[key] = value
            else:
                self.meta[key] = value

    def dump(
        self,
        stream=None,
        indent: int = 2,
    ) -> str:
        r"""Serialize object to YAML.

        Args:
            stream: file-like object. If ``None`` serializes to string
            indent: indent

        Returns:
            YAML string

        """
        return yaml.dump(
            self.to_dict(),
            stream=stream,
            default_flow_style=None,
            indent=indent,
            allow_unicode=True,
        )

    def __eq__(
        self,
        other: HeaderBase,
    ) -> bool:
        return self.dump() == other.dump()

    def __hash__(self) -> int:
        return hash(self.dump())

    def __repr__(self):
        s = self.dump()
        return s[:-1] if s.endswith("\n") else s

    def __str__(self):
        return repr(self)


class DefineBase:
    @classmethod
    def _assert_has_attribute_value(cls, value):
        valid_values = cls._attribute_values()
        if value not in valid_values:
            raise BadValueError(value, valid_values)

    @classmethod
    def _attribute_values(cls):
        attributes = inspect.getmembers(cls, lambda x: not inspect.isroutine(x))
        return sorted(
            [
                a[1]
                for a in attributes
                if not (a[0].startswith("__") and a[0].endswith("__"))
            ]
        )


def format_series_as_html():  # pragma: no cover (only used in documentation)
    setattr(pd.Series, "_repr_html_", series_to_html)
    setattr(pd.Index, "_repr_html_", index_to_html)


def index_to_html(self):  # pragma: no cover
    df = self.to_frame()
    df = df.drop(columns=df.columns)
    return df.to_html()


def is_relative_path(path):
    return not (
        os.path.isabs(path)
        or "\\" in path
        or path.startswith("./")
        or "/./" in path
        or path.startswith("../")
        or "/../" in path
    )


def series_to_html(self):  # pragma: no cover
    df = self.to_frame()
    return df.to_html()


def to_audformat_dtype(dtype: str | type) -> str:
    r"""Convert pandas to audformat dtype."""
    # bool
    if pd.api.types.is_bool_dtype(dtype):
        return define.DataType.BOOL
    # date
    elif pd.api.types.is_datetime64_dtype(dtype) or dtype in ["datetime", "date"]:
        return define.DataType.DATE
    # float
    elif pd.api.types.is_float_dtype(dtype) or dtype in [
        "floating",
        "decimal",
        "mixed-integer-float",
    ]:
        return define.DataType.FLOAT
    # int
    elif pd.api.types.is_integer_dtype(dtype) or dtype == "integer":
        return define.DataType.INTEGER
    # time
    elif pd.api.types.is_timedelta64_dtype(dtype) or dtype in ["timedelta", "time"]:
        return define.DataType.TIME
    # string
    # We cannot use pd.api.types.is_string_dtype()
    # as it returns `True` for list, object, etc.
    elif dtype in [str, "str", "string"]:
        return define.DataType.STRING
    # object
    else:
        return define.DataType.OBJECT


def to_pandas_dtype(dtype: str) -> str | None:
    r"""Convert audformat to pandas dtype.

    We use ``"Int64"`` instead of ``"int64"``,
    and ``"boolean"`` instead of ``"bool"``
    to allow for nullable entries,
    e.g. ``[0, 2, <NA>]``.

    Args:
        dtype: audformat dtype

    Returns:
        pandas dtype

    """
    if dtype == define.DataType.BOOL:
        return "boolean"
    elif dtype == define.DataType.DATE:
        return "datetime64[ns]"
    elif dtype == define.DataType.FLOAT:
        return "float"
    elif dtype == define.DataType.INTEGER:
        return "Int64"
    elif dtype == define.DataType.OBJECT:
        return "object"
    elif dtype == define.DataType.STRING:
        return "string"
    elif dtype == define.DataType.TIME:
        return "timedelta64[ns]"
