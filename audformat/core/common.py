import inspect
import oyaml as yaml
from typing import Sequence, Callable
import textwrap
from collections import OrderedDict

import pandas as pd

from audformat.core.errors import BadTypeError, BadValueError


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
        sorted_iter: if set, ``items()`` returns a sorted iterator
        value_type: only accept values of this type
        get_callback: call this function when an item is requested
        set_callback: call this function when an item is added
        **kwargs: default keyword arguments

    Raises:
        BadTypeError: if a value of an invalid type is added

    Example:
        >>> HeaderDict()

        >>> HeaderDict(
        ...     get_callback=lambda key, value: value + value,
        ...     set_callback=lambda key, value: value.upper(),
        ...     foo='bar',
        ... )
        foo:
          BARBAR

    """
    def __init__(
            self,
            *args,
            sorted_iter: bool = True,
            value_type: type = None,
            get_callback: Callable = None,
            set_callback: Callable = None,
            **kwargs,
    ):
        self.sorted_iter = sorted_iter
        r"""Use sorted iterator"""
        self.value_type = value_type
        r"""Only accept values of this type"""
        self.get_callback = get_callback
        r"""Callback function when item is requested"""
        self.set_callback = set_callback
        r"""Callback function when item is added"""
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        if self.set_callback is not None:
            value = self.set_callback(key, value)
        if self.value_type is not None:
            if not isinstance(value, self.value_type):
                raise BadTypeError(value, self.value_type)
        super().__setitem__(key, value)

    def __getitem__(self, key):
        value = super().__getitem__(key)
        if self.get_callback is not None:
            value = self.get_callback(key, value)
        return value

    def items(self):
        if self.sorted_iter:
            return iter(sorted(super().items()))
        else:
            return super().items()

    def dump(self) -> str:
        if not self:
            return ''
        else:
            return '\n'.join(
                [
                    '{}:\n{}'.format(
                        key, textwrap.indent(str(self[key]), '  '))
                    for key in self
                ]
            )

    def __repr__(self) -> str:
        return self.dump()


class HeaderBase:
    r"""Base class for header objects.

    Args:
        description: description string
        meta: dictionary with meta fields

    """
    def __init__(
            self, *,
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
        if type(value) is list:
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
        if key == 'meta':
            return False
        if key.startswith('_'):
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
            if key in self.__dict__:
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
            self.to_dict(), stream=stream, default_flow_style=None,
            indent=indent, allow_unicode=True,
        )

    def __eq__(
            self,
            other: 'HeaderBase',
    ) -> bool:
        return self.dump() == other.dump()

    def __hash__(self) -> int:
        return hash(self.dump())

    def __repr__(self):
        s = self.dump()
        return s[:-1] if s.endswith('\n') else s

    def __str__(self):
        return repr(self)


class DefineBase:

    @classmethod
    def assert_has_attribute_value(cls, value):
        valid_values = cls.attribute_values()
        if value not in valid_values:
            raise BadValueError(value, valid_values)

    @classmethod
    def attribute_values(cls):
        attributes = inspect.getmembers(
            cls, lambda x: not inspect.isroutine(x)
        )
        return sorted(
            [
                a[1] for a in attributes
                if not(a[0].startswith('__') and a[0].endswith('__'))
            ]
        )


def format_series_as_html():  # pragma: no cover (only used in documentation)
    setattr(pd.Series, '_repr_html_', series_to_html)
    setattr(pd.Index, '_repr_html_', index_to_html)


def index_to_html(self):  # pragma: no cover
    return self.to_frame(index=False)._repr_html_()


def series_to_html(self):  # pragma: no cover
    df = self.to_frame()
    return df._repr_html_()
