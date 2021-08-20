import typing

from audformat.core import define
from audformat.core.common import HeaderBase


class Split(HeaderBase):
    r"""Database split.

    Defines if a subset of a database should be used for training,
    development or testing.

    Args:
        type: split type,
            see :class:`audformat.define.SplitType`
            for available split types
        description: split description
        meta: additional meta fields

    Raises:
        BadValueError: if an invalid ``type`` is passed

    Example:
        >>> Split(define.SplitType.TEST)
        {type: test}

    """
    def __init__(
            self,
            type: str = define.SplitType.OTHER,
            *,
            description: str = None,
            meta: dict = None,
    ):
        super().__init__(description=description, meta=meta)
        define.SplitType.assert_has_attribute_value(type)
        self.type = type
        r"""Split type"""
