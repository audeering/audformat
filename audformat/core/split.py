import typing

from audformat.core import define
from audformat.core.common import HeaderBase


class Split(HeaderBase):
    r"""Defines if a subset of a database should be used for training,
    development or testing.

    Args:
        type: split type
        description: split description
        meta: additional meta fields

    Raises:
        BadValueError: if an invalid ``type`` is passed

    """
    def __init__(
            self,
            type: define.SplitType = define.SplitType.UNDEFINED,
            *,
            description: str = None,
            meta: dict = None,
    ):
        super().__init__(description=description, meta=meta)
        define.SplitType.assert_has_value(type)
        self.type = type
        r"""Split type"""
