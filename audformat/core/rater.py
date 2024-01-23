from audformat.core import define
from audformat.core.common import HeaderBase


class Rater(HeaderBase):
    r"""A rater is the author of an annotation.

    Args:
        type: rater type,
            see :class:`audformat.define.RaterType`
            for available rater types
        description: rater description
        meta: additional meta fields

    Raises:
        BadValueError: if an invalid ``type`` is passed


    Examples:
        >>> Rater(define.RaterType.HUMAN)
        {type: human}

    """

    def __init__(
        self,
        type: str = define.RaterType.HUMAN,
        *,
        description: str = None,
        meta: dict = None,
    ):
        super().__init__(description=description, meta=meta)
        define.RaterType._assert_has_attribute_value(type)
        self.type = type
        r"""Rater type"""
