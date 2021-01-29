from audformat.core import define
from audformat.core.common import HeaderBase


class Rater(HeaderBase):
    r"""A rater is the author of an annotation.

    Args:
        type: rater type
        description: rater description
        meta: additional meta fields

    Raises:
        BadValueError: if an invalid ``type`` is passed


    Example:
        >>> Rater(define.RaterType.HUMAN)
        {type: human}

    """
    def __init__(
            self,
            type: define.RaterType = define.RaterType.HUMAN,
            *,
            description: str = None,
            meta: dict = None,
    ):
        super().__init__(description=description, meta=meta)
        define.RaterType.assert_has_attribute_value(type)
        self.type = type
        r"""Rater type"""
