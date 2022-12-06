import typing

import numpy as np
import pandas as pd

from audformat.core.common import DefineBase


class DataType(DefineBase):
    r"""Data types of column content.

    Use ``DATE``
    to handle time and date information,
    e.g. as provided by :class:`datetime.datetime`.
    Use ``TIME``
    to handle duration values.

    """
    BOOL = 'bool'
    DATE = 'date'
    INTEGER = 'int'
    FLOAT = 'float'
    OBJECT = 'object'
    STRING = 'str'
    TIME = 'time'


class Gender(DefineBase):
    r"""Gender scheme definitions."""
    CHILD = 'child'
    FEMALE = 'female'
    MALE = 'male'
    OTHER = 'other'


class IndexField(DefineBase):
    r"""Index fields of a table.

    Specifies the string values
    representing column/field names
    for a filewise
    and segmented index.
    The exact string values are part
    of the :ref:`table specifications <data-tables:Tables>`,
    and should never be changed by a user.

    """
    FILE = 'file'
    START = 'start'
    END = 'end'


class IndexType(DefineBase):
    r"""Index types of a table.

    Specifies the string values
    representing a filewise or segmented index.
    Those string values are returned by
    :attr:`audformat.Table.type`
    and :func:`audformat.index_type`.
    The exact string values are not part
    of the :ref:`table specifications <data-tables:Tables>`,
    but it is recommended
    to not change them.

    """
    FILEWISE = 'filewise'
    SEGMENTED = 'segmented'


class MediaType(DefineBase):
    r"""Media type of table."""
    AUDIO = 'audio'
    OTHER = 'other'
    VIDEO = 'video'


class License(DefineBase):
    r"""Common public licenses recommended to use with your data."""
    CC0_1_0 = 'CC0-1.0'
    CC_BY_4_0 = 'CC-BY-4.0'
    CC_BY_NC_4_0 = 'CC-BY-NC-4.0'
    CC_BY_NC_SA_4_0 = 'CC-BY-NC-SA-4.0'
    CC_BY_SA_4_0 = 'CC-BY-SA-4.0'


LICENSE_URLS = {
    License.CC0_1_0: 'https://creativecommons.org/publicdomain/zero/1.0/',
    License.CC_BY_4_0: 'https://creativecommons.org/licenses/by/4.0/',
    License.CC_BY_NC_4_0: 'https://creativecommons.org/licenses/by-nc/4.0/',
    License.CC_BY_NC_SA_4_0:
        'https://creativecommons.org/licenses/by-nc-sa/4.0/',
    License.CC_BY_SA_4_0: 'https://creativecommons.org/licenses/by-sa/4.0/',
}


class RaterType(DefineBase):
    r"""Rater type of column."""
    HUMAN = 'human'
    MACHINE = 'machine'
    OTHER = 'other'
    TRUTH = 'ground truth'
    VOTE = 'vote'


class SplitType(DefineBase):
    r"""Split type of table."""
    TRAIN = 'train'
    DEVELOP = 'dev'
    OTHER = 'other'
    TEST = 'test'


class TableStorageFormat(DefineBase):
    r"""Storage format of tables."""
    CSV = 'csv'
    PICKLE = 'pkl'


class Usage(DefineBase):
    r"""Usage permission of database."""
    COMMERCIAL = 'commercial'
    OTHER = 'other'
    RESEARCH = 'research'
    RESTRICTED = 'restricted'
    UNRESTRICTED = 'unrestricted'
