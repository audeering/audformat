import typing

import numpy as np
import pandas as pd

from audformat.core.common import DefineBase


class DataType(DefineBase):
    r"""Data types of column content."""
    BOOL = 'bool'
    DATE = 'date'
    INTEGER = 'int'
    FLOAT = 'float'
    STRING = 'str'
    TIME = 'time'


class Gender(DefineBase):
    r"""Gender scheme definitions."""
    CHILD = 'child'
    FEMALE = 'female'
    MALE = 'male'
    OTHER = 'other'


class IndexField(DefineBase):
    r"""Index fields defined in
        :ref:`table specifications <data-tables:Tables>`.
    """
    FILE = 'file'
    START = 'start'
    END = 'end'


class IndexType(DefineBase):
    r"""Index types defined in
        :ref:`table specifications <data-tables:Tables>`.
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
