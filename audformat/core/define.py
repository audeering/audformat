import typing

import numpy as np
import pandas as pd

from audformat.core.common import DefineBase


class DataType(DefineBase):
    r"""Data types"""
    BOOL = 'bool'
    DATE = 'date'
    INTEGER = 'int'
    FLOAT = 'float'
    STRING = 'str'
    TIME = 'time'


class Gender(DefineBase):
    r"""Gender"""
    CHILD = 'child'
    FEMALE = 'female'
    MALE = 'male'
    OTHER = 'other'


class IndexField(DefineBase):
    r"""Index field"""
    FILE = 'file'
    START = 'start'
    END = 'end'


class IndexType(DefineBase):
    r"""Index type"""
    FILEWISE = 'filewise'
    SEGMENTED = 'segmented'


class MediaType(DefineBase):
    r"""Media type"""
    AUDIO = 'audio'
    VIDEO = 'video'


class RaterType(DefineBase):
    r"""Rater type"""
    TRUTH = 'ground truth'
    HUMAN = 'human'
    MACHINE = 'machine'
    VOTE = 'vote'


class SplitType(DefineBase):
    r"""Split type"""
    TRAIN = 'train'
    DEVELOP = 'dev'
    TEST = 'test'
    UNDEFINED = 'undefined'


class SpeakerProfession(DefineBase):
    r"""Speaker profession"""
    ACTOR = 'actor'
    AMATEUR = 'amateur'


class Typing:
    r"""Type hints"""
    FILES = typing.Union[
        str, typing.Sequence[str], pd.Index, pd.Series,
    ]
    TIMESTAMPS = typing.Union[
        pd.Timedelta, typing.Sequence[pd.Timedelta], pd.Index, pd.Series,
    ]
    VALUES = typing.Union[
        int, float, str, pd.Timedelta,
        typing.Sequence[
            typing.Union[int, float, str, pd.Timedelta],
        ],
        np.ndarray,
        pd.Series,
    ]


class Usage(DefineBase):
    COMMERCIAL = 'commercial'
    RESEARCH = 'research'
    RESTRICTED = 'restricted'
