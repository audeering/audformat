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
    OTHER = 'other'
    VIDEO = 'video'


class RaterType(DefineBase):
    r"""Rater type"""
    HUMAN = 'human'
    MACHINE = 'machine'
    TRUTH = 'ground truth'
    VOTE = 'vote'


class SplitType(DefineBase):
    r"""Split type"""
    TRAIN = 'train'
    DEVELOP = 'dev'
    OTHER = 'other'
    TEST = 'test'


class SpeakerProfession(DefineBase):
    r"""Speaker profession"""
    ACTOR = 'actor'
    AMATEUR = 'amateur'


class Usage(DefineBase):
    COMMERCIAL = 'commercial'
    RESEARCH = 'research'
    RESTRICTED = 'restricted'
