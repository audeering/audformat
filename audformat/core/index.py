import typing

import numpy as np
import pandas as pd

from audformat.core import define
from audformat.errors import (
    CannotCreateSegmentedIndex,
    NotConformToUnifiedFormat,
)


def is_scalar(value: typing.Any) -> bool:
    r"""Check if value is scalar"""
    return (value is not None) and \
           (isinstance(value, str) or not hasattr(value, '__len__'))


def to_array(value: typing.Any) -> typing.Union[list, np.ndarray]:
    r"""Convert value to list or array."""
    if value is not None:
        if isinstance(value, (pd.Series, pd.DataFrame, pd.Index)):
            value = value.to_numpy()
        elif is_scalar(value):
            value = [value]
    return value


def create_filewise_index(
        files: define.Typing.FILES = None,
) -> pd.Index:
    r"""Creates a filewise index conform to
        :ref:`table specifications <data-tables:Tables>`.

    Args:
        files: list of files

    Returns:
        filewise index

    """
    if files is None:
        files = []
    files = to_array(files)
    return pd.Index(files, name=define.IndexField.FILE)


def create_segmented_index(
        files: define.Typing.FILES = None,
        starts: define.Typing.TIMESTAMPS = None,
        ends: define.Typing.TIMESTAMPS = None,
) -> pd.Index:
    r"""Create segmented index conform to
        :ref:`table specifications <data-tables:Tables>`.

    If a non-empty index is created and ``starts`` is set to ``None``,
    the level will be filled up with ``0``.
    If a non-empty index is created and ``ends`` is set to ``None``,
    the level will be filled up with ``NaT``.

    Args:
        files: set confidence values only on a sub-set of files
        starts: segment start positions
        ends: segment end positions

    Returns:
        segmented index

    Raises:
        CannotCreateSegmentedIndex: if ``files``, ``start`` and ``ends``
            differ in size

    """
    files = to_array(files)
    starts = to_array(starts)
    ends = to_array(ends)

    if files is None:
        files = []

    num_files = len(files)

    if starts is None:
        starts = [0] * num_files

    if ends is None:
        ends = [pd.NaT] * num_files

    if num_files != len(starts) or num_files != len(ends):
        raise CannotCreateSegmentedIndex()

    return pd.MultiIndex.from_arrays(
        [files, pd.to_timedelta(starts), pd.to_timedelta(ends)],
        names=[
            define.IndexField.FILE,
            define.IndexField.START,
            define.IndexField.END,
        ])


def index_type(
        obj: typing.Union[pd.Index, pd.Series, pd.DataFrame]
) -> define.IndexType:
    r"""Derive index type.

    Args:
        obj: object conform to
            :ref:`table specifications <data-tables:Tables>`

    Returns:
        table type

    Raises:
        NotConformToUnifiedFormat: if not conform to
            :ref:`table specifications <data-tables:Tables>`

    """
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        obj = obj.index

    num = len(obj.names)
    if num == 1 and obj.names[0] == define.IndexField.FILE:
        return define.IndexType.FILEWISE
    elif num == 3 and \
            obj.names[0] == define.IndexField.FILE and \
            obj.names[1] == define.IndexField.START and \
            obj.names[2] == define.IndexField.END:
        return define.IndexType.SEGMENTED

    raise NotConformToUnifiedFormat()
