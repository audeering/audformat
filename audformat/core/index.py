import typing

import numpy as np
import pandas as pd

from audformat.core import define
from audformat.core.typing import (
    Files,
    Timestamps,
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


def to_timedelta(times):
    r"""Convert time value to pd.Timedelta."""
    try:
        return pd.to_timedelta(times, unit='s')
    except ValueError:  # catches values like '1s'
        return pd.to_timedelta(times)


def filewise_index(
        files: Files = None,
) -> pd.Index:
    r"""Creates a filewise index.

    Index is conform to :ref:`table specifications <data-tables:Tables>`.

    Args:
        files: list of files

    Returns:
        filewise index

    Example:
        >>> filewise_index(['a.wav', 'b.wav'])
        Index(['a.wav', 'b.wav'], dtype='object', name='file')

    """
    if files is None:
        files = []
    files = to_array(files)
    return pd.Index(files, name=define.IndexField.FILE)


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
        ValueError: if not conform to
            :ref:`table specifications <data-tables:Tables>`

    Example:
        >>> index_type(filewise_index())
        'filewise'
        >>> index_type(segmented_index())
        'segmented'

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

    raise ValueError('Index not conform to audformat.')


def segmented_index(
        files: Files = None,
        starts: Timestamps = None,
        ends: Timestamps = None,
) -> pd.Index:
    r"""Create segmented index.

    Index is conform to :ref:`table specifications <data-tables:Tables>`.

    If a non-empty index is created and ``starts`` is set to ``None``,
    the level will be filled up with ``0``.
    If a non-empty index is created and ``ends`` is set to ``None``,
    the level will be filled up with ``NaT``.

    Args:
        files: set confidence values only on a sub-set of files
        starts: segment start positions.
            Time values given as float or integers are treated as seconds
        ends: segment end positions.
            Time values given as float or integers are treated as seconds

    Returns:
        segmented index

    Raises:
        ValueError: if ``files``, ``start`` and ``ends`` differ in size

    Example:
        >>> segmented_index('a.wav', 0, 1.1)
        MultiIndex([('a.wav', '0 days', '0 days 00:00:01.100000')],
                   names=['file', 'start', 'end'])
        >>> segmented_index('a.wav', '0ms', '1ms')
        MultiIndex([('a.wav', '0 days', '0 days 00:00:00.001000')],
                   names=['file', 'start', 'end'])
        >>> segmented_index(['a.wav', 'b.wav'])
        MultiIndex([('a.wav', '0 days', NaT),
                    ('b.wav', '0 days', NaT)],
                   names=['file', 'start', 'end'])
        >>> segmented_index(['a.wav', 'b.wav'], [None, 1], [1, None])
        MultiIndex([('a.wav',               NaT, '0 days 00:00:01'),
                    ('b.wav', '0 days 00:00:01',               NaT)],
                   names=['file', 'start', 'end'])
        >>> segmented_index(
        ...     files=['a.wav', 'a.wav'],
        ...     starts=[0, 1],
        ...     ends=pd.to_timedelta([1000, 2000], unit='ms'),
        ... )
        MultiIndex([('a.wav', '0 days 00:00:00', '0 days 00:00:01'),
                    ('a.wav', '0 days 00:00:01', '0 days 00:00:02')],
                   names=['file', 'start', 'end'])

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
        raise ValueError(
            "Cannot create segmented table if 'files', "
            "'starts', and 'ends' differ in size",
        )

    return pd.MultiIndex.from_arrays(
        [files, to_timedelta(starts), to_timedelta(ends)],
        names=[
            define.IndexField.FILE,
            define.IndexField.START,
            define.IndexField.END,
        ])
