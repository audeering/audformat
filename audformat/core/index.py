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


def assert_index(
    obj: typing.Union[pd.Index, pd.Series, pd.DataFrame],
):
    r"""Assert object is conform to :ref:`table specifications
    <data-tables:Tables>`.

    This does not check for duplicates in the index.
    If you need that check
    use :func:`audformat.assert_no_duplicates` in addition.

    If the index is a :class:`pandas.MultiIndex`
    the first level is considered
    to have the name ``'file'``
    and data type ``string``.
    If the name of the second level is ``'start'``
    and the name of the third level ``'end'``
    both need to be of type ``timedelta64[ns]``.

    Args:
        obj: object

    Raises:
        ValueError: if not conform to
            :ref:`table specifications <data-tables:Tables>`

    Example:
        >>> assert_index(pd.Index(['f1'], name='file'))
        >>> assert_index(
        ...     pd.MultiIndex.from_arrays(
        ...         [
        ...             ['f1'],
        ...             pd.to_timedelta([0], unit='s'),
        ...             pd.to_timedelta([1], unit='s'),
        ...         ],
        ...         names=['file', 'start', 'end'],
        ...     )
        ... )

    """
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        obj = obj.index

    if obj.names[0] != define.IndexField.FILE:
        raise ValueError(
            'Index not conform to audformat. '
            'Found first level with name '
            f'{obj.names[0]}, '
            f'but expected name '
            f"'{define.IndexField.FILE}'."
        )

    if len(obj.names) == 1:
        file_dtype = obj.dtype
    else:
        file_dtype = obj.levels[0].dtype
    if not pd.api.types.is_string_dtype(file_dtype):
        raise ValueError(
            "Index not conform to audformat. "
            "Level 'file' must contain values of type 'string'."
        )

    if len(obj.names) > 2:
        if (
                obj.names[1] == define.IndexField.START
                and not pd.api.types.is_timedelta64_dtype(obj.levels[1].dtype)
        ):
            raise ValueError(
                "Index not conform to audformat. "
                "Level 'start' must contain values of type 'timedelta64[ns]'."
            )
        if (
                obj.names[2] == define.IndexField.END
                and not pd.api.types.is_timedelta64_dtype(obj.levels[2].dtype)
        ):
            raise ValueError(
                "Index not conform to audformat. "
                "Level 'end' must contain values of type 'timedelta64[ns]'."
            )


def assert_no_duplicates(
    obj: typing.Union[pd.Index, pd.Series, pd.DataFrame],
):
    r"""Assert object contains no duplicates in its index.

    The :ref:`table specifications <data-tables:Tables>`
    allow no duplicated index entries.
    To save time we do not test for this
    in :func:`audformat.assert_index`.

    Args:
        obj: object

    Raises:
        ValueError: if duplicates are found

    """
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        obj = obj.index

    if obj.has_duplicates:
        max_display = 10
        duplicates = obj[obj.duplicated()]
        msg_tail = '\n...' if len(duplicates) > max_display else ''
        msg_duplicates = '\n'.join(
            [
                str(duplicate) for duplicate
                in duplicates[:max_display].tolist()
            ]
        )
        raise ValueError(
            'Index not conform to audformat. '
            'Found duplicates:\n'
            f'{msg_duplicates}{msg_tail}'
        )


def filewise_index(
        files: Files = None,
) -> pd.Index:
    r"""Creates a filewise index.

    Index is conform to :ref:`table specifications <data-tables:Tables>`.

    Args:
        files: list of files

    Returns:
        filewise index

    Raises:
        ValueError: if created index contains duplicates

    Example:
        >>> filewise_index(['a.wav', 'b.wav'])
        Index(['a.wav', 'b.wav'], dtype='object', name='file')

    """
    if files is None:
        files = []

    files = to_array(files)
    index = pd.Index(files, name=define.IndexField.FILE)
    assert_index(index)

    return index


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
        >>> index_type(pd.Index(['f1'], name='file'))
        'filewise'
        >>> index_type(
        ...     pd.MultiIndex.from_arrays(
        ...         [
        ...             ['f1'],
        ...             ['f2'],
        ...         ],
        ...         names=['file', 'verification-file'],
        ...     )
        ... )
        'filewise'
        >>> index_type(
        ...     pd.MultiIndex.from_arrays(
        ...         [
        ...             ['f1'],
        ...             pd.to_timedelta([0], unit='s'),
        ...             pd.to_timedelta([1], unit='s'),
        ...             [1],
        ...         ],
        ...         names=['file', 'start', 'end', 'version'],
        ...     )
        ... )
        'segmented'

    """
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        obj = obj.index

    assert_index(obj)

    if (
            len(obj.names) < 3
            or (
                obj.names[1] != define.IndexField.START
                and obj.names[2] != define.IndexField.END
            )
    ):
        return define.IndexType.FILEWISE
    else:
        return define.IndexType.SEGMENTED


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
        ValueError: if created index contains duplicates

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

    index = pd.MultiIndex.from_arrays(
        [files, to_timedelta(starts), to_timedelta(ends)],
        names=[
            define.IndexField.FILE,
            define.IndexField.START,
            define.IndexField.END,
        ])
    assert_index(index)

    return index
