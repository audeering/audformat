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
    r"""Converts value to list or array."""
    if value is not None:
        if isinstance(value, (pd.Series, pd.DataFrame, pd.Index)):
            value = value.to_numpy()
        elif is_scalar(value):
            value = [value]
    return value


def index(
        files: define.Typing.FILES,
        *,
        starts: define.Typing.TIMESTAMPS = None,
        ends: define.Typing.TIMESTAMPS = None,
) -> pd.Index:
    r"""Creates index conform to Unified Format.

    Creates a filewise index if only ``files`` is given.
    Otherwise creates a segmented index,
    where ``starts`` default to ``0`` and ``ends`` to ``NaT``.

    Args:
        files: set confidence values only on a sub-set of files
        starts: segment start positions
        ends: segment end positions

    Returns:
        index

    Raises:
        CannotCreateSegmentedIndex: if ``files``, ``start`` and ``ends``
            differ in size

    """
    files = to_array(files)
    starts = to_array(starts)
    ends = to_array(ends)

    if starts is not None or ends is not None:
        if starts is None:
            starts = [pd.to_timedelta(0)] * len(ends)
        elif ends is None:
            ends = [pd.NaT] * len(starts)

        if not len(files) == len(starts) and len(files) == len(ends):
            raise CannotCreateSegmentedIndex()

        idx = pd.MultiIndex.from_arrays(
            [files, starts, ends],
            names=[
                define.IndexField.FILE,
                define.IndexField.START,
                define.IndexField.END,
            ])
    else:
        idx = pd.Index(files, name=define.IndexField.FILE)

    return idx


def index_type(
        obj: typing.Union[pd.Index, pd.Series, pd.DataFrame]
) -> define.IndexType:
    r"""Derive table type.

    Args:
        obj: object in Unified Format

    Returns:
        table type

    Raises:
        NotConformToUnifiedFormat: if not conform to Unified Format

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
