from collections import OrderedDict
import os
import typing as typing

import iso639
import numpy as np
import pandas as pd

import audeer
import audiofile

from audformat.core import define
from audformat.core.index import index_type
from audformat.core.errors import (
    CannotCreateSegmentedIndex,
    NotConformToUnifiedFormat,
    RedundantArgumentError,
)


def concat(frames: typing.Sequence[pd.DataFrame]) -> pd.DataFrame:
    r"""Concatenate sequence of frames along index and columns.

    If at least one frame is segmented, the output will be a segmented table.

    Args:
        frames: sequence of :class:`pandas.DataFrame`

    Returns:
        concatenated frame

    Raises:
        ValueError: if one or more frame are not in Unified Format

    """
    if not frames:
        return pd.DataFrame([], columns=[define.IndexField.FILE])

    concat_table_type = define.IndexType.FILEWISE
    for frame in frames:
        if index_type(frame) == define.IndexType.SEGMENTED:
            concat_table_type = define.IndexType.SEGMENTED

    if concat_table_type == define.IndexType.SEGMENTED:
        index = pd.MultiIndex(
            levels=[[], [], []],
            codes=[[], [], []],
            names=[
                define.IndexField.FILE,
                define.IndexField.START,
                define.IndexField.END,
            ],
        )
        frames = [
            to_segmented(frame) for frame in frames
        ]
    else:
        index = pd.Index([], name=define.IndexField.FILE)

    index = index.append([frame.index for frame in frames])
    index = index.drop_duplicates()

    columns = OrderedDict()
    for df in frames:
        for c, d in zip(df.columns, df.dtypes):
            if c not in columns:
                columns[c] = d

    df_concat = pd.DataFrame(index=index, columns=columns.keys()).sort_index()
    for df in frames:
        if not df.empty:
            df_concat.loc[df.index, df.columns] = df
    if not df_concat.empty:
        df_concat = df_concat.astype(columns)

    return df_concat


def is_scalar(value: typing.Any) -> bool:
    r"""Check if value is scalar"""
    return (value is not None) and \
           (isinstance(value, str) or not hasattr(value, '__len__'))


def map_language(language: str) -> typing.Optional[str]:
    r"""Map language to ISO 639-3.

    Args:
        language: language string

    Returns:
        mapped string

    Raises:
        ValueError: if language is not supported

    """
    result = None

    if len(language) == 2:
        try:
            result = iso639.languages.get(alpha2=language.lower())
        except KeyError:
            pass
    elif len(language) == 3:
        try:
            result = iso639.languages.get(part3=language.lower())
        except KeyError:
            pass
    else:
        try:
            result = iso639.languages.get(name=language.title())
        except KeyError:
            pass

    if result is not None:
        result = result.part3

    if not result:
        raise ValueError(
            f"'{language}' is not supported by ISO 639-3."
        )

    return result


def remove_duplicates(x: typing.Sequence[typing.Any]) \
        -> typing.Sequence[typing.Any]:
    r""""Remove duplicates from list by keeping the order."""
    return list(dict.fromkeys(x))


def to_array(value: typing.Any) -> typing.Union[list, np.ndarray]:
    r"""Converts value to list or array."""
    if value is not None:
        if isinstance(value, (pd.Series, pd.DataFrame, pd.Index)):
            value = value.to_numpy()
        elif is_scalar(value):
            value = [value]
    return value


def to_filewise(
        obj: typing.Union[pd.Index, pd.Series, pd.DataFrame],
        root: str,
        output_folder: str,
        *,
        num_workers: int = 1,
        progress_bar: bool = True
) -> typing.Union[pd.Index, pd.Series, pd.DataFrame]:
    r"""Convert to filewise index.

    If input is segmented, each segment is saved to a separate file
    in ``output_folder``. The directory structure of the original data is
    preserved within ``output folder``.
    If input is filewise no action is applied.

    Args:
        obj: object in Unified Format
        root: path to root folder of data. Even if the file paths of ``frame``
            are absolute, this argument is needed in order to reconstruct
            the directory structure of the original data
        output_folder: path to folder of the created audio segments.
            If it's relative (absolute), then the file paths of the returned
            data frame are also relative (absolute)
        num_workers: number of threads to spawn
        progress_bar: show progress bar

    Returns:
        object with filewise index

    Raises:
        ValueError: if ``output_folder`` contained in path to files of
            original data

    """
    if index_type(obj) == define.IndexType.FILEWISE:
        return obj

    test_path = obj.index.get_level_values(define.IndexField.FILE)[0]
    is_abs = os.path.isabs(test_path)
    test_path = audeer.safe_path(test_path)

    # keep ``output_folder`` relative if it's relative
    if audeer.safe_path(output_folder) in test_path:
        raise ValueError(
            f'``output_folder`` may not be contained in path to files of '
            f'original data: {audeer.safe_path(output_folder)} != {test_path}')

    obj = obj.copy()
    original_files = obj.index.get_level_values(define.IndexField.FILE)
    if not is_abs:
        original_files = [os.path.join(root, f) for f in original_files]
    starts = obj.index.get_level_values(define.IndexField.START)
    ends = obj.index.get_level_values(define.IndexField.END)

    # order of rows within group is preserved:
    # "https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.html"  # noqa
    files = obj.groupby(define.IndexField.FILE, sort=False)
    segments = []

    for _, group in files:
        width = len(str(len(group) - 1))  # -1 because count starts at `0`
        f = group.index.get_level_values(define.IndexField.FILE)[0]
        f = os.path.relpath(f, root) if is_abs else f
        segments.extend(
            [os.path.join(
                output_folder,
                "_{}.".format(str(count).zfill(width)).join(f.rsplit('.', 1))
            )
                for count in range(len(group))]
        )
        audeer.mkdir(os.path.dirname(segments[-1]))

    def _split_files(original, start, end, segment):
        signal, sr = audiofile.read(
            file=original,
            duration=end.total_seconds() - start.total_seconds(),
            offset=start.total_seconds())
        audiofile.write(file=segment, signal=signal, sampling_rate=sr)

    params = [
        ([file, start, end, segment], {}) for
        file, start, end, segment in
        zip(original_files, starts, ends, segments)
    ]

    audeer.run_tasks(
        task_func=_split_files,
        params=params,
        task_description='To filewise index',
        num_workers=num_workers,
        progress_bar=progress_bar,
    )

    obj = obj.reset_index(drop=True)
    obj[define.IndexField.FILE] = segments
    return obj.set_index(keys=define.IndexField.FILE)


def to_segmented(
        obj: typing.Union[pd.Index, pd.Series, pd.DataFrame],
) -> typing.Union[pd.Index, pd.Series, pd.DataFrame]:
    r"""Convert to segmented index.

    Args:
        obj: object in Unified Format

    Returns:
        object with segmented index

    Raises:
        ValueError: if not conform to Unified Format.

    """
    table_type = index_type(obj)

    if obj.empty or table_type == define.IndexType.SEGMENTED:
        return obj

    if isinstance(obj, (pd.Series, pd.DataFrame)):
        index = obj.index
    else:
        index = obj

    df = index.to_frame(index=False)
    df[define.IndexField.START] = pd.to_timedelta(0)
    df[define.IndexField.END] = pd.NaT

    index = pd.MultiIndex.from_frame(
        df,
        names=[
            define.IndexField.FILE,
            define.IndexField.START,
            define.IndexField.END,
        ]
    )

    if isinstance(obj, pd.Index):
        return index

    obj = obj.reset_index(drop=True)
    obj.index = index
    return obj


# TODO: try to get rid of following functions

def check_redundant_arguments(**kwargs):
    r"""Check for redundant arguments."""
    redundant = []
    for key, value in kwargs.items():
        if value is not None:
            redundant.append(key)
    if redundant:
        raise RedundantArgumentError(redundant)


def series_to_array(series: pd.Series) -> np.ndarray:
    r"""Convert :class:`pandas.Series` to a ::`numpy.ndarray`.

    .. note:: If ``series`` holds categorical data,``NaN``s will be
        replaced with ``None``.

    Args:
        series: series

    Returns:
        array

    """
    values = series.to_numpy()
    if series.dtype.name == 'category':
        for idx, item in enumerate(values):
            if isinstance(item, float) and np.isnan(item):
                values[idx] = None
    return values


def index_to_dict(index: typing.Union[pd.Index, pd.Series,
                                      pd.DataFrame]) -> dict:
    r"""Convert :class:`pandas.Index` to a dictionary.

    Returns a dictionary with keys files, starts, and ends.

    """
    if isinstance(index, (pd.Series, pd.DataFrame)):
        index = index.index

    d = dict([(define.IndexField.FILE + 's', None),
              (define.IndexField.START + 's', None),
              (define.IndexField.END + 's', None)])

    table_type = index_type(index)

    d[define.IndexField.FILE + 's'] = index.get_level_values(
        define.IndexField.FILE).values
    if table_type == define.IndexType.SEGMENTED:
        d[define.IndexField.START + 's'] = index.get_level_values(
            define.IndexField.START).values.astype(np.timedelta64)
        d[define.IndexField.END + 's'] = index.get_level_values(
            define.IndexField.END).values.astype(np.timedelta64)

    return d


def series_to_dict(series: pd.Series) -> dict:
    r"""Convert :class:`pandas.Series` to a dictionary.

    Returns a dictionary with keys values, files, starts, and ends.

    """
    d = index_to_dict(series)
    d['values'] = series_to_array(series)
    return d


def frame_to_dict(frame: pd.DataFrame) -> dict:
    r"""Convert :class:`pandas.DataFrame` to a dictionary.

    Returns a dictionary with keys values, files, starts, and ends.

    """
    d = index_to_dict(frame)
    d['values'] = {}
    for column_id, column in frame.items():
        d['values'][column_id] = series_to_array(column)
    return d
