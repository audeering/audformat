from collections import OrderedDict
import os
import typing as typing

import iso639
import pandas as pd

import audeer
import audiofile

from audformat.core import define
from audformat.core.index import index_type
from audformat.core.index import index as create_index
from audformat.core.errors import NotConformToUnifiedFormat


def concat(
        objs: typing.Sequence[typing.Union[pd.Series, pd.DataFrame]],
) -> pd.DataFrame:
    r"""Concatenate index and columns.

    If at least one object is segmented, the output has a segmented index.

    Args:
        objs: objects in Unified Format

    Returns:
        concatenated objects

    Raises:
        ValueError: if one or more objects are not in Unified Format

    """
    if not objs:
        return pd.DataFrame([], index=create_index([]))

    concat_table_type = define.IndexType.FILEWISE
    for frame in objs:
        if index_type(frame) == define.IndexType.SEGMENTED:
            concat_table_type = define.IndexType.SEGMENTED

    if concat_table_type == define.IndexType.SEGMENTED:
        index = create_index([], starts=[], ends=[])
        objs = [to_segmented(frame) for frame in objs]
    else:
        index = create_index([])

    index = index.append([frame.index for frame in objs])
    index = index.drop_duplicates()

    columns = OrderedDict()
    for obj in objs:
        if isinstance(obj, pd.Series):
            columns[obj.name] = obj.dtype
        else:
            for c, d in zip(obj.columns, obj.dtypes):
                if c not in columns:
                    columns[c] = d

    df_concat = pd.DataFrame(index=index, columns=columns.keys()).sort_index()
    for obj in objs:
        if not obj.empty:
            if isinstance(obj, pd.Series):
                df_concat.loc[obj.index, obj.name] = obj
            else:
                df_concat.loc[obj.index, obj.columns] = obj

    if not df_concat.empty:
        df_concat = df_concat.astype(columns)

    return df_concat


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


def read_csv(
        *args,
        **kwargs,
) -> typing.Union[pd.Index, pd.Series, pd.DataFrame]:
    r"""Read object in Unified Format from CSV file.

    See :meth:`pandas.read_csv` for supported arguments.

    Args:
        *args: arguments
        **kwargs: keyword arguments

    Raises:
        NotConformToUnifiedFormat: if CSV file cannot
            be converted to the :ref:`Unified Format
            <data-format:Unified Format>`

    """
    frame = pd.read_csv(*args, **kwargs)

    drop = [define.IndexField.FILE]
    if define.IndexField.FILE in frame.columns:
        files = frame[define.IndexField.FILE].astype(str)
    else:
        raise NotConformToUnifiedFormat()

    starts = None
    if define.IndexField.START in frame.columns:
        starts = pd.to_timedelta(frame[define.IndexField.START])
        drop.append(define.IndexField.START)

    ends = None
    if define.IndexField.END in frame.columns:
        ends = pd.to_timedelta(frame[define.IndexField.END])
        drop.append(define.IndexField.END)

    index = create_index(files, starts=starts, ends=ends)
    frame.drop(drop, axis='columns', inplace=True)

    if len(frame.columns) == 0:
        return index

    frame = frame.set_index(index)
    if len(frame.columns) == 1:
        return frame[frame.columns[0]]
    else:
        return frame


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
