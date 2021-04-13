import errno
import os
import typing as typing

import iso639
import numpy as np
import pandas as pd

import audeer
import audiofile

from audformat.core import define
from audformat.core.index import index_type
from audformat.core.index import (
    filewise_index,
    segmented_index,
)


def concat(
        objs: typing.Sequence[typing.Union[pd.Series, pd.DataFrame]],
        *,
        overwrite: bool = False,
) -> typing.Union[pd.Series, pd.DataFrame]:
    r"""Concatenate objects.

    Objects must be conform to
    :ref:`table specifications <data-tables:Tables>`.

    The new object contains index and columns of both objects.
    Missing values will be set to ``NaN``.
    If at least one object is segmented, the output has a segmented index.

    Columns with the same identifier are combined to a single column.
    This requires that both columns have the same dtype
    and if ``overwrite`` is set to ``False``,
    values in places where the indices overlap have to match
    or one column contains ``NaN``.
    If ``overwrite`` is set to ``True``,
    the value of the last object in the list is kept.

    Args:
        objs: objects conform to
            :ref:`table specifications <data-tables:Tables>`
        overwrite: overwrite values where indices overlap

    Returns:
        concatenated objects

    Raises:
        ValueError: if one or more objects are not conform to
            :ref:`table specifications <data-tables:Tables>`
        ValueError: if columns with the same name have different dtypes
        ValueError: if values in the same position do not match

    Example:
        >>> obj1 = pd.Series(
        ...     [0., 1.],
        ...     index=filewise_index(['f1', 'f2']),
        ...     name='float',
        ... )
        >>> obj2 = pd.DataFrame(
        ...     {
        ...         'float': [1., 2.],
        ...         'string': ['a', 'b'],
        ...     },
        ...     index=segmented_index(['f2', 'f3']),
        ... )
        >>> concat([obj1, obj2])
                         float string
        file start  end
        f1   0 days NaT    0.0    NaN
        f2   0 days NaT    1.0      a
        f3   0 days NaT    2.0      b
        >>> obj1 = pd.Series(
        ...     [0., 1.],
        ...     index=filewise_index(['f1', 'f2']),
        ...     name='float',
        ... )
        >>> obj2 = pd.DataFrame(
        ...     {
        ...         'float': [np.nan, 2.],
        ...         'string': ['a', 'b'],
        ...     },
        ...     index=filewise_index(['f2', 'f3']),
        ... )
        >>> concat([obj1, obj2])
              float string
        file
        f1      0.0    NaN
        f2      1.0      a
        f3      2.0      b
        >>> obj1 = pd.Series(
        ...     [0., 0.],
        ...     index=filewise_index(['f1', 'f2']),
        ...     name='float',
        ... )
        >>> obj2 = pd.DataFrame(
        ...     {
        ...         'float': [1., 2.],
        ...         'string': ['a', 'b'],
        ...     },
        ...     index=segmented_index(['f2', 'f3']),
        ... )
        >>> concat([obj1, obj2], overwrite=True)
                         float string
        file start  end
        f1   0 days NaT    0.0    NaN
        f2   0 days NaT    1.0      a
        f3   0 days NaT    2.0      b

    """
    if not objs:
        return pd.Series([], index=filewise_index(), dtype='object')

    # the new index is a union of the individual objects
    index = union([obj.index for obj in objs])
    as_segmented = index_type(index) == define.IndexType.SEGMENTED

    # list with all columns we need to concatenate
    columns = []
    for obj in objs:
        if isinstance(obj, pd.Series):
            columns.append(obj)
        else:
            for column in obj:
                columns.append(obj[column])

    # reindex all columns to the new index
    columns_reindex = {}
    for column in columns:

        if as_segmented:
            column = to_segmented_index(column)

        # if we already have a column with that name, we have to merge them
        if column.name in columns_reindex:

            # assert same dtype
            if not same_dtype(
                    columns_reindex[column.name].dtype, column.dtype
            ):
                dtype_1 = columns_reindex[column.name].dtype
                if dtype_1.name == 'category':
                    dtype_1 = repr(dtype_1)
                dtype_2 = column.dtype
                if dtype_2.name == 'category':
                    dtype_2 = repr(dtype_2)
                raise ValueError(
                    "Found two columns with name "
                    f"'{column.name}' "
                    "buf different dtypes:\n"
                    f"{dtype_1} "
                    "!= "
                    f"{dtype_2}."
                )

            # overlapping values must match or have to be nan in one column
            if not overwrite:
                intersection = intersect(
                    [
                        columns_reindex[column.name].index,
                        column.index,
                    ]
                )
                if not intersection.empty:
                    combine = pd.DataFrame(
                        {
                            'left': columns_reindex[column.name][intersection],
                            'right': column[intersection]
                        }
                    )
                    combine.dropna(inplace=True)
                    differ = combine['left'] != combine['right']
                    if np.any(differ):
                        max_display = 10
                        overlap = combine[differ]
                        msg_overlap = str(overlap[:max_display])
                        msg_tail = '\n...' \
                            if len(overlap) > max_display \
                            else ''
                        raise ValueError(
                            "Found overlapping data in column "
                            f"'{column.name}':\n"
                            f"{msg_overlap}{msg_tail}"
                        )

            # drop NaN to avoid overwriting values from other column
            column = column.dropna()
        else:
            columns_reindex[column.name] = pd.Series(
                index=index,
                dtype=column.dtype,
            )
        columns_reindex[column.name][column.index] = column

    df = pd.DataFrame(columns_reindex, index=index)

    if len(df.columns) == 1:
        return df[df.columns[0]]
    else:
        return df


def intersect(
    objs: typing.Sequence[typing.Union[pd.Index]],
) -> pd.Index:
    r"""Intersect index objects.

    Index objects must be conform to
    :ref:`table specifications <data-tables:Tables>`.

    If at least one object is segmented, the output is a segmented index.

    Args:
        objs: index objects conform to
            :ref:`table specifications <data-tables:Tables>`

    Returns:
        intersection of index objects

    Raises:
        ValueError: if one or more objects are not conform to
            :ref:`table specifications <data-tables:Tables>`

    Example:
        >>> i1 = filewise_index(['f1', 'f2', 'f3'])
        >>> i2 = filewise_index(['f2', 'f3', 'f4'])
        >>> intersect([i1, i2])
        Index(['f2', 'f3'], dtype='object', name='file')
        >>> i3 = segmented_index(
        ...     ['f1', 'f2', 'f3', 'f4'],
        ...     [0, 0, 0, 0],
        ...     [1, 1, 1, 1],
        ... )
        >>> i4 = segmented_index(
        ...     ['f1', 'f2', 'f3'],
        ...     [0, 0, 1],
        ...     [1, 1, 2],
        ... )
        >>> intersect([i3, i4])
        MultiIndex([('f1', '0 days', '0 days 00:00:01'),
                    ('f2', '0 days', '0 days 00:00:01')],
                   names=['file', 'start', 'end'])
        >>> intersect([i1, i2, i3, i4])
        MultiIndex([('f2', '0 days', '0 days 00:00:01')],
                   names=['file', 'start', 'end'])

    """
    if not objs:
        return filewise_index()

    types = [index_type(obj) for obj in objs]

    if len(set(types)) == 1:

        index = objs[0]
        for obj in objs[1:]:
            index = index.intersection(obj)

    else:

        # intersect only filewise
        objs_filewise = [
            obj for obj, type in zip(objs, types)
            if type == define.IndexType.FILEWISE
        ]
        index_filewise = intersect(objs_filewise)

        # intersect only segmented
        objs_segmented = [
            obj for obj, type in zip(objs, types)
            if type == define.IndexType.SEGMENTED
        ]
        index_segmented = intersect(objs_segmented)

        # intersect segmented and filewise
        index = index_segmented[
            index_segmented.isin(index_filewise, 0)
        ]

    if index.empty and index_type(index) == define.IndexType.SEGMENTED:
        # asserts that start and end are of type 'timedelta64[ns]'
        index = segmented_index()

    return index


def map_language(language: str) -> str:
    r"""Map language to ISO 639-3.

    Args:
        language: language string

    Returns:
        mapped string

    Raises:
        ValueError: if language is not supported

    Example:
        >>> map_language('en')
        'eng'
        >>> map_language('eng')
        'eng'
        >>> map_language('English')
        'eng'

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
    r"""Read object from CSV file.

    Automatically detects the index type and returns an object that is
    conform to :ref:`table specifications <data-tables:Tables>`.
    If conversion is not possible, an error is raised.

    See :meth:`pandas.read_csv` for supported arguments.

    Args:
        *args: arguments
        **kwargs: keyword arguments

    Returns:
        object conform to :ref:`table specifications <data-tables:Tables>`

    Raises:
        ValueError: if CSV file is not conform to
            :ref:`table specifications <data-tables:Tables>`

    Example:
        >>> from io import StringIO
        >>> string = StringIO('''file,start,end,value
        ... f1,00:00:00,00:00:01,0.0
        ... f1,00:00:01,00:00:02,1.0
        ... f2,00:00:02,00:00:03,2.0''')
        >>> read_csv(string)
        file  start            end
        f1    0 days 00:00:00  0 days 00:00:01    0.0
              0 days 00:00:01  0 days 00:00:02    1.0
        f2    0 days 00:00:02  0 days 00:00:03    2.0
        Name: value, dtype: float64

    """
    frame = pd.read_csv(*args, **kwargs)

    drop = [define.IndexField.FILE]
    if define.IndexField.FILE in frame.columns:
        files = frame[define.IndexField.FILE].astype(str)
    else:
        raise ValueError('Index not conform to audformat.')

    starts = None
    if define.IndexField.START in frame.columns:
        starts = pd.to_timedelta(frame[define.IndexField.START])
        drop.append(define.IndexField.START)

    ends = None
    if define.IndexField.END in frame.columns:
        ends = pd.to_timedelta(frame[define.IndexField.END])
        drop.append(define.IndexField.END)

    if starts is None and ends is None:
        index = filewise_index(files)
    else:
        index = segmented_index(files, starts=starts, ends=ends)
    frame.drop(drop, axis='columns', inplace=True)

    if len(frame.columns) == 0:
        return index

    frame = frame.set_index(index)
    if len(frame.columns) == 1:
        return frame[frame.columns[0]]
    else:
        return frame


def same_dtype(d1, d2) -> bool:
    r"""Helper function to compare pandas dtype."""
    if d1.name.lower().startswith('int') and d2.name.lower().startswith('int'):
        # match different int types, e.g. int64 and Int64
        return True
    if d1.name.startswith('float') and d2.name.startswith('float'):
        # match different float types, e.g. float32 and float64
        return True
    if d1.name == 'category' and d2.name == 'category':
        # match only if categories are the same
        return d1 == d2
    return d1.name == d2.name


def to_filewise_index(
        obj: typing.Union[pd.Index, pd.Series, pd.DataFrame],
        root: str,
        output_folder: str,
        *,
        num_workers: int = 1,
        progress_bar: bool = False,
) -> typing.Union[pd.Index, pd.Series, pd.DataFrame]:
    r"""Convert to filewise index.

    If input is segmented, each segment is saved to a separate file
    in ``output_folder``. The directory structure of the original data is
    preserved within ``output_folder``.
    If input is filewise no action is applied.

    Args:
        obj: object conform to
            :ref:`table specifications <data-tables:Tables>`
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


def to_segmented_index(
        obj: typing.Union[pd.Index, pd.Series, pd.DataFrame],
        *,
        allow_nat: bool = True,
        root: str = None,
        num_workers: typing.Optional[int] = 1,
        verbose: bool = False,
) -> typing.Union[pd.Index, pd.Series, pd.DataFrame]:
    r"""Convert to segmented index.

    If the input a filewise table,
    ``start`` and ``end`` will be added as new levels to the index.
    By default, ``start`` will be set to 0 and ``end`` to ``NaT``.

    If ``allow_nat`` is set to ``False``,
    all occurrences of ``end=NaT``
    are replaced with the duration of the file.
    This, however, requires that the referenced file exists.
    If file names in the index are relative,
    the ``root`` argument can be used to provide
    the location where the files are stored.

    Args:
        obj: object conform to
            :ref:`table specifications <data-tables:Tables>`
        allow_nat: if set to ``False``, ``end=NaT`` is replaced with file
            duration
        root: root directory under which the files referenced in the index
            are stored
        num_workers: number of parallel jobs.
            If ``None`` will be set to the number of processors
            on the machine multiplied by 5
        verbose: show progress bar

    Returns:
        object with segmented index

    Raises:
        ValueError: if object not conform to
            :ref:`table specifications <data-tables:Tables>`
        FileNotFoundError: if file is not found

    """
    is_segmented = index_type(obj) == define.IndexType.SEGMENTED

    if is_segmented and allow_nat:
        return obj

    if isinstance(obj, (pd.Series, pd.DataFrame)):
        index = obj.index
    else:
        index = obj

    if not is_segmented:
        index = segmented_index(
            files=list(index),
            starts=[0] * len(index),
            ends=[pd.NaT] * len(index),
        )

    if not allow_nat:

        ends = index.get_level_values(define.IndexField.END)
        has_nat = pd.isna(ends)

        if any(has_nat):

            idx_nat = np.where(has_nat)[0]
            files = index.get_level_values(define.IndexField.FILE)
            starts = index.get_level_values(define.IndexField.START)

            def job(file: str) -> float:
                if root is not None and not os.path.isabs(file):
                    file = os.path.join(root, file)
                if not os.path.exists(file):
                    raise FileNotFoundError(
                        errno.ENOENT,
                        os.strerror(errno.ENOENT),
                        file,
                    )
                return audiofile.duration(file)

            params = [([file], {}) for file in files[idx_nat]]
            durs = audeer.run_tasks(
                job,
                params,
                num_workers=num_workers,
                progress_bar=verbose,
                task_description='Read duration',
            )
            ends.values[idx_nat] = pd.to_timedelta(durs, unit='s')

            index = segmented_index(files, starts, ends)

    if isinstance(obj, pd.Index):
        return index

    obj = obj.reset_index(drop=True)
    obj.index = index

    return obj


def union(
    objs: typing.Sequence[pd.Index],
) -> pd.Index:
    r"""Create union of index objects.

    Index objects must be conform to
    :ref:`table specifications <data-tables:Tables>`.

    If at least one object is segmented, the output is a segmented index.

    Args:
        objs: index objects conform to
            :ref:`table specifications <data-tables:Tables>`

    Returns:
        union of index objects

    Raises:
        ValueError: if one or more objects are not conform to
            :ref:`table specifications <data-tables:Tables>`

    Example:
        >>> i1 = filewise_index(['f1', 'f2', 'f3'])
        >>> i2 = filewise_index(['f2', 'f3', 'f4'])
        >>> union([i1, i2])
        Index(['f1', 'f2', 'f3', 'f4'], dtype='object', name='file')
        >>> i3 = segmented_index(
        ...     ['f1', 'f2', 'f3', 'f4'],
        ...     [0, 0, 0, 0],
        ...     [1, 1, 1, 1],
        ... )
        >>> i4 = segmented_index(
        ...     ['f1', 'f2', 'f3'],
        ...     [0, 0, 1],
        ...     [1, 1, 2],
        ... )
        >>> union([i3, i4])
        MultiIndex([('f1', '0 days 00:00:00', '0 days 00:00:01'),
                    ('f2', '0 days 00:00:00', '0 days 00:00:01'),
                    ('f3', '0 days 00:00:00', '0 days 00:00:01'),
                    ('f3', '0 days 00:00:01', '0 days 00:00:02'),
                    ('f4', '0 days 00:00:00', '0 days 00:00:01')],
                   names=['file', 'start', 'end'])

        >>> union([i1, i2, i3, i4])
        MultiIndex([('f1', '0 days 00:00:00',               NaT),
                    ('f1', '0 days 00:00:00', '0 days 00:00:01'),
                    ('f2', '0 days 00:00:00',               NaT),
                    ('f2', '0 days 00:00:00', '0 days 00:00:01'),
                    ('f3', '0 days 00:00:00',               NaT),
                    ('f3', '0 days 00:00:00', '0 days 00:00:01'),
                    ('f3', '0 days 00:00:01', '0 days 00:00:02'),
                    ('f4', '0 days 00:00:00',               NaT),
                    ('f4', '0 days 00:00:00', '0 days 00:00:01')],
                   names=['file', 'start', 'end'])

    """
    if not objs:
        return filewise_index()

    types = [index_type(obj) for obj in objs]

    if len(set(types)) != 1:
        objs = [to_segmented_index(obj) for obj in objs]

    index = objs[0]
    for obj in objs[1:]:
        index = index.union(obj)

    if isinstance(index, pd.MultiIndex) and len(index.levels) == 3:
        # asserts that start and end are of type 'timedelta64[ns]'
        if index.empty:
            index = segmented_index()
        elif index.levels[2].empty:
            # If all end values are NaT, pandas stores an empty array and
            # since pd.Index.union() in that case sets the type to
            # DatetimeArray, we need to set it back to 'timedelta64[ns]'.
            index.set_levels(
                [pd.to_timedelta([])],
                level=[2],
                inplace=True,
            )

    return index
