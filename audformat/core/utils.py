import errno
import os
import re
import typing as typing

import iso639
import numpy as np
import pandas as pd

import audeer
import audiofile

from audformat.core import define
from audformat.core.database import Database
from audformat.core.index import index_type
from audformat.core.index import (
    filewise_index,
    segmented_index,
)
from audformat.core.scheme import Scheme


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
    return_as_frame = False
    for obj in objs:
        if isinstance(obj, pd.Series):
            columns.append(obj)
        else:
            return_as_frame = True
            for column in obj:
                columns.append(obj[column])

    # reindex all columns to the new index
    columns_reindex = {}
    for column in columns:

        if as_segmented:
            column = to_segmented_index(column)

        # if we already have a column with that name, we have to merge them
        if column.name in columns_reindex:

            dtype_1 = columns_reindex[column.name].dtype
            dtype_2 = column.dtype

            # assert same dtype
            if not same_dtype(dtype_1, dtype_2):
                if dtype_1.name == 'category':
                    dtype_1 = repr(dtype_1)
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

            # Fix changed handling of float32/float64 in pandas>=1.3
            if 'float64' in [dtype_1, dtype_2]:
                columns_reindex[column.name] = (
                    columns_reindex[column.name].astype('float64')
                )

            # overlapping values must match or have to be nan in one column
            if not overwrite:
                intersection = intersect(
                    [
                        columns_reindex[column.name].index,
                        column.index,
                    ]
                )
                # We use len() here as index.empty takes a very long time
                if len(intersection) > 0:
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
            if column.dtype.name.startswith('int'):
                dtype = 'Int64'
            elif column.dtype.name == 'bool':
                dtype = 'boolean'
            else:
                dtype = column.dtype
            columns_reindex[column.name] = pd.Series(
                index=index,
                dtype=dtype,
            )
        columns_reindex[column.name][column.index] = column

    df = pd.DataFrame(columns_reindex, index=index)

    if not return_as_frame and len(df.columns) == 1:
        return df[df.columns[0]]
    else:
        return df


def duration(
        obj: typing.Union[pd.Index, pd.Series, pd.DataFrame],
        *,
        root: str = None,
        num_workers: int = 1,
        verbose: bool = False,
) -> pd.Timedelta:
    r"""Total duration of all entries present in the object.

    The object might contain a segmented or a filewise index.
    For a segmented index the duration is calculated
    from its start and end values.
    If an end value is ``NaT``
    or the object contains a filewise index
    the duration is calculated from the media file
    by calling :func:`audiofile.duration`.

    Args:
        obj: object conform to
            :ref:`table specifications <data-tables:Tables>`
        root: root directory under which the files referenced in the index
            are stored.
            Only relevant when the duration of the files
            needs to be detected from the file
        num_workers: number of parallel jobs.
            Only relevant when the duration of the files
            needs to be detected from the file
            If ``None`` will be set to the number of processors
            on the machine multiplied by 5
        verbose: show progress bar.
            Only relevant when the duration of the files
            needs to be detected from the file

    Returns:
        duration

    Example:

        >>> index = segmented_index(
        ...     files=['a', 'b', 'c'],
        ...     starts=[0, 1, 3],
        ...     ends=[1, 2, 4],
        ... )
        >>> duration(index)
        Timedelta('0 days 00:00:03')

    """
    obj = to_segmented_index(
        obj,
        allow_nat=False,
        root=root,
        num_workers=num_workers,
        verbose=verbose,
    )

    if not isinstance(obj, pd.MultiIndex):
        obj = obj.index

    # We use len() here as index.empty takes a very long time
    if len(obj) == 0:
        return pd.Timedelta(0, unit='s')

    starts = obj.get_level_values(define.IndexField.START)
    ends = obj.get_level_values(define.IndexField.END)
    return (ends - starts).sum()


def expand_file_path(
        index: pd.Index,
        root: str,
) -> pd.Index:
    r"""Expand relative path in index with root.

    Args:
        index: index with relative file path conform to
            :ref:`table specifications <data-tables:Tables>`
        root: directory added in front of the relative file path

    Returns:
        index with absolute file path

    Raises:
        ValueError: if index is not conform to
            :ref:`table specifications <data-tables:Tables>`

    Example:
        >>> index = filewise_index(['f1', 'f2'])
        >>> index
        Index(['f1', 'f2'], dtype='object', name='file')
        >>> expand_file_path(index, '/some/where')  # doctest: +SKIP
        Index(['/some/where/f1', '/some/where/f2'], dtype='object', name='file')

    """  # noqa: E501
    if len(index) == 0:
        return index

    root = audeer.safe_path(root) + os.path.sep
    is_segmented = index_type(index) == define.IndexType.SEGMENTED

    if is_segmented:
        index = index.set_levels(root + index.levels[0], level=0)
    else:
        index = root + index

    return index


def hash(
        obj: typing.Union[pd.Index, pd.Series, pd.DataFrame],
) -> str:
    r"""Create hash from object.

    Objects with the same elements
    produce the same hash string
    independent of the ordering of the elements.

    Args:
        obj: object

    Returns:
        hash string

    Example:
        >>> index = filewise_index(['f1', 'f2'])
        >>> hash(index)
        '-4231615416436839963'
        >>> y = pd.Series(0, index)
        >>> hash(y)
        '5251663970176285425'

    """
    return str(pd.util.hash_pandas_object(obj).sum())


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
        >>> index1 = filewise_index(['f1', 'f2', 'f3'])
        >>> index2 = filewise_index(['f2', 'f3', 'f4'])
        >>> intersect([index1, index2])
        Index(['f2', 'f3'], dtype='object', name='file')
        >>> index3 = segmented_index(
        ...     ['f1', 'f2', 'f3', 'f4'],
        ...     [0, 0, 0, 0],
        ...     [1, 1, 1, 1],
        ... )
        >>> index4 = segmented_index(
        ...     ['f1', 'f2', 'f3'],
        ...     [0, 0, 1],
        ...     [1, 1, 2],
        ... )
        >>> intersect([index3, index4])
        MultiIndex([('f1', '0 days', '0 days 00:00:01'),
                    ('f2', '0 days', '0 days 00:00:01')],
                   names=['file', 'start', 'end'])
        >>> intersect([index1, index2, index3, index4])
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

    # We use len() here as index.empty takes a very long time
    if len(index) == 0 and index_type(index) == define.IndexType.SEGMENTED:
        # asserts that start and end are of type 'timedelta64[ns]'
        index = segmented_index()

    return index


def join_labels(
        labels: typing.Sequence[typing.Union[typing.List, typing.Dict]],
):
    r"""Combine scheme labels.

    Args:
        labels: sequence of labels to join.
            For dictionary labels,
            labels further to the right
            can overwrite previous labels

    Returns:
        joined labels

    Raises:
        ValueError: if labels are of different type
        ValueError: if label type is not ``list`` or ``dict``

    Example:
        >>> join_labels([{'a': 0, 'b': 1}, {'b': 2, 'c': 2}])
        {'a': 0, 'b': 2, 'c': 2}

    """
    if len(labels) == 0:
        return labels

    if not isinstance(labels, list):
        labels = list(labels)

    label_type = type(labels[0])
    joined_labels = labels[0]

    for label in labels[1:]:
        if type(label) != label_type:
            raise ValueError(
                f"Labels are of different type:\n"
                f"{label_type}\n"
                f"!=\n"
                f"{type(label)}"
            )

    if label_type == dict:
        for label in labels[1:]:
            for key, value in label.items():
                if key not in joined_labels or joined_labels[key] != value:
                    joined_labels[key] = value
    elif label_type == list:
        joined_labels = list(
            set(list(joined_labels) + audeer.flatten_list(labels[1:]))
        )
        joined_labels = sorted(audeer.flatten_list(joined_labels))
    else:
        raise ValueError(
            f"Supported label types are 'list' and 'dict', "
            f"but your is '{label_type}'"
        )

    # Check if joined labels have a valid format,
    # e.g. {0: {'age': 20}, '0': {'age': 30}} is not allowed
    Scheme(labels=joined_labels)

    return joined_labels


def join_schemes(
        dbs: typing.Sequence[Database],
        scheme_id: str,
):
    r"""Join and update scheme of databases.

    This joins the given scheme of several databases
    using :func:`audformat.utils.join_labels`
    and replaces the scheme in each database
    with the joined one.
    The dtype of all :class:`audformat.Column` objects
    that reference the scheme in the databases
    will be updated.
    Removed labels are set to ``NaN``.

    This might be useful,
    if you want to combine databases
    with :meth:`audformat.Database.update`.

    Args:
        dbs: sequence of databases
        scheme_id: scheme ID of a scheme with labels
            that should be joined

    Example:
        >>> db1 = Database('db1')
        >>> db2 = Database('db2')
        >>> db1.schemes['scheme_id'] = Scheme(labels=['a'])
        >>> db2.schemes['scheme_id'] = Scheme(labels=['b'])
        >>> join_schemes([db1, db2], 'scheme_id')
        >>> db1.schemes
        scheme_id:
          dtype: str
          labels: [a, b]

    """
    labels = join_labels([db.schemes[scheme_id].labels for db in dbs])
    for db in dbs:
        db.schemes[scheme_id].replace_labels(labels)


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
    if d1.name.startswith('bool') and d2.name.startswith('bool'):
        # match different bool types, i.e. bool and boolean
        return True
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


def replace_file_extension(
        index: pd.Index,
        extension: str,
        pattern: str = None,
) -> pd.Index:
    r"""Change the file extension of index entries.

    It replaces all existing file extensions
    in the index file path
    by the new provided one.

    Args:
        index: index with file path
            conform to :ref:`table specifications <data-tables:Tables>`
        extension: new file extension without ``'.'``.
            If set to ``''``,
            the current file extension is removed
        pattern: regexp pattern to match current extensions.
            In contrast to ``extension``,
            you have to include ``'.'``.
            If ``None`` the default of ``r'\.[a-zA-Z0-9]+$'`` is used

    Returns:
        updated index

    Example:
        >>> index = filewise_index(['f1.wav', 'f2.flac'])
        >>> replace_file_extension(index, 'mp3')
        Index(['f1.mp3', 'f2.mp3'], dtype='object', name='file')
        >>> index = filewise_index(['f1.wav.gz', 'f2.wav.gz'])
        >>> replace_file_extension(index, '')
        Index(['f1.wav', 'f2.wav'], dtype='object', name='file')
        >>> replace_file_extension(index, 'flac', pattern=r'\.wav\.gz$')
        Index(['f1.flac', 'f2.flac'], dtype='object', name='file')

    """
    if len(index) == 0:
        return index

    if pattern is None:
        pattern = r'\.[a-zA-Z0-9]+$'
    cur_ext = re.compile(pattern)
    if extension:
        new_ext = f'.{extension}'
    else:
        new_ext = ''
    is_segmented = index_type(index) == define.IndexType.SEGMENTED

    if is_segmented:
        index = index.set_levels(
            index.levels[0].str.replace(cur_ext, new_ext, regex=True),
            level='file',
        )
    else:
        index = index.str.replace(cur_ext, new_ext, regex=True)

    return index


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
        files_duration: typing.MutableMapping[str, pd.Timedelta] = None,
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
        files_duration: mapping from file to duration.
            If not ``None``,
            used to look up durations.
            If no entry is found for a file,
            it is added to the mapping.
            Expects absolute file names.
            Only relevant if ``allow_nat`` is set to ``False``
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

            def job(file: str) -> pd.Timedelta:

                if root is not None and not os.path.isabs(file):
                    file = os.path.join(root, file)
                if files_duration is not None and file in files_duration:
                    return files_duration[file]

                if not os.path.exists(file):
                    raise FileNotFoundError(
                        errno.ENOENT,
                        os.strerror(errno.ENOENT),
                        file,
                    )
                dur = audiofile.duration(file)
                dur = pd.to_timedelta(dur, unit='s')

                if files_duration is not None:
                    files_duration[file] = dur

                return dur

            params = [([file], {}) for file in files[idx_nat]]
            durs = audeer.run_tasks(
                job,
                params,
                num_workers=num_workers,
                progress_bar=verbose,
                task_description='Read duration',
            )
            ends.values[idx_nat] = durs

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
        >>> index1 = filewise_index(['f1', 'f2', 'f3'])
        >>> index2 = filewise_index(['f2', 'f3', 'f4'])
        >>> union([index1, index2])
        Index(['f1', 'f2', 'f3', 'f4'], dtype='object', name='file')
        >>> index3 = segmented_index(
        ...     ['f1', 'f2', 'f3', 'f4'],
        ...     [0, 0, 0, 0],
        ...     [1, 1, 1, 1],
        ... )
        >>> index4 = segmented_index(
        ...     ['f1', 'f2', 'f3'],
        ...     [0, 0, 1],
        ...     [1, 1, 2],
        ... )
        >>> union([index3, index4])
        MultiIndex([('f1', '0 days 00:00:00', '0 days 00:00:01'),
                    ('f2', '0 days 00:00:00', '0 days 00:00:01'),
                    ('f3', '0 days 00:00:00', '0 days 00:00:01'),
                    ('f3', '0 days 00:00:01', '0 days 00:00:02'),
                    ('f4', '0 days 00:00:00', '0 days 00:00:01')],
                   names=['file', 'start', 'end'])

        >>> union([index1, index2, index3, index4])
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

    # Combine all MultiIndex entries and drop duplicates afterwards,
    # faster than using index.union(),
    # compare https://github.com/audeering/audformat/pull/98
    df = pd.concat([o.to_frame() for o in objs])
    index = df.index
    index = index.drop_duplicates()
    index, _ = index.sortlevel()

    return index
