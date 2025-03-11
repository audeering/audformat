from __future__ import annotations

import collections
from collections.abc import Callable
from collections.abc import Iterator
from collections.abc import MutableMapping
from collections.abc import Sequence
import errno
import hashlib
import os
import platform
import re
import sys

import iso639
from iso639.exceptions import InvalidLanguageValue
import iso3166
import numpy as np
import pandas as pd
import pyarrow as pa

import audeer
import audiofile

from audformat.core import define
from audformat.core.common import to_audformat_dtype
from audformat.core.database import Database
from audformat.core.index import filewise_index
from audformat.core.index import is_filewise_index
from audformat.core.index import is_segmented_index
from audformat.core.index import segmented_index
from audformat.core.scheme import Scheme


# Exclude examples that return a path containing `\`
# from doctest on Windows
if platform.system() in ["Windows"]:  # pragma: no cover
    __doctest_skip__ = [
        "expand_file_path",
        "to_filewise_index",
    ]


def concat(
    objs: Sequence[pd.Series | pd.DataFrame],
    *,
    overwrite: bool = False,
    aggregate_function: Callable[[pd.Series], object] = None,
    aggregate_strategy: str = "mismatch",
) -> pd.Series | pd.DataFrame:
    r"""Concatenate objects.

    If all objects are conform to
    :ref:`table specifications <data-tables:Tables>`
    and at least one object is segmented,
    the output has a segmented index.
    Otherwise,
    requires that levels and dtypes
    of all objects match,
    see :func:`audformat.utils.is_index_alike`.
    When a :class:`pandas.Index`
    is concatenated with a single-level
    :class:`pandas.MultiIndex`,
    the result is a
    :class:`pandas.Index`.

    The new object contains index and columns of all objects.
    Missing values will be set to ``NaN``.

    Columns with the same identifier are combined to a single column.
    This requires that both columns have the same dtype
    and if ``overwrite`` is set to ``False``,
    values in places where the indices overlap have to match
    or one column contains ``NaN``.
    If ``overwrite`` is set to ``True``,
    the value of the last object in the list is kept.
    If ``overwrite`` is set to ``False``,
    a custom aggregation function can be provided
    with ``aggregate_function``
    that converts the overlapping values
    into a single value.

    Args:
        objs: objects
        overwrite: overwrite values where indices overlap
        aggregate_function: function to aggregate overlapping values,
            that cannot be joined
            when ``overwrite`` is ``False``.
            The function gets a :class:`pandas.Series`
            with overlapping values
            as input.
            E.g. set to
            ``lambda y: y.mean()``
            to average the values
            or to
            ``tuple``
            to return them as a tuple
        aggregate_strategy: if ``aggregate_function`` is not ``None``,
            ``aggregate_strategy`` decides
            when ``aggregate_function`` is applied.
            ``'overlap'``: apply to all samples
            that have an overlapping index;
            ``'mismatch'``: apply to all samples
            that have an overlapping index
            and a different value

    Returns:
        concatenated objects

    Raises:
        ValueError: if level and dtypes of object indices do not match
        ValueError: if columns with the same name have different dtypes
        ValueError: if ``aggregate_strategy`` is not one of
            ``'overlap'``, ``'mismatch'``
        ValueError: if ``aggregate_function`` is ``None``,
            ``overwrite`` is ``False``,
            and values in the same position do not match

    Examples:
        >>> concat(
        ...     [
        ...         pd.Series([0], index=pd.Index([0])),
        ...         pd.Series([1], index=pd.Index([1])),
        ...     ]
        ... )
        0    0
        1    1
        dtype: Int64
        >>> concat(
        ...     [
        ...         pd.Series([0], index=pd.Index([0]), name="col1"),
        ...         pd.Series([1], index=pd.Index([0]), name="col2"),
        ...     ]
        ... )
           col1  col2
        0     0     1
        >>> concat(
        ...     [
        ...         pd.Series([1, 1], index=pd.Index([0, 1])),
        ...         pd.Series([1, 1], index=pd.Index([0, 1])),
        ...     ],
        ...     aggregate_function=np.sum,
        ... )
        0    1
        1    1
        dtype: Int64
        >>> concat(
        ...     [
        ...         pd.Series([1, 1], index=pd.Index([0, 1])),
        ...         pd.Series([1, 2], index=pd.Index([0, 1])),
        ...     ],
        ...     aggregate_function=np.sum,
        ... )
        0    1
        1    3
        dtype: Int64
        >>> concat(
        ...     [
        ...         pd.Series([1, 1], index=pd.Index([0, 1])),
        ...         pd.Series([1, 1], index=pd.Index([0, 1])),
        ...     ],
        ...     aggregate_function=np.sum,
        ...     aggregate_strategy="overlap",
        ... )
        0    2
        1    2
        dtype: Int64
        >>> concat(
        ...     [
        ...         pd.Series(
        ...             [0.0, 1.0],
        ...             index=pd.Index(
        ...                 [0, 1],
        ...                 dtype="int",
        ...                 name="idx",
        ...             ),
        ...             name="float",
        ...         ),
        ...         pd.DataFrame(
        ...             {
        ...                 "float": [np.nan, 2.0],
        ...                 "string": ["a", "b"],
        ...             },
        ...             index=pd.MultiIndex.from_arrays(
        ...                 [[0, 2]],
        ...                 names=["idx"],
        ...             ),
        ...         ),
        ...     ]
        ... )
             float string
        idx
        0      0.0      a
        1      1.0    NaN
        2      2.0      b
        >>> concat(
        ...     [
        ...         pd.Series(
        ...             [0.0, 1.0],
        ...             index=filewise_index(["f1", "f2"]),
        ...             name="float",
        ...         ),
        ...         pd.DataFrame(
        ...             {
        ...                 "float": [1.0, 2.0],
        ...                 "string": ["a", "b"],
        ...             },
        ...             index=segmented_index(["f2", "f3"]),
        ...         ),
        ...     ]
        ... )
                         float string
        file start  end
        f1   0 days NaT    0.0    NaN
        f2   0 days NaT    1.0      a
        f3   0 days NaT    2.0      b
        >>> concat(
        ...     [
        ...         pd.Series(
        ...             [0.0, 0.0],
        ...             index=filewise_index(["f1", "f2"]),
        ...             name="float",
        ...         ),
        ...         pd.DataFrame(
        ...             {
        ...                 "float": [1.0, 2.0],
        ...                 "string": ["a", "b"],
        ...             },
        ...             index=segmented_index(["f2", "f3"]),
        ...         ),
        ...     ],
        ...     overwrite=True,
        ... )
                         float string
        file start  end
        f1   0 days NaT    0.0    NaN
        f2   0 days NaT    1.0      a
        f3   0 days NaT    2.0      b

    """
    allowed_values = ["overlap", "mismatch"]
    if aggregate_strategy not in allowed_values:
        raise ValueError(
            "aggregate_strategy needs to be one of: " f"{', '.join(allowed_values)}"
        )

    if not objs:
        return pd.Series([], index=pd.Index([]), dtype="object")

    if len(objs) == 1:
        return objs[0]

    objs = _maybe_convert_filewise_index(objs)
    objs = _maybe_convert_single_level_multi_index(objs)
    _assert_index_alike(objs)

    # the new index is a union of the individual objects
    index = union([obj.index for obj in objs])

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
    overlapping_values = {}
    for column in columns:
        # if we already have a column with that name, we have to merge them
        if column.name in columns_reindex:
            dtype_1 = columns_reindex[column.name].dtype
            dtype_2 = column.dtype

            # assert same dtype
            if not _is_same_dtype(dtype_1, dtype_2):
                if dtype_1.name == "category":
                    dtype_1 = repr(dtype_1)
                if dtype_2.name == "category":
                    dtype_2 = repr(dtype_2)
                raise ValueError(
                    "Found two columns with name "
                    f"'{column.name}' "
                    "but different dtypes:\n"
                    f"{dtype_1} "
                    "!= "
                    f"{dtype_2}."
                )

            # Fix changed handling of float32/float64 in pandas>=1.3
            if "float64" in [dtype_1, dtype_2]:
                columns_reindex[column.name] = columns_reindex[column.name].astype(
                    "float64"
                )

            # Handle overlapping values
            if not overwrite:

                def collect_overlap(overlapping_values, column, index):
                    """Collect overlap for aggregate function."""
                    if column.name not in overlapping_values:
                        overlapping_values[column.name] = []
                    overlapping_values[column.name].append(column.loc[index])
                    column = column.loc[~column.index.isin(index)]
                    column = column.dropna()
                    return column, overlapping_values

                # Apply aggregate function only to overlapping entries
                intersection = intersect(
                    [
                        columns_reindex[column.name].dropna().index,
                        column.dropna().index,
                    ]
                )
                # We use len() here as index.empty takes a very long time
                if len(intersection) > 0:
                    # Apply aggregate function
                    # to all overlapping entries
                    if (
                        aggregate_function is not None
                        and aggregate_strategy == "overlap"
                    ):
                        column, overlapping_values = collect_overlap(
                            overlapping_values,
                            column,
                            intersection,
                        )
                        columns_reindex[column.name][column.index] = column
                        continue

                    # Find data that differ and cannot be joined
                    combine = pd.DataFrame(
                        {
                            "left": columns_reindex[column.name][intersection],
                            "right": column[intersection],
                        }
                    )
                    combine.dropna(inplace=True)
                    differ = combine["left"] != combine["right"]

                    if np.any(differ):
                        # Apply aggregate function
                        # to overlapping entries
                        # that do not match in value
                        if (
                            aggregate_function is not None
                            and aggregate_strategy == "mismatch"
                        ):
                            column, overlapping_values = collect_overlap(
                                overlapping_values,
                                column,
                                intersection[differ],
                            )
                            columns_reindex[column.name][column.index] = column
                            continue

                        # Raise error if values don't match and are not NaN
                        else:
                            max_display = 10
                            overlap = combine[differ]
                            msg_overlap = str(overlap[:max_display])
                            msg_tail = "\n..." if len(overlap) > max_display else ""
                            raise ValueError(
                                "Found overlapping data in column "
                                f"'{column.name}':\n"
                                f"{msg_overlap}{msg_tail}"
                            )

            # drop NaN to avoid overwriting values from other column
            column = column.dropna()
        else:
            # Adjust dtype and initialize empty column
            if pd.api.types.is_integer_dtype(column.dtype):
                dtype = "Int64"
            elif pd.api.types.is_bool_dtype(column.dtype):
                dtype = "boolean"
            else:
                dtype = column.dtype
            columns_reindex[column.name] = pd.Series(
                index=index,
                dtype=dtype,
            )
        columns_reindex[column.name].loc[column.index] = column

    # Apply custom aggregation function
    # on collected overlapping data
    # (no overlapping data is collected
    #  when no aggregation function is provided)
    if len(overlapping_values) > 0:
        for column in overlapping_values:
            # Add data of first column
            # overlapping with all other columns
            union_index = union([y.index for y in overlapping_values[column]])
            overlapping_values[column].insert(
                0, columns_reindex[column].loc[union_index]
            )

            # Convert list of overlapping data series to data frame
            # and apply aggregate function
            df = pd.concat(
                overlapping_values[column],
                axis=1,
                ignore_index=True,
            )
            dtype = columns_reindex[column].dtype
            y = df.apply(aggregate_function, axis=1)

            # Restore the original dtype if possible
            try:
                y = y.astype(dtype)
            except (TypeError, ValueError):
                columns_reindex[column] = columns_reindex[column].astype(y.dtype)
            columns_reindex[column].loc[y.index] = y

    # Use `None` to force `{}` return the correct index, see
    # https://github.com/pandas-dev/pandas/issues/52404
    df = pd.DataFrame(columns_reindex or None, index=index)

    if not return_as_frame and len(df.columns) == 1:
        return df[df.columns[0]]
    else:
        return df


def difference(
    objs: Sequence[pd.Index],
) -> pd.Index:
    r"""Difference of index objects.

    Returns index items
    that are not shared by two or more objects.
    For two objects this is identical
    to their `symmetric difference`_.

    If all index objects are conform to
    :ref:`table specifications <data-tables:Tables>`
    and at least one object is segmented,
    the output is a segmented index.
    Otherwise,
    requires that levels and dtypes
    of all objects match,
    see :func:`audformat.utils.is_index_alike`.
    Integer dtypes don't have to match,
    but the result will always be of dtype ``Int64``.
    When the symmetric difference of a
    :class:`pandas.Index`
    with a single-level
    :class:`pandas.MultiIndex`,
    is calculated,
    the result is a
    :class:`pandas.Index`.

    The order of the resulting index
    depends on the order of ``objs``.
    If you require :func:`audformat.utils.difference`
    to be commutative_,
    you have to sort its output.

    .. _symmetric difference: https://en.wikipedia.org/wiki/Symmetric_difference
    .. _commutative: https://en.wikipedia.org/wiki/Commutative_property

    Args:
        objs: index objects

    Returns:
        difference of index objects

    Raises:
        ValueError: if level and dtypes of objects do not match

    Examples:
        >>> difference(
        ...     [
        ...         pd.Index([1, 2, 3], name="idx"),
        ...     ]
        ... )
        Index([1, 2, 3], dtype='Int64', name='idx')
        >>> difference(
        ...     [
        ...         pd.Index([0, 1], name="idx"),
        ...         pd.Index([1, np.nan], dtype="Int64", name="idx"),
        ...     ]
        ... )
        Index([0, <NA>], dtype='Int64', name='idx')
        >>> difference(
        ...     [
        ...         pd.Index([0, 1], name="idx"),
        ...         pd.MultiIndex.from_arrays([[1, 2]], names=["idx"]),
        ...     ]
        ... )
        Index([0, 2], dtype='Int64', name='idx')
        >>> difference(
        ...     [
        ...         pd.MultiIndex.from_arrays(
        ...             [["a", "b", "c"], [0, 1, 2]],
        ...             names=["idx1", "idx2"],
        ...         ),
        ...         pd.MultiIndex.from_arrays(
        ...             [["b", "c"], [1, 3]],
        ...             names=["idx1", "idx2"],
        ...         ),
        ...     ]
        ... )
        MultiIndex([('a', 0),
                    ('c', 2),
                    ('c', 3)],
                   names=['idx1', 'idx2'])
        >>> difference(
        ...     [
        ...         filewise_index(["f1", "f2", "f3"]),
        ...         filewise_index(["f2", "f3", "f4"]),
        ...     ]
        ... )
        Index(['f1', 'f4'], dtype='string', name='file')
        >>> difference(
        ...     [
        ...         segmented_index(["f1"], [0], [1]),
        ...         segmented_index(["f1", "f2"], [0, 1], [1, 2]),
        ...     ]
        ... )
        MultiIndex([('f2', '0 days 00:00:01', '0 days 00:00:02')],
                   names=['file', 'start', 'end'])
        >>> difference(
        ...     [
        ...         filewise_index(["f1", "f2"]),
        ...         segmented_index(["f1", "f2"], [0, 0], [pd.NaT, 1]),
        ...     ]
        ... )
        MultiIndex([('f2', '0 days',               NaT),
                    ('f2', '0 days', '0 days 00:00:01')],
                   names=['file', 'start', 'end'])

    """  # noqa: E501
    if not objs:
        return pd.Index([])

    objs = [_maybe_convert_pandas_dtype(obj) for obj in objs]

    if len(objs) == 1:
        return objs[0]

    objs = _maybe_convert_filewise_index(objs)
    objs = _maybe_convert_single_level_multi_index(objs)
    _assert_index_alike(objs)

    index = list(objs[0])
    for obj in objs[1:]:
        index += list(obj)

    counting = collections.Counter(index)
    index = [idx for idx, count in counting.items() if count == 1]

    index = _alike_index(objs[0], index)

    return index


def duration(
    obj: pd.Index | pd.Series | pd.DataFrame,
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

    Examples:
        >>> index = segmented_index(
        ...     files=["a", "b", "c"],
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
        return pd.Timedelta(0, unit="s")

    starts = obj.get_level_values(define.IndexField.START)
    ends = obj.get_level_values(define.IndexField.END)
    return (ends - starts).sum()


def expand_file_path(
    index: pd.Index,
    root: str,
) -> pd.Index:
    r"""Expand path in index with root.

    It applies :func:`os.path.normpath`
    to the provided path ``root``,
    adds a file separator at its end
    and puts it in front of the file path in the index.

    Args:
        index: index conform to
            :ref:`table specifications <data-tables:Tables>`
        root: relative or absolute path added in front
            of the index file path

    Returns:
        index with root added to file path

    Raises:
        ValueError: if index is not conform to
            :ref:`table specifications <data-tables:Tables>`

    Examples:

        .. skip: start if(sys.platform.startswith("win"))

        >>> expand_file_path(filewise_index(["f1", "f2"]), "/a")
        Index(['/a/f1', '/a/f2'], dtype='string', name='file')
        >>> expand_file_path(filewise_index(["f1", "f2"]), "./a")
        Index(['a/f1', 'a/f2'], dtype='string', name='file')

        .. skip: end

    """  # noqa: E501
    if len(index) == 0:
        return index

    root = os.path.normpath(root) + os.path.sep

    if is_segmented_index(index):
        index = index.set_levels(root + index.levels[0], level=0)
    else:
        index = root + index

    return index


def hash(
    obj: pd.Index | pd.Series | pd.DataFrame,
    strict: bool = False,
) -> str:
    r"""Create hash from object.

    If ``strict`` is ``False``,
    objects with the same elements
    produce the same hash string
    independent of the ordering of the elements,
    and level or column names.

    .. warning::

        If ``obj`` is a dataframe or series
        with data type ``"Int64"``,
        and ``strict`` is ``False``,
        the returned hash value changes with ``pandas>=2.2.0``.

    Args:
        obj: object
        strict: if ``True``,
            the hash takes into account
            the order of rows
            and column/level names

    Returns:
        hash string with 19 characters,
        or 32 characters if ``strict`` is ``True``

    Examples:
        >>> index = filewise_index(["f1", "f2"])
        >>> hash(index)
        '-4231615416436839963'
        >>> hash(index[::-1])  # reversed index
        '-4231615416436839963'
        >>> y = pd.Series(0, index)
        >>> hash(y)
        '5251663970176285425'
        >>> hash(index, strict=True)
        '0741235e2250e0fcd9ab7b64972f5047'
        >>> hash(index[::-1], strict=True)  # reversed index
        'c6639d377897dd9353dc3e8b2968170d'

    """
    if strict:
        if isinstance(obj, pd.Index):
            df = obj.to_frame()
        elif isinstance(obj, pd.Series):
            df = obj.to_frame().reset_index()
        else:
            df = obj.reset_index()
        # Handle column names and dtypes
        table = pa.Table.from_pandas(df, preserve_index=False)
        schema_str = table.schema.to_string(
            # schema.metadata contains pandas related information,
            # and the used pyarrow and pandas version,
            # and needs to be excluded
            show_field_metadata=False,
            show_schema_metadata=False,
        )
        schema_md5 = hashlib.md5(schema_str.encode())
        # Handle index, values, and row order
        data_md5 = hashlib.md5()
        for _, y in df.items():
            # Convert every column to a numpy array,
            # and hash its string representation
            if y.dtype == "Int64":
                # Enforce consistent conversion to numpy.array
                # for integers across different pandas versions
                # (since pandas 2.2.x, Int64 is converted to float if it contains <NA>)
                y = y.astype("float")
            data_md5.update(bytes(str(y.to_numpy()), "utf-8"))
        md5 = hashlib.md5()
        md5.update(schema_md5.digest())
        md5.update(data_md5.digest())
        md5 = md5.hexdigest()
    else:
        # Convert to int64
        # to enforce same behavior
        # across different pandas versions,
        # see
        # https://github.com/pandas-dev/pandas/issues/55452
        md5 = str(pd.util.hash_pandas_object(obj).astype("int64").sum())
    return md5


def index_has_overlap(
    obj: pd.Index | pd.DataFrame | pd.Series,
) -> bool:
    r"""Check if one or more segments in the index overlap.

    If the index is filewise, the result will always be ``False``.

    Args:
        obj: object conform to
            :ref:`table specifications <data-tables:Tables>`

    Returns:
        ``True`` if overlap is detected, otherwise ``False``

    Examples:
        >>> index = filewise_index(["f1", "f2"])
        >>> index_has_overlap(index)
        False
        >>> index = segmented_index(
        ...     ["f1", "f2"],
        ...     [0, 1],
        ...     [2, 3],
        ... )
        >>> index_has_overlap(index)
        False
        >>> index = segmented_index(
        ...     ["f1"] * 2,
        ...     [0, 1],
        ...     [2, 3],
        ... )
        >>> index_has_overlap(index)
        True

    """
    index = obj if isinstance(obj, pd.Index) else obj.index

    if is_filewise_index(index):
        return False

    for _, sub_index in iter_by_file(index):
        sub_index = sub_index.sortlevel(define.IndexField.START)[0]
        starts = sub_index.get_level_values(define.IndexField.START)
        ends = sub_index.get_level_values(define.IndexField.END)
        ends = ends.fillna(pd.Timedelta(sys.maxsize))
        if any(ends[:-1] > starts[1:]):
            return True

    return False


def intersect(
    objs: Sequence[pd.Index],
) -> pd.Index:
    r"""Intersect index objects.

    If all index objects are conform to
    :ref:`table specifications <data-tables:Tables>`
    and at least one object is segmented,
    the output is a segmented index.
    Otherwise,
    requires that levels and dtypes
    of all objects match,
    see :func:`audformat.utils.is_index_alike`.
    Integer dtypes don't have to match,
    but the result will always be of dtype ``Int64``.
    When a :class:`pandas.Index`
    is intersected with a single-level
    :class:`pandas.MultiIndex`,
    the result is a
    :class:`pandas.Index`.

    The order of the resulting index
    depends on the order of ``objs``.
    If you require :func:`audformat.utils.intersect`
    to be commutative_,
    you have to sort its output.

    .. _commutative: https://en.wikipedia.org/wiki/Commutative_property

    Args:
        objs: index objects

    Returns:
        intersection of index objects

    Raises:
        ValueError: if level and dtypes of objects do not match


    Examples:
        >>> intersect(
        ...     [
        ...         pd.Index([1, 2, 3], name="idx"),
        ...     ]
        ... )
        Index([], dtype='Int64', name='idx')
        >>> intersect(
        ...     [
        ...         pd.Index([1, np.nan], dtype="Int64", name="idx"),
        ...         pd.Index([1, 2, 3], name="idx"),
        ...     ]
        ... )
        Index([1], dtype='Int64', name='idx')
        >>> intersect(
        ...     [
        ...         pd.Index([0, 1], name="idx"),
        ...         pd.MultiIndex.from_arrays([[1, 2]], names=["idx"]),
        ...     ]
        ... )
        Index([1], dtype='Int64', name='idx')
        >>> intersect(
        ...     [
        ...         pd.MultiIndex.from_arrays(
        ...             [["a", "b", "c"], [0, 1, 2]],
        ...             names=["idx1", "idx2"],
        ...         ),
        ...         pd.MultiIndex.from_arrays(
        ...             [["b", "c"], [1, 3]],
        ...             names=["idx1", "idx2"],
        ...         ),
        ...     ]
        ... )
        MultiIndex([('b', 1)],
                   names=['idx1', 'idx2'])
        >>> intersect(
        ...     [
        ...         filewise_index(["f1", "f2", "f3"]),
        ...         filewise_index(["f2", "f3", "f4"]),
        ...     ]
        ... )
        Index(['f2', 'f3'], dtype='string', name='file')
        >>> intersect(
        ...     [
        ...         segmented_index(["f1"], [0], [1]),
        ...         segmented_index(["f1", "f2"], [0, 1], [1, 2]),
        ...     ]
        ... )
        MultiIndex([('f1', '0 days', '0 days 00:00:01')],
                   names=['file', 'start', 'end'])
        >>> intersect(
        ...     [
        ...         filewise_index(["f1", "f2"]),
        ...         segmented_index(["f1", "f2"], [0, 0], [pd.NaT, 1]),
        ...     ]
        ... )
        MultiIndex([('f1', '0 days', NaT)],
                   names=['file', 'start', 'end'])

    """
    if not objs:
        return pd.Index([])

    objs = [_maybe_convert_pandas_dtype(obj) for obj in objs]

    if len(objs) == 1:
        return _alike_index(objs[0])

    objs = _maybe_convert_filewise_index(objs)
    objs = _maybe_convert_single_level_multi_index(objs)
    _assert_index_alike(objs)

    # sort objects by length
    objs_sorted = sorted(objs, key=lambda obj: len(obj))

    # return if the shortest obj has no entries
    if len(objs_sorted[0]) == 0:
        return _alike_index(objs[0])

    # start from shortest object
    index = list(objs_sorted[0])
    for obj in objs_sorted[1:]:
        index = [idx for idx in index if idx in obj]
        if len(index) == 0:
            # break early if no more intersection is possible
            break

    index = _alike_index(objs[0], index)

    # Ensure we have order of first object
    index = objs[0].intersection(index)
    if isinstance(index, pd.MultiIndex):
        index = set_index_dtypes(index, objs[0].dtypes.to_dict())

    return index


def is_index_alike(
    objs: Sequence[pd.Index | pd.Series | pd.DataFrame],
) -> bool:
    r"""Check if index objects are alike.

    Two index objects are alike
    if they have the same number of levels
    and share the same level names.
    In addition,
    the dtypes have to match the the same audformat dtypes category,
    compare :class:`audformat.define.DataType`.

    Args:
        objs: objects

    Returns:
        ``True`` if index objects are alike, otherwise ``False``

    Examples:
        >>> index1 = pd.Index([1, 2, 3], dtype="Int64", name="l")
        >>> index2 = pd.MultiIndex.from_arrays([[10, 20]], names=["l"])
        >>> is_index_alike([index1, index2])
        True
        >>> is_index_alike([index1, pd.Series(["a", "b"], index=index2)])
        True
        >>> index3 = index2.set_names(["L"])
        >>> is_index_alike([index2, index3])
        False
        >>> index4 = index2.set_levels([["10", "20"]])
        >>> is_index_alike([index2, index4])
        False
        >>> index5 = pd.MultiIndex.from_arrays([[1], ["a"]], names=["l1", "l2"])
        >>> is_index_alike([index2, index5])
        False
        >>> index6 = pd.MultiIndex.from_arrays([["a"], [1]], names=["l2", "l1"])
        >>> is_index_alike([index5, index6])
        False

    """  # noqa: E501
    objs = [obj if isinstance(obj, pd.Index) else obj.index for obj in objs]

    # check names
    levels = {obj.names for obj in objs}
    if len(levels) > 1:
        return False

    # check dtypes
    dtypes = set()
    for obj in objs:
        dtypes.add(tuple(_audformat_dtypes(obj)))
    if len(dtypes) > 1:
        return False

    return True


def iter_by_file(
    obj: (pd.Index | pd.Series | pd.DataFrame),
) -> Iterator[
    tuple[
        str,
        pd.Index | pd.Series | pd.DataFrame,
    ],
]:
    r"""Iterate over object by file.

    Each iteration returns a file and the according sub-object.

    Args:
        obj: object conform to
            :ref:`table specifications <data-tables:Tables>`

    Returns:
        iterator in form of (file, sub_obj)

    Examples:
        >>> index = filewise_index(["f1", "f1", "f2"])
        >>> next(iter_by_file(index))
        ('f1', Index(['f1'], dtype='string', name='file'))
        >>> index = segmented_index(["f1", "f1", "f2"], [0, 1, 0], [2, 3, 1])
        >>> next(iter_by_file(index))
        ('f1', MultiIndex([('f1', '0 days 00:00:00', '0 days 00:00:02'),
            ('f1', '0 days 00:00:01', '0 days 00:00:03')],
           names=['file', 'start', 'end']))
        >>> obj = pd.Series(["a", "b", "b"], index)
        >>> next(iter_by_file(obj))
        ('f1', file  start            end
        f1    0 days 00:00:00  0 days 00:00:02    a
              0 days 00:00:01  0 days 00:00:03    b
        dtype: object)

    """
    is_index = isinstance(obj, pd.Index)
    index = obj if is_index else obj.index

    # We use len() here as index.empty takes a very long time
    if len(index) != 0:
        files = index.get_level_values("file").drop_duplicates()
        if is_filewise_index(index):
            for file in files:
                sub_index = filewise_index(file)
                sub_obj = sub_index if is_index else obj.loc[sub_index]
                yield file, sub_obj
        else:
            for file in files:
                sub_index = index[index.get_loc(file)]
                sub_obj = sub_index if is_index else obj.loc[sub_index]
                yield file, sub_obj


def join_labels(
    labels: Sequence[list | dict],
) -> list | dict:
    r"""Combine scheme labels.

    Args:
        labels: sequence of labels to join.
            For dictionary labels,
            labels further to the right
            can overwrite previous labels

    Returns:
        joined labels

    Raises:
        ValueError: if labels are of different dtype
            or not ``list`` or ``dict``

    Examples:
        >>> join_labels([{"a": 0, "b": 1}, {"b": 2, "c": 2}])
        {'a': 0, 'b': 2, 'c': 2}

    """
    if len(labels) == 0:
        return []

    if not isinstance(labels, list):
        labels = list(labels)

    if misc_table_ids := [x for x in labels if isinstance(x, str)]:
        raise ValueError(
            f"The following string values were provided: '"
            f"{misc_table_ids}'. "
            "This assumes that labels are defined "
            "in misc tables with according IDs, "
            "which is not supported by 'join_labels()'."
        )

    if not (
        all(isinstance(x, list) for x in labels)
        or all(isinstance(x, dict) for x in labels)
    ):
        raise ValueError("All labels must be either of type 'list' or 'dict'.")

    if len(labels) == 1:
        return labels[0]

    items = audeer.flatten_list([list(x) for x in labels])
    dtypes = sorted(list({str(type(x)) for x in items}))
    if len(dtypes) > 1:
        raise ValueError(
            f"Elements or keys must "
            f"have the same dtype, "
            f"but yours have "
            f"{dtypes}.",
        )

    if isinstance(labels[0], dict):
        joined_labels = labels[0]
        for label in labels[1:]:
            for key, value in label.items():
                if key not in joined_labels or joined_labels[key] != value:
                    joined_labels[key] = value
    else:
        joined_labels = sorted(list(set(items)))

    # Check if joined labels have a valid format,
    # e.g. {0: {'age': 20}, '0': {'age': 30}} is not allowed
    Scheme(labels=joined_labels)

    return joined_labels


def join_schemes(
    dbs: Sequence[Database],
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

    Joining schemes that use labels
    from a misc table is not supported.
    Please use
    :meth:`audformat.Database.update`
    instead.

    Args:
        dbs: sequence of databases
        scheme_id: scheme ID of a scheme with labels
            that should be joined

    Raises:
        ValueError: if scheme labels are of different dtype
            or not ``list`` or ``dict``

    Examples:
        >>> db1 = Database("db1")
        >>> db2 = Database("db2")
        >>> db1.schemes["scheme_id"] = Scheme(labels=["a"])
        >>> db2.schemes["scheme_id"] = Scheme(labels=["b"])
        >>> join_schemes([db1, db2], "scheme_id")
        >>> db1.schemes
        scheme_id:
          dtype: str
          labels: [a, b]

    """
    labels = join_labels([db.schemes[scheme_id].labels for db in dbs])
    for db in dbs:
        db.schemes[scheme_id].replace_labels(labels)


def map_country(country: str) -> str:
    r"""Map country to ISO 3166-1.

    Args:
        country: country string

    Returns:
        mapped string

    Raises:
        ValueError: if country is not supported

    Examples:
        >>> map_country("gb")
        'GBR'
        >>> map_country("gbr")
        'GBR'
        >>> map_country("United Kingdom of Great Britain and Northern Ireland")
        'GBR'

    """
    try:
        result = iso3166.countries.get(country.lower())
    except KeyError:
        raise ValueError(f"'{country}' is not supported by ISO 3166-1.")

    return result.alpha3


def map_file_path(
    index: pd.Index,
    func: Callable[[str], str],
) -> pd.Index:
    r"""Apply callable to file path in index.

    Relies on :meth:`pandas.Index.map`,
    which can be slow.
    If speed is crucial,
    consider to change the index directly.
    In the following example we prefix every file with a folder
    and add a new extension,
    compare also :func:`audformat.utils.expand_file_path`
    and :func:`audformat.utils.replace_file_extension`:

    .. code-block:: python

        root = "/root/"
        ext = ".new"
        if table.is_filewise:
            table.df.index = root + table.df.index + ext
            table.df.index.name = audformat.define.IndexField.FILE
        elif len(table.df.index) > 0:
            table.df.index = table.df.index.set_levels(
                root + table.df.index.levels[0] + ext,
                level=audformat.define.IndexField.FILE,
            )

    Args:
        index: index with file path conform to
            :ref:`table specifications <data-tables:Tables>`
        func: callable

    Returns:
        index modified by ``func``

    Raises:
        ValueError: if index is not conform to
            :ref:`table specifications <data-tables:Tables>`

    Examples:
        >>> index = filewise_index(["a/f1", "a/f2"])
        >>> index
        Index(['a/f1', 'a/f2'], dtype='string', name='file')
        >>> map_file_path(index, lambda x: x.replace("a", "b"))
        Index(['b/f1', 'b/f2'], dtype='string', name='file')

    """
    if len(index) == 0:
        return index

    if is_segmented_index(index):
        index = index.set_levels(
            index.levels[0].map(func),
            level=0,
        )
    else:
        index = index.map(func)

    return index


def map_language(language: str) -> str:
    r"""Map language to ISO 639-3.

    `ISO 639-3`_ is an international standard for language codes.
    It defines three-letter codes for identifying languages,
    with the aim to cover all known languages.

    .. _ISO 639-3: https://iso639-3.sil.org/code_tables/639/data/

    Args:
        language: language string

    Returns:
        mapped string

    Raises:
        ValueError: if language is not supported

    Examples:
        >>> map_language("en")
        'eng'
        >>> map_language("eng")
        'eng'
        >>> map_language("English")
        'eng'

    """
    try:
        return iso639.Lang(
            language.title() if len(language) > 3 else language.lower()
        ).pt3
    except InvalidLanguageValue:
        raise ValueError(f"'{language}' is not supported by ISO 639-3.")


def read_csv(
    *args,
    as_dataframe: bool = False,
    **kwargs,
) -> pd.Index | pd.Series | pd.DataFrame:
    r"""Read object from CSV file.

    Automatically detects the index type and returns an object that is
    conform to :ref:`table specifications <data-tables:Tables>`.
    If conversion is not possible, an error is raised.

    Time values for the ``start`` and ``end`` column
    are converted using :func:`pandas.to_timedelta`,
    whereby float and integers are treated as seconds.


    Args:
        *args: arguments passed on to :func:`pandas.read_csv`
        as_dataframe: if ``False``,
            a dataframe is only returned for data with two or more columns,
            a series for data with one column,
            an index for data with zero columns
        **kwargs: keyword arguments passed on to :func:`pandas.read_csv`

    Returns:
        object conform to :ref:`table specifications <data-tables:Tables>`

    Raises:
        ValueError: if CSV file is not conform to
            :ref:`table specifications <data-tables:Tables>`

    Examples:
        >>> string = '''file,start,end,value
        ... f1,00:00:00,00:00:01,0.0
        ... f1,00:00:01,00:00:02,1.0
        ... f2,00:00:02,00:00:03,2.0'''
        >>> with open("file.csv", "w") as file:
        ...     _ = file.write(string)
        >>> read_csv("file.csv")
        file  start            end
        f1    0 days 00:00:00  0 days 00:00:01    0.0
              0 days 00:00:01  0 days 00:00:02    1.0
        f2    0 days 00:00:02  0 days 00:00:03    2.0
        Name: value, dtype: float64
        >>> string = '''file,start,end,value
        ... f1,0,1,0.0
        ... f1,1,2,1.0
        ... f2,2,3,2.0'''
        >>> with open("file.csv", "w") as file:
        ...     _ = file.write(string)
        >>> read_csv("file.csv")
        file  start            end
        f1    0 days 00:00:00  0 days 00:00:01    0.0
              0 days 00:00:01  0 days 00:00:02    1.0
        f2    0 days 00:00:02  0 days 00:00:03    2.0
        Name: value, dtype: float64
        >>> read_csv("file.csv", as_dataframe=True)
                                              value
        file start           end
        f1   0 days 00:00:00 0 days 00:00:01    0.0
             0 days 00:00:01 0 days 00:00:02    1.0
        f2   0 days 00:00:02 0 days 00:00:03    2.0

    """
    frame = pd.read_csv(*args, **kwargs)

    drop = [define.IndexField.FILE]
    if define.IndexField.FILE in frame.columns:
        files = frame[define.IndexField.FILE].astype("string")
    else:
        raise ValueError("Index not conform to audformat.")

    starts = None
    if define.IndexField.START in frame.columns:
        starts = frame[define.IndexField.START]
        drop.append(define.IndexField.START)

    ends = None
    if define.IndexField.END in frame.columns:
        ends = frame[define.IndexField.END]
        drop.append(define.IndexField.END)

    if starts is None and ends is None:
        index = filewise_index(files)
    else:
        index = segmented_index(files, starts=starts, ends=ends)
    frame.drop(drop, axis="columns", inplace=True)

    if len(frame.columns) == 0 and not as_dataframe:
        return index

    frame = frame.set_index(index)
    if len(frame.columns) == 1 and not as_dataframe:
        return frame[frame.columns[0]]
    else:
        return frame


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

    Examples:
        >>> index = filewise_index(["f1.wav", "f2.flac"])
        >>> replace_file_extension(index, "mp3")
        Index(['f1.mp3', 'f2.mp3'], dtype='string', name='file')
        >>> index = filewise_index(["f1.wav.gz", "f2.wav.gz"])
        >>> replace_file_extension(index, "")
        Index(['f1.wav', 'f2.wav'], dtype='string', name='file')
        >>> replace_file_extension(index, "flac", pattern=r"\.wav\.gz$")
        Index(['f1.flac', 'f2.flac'], dtype='string', name='file')

    """
    if len(index) == 0:
        return index

    if pattern is None:
        pattern = r"\.[a-zA-Z0-9]+$"
    cur_ext = re.compile(pattern)
    if extension:
        new_ext = f".{extension}"
    else:
        new_ext = ""

    if is_segmented_index(index):
        index = index.set_levels(
            index.levels[0].str.replace(cur_ext, new_ext, regex=True),
            level="file",
        )
    else:
        index = index.str.replace(cur_ext, new_ext, regex=True)

    return index


def set_index_dtypes(
    index: pd.Index,
    dtypes: (str | dict[str, str]),
) -> pd.Index:
    r"""Set the dtypes of an index for the given level names.

    Args:
        index: index object
        dtypes: dictionary mapping level names to new dtype.
            If a single dtype is given,
            it will be applied to all levels

    Raises:
        ValueError: if level names are not unique
        ValueError: if level does not exist

    Returns:
        index with new dtypes

    Examples:
        >>> index1 = pd.Index(["a", "b"])
        >>> index1
        Index(['a', 'b'], dtype='object')
        >>> index2 = set_index_dtypes(index1, "string")
        >>> index2
        Index(['a', 'b'], dtype='string')
        >>> index3 = pd.MultiIndex.from_arrays(
        ...     [["a", "b"], [1, 2]],
        ...     names=["level1", "level2"],
        ... )
        >>> index3.dtypes
        level1    object
        level2     int64
        dtype: object
        >>> index4 = set_index_dtypes(index3, {"level2": "float"})
        >>> index4.dtypes
        level1    object
        level2   float64
        dtype: object
        >>> index5 = set_index_dtypes(index3, "string")
        >>> index5.dtypes
        level1    string[python]
        level2    string[python]
        dtype: object

    """
    levels = index.names if isinstance(index, pd.MultiIndex) else [index.name]

    if len(set(levels)) != len(levels):
        raise ValueError(
            f"Got index with levels " f"{levels}, " f"but names must be unique."
        )

    if not isinstance(dtypes, dict):
        dtypes = {level: dtypes for level in levels}

    for name in dtypes:
        if name not in levels:
            raise ValueError(
                f"A level with name "
                f"'{name}' "
                f"does not exist. "
                f"Level names are: "
                f"{levels}."
            )

    if len(dtypes) == 0:
        return index

    if isinstance(index, pd.MultiIndex):
        # MultiIndex
        if any([len(index.levels[index.names.index(level)]) == 0 for level in dtypes]):
            # set_levels() does not work on empty levels,
            # so we convert to a dataframe instead
            df = index.to_frame()
            for level, dtype in dtypes.items():
                if dtype != df[level].dtype:
                    if pd.api.types.is_timedelta64_dtype(dtype):
                        # avoid: TypeError: Cannot cast DatetimeArray
                        # to dtype timedelta64[ns]
                        df[level] = pd.to_timedelta(list(df[level]))
                    else:
                        df[level] = df[level].astype(dtype)
            index = pd.MultiIndex.from_frame(df)
        else:
            for level, dtype in dtypes.items():
                # get_level_values() does not work
                # for levels containing non-unique entries,
                # hence we access the data directly with
                # index.levels[idx]
                idx = index.names.index(level)
                if dtype != index.levels[idx].dtype:
                    index = index.set_levels(
                        index.levels[idx].astype(dtype),
                        level=level,
                        verify_integrity=False,
                    )
    else:
        # Index
        dtype = next(iter(dtypes.values()))
        if dtype != index.dtype:
            index = index.astype(dtype)

    return index


def to_filewise_index(
    obj: pd.Index | pd.Series | pd.DataFrame,
    root: str,
    output_folder: str,
    *,
    num_workers: int = 1,
    progress_bar: bool = False,
) -> pd.Index | pd.Series | pd.DataFrame:
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

    Examples:

        .. skip: start if(sys.platform.startswith("win"))

        >>> index = segmented_index(
        ...     files=["f.wav", "f.wav"],
        ...     starts=[0, 0.5],
        ...     ends=[0.5, 1],
        ... )
        >>> to_filewise_index(index, ".", "split")
        Index(['split/f_0.wav', 'split/f_1.wav'], dtype='string', name='file')

        .. skip: end

    """
    if is_filewise_index(obj):
        return obj

    obj = obj.copy()

    if len(obj) == 0:
        index = filewise_index()
        if not isinstance(obj, pd.Index):
            obj.index = index
        else:
            obj = index
        return obj

    index = obj if isinstance(obj, pd.Index) else obj.index
    test_path = index.get_level_values(define.IndexField.FILE)[0]
    is_abs = os.path.isabs(test_path)
    test_path = audeer.path(test_path)

    # keep ``output_folder`` relative if it's relative
    if test_path.startswith(audeer.path(output_folder)):
        raise ValueError(
            f"``output_folder`` may not be contained in path to files of "
            f"original data: {audeer.path(output_folder)} != {test_path}"
        )

    original_files = index.get_level_values(define.IndexField.FILE)
    if not is_abs:
        original_files = [os.path.join(root, f) for f in original_files]
    starts = index.get_level_values(define.IndexField.START)
    ends = index.get_level_values(define.IndexField.END)

    # order of rows within group is preserved:
    # "https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.html"  # noqa
    if isinstance(obj, pd.Index):
        groups = pd.Series(index=obj, dtype="object").groupby(
            define.IndexField.FILE,
            sort=False,
        )
    else:
        groups = obj.groupby(define.IndexField.FILE, sort=False)
    new_files = []

    for _, group in groups:
        width = len(str(len(group) - 1))  # -1 because count starts at `0`
        f = group.index.get_level_values(define.IndexField.FILE)[0]
        f = os.path.relpath(f, root) if is_abs else f
        new_files.extend(
            [
                os.path.join(
                    output_folder,
                    f"_{str(count).zfill(width)}.".join(f.rsplit(".", 1)),
                )
                for count in range(len(group))
            ]
        )
        audeer.mkdir(os.path.dirname(new_files[-1]))

    def _split_files(original, start, end, segment):
        signal, sr = audiofile.read(
            file=original,
            duration=end.total_seconds() - start.total_seconds(),
            offset=start.total_seconds(),
        )
        audiofile.write(file=segment, signal=signal, sampling_rate=sr)

    params = [
        ([file, start, end, segment], {})
        for file, start, end, segment in zip(original_files, starts, ends, new_files)
    ]

    audeer.run_tasks(
        task_func=_split_files,
        params=params,
        task_description="To filewise index",
        num_workers=num_workers,
        progress_bar=progress_bar,
    )

    index = filewise_index(new_files)
    if not isinstance(obj, pd.Index):
        obj.index = index
    else:
        obj = index

    return obj


def to_segmented_index(
    obj: pd.Index | pd.Series | pd.DataFrame,
    *,
    allow_nat: bool = True,
    files_duration: MutableMapping[str, pd.Timedelta] = None,
    root: str = None,
    num_workers: int | None = 1,
    verbose: bool = False,
) -> pd.Index | pd.Series | pd.DataFrame:
    r"""Convert to segmented index.

    If the input a filewise table,
    ``start`` and ``end`` will be added as new levels to the index.
    By default, ``start`` will be set to 0 and ``end`` to ``NaT``.

    If ``allow_nat`` is set to ``False``,
    all occurrences of ``end=NaT``
    are replaced with the duration of the file.
    This requires that the referenced file exists,
    or that the durations are provided with ``files_duration``.
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
            Expects absolute file names
            and durations as :class:`pd.Timedelta` objects.
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

    Examples:
        >>> index = filewise_index(["f1", "f2"])
        >>> to_segmented_index(index)
        MultiIndex([('f1', '0 days', NaT),
                    ('f2', '0 days', NaT)],
                   names=['file', 'start', 'end'])
        >>> to_segmented_index(
        ...     index,
        ...     allow_nat=False,
        ...     files_duration={
        ...         "f1": pd.to_timedelta(1.1, unit="s"),
        ...         "f2": pd.to_timedelta(2.2, unit="s"),
        ...     },
        ... )
        MultiIndex([('f1', '0 days', '0 days 00:00:01.100000'),
                    ('f2', '0 days', '0 days 00:00:02.200000')],
                   names=['file', 'start', 'end'])

    """
    is_segmented = is_segmented_index(obj)

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
            # Gather duration values
            # for all NaT end entries

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
                dur = pd.to_timedelta(dur, unit="s")

                if files_duration is not None:
                    files_duration[file] = dur

                return dur

            params = [([file], {}) for file in files[idx_nat]]
            durs = audeer.run_tasks(
                job,
                params,
                num_workers=num_workers,
                progress_bar=verbose,
                task_description="Read duration",
            )

            # Replace all NaT entries in end
            # by the collected duration values.
            # We have to convert ends to a series first
            # in order to preserve precision of duration values

            ends = ends.to_series()
            ends.iloc[idx_nat] = durs

            # Create a new index
            index = segmented_index(files, starts, ends)

    if isinstance(obj, pd.Index):
        return index

    obj = obj.reset_index(drop=True)
    obj.index = index

    return obj


UNION_MAX_INDEX_LEN_THRES = 500


def union(
    objs: Sequence[pd.Index],
) -> pd.Index:
    r"""Create union of index objects.

    If all index objects are conform to
    :ref:`table specifications <data-tables:Tables>`
    and at least one object is segmented,
    the output is a segmented index.
    Otherwise,
    requires that levels and dtypes
    of all objects match,
    see :func:`audformat.utils.is_index_alike`.
    Integer dtypes don't have to match,
    but the result will always be of dtype ``Int64``.
    When a :class:`pandas.Index`
    is combined with a single-level
    :class:`pandas.MultiIndex`,
    the result is a
    :class:`pandas.Index`.

    The order of the resulting index
    depends on the order of ``objs``.
    If you require :func:`audformat.utils.union`
    to be commutative_,
    you have to sort its output.

    .. _commutative: https://en.wikipedia.org/wiki/Commutative_property

    Args:
        objs: index objects

    Returns:
        union of index objects

    Raises:
        ValueError: if level and dtypes of objects do not match

    Examples:
        >>> union(
        ...     [
        ...         pd.Index([0, 1], name="idx"),
        ...         pd.Index([1, 2], dtype="Int64", name="idx"),
        ...     ]
        ... )
        Index([0, 1, 2], dtype='Int64', name='idx')
        >>> union(
        ...     [
        ...         pd.Index([0, 1], name="idx"),
        ...         pd.MultiIndex.from_arrays([[1, 2]], names=["idx"]),
        ...     ]
        ... )
        Index([0, 1, 2], dtype='Int64', name='idx')
        >>> union(
        ...     [
        ...         pd.MultiIndex.from_arrays(
        ...             [["a", "b", "c"], [0, 1, 2]],
        ...             names=["idx1", "idx2"],
        ...         ),
        ...         pd.MultiIndex.from_arrays(
        ...             [["b", "c"], [1, 3]],
        ...             names=["idx1", "idx2"],
        ...         ),
        ...     ]
        ... )
        MultiIndex([('a', 0),
                    ('b', 1),
                    ('c', 2),
                    ('c', 3)],
                   names=['idx1', 'idx2'])
        >>> union(
        ...     [
        ...         filewise_index(["f1", "f2", "f3"]),
        ...         filewise_index(["f2", "f3", "f4"]),
        ...     ]
        ... )
        Index(['f1', 'f2', 'f3', 'f4'], dtype='string', name='file')
        >>> union(
        ...     [
        ...         segmented_index(["f2"], [0], [1]),
        ...         segmented_index(["f1", "f2"], [0, 1], [1, 2]),
        ...     ]
        ... )
        MultiIndex([('f2', '0 days 00:00:00', '0 days 00:00:01'),
                    ('f1', '0 days 00:00:00', '0 days 00:00:01'),
                    ('f2', '0 days 00:00:01', '0 days 00:00:02')],
                   names=['file', 'start', 'end'])
        >>> union(
        ...     [
        ...         filewise_index(["f1", "f2"]),
        ...         segmented_index(["f1", "f2"], [0, 0], [1, 1]),
        ...     ]
        ... )
        MultiIndex([('f1', '0 days',               NaT),
                    ('f2', '0 days',               NaT),
                    ('f1', '0 days', '0 days 00:00:01'),
                    ('f2', '0 days', '0 days 00:00:01')],
                   names=['file', 'start', 'end'])

    """
    if not objs:
        return pd.Index([])

    objs = [_maybe_convert_pandas_dtype(obj) for obj in objs]

    if len(objs) == 1:
        return objs[0]

    objs = _maybe_convert_filewise_index(objs)
    objs = _maybe_convert_single_level_multi_index(objs)
    _assert_index_alike(objs)

    # Combine all index entries and drop duplicates afterwards,
    # faster than using index.union(),
    # compare https://github.com/audeering/audformat/pull/98

    # Use pd.concat() if at least one index has
    # more than 500 segments
    # otherwise create index from lists,
    # compare https://github.com/audeering/audformat/pull/354

    max_num_seg = max([len(obj) for obj in objs])
    if max_num_seg > UNION_MAX_INDEX_LEN_THRES:
        df = pd.concat([o.to_frame() for o in objs])
        index = df.index

    elif isinstance(objs[0], pd.MultiIndex):
        names = objs[0].names
        num_levels = len(names)
        dtypes = {name: dtype for name, dtype in zip(names, objs[0].dtypes)}
        values = [[] for _ in range(num_levels)]

        for obj in objs:
            for idx in range(num_levels):
                values[idx].extend(obj.get_level_values(idx))

        index = pd.MultiIndex.from_arrays(
            values,
            names=names,
        )
        index = set_index_dtypes(index, dtypes)

    else:
        name = objs[0].name
        values = []

        for obj in objs:
            values.extend(obj.to_list())

        index = pd.Index(values, name=name)
        index = set_index_dtypes(index, objs[0].dtype)

    index = index.drop_duplicates()

    return index


def _alike_index(
    index: pd.Index,
    data: Sequence = [],
) -> pd.Index:
    if isinstance(index, pd.MultiIndex):
        return set_index_dtypes(
            pd.MultiIndex.from_tuples(data, names=list(index.names)),
            index.dtypes.to_dict(),
        )
    else:
        return pd.Index(
            data,
            dtype=index.dtype,
            name=index.name,
        )


def _assert_index_alike(
    objs: Sequence[pd.Index | pd.Series | pd.DataFrame],
):
    r"""Raise if index objects are not alike.

    Args:
        objs: objects

    Raises:
        ValueError: if index objects are not alike

    """
    if is_index_alike(objs):
        return

    objs = [obj if isinstance(obj, pd.Index) else obj.index for obj in objs]
    msg = "Levels and dtypes of all objects must match."

    dims = list(dict.fromkeys(obj.nlevels for obj in objs))
    if len(dims) > 1:
        msg += f" Found different number of levels: {dims}."
        raise ValueError(msg)

    names = []
    for obj in objs:
        if len(obj.names) > 1:
            names.append(tuple([name for name in obj.names]))
        else:
            names.append(obj.names[0])
    names = list(dict.fromkeys(names))
    if len(names) > 1:
        msg += f" Found different level names: {names}."
        raise ValueError(msg)

    dtypes = []
    for obj in objs:
        ds = _audformat_dtypes(obj)
        dtypes.append(tuple(ds) if len(ds) > 1 else ds[0])
    dtypes = list(dict.fromkeys(dtypes))
    if len(dtypes) > 1:
        msg += f" Found different level dtypes: {dtypes}."

    raise ValueError(msg)


def _audformat_dtypes(index) -> list[str]:
    r"""List of audformat data types of index.

    Args:
        index: index

    Returns:
        audformat data types of index

    """
    dtypes = _pandas_dtypes(index)
    return [to_audformat_dtype(dtype) for dtype in dtypes]


def _is_same_dtype(d1, d2) -> bool:
    r"""Helper function to compare pandas dtype."""
    if d1.name.startswith("bool") and d2.name.startswith("bool"):
        # match different bool types, i.e. bool and boolean
        return True
    if d1.name.lower().startswith("int") and d2.name.lower().startswith("int"):
        # match different int types, e.g. int64 and Int64
        return True
    if d1.name.startswith("float") and d2.name.startswith("float"):
        # match different float types, e.g. float32 and float64
        return True
    if d1.name == "category" and d2.name == "category":
        # match only if categories are the same
        return d1 == d2
    return d1.name == d2.name


def _levels(index) -> list[str]:
    r"""List of levels of index.

    Args:
        index: index

    Returns:
        index levels

    """
    if isinstance(index, pd.MultiIndex):
        return list(index.names)
    else:
        return [index.name]


def _maybe_convert_filewise_index(
    objs: Sequence[pd.Index | pd.Series | pd.DataFrame],
) -> Sequence[pd.Index | pd.Series | pd.DataFrame]:
    r"""Convert filewise to segmented index.

    Checks if all index objects are either filewise or segmented,
    if this is the case possibly convert filewise to segmented indices

    Args:
        objs: list with objects

    Returns:
        list with possibly converted index objects

    """
    filewise = np.array([is_filewise_index(obj) for obj in objs])
    segmented = np.array([is_segmented_index(obj) for obj in objs])
    if (filewise | segmented).all():
        if not filewise.all():
            objs = [to_segmented_index(obj) for obj in objs]

    return objs


def _maybe_convert_pandas_dtype(
    index: pd.Index,
) -> pd.Index:
    r"""Ensure desired pandas dtypes.

    Applies the following conversions:

    * integer -> Int64
    * bool -> boolean

    Args:
        index: index object

    Returns:
        index object

    """
    levels = _levels(index)
    dtypes = _pandas_dtypes(index)

    # Ensure integers are stored as Int64
    int_dtypes = {
        level: "Int64"
        for level, dtype in zip(levels, dtypes)
        if pd.api.types.is_integer_dtype(dtype)
    }
    # Ensure bool values are stored as boolean
    bool_dtypes = {
        level: "boolean"
        for level, dtype in zip(levels, dtypes)
        if pd.api.types.is_bool_dtype(dtype)
    }
    # Merge dictionaries
    dtypes = {**int_dtypes, **bool_dtypes}

    index = set_index_dtypes(index, dtypes)
    return index


def _maybe_convert_single_level_multi_index(
    objs: Sequence[pd.Index | pd.Series | pd.DataFrame],
) -> Sequence[pd.Index | pd.Series | pd.DataFrame]:
    r"""Convert single-level pd.MultiIndex to pd.Index.

    If input is a mixture of single-level
    pd.MultiIndex and pd.Index objects,
    all objects are converted to pd.Index.
    Assumes that list is not empty.

    Args:
        objs: list with objects

    Returns:
        list with possibly converted objects

    """
    indices = [obj if isinstance(obj, pd.Index) else obj.index for obj in objs]
    is_single_level = indices[0].nlevels == 1
    is_mix = len({isinstance(index, pd.MultiIndex) for index in indices}) == 2

    if is_single_level and is_mix:
        objs = list(objs)
        for idx, obj in enumerate(objs):
            if isinstance(obj, pd.MultiIndex):
                objs[idx] = obj.get_level_values(0)
            elif not isinstance(obj, pd.Index) and isinstance(obj.index, pd.MultiIndex):
                objs[idx].index = obj.index.get_level_values(0)

    return objs


def _pandas_dtypes(index) -> list[object]:
    r"""List of pandas dtypes of index.

    Args:
        index: index

    Returns:
        pandas data types of index

    """
    if isinstance(index, pd.MultiIndex):
        return list(index.dtypes)
    else:
        return [index.dtype]
