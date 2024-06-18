import re

import numpy as np
import pandas as pd
import pytest

import audformat


@pytest.mark.parametrize(
    "objs, overwrite, expected",
    [
        # empty
        (
            [],
            False,
            pd.Series([], pd.Index([]), dtype="object"),
        ),
        (
            [pd.Series([], pd.Index([]), dtype="object")],
            False,
            pd.Series([], pd.Index([]), dtype="object"),
        ),
        (
            [pd.Series([], audformat.filewise_index(), dtype="object")],
            False,
            pd.Series([], audformat.filewise_index(), dtype="object"),
        ),
        (
            [pd.Series([], audformat.segmented_index(), dtype="object")],
            False,
            pd.Series([], audformat.segmented_index(), dtype="object"),
        ),
        (
            [pd.DataFrame([], audformat.segmented_index(), dtype="object")],
            False,
            pd.DataFrame([], audformat.segmented_index(), dtype="object"),
        ),
        # combine series with same name
        (
            [
                pd.Series([], audformat.filewise_index(), dtype=float),
                pd.Series([1.0, 2.0], audformat.filewise_index(["f1", "f2"])),
            ],
            False,
            pd.Series([1.0, 2.0], audformat.filewise_index(["f1", "f2"])),
        ),
        (
            [
                pd.Series([1.0, 2.0], pd.Index(["f1", "f2"])),
                pd.Series([1.0, 2.0], pd.Index(["f1", "f2"])),
            ],
            False,
            pd.Series([1.0, 2.0], pd.Index(["f1", "f2"])),
        ),
        (
            [
                pd.Series([1.0], audformat.filewise_index("f1")),
                pd.Series([2.0], audformat.filewise_index("f2")),
            ],
            False,
            pd.Series([1.0, 2.0], audformat.filewise_index(["f1", "f2"])),
        ),
        (
            [
                pd.Series([1.0], audformat.segmented_index("f1")),
                pd.Series([2.0], audformat.segmented_index("f2")),
            ],
            False,
            pd.Series([1.0, 2.0], audformat.segmented_index(["f1", "f2"])),
        ),
        (
            [
                pd.Series([1.0], audformat.filewise_index("f1")),
                pd.Series([2.0], audformat.segmented_index("f2")),
            ],
            False,
            pd.Series([1.0, 2.0], audformat.segmented_index(["f1", "f2"])),
        ),
        (
            [
                pd.Series([1.0], pd.Index(["f1"])),
                pd.Series(
                    [2.0],
                    pd.MultiIndex.from_arrays([["f2"]]),
                ),
            ],
            False,
            pd.Series(
                [1.0, 2.0],
                pd.Index(["f1", "f2"]),
            ),
        ),
        (
            [
                pd.Series([1.0], pd.Index(["f1"], name="idx")),
                pd.Series(
                    [2.0],
                    pd.MultiIndex.from_arrays([["f2"]], names=["idx"]),
                ),
            ],
            False,
            pd.Series(
                [1.0, 2.0],
                pd.Index(["f1", "f2"], name="idx"),
            ),
        ),
        # combine values in same location
        (
            [
                pd.Series([np.nan], audformat.filewise_index("f1")),
                pd.Series([np.nan], audformat.filewise_index("f1")),
            ],
            False,
            pd.Series([np.nan], audformat.filewise_index("f1")),
        ),
        (
            [
                pd.Series([1.0], audformat.filewise_index("f1")),
                pd.Series([np.nan], audformat.filewise_index("f1")),
            ],
            False,
            pd.Series([1.0], audformat.filewise_index("f1")),
        ),
        (
            [
                pd.Series([1.0], audformat.filewise_index("f1")),
                pd.Series([1.0], audformat.filewise_index("f1")),
            ],
            False,
            pd.Series([1.0], audformat.filewise_index("f1")),
        ),
        # combine series and overwrite values
        (
            [
                pd.Series([1.0], audformat.filewise_index("f1")),
                pd.Series([np.nan], audformat.filewise_index("f1")),
            ],
            True,
            pd.Series([1.0], audformat.filewise_index("f1")),
        ),
        (
            [
                pd.Series([1.0], audformat.filewise_index("f1")),
                pd.Series([2.0], audformat.filewise_index("f1")),
            ],
            True,
            pd.Series([2.0], audformat.filewise_index("f1")),
        ),
        # combine values with matching dtype
        (
            [
                pd.Series(
                    [1, 2],
                    audformat.filewise_index(["f1", "f2"]),
                    dtype="int64",
                ),
                pd.Series(
                    [1, 2],
                    audformat.filewise_index(["f1", "f2"]),
                    dtype="Int64",
                ),
            ],
            False,
            pd.Series(
                [1, 2],
                audformat.filewise_index(["f1", "f2"]),
                dtype="Int64",
            ),
        ),
        (
            [
                pd.Series(
                    [1.0, 2.0],
                    audformat.filewise_index(["f1", "f2"]),
                    dtype="float32",
                ),
                pd.Series(
                    [1.0, 2.0],
                    audformat.filewise_index(["f1", "f2"]),
                    dtype="float64",
                ),
            ],
            False,
            pd.Series(
                [1.0, 2.0],
                audformat.filewise_index(["f1", "f2"]),
                dtype="float64",
            ),
        ),
        (
            [
                pd.Series(
                    [1.0, 2.0],
                    audformat.filewise_index(["f1", "f2"]),
                    dtype="float32",
                ),
                pd.Series(
                    [1.0, 2.0],
                    audformat.filewise_index(["f1", "f2"]),
                    dtype="float64",
                ),
            ],
            False,
            pd.Series(
                [1.0, 2.0],
                audformat.filewise_index(["f1", "f2"]),
                dtype="float64",
            ),
        ),
        (
            [
                pd.Series(
                    ["a", "b", "a"],
                    index=audformat.filewise_index(["f1", "f2", "f3"]),
                ),
                pd.Series(
                    ["a", "b", "a"],
                    index=audformat.filewise_index(["f1", "f2", "f3"]),
                ),
            ],
            False,
            pd.Series(
                ["a", "b", "a"],
                index=audformat.filewise_index(["f1", "f2", "f3"]),
            ),
        ),
        (
            [
                pd.Series(
                    ["a", "b", "a"],
                    index=audformat.filewise_index(["f1", "f2", "f3"]),
                    dtype="category",
                ),
                pd.Series(
                    ["a", "b", "a"],
                    index=audformat.filewise_index(["f1", "f2", "f3"]),
                    dtype="category",
                ),
            ],
            False,
            pd.Series(
                ["a", "b", "a"],
                index=audformat.filewise_index(["f1", "f2", "f3"]),
                dtype="category",
            ),
        ),
        # combine series with non-nullable dtype
        (
            [
                pd.Series([1, 2], audformat.filewise_index(["f1", "f2"])),
                pd.Series([1, 2], audformat.filewise_index(["f1", "f2"])),
            ],
            False,
            pd.Series([1, 2], audformat.filewise_index(["f1", "f2"]), dtype="Int64"),
        ),
        (
            [
                pd.Series(
                    True,
                    audformat.filewise_index("f1"),
                    dtype="bool",
                ),
                pd.Series(
                    True,
                    audformat.filewise_index("f2"),
                    dtype="bool",
                ),
            ],
            False,
            pd.Series(
                True,
                audformat.filewise_index(["f1", "f2"]),
                dtype="boolean",
            ),
        ),
        (
            [
                pd.Series(
                    1,
                    audformat.filewise_index("f1"),
                    dtype="int64",
                ),
                pd.Series(
                    2,
                    audformat.filewise_index("f2"),
                    dtype="int64",
                ),
            ],
            False,
            pd.Series(
                [1, 2],
                audformat.filewise_index(["f1", "f2"]),
                dtype="Int64",
            ),
        ),
        # combine series with different names
        (
            [
                pd.Series([1.0], audformat.filewise_index("f1"), name="c1"),
                pd.Series([2.0], audformat.filewise_index("f1"), name="c2"),
            ],
            False,
            pd.DataFrame(
                {
                    "c1": [1.0],
                    "c2": [2.0],
                },
                audformat.filewise_index("f1"),
            ),
        ),
        (
            [
                pd.Series([1.0], audformat.filewise_index("f1"), name="c1"),
                pd.Series([2.0], audformat.filewise_index("f2"), name="c2"),
            ],
            False,
            pd.DataFrame(
                {
                    "c1": [1.0, np.nan],
                    "c2": [np.nan, 2.0],
                },
                audformat.filewise_index(["f1", "f2"]),
            ),
        ),
        (
            [
                pd.Series(
                    [1.0, 2.0],
                    audformat.filewise_index(["f1", "f2"]),
                    name="c1",
                ),
                pd.Series(
                    [2.0],
                    audformat.filewise_index("f2"),
                    name="c2",
                ),
            ],
            False,
            pd.DataFrame(
                {
                    "c1": [1.0, 2.0],
                    "c2": [np.nan, 2.0],
                },
                audformat.filewise_index(["f1", "f2"]),
            ),
        ),
        (
            [
                pd.Series([1.0], audformat.filewise_index("f1"), name="c1"),
                pd.Series(
                    [2.0],
                    audformat.segmented_index("f1", 0, 1),
                    name="c2",
                ),
            ],
            False,
            pd.DataFrame(
                {
                    "c1": [1.0, np.nan],
                    "c2": [np.nan, 2.0],
                },
                audformat.segmented_index(
                    ["f1", "f1"],
                    [0, 0],
                    [None, 1],
                ),
            ),
        ),
        # combine series and data frame
        (
            [
                pd.Series(
                    [1.0, 2.0],
                    audformat.filewise_index(["f1", "f2"]),
                    name="c",
                ),
                pd.DataFrame(
                    {"c": [2.0, 3.0]},
                    audformat.filewise_index(["f2", "f3"]),
                ),
            ],
            False,
            pd.DataFrame(
                {
                    "c": [1.0, 2.0, 3.0],
                },
                audformat.filewise_index(["f1", "f2", "f3"]),
            ),
        ),
        (
            [
                pd.Series(
                    [1.0, 2.0],
                    audformat.filewise_index(["f1", "f2"]),
                    name="c1",
                ),
                pd.Series(
                    ["a", np.nan, "c"],
                    audformat.filewise_index(["f1", "f2", "f3"]),
                    name="c2",
                ),
                pd.DataFrame(
                    {
                        "c1": [np.nan, 4.0],
                        "c2": ["b", "d"],
                    },
                    audformat.segmented_index(["f2", "f4"]),
                ),
            ],
            False,
            pd.DataFrame(
                {"c1": [1.0, 2.0, np.nan, 4.0], "c2": ["a", "b", "c", "d"]},
                audformat.segmented_index(["f1", "f2", "f3", "f4"]),
            ),
        ),
    ],
)
def test_concat(objs, overwrite, expected):
    obj = audformat.utils.concat(objs, overwrite=overwrite)
    if isinstance(obj, pd.Series):
        pd.testing.assert_series_equal(obj, expected)
    else:
        pd.testing.assert_frame_equal(obj, expected)


@pytest.mark.parametrize(
    "objs, aggregate_function, expected",
    [
        # empty
        (
            [],
            None,
            pd.Series([], pd.Index([]), dtype="object"),
        ),
        # identical values
        (
            [
                pd.Series([1, 2], pd.Index(["a", "b"]), dtype="float"),
                pd.Series([1, 2], pd.Index(["a", "b"]), dtype="float"),
            ],
            None,
            pd.Series([1, 2], pd.Index(["a", "b"]), dtype="float"),
        ),
        (
            [
                pd.Series([1, 2], pd.Index(["a", "b"]), dtype="float"),
                pd.Series([1, 2], pd.Index(["a", "b"]), dtype="float"),
            ],
            np.mean,
            pd.Series([1, 2], pd.Index(["a", "b"]), dtype="float"),
        ),
        (
            [
                pd.Series([1, 2], pd.Index(["a", "b"]), dtype="float"),
                pd.Series([1, 2], pd.Index(["a", "b"]), dtype="float"),
            ],
            np.sum,
            pd.Series([1, 2], pd.Index(["a", "b"]), dtype="float"),
        ),
        (
            [
                pd.Series([1, 2], pd.Index(["a", "b"]), dtype="float"),
                pd.Series([1, 2], pd.Index(["a", "b"]), dtype="float"),
            ],
            np.var,
            pd.Series([1, 2], pd.Index(["a", "b"]), dtype="float"),
        ),
        (
            [
                pd.Series([1, 2], pd.Index(["a", "b"]), dtype="float"),
                pd.Series([1, 2], pd.Index(["a", "b"]), dtype="float"),
            ],
            lambda y: "a",
            pd.Series([1, 2], pd.Index(["a", "b"]), dtype="float"),
        ),
        (
            [
                pd.Series([1, 2], pd.Index(["a", "b"]), dtype="float"),
                pd.Series([1, 2], pd.Index(["a", "b"]), dtype="float"),
            ],
            lambda y: 0,
            pd.Series([1, 2], pd.Index(["a", "b"]), dtype="float"),
        ),
        (
            [
                pd.Series([1, 2], pd.Index(["a", "b"]), dtype="int"),
                pd.Series([1, 2], pd.Index(["a", "b"]), dtype="int"),
            ],
            lambda y: 0,
            pd.Series([1, 2], pd.Index(["a", "b"]), dtype="Int64"),
        ),
        (
            [
                pd.Series([1, 2], pd.Index(["a", "b"]), dtype="int"),
                pd.Series([1, 2], pd.Index(["a", "b"]), dtype="int"),
            ],
            lambda y: 0.5,
            pd.Series([1, 2], pd.Index(["a", "b"]), dtype="Int64"),
        ),
        (
            [
                pd.Series([1, 2], pd.Index(["a", "b"]), dtype="float"),
                pd.Series([1, 2], pd.Index(["a", "b"]), dtype="float"),
            ],
            lambda y: ("a", "b"),
            pd.Series([1, 2], pd.Index(["a", "b"]), dtype="float"),
        ),
        (
            [
                pd.Series([1, 2], pd.Index(["a", "b"]), dtype="float"),
                pd.Series([1, 2], pd.Index(["a", "b"]), dtype="float"),
                pd.Series([1, 2], pd.Index(["a", "b"]), dtype="float"),
            ],
            np.sum,
            pd.Series([1, 2], pd.Index(["a", "b"]), dtype="float"),
        ),
        (
            [
                pd.Series(
                    [1, 2],
                    audformat.filewise_index(["a", "b"]),
                    dtype="float",
                ),
                pd.Series(
                    [1, 2],
                    audformat.filewise_index(["a", "b"]),
                    dtype="float",
                ),
            ],
            np.sum,
            pd.Series(
                [1, 2],
                audformat.filewise_index(["a", "b"]),
                dtype="float",
            ),
        ),
        (
            [
                pd.Series(["a", "b"], pd.Index(["a", "b"]), dtype="string"),
                pd.Series(["a", "b"], pd.Index(["a", "b"]), dtype="string"),
            ],
            lambda y: np.char.add(y[0], y[1]),
            pd.Series(["a", "b"], pd.Index(["a", "b"]), dtype="string"),
        ),
        (
            [
                pd.DataFrame(
                    {
                        "A": [1, 3],
                        "B": [2, 4],
                    },
                    pd.Index(["a", "b"]),
                    dtype="float",
                ),
                pd.DataFrame(
                    {
                        "A": [1, 3],
                        "B": [2, 4],
                    },
                    pd.Index(["a", "b"]),
                    dtype="float",
                ),
            ],
            None,
            pd.DataFrame(
                {
                    "A": [1, 3],
                    "B": [2, 4],
                },
                pd.Index(["a", "b"]),
                dtype="float",
            ),
        ),
        (
            [
                pd.DataFrame(
                    {
                        "A": [1, 3],
                        "B": [2, 4],
                    },
                    pd.Index(["a", "b"]),
                    dtype="float",
                ),
                pd.DataFrame(
                    {
                        "A": [1, 3],
                        "B": [2, 4],
                    },
                    pd.Index(["a", "b"]),
                    dtype="float",
                ),
            ],
            np.sum,
            pd.DataFrame(
                {
                    "A": [1, 3],
                    "B": [2, 4],
                },
                pd.Index(["a", "b"]),
                dtype="float",
            ),
        ),
        (
            [
                pd.DataFrame(
                    {
                        "A": [1, 3],
                        "B": [2, 4],
                    },
                    pd.Index(["a", "b"]),
                    dtype="float",
                ),
                pd.DataFrame(
                    {
                        "A": [1, 3],
                    },
                    pd.Index(["a", "b"]),
                    dtype="float",
                ),
            ],
            np.sum,
            pd.DataFrame(
                {
                    "A": [1, 3],
                    "B": [2, 4],
                },
                pd.Index(["a", "b"]),
                dtype="float",
            ),
        ),
        (
            [
                pd.DataFrame(
                    {
                        "A": [1, 3],
                        "B": [2, 4],
                    },
                    pd.Index(["a", "b"]),
                    dtype="float",
                ),
                pd.DataFrame(
                    {
                        "B": [2, 4],
                    },
                    pd.Index(["a", "b"]),
                    dtype="float",
                ),
            ],
            np.sum,
            pd.DataFrame(
                {
                    "A": [1, 3],
                    "B": [2, 4],
                },
                pd.Index(["a", "b"]),
                dtype="float",
            ),
        ),
        (
            [
                pd.DataFrame(
                    {
                        "A": [1, 3],
                    },
                    pd.Index(["a", "b"]),
                    dtype="float",
                ),
                pd.DataFrame(
                    {
                        "A": [1, 3],
                        "B": [2, 4],
                    },
                    pd.Index(["a", "b"]),
                    dtype="float",
                ),
            ],
            np.sum,
            pd.DataFrame(
                {
                    "A": [1, 3],
                    "B": [2, 4],
                },
                pd.Index(["a", "b"]),
                dtype="float",
            ),
        ),
        (
            [
                pd.DataFrame(
                    {
                        "B": [2, 4],
                    },
                    pd.Index(["a", "b"]),
                    dtype="float",
                ),
                pd.DataFrame(
                    {
                        "A": [1, 3],
                        "B": [2, 4],
                    },
                    pd.Index(["a", "b"]),
                    dtype="float",
                ),
            ],
            np.sum,
            pd.DataFrame(
                {
                    "B": [2, 4],
                    "A": [1, 3],
                },
                pd.Index(["a", "b"]),
                dtype="float",
            ),
        ),
        # different values
        (
            [
                pd.Series([1, 2], pd.Index(["a", "b"]), dtype="float"),
                pd.Series([2, 3], pd.Index(["a", "b"]), dtype="float"),
            ],
            np.mean,
            pd.Series([1.5, 2.5], pd.Index(["a", "b"]), dtype="float"),
        ),
        (
            [
                pd.Series([1, 2], pd.Index(["a", "b"]), dtype="float"),
                pd.Series([2, 3], pd.Index(["a", "b"]), dtype="float"),
            ],
            np.sum,
            pd.Series([3, 5], pd.Index(["a", "b"]), dtype="float"),
        ),
        (
            [
                pd.Series([1, 2], pd.Index(["a", "b"]), dtype="float"),
                pd.Series([2, 3], pd.Index(["a", "b"]), dtype="float"),
                pd.Series([3, 4], pd.Index(["a", "b"]), dtype="float"),
            ],
            np.mean,
            pd.Series([2, 3], pd.Index(["a", "b"]), dtype="float"),
        ),
        (
            [
                pd.Series([1, 2], pd.Index(["a", "b"]), dtype="float"),
                pd.Series([2, 3], pd.Index(["a", "b"]), dtype="float"),
                pd.Series([3, 4], pd.Index(["a", "b"]), dtype="float"),
            ],
            np.sum,
            pd.Series([6, 9], pd.Index(["a", "b"]), dtype="float"),
        ),
        (
            [
                pd.Series(
                    [1, 2],
                    audformat.filewise_index(["a", "b"]),
                    dtype="float",
                ),
                pd.Series(
                    [2, 3],
                    audformat.filewise_index(["a", "b"]),
                    dtype="float",
                ),
            ],
            np.sum,
            pd.Series(
                [3, 5],
                audformat.filewise_index(["a", "b"]),
                dtype="float",
            ),
        ),
        (
            [
                pd.Series(["a", "b"], pd.Index(["a", "b"]), dtype="string"),
                pd.Series(["b", "a"], pd.Index(["a", "b"]), dtype="string"),
            ],
            lambda y: np.char.add(y[0], y[1]),
            pd.Series(["ab", "ba"], pd.Index(["a", "b"]), dtype="string"),
        ),
        (
            [
                pd.Series([1, 2], pd.Index(["a", "b"]), dtype="float"),
                pd.Series([2, 3], pd.Index(["a", "b"]), dtype="float"),
                pd.Series([3, 4], pd.Index(["a", "b"]), dtype="float"),
            ],
            lambda y: y[2],
            pd.Series([3, 4], pd.Index(["a", "b"]), dtype="float"),
        ),
        (
            [
                pd.DataFrame(
                    {
                        "A": [1, 3],
                        "B": [2, 4],
                    },
                    pd.Index(["a", "b"]),
                    dtype="float",
                ),
                pd.DataFrame(
                    {
                        "A": [2, 4],
                        "B": [3, 5],
                    },
                    pd.Index(["a", "b"]),
                    dtype="float",
                ),
            ],
            np.sum,
            pd.DataFrame(
                {
                    "A": [3, 7],
                    "B": [5, 9],
                },
                pd.Index(["a", "b"]),
                dtype="float",
            ),
        ),
        (
            [
                pd.DataFrame(
                    {
                        "A": [1, 3],
                        "B": [2, 4],
                    },
                    pd.Index(["a", "b"]),
                    dtype="float",
                ),
                pd.DataFrame(
                    {
                        "A": [2, 4],
                        "B": [3, 5],
                    },
                    pd.Index(["a", "b"]),
                    dtype="float",
                ),
            ],
            lambda y: y[1],
            pd.DataFrame(
                {
                    "A": [2, 4],
                    "B": [3, 5],
                },
                pd.Index(["a", "b"]),
                dtype="float",
            ),
        ),
        # different index
        (
            [
                pd.DataFrame(
                    {
                        "A": [1, 1, 1],
                        "B": [1, 1, 1],
                    },
                    index=pd.Index(["a", "b", "c"]),
                    dtype="float",
                ),
                pd.DataFrame(
                    {
                        "A": [2, 2, 2],
                        "B": [2, 2, 2],
                    },
                    index=pd.Index(["b", "c", "d"]),
                    dtype="float",
                ),
            ],
            np.sum,
            pd.DataFrame(
                {
                    "A": [1, 3, 3, 2],
                    "B": [1, 3, 3, 2],
                },
                index=pd.Index(["a", "b", "c", "d"]),
                dtype="float",
            ),
        ),
        (
            [
                pd.DataFrame(
                    {
                        "A": [1, 1, 1],
                        "B": [1, 1, 1],
                    },
                    index=pd.Index(["b", "c", "d"]),
                    dtype="float",
                ),
                pd.DataFrame(
                    {
                        "A": [2, 2, 2],
                        "B": [2, 2, 2],
                    },
                    index=pd.Index(["a", "b", "c"]),
                    dtype="float",
                ),
            ],
            np.sum,
            pd.DataFrame(
                {
                    "A": [3, 3, 1, 2],
                    "B": [3, 3, 1, 2],
                },
                index=pd.Index(["b", "c", "d", "a"]),
                dtype="float",
            ),
        ),
        (
            [
                pd.DataFrame(
                    {
                        "A": [1.0, 1.0, 1.0],
                        "B": ["A", "A", "A"],
                    },
                    index=pd.Index(["a", "b", "c"]),
                ),
                pd.DataFrame(
                    {
                        "A": [2.0, 2.0, 2.0],
                        "B": ["B", "B", "B"],
                    },
                    index=pd.Index(["b", "c", "d"]),
                ),
            ],
            lambda y: y[0],
            pd.DataFrame(
                {
                    "A": [1.0, 1.0, 1.0, 2.0],
                    "B": ["A", "A", "A", "B"],
                },
                index=pd.Index(["a", "b", "c", "d"]),
            ),
        ),
        (
            [
                pd.DataFrame(
                    {
                        "A": [1, 1, 1],
                        "B": [1, 1, 1],
                    },
                    index=pd.Index(["a", "b", "c"]),
                ),
                pd.DataFrame(
                    {
                        "A": [2, 2, 2],
                        "B": [2, 2, 2],
                    },
                    index=pd.Index(["b", "c", "d"]),
                ),
                pd.DataFrame(
                    {
                        "A": [3],
                        "B": [3],
                    },
                    index=pd.Index(["a"]),
                ),
            ],
            np.sum,
            pd.DataFrame(
                {
                    "A": [4, 3, 3, 2],
                    "B": [4, 3, 3, 2],
                },
                index=pd.Index(["a", "b", "c", "d"]),
                dtype="Int64",
            ),
        ),
        (
            [
                pd.DataFrame(
                    {
                        "A": [1, 1, 1],
                        "B": [1, 1, 1],
                    },
                    index=pd.Index(["a", "b", "c"]),
                ),
                pd.DataFrame(
                    {
                        "A": [2, 2, 2],
                        "B": [2, 2, 2],
                    },
                    index=pd.Index(["b", "c", "d"]),
                ),
                pd.DataFrame(
                    {
                        "A": [3, 3],
                        "B": [3, 3],
                    },
                    index=pd.Index(["a", "d"]),
                ),
            ],
            np.sum,
            pd.DataFrame(
                {
                    "A": [4, 3, 3, 5],
                    "B": [4, 3, 3, 5],
                },
                index=pd.Index(["a", "b", "c", "d"]),
                dtype="Int64",
            ),
        ),
        (
            [
                pd.DataFrame(
                    {
                        "A": [1, 1, 1],
                        "B": [1, 1, 1],
                        "C": [1, 1, 1],
                    },
                    index=pd.Index(["a", "b", "c"]),
                ),
                pd.DataFrame(
                    {
                        "A": [2, 2, 2],
                        "B": [2, 2, 2],
                    },
                    index=pd.Index(["b", "c", "d"]),
                ),
                pd.DataFrame(
                    {
                        "A": [3, 3],
                        "B": [3, 3],
                    },
                    index=pd.Index(["a", "d"]),
                ),
            ],
            np.sum,
            pd.DataFrame(
                {
                    "A": [4, 3, 3, 5],
                    "B": [4, 3, 3, 5],
                    "C": [1, 1, 1, np.nan],
                },
                index=pd.Index(["a", "b", "c", "d"]),
                dtype="Int64",
            ),
        ),
        (
            [
                pd.DataFrame(
                    {
                        "A": [1, 1, 1],
                        "B": [1, 1, 1],
                        "C": [1, 1, 1],
                    },
                    index=pd.Index(["a", "b", "c"]),
                    dtype="Int64",
                ),
                pd.DataFrame(
                    {
                        "A": [2, 2, 2],
                        "B": [2, 2, 2],
                    },
                    index=pd.Index(["b", "c", "d"]),
                    dtype="Int64",
                ),
                pd.DataFrame(
                    {
                        "A": [3, 3, 4],
                        "B": [3, 3, np.nan],
                    },
                    index=pd.Index(["a", "d", "e"]),
                    dtype="Int64",
                ),
            ],
            lambda y: y[0],
            pd.DataFrame(
                {
                    "A": [1, 1, 1, 2, 4],
                    "B": [1, 1, 1, 2, np.nan],
                    "C": [1, 1, 1, np.nan, np.nan],
                },
                index=pd.Index(["a", "b", "c", "d", "e"]),
                dtype="Int64",
            ),
        ),
        (
            [
                pd.DataFrame(
                    {
                        "A": [1, 1, 1],
                        "B": [1, 1, 1],
                    },
                    index=pd.Index(["a", "b", "c"]),
                ),
                pd.DataFrame(
                    {
                        "A": [2, 1, 2],
                        "B": [2, 1, 2],
                        "C": [2, 1, 2],
                    },
                    index=pd.Index(["b", "c", "d"]),
                ),
                pd.DataFrame(
                    {
                        "A": [3, 3],
                        "B": [3, 3],
                    },
                    index=pd.Index(["a", "d"]),
                ),
            ],
            np.sum,
            pd.DataFrame(
                {
                    "A": [4, 3, 1, 5],
                    "B": [4, 3, 1, 5],
                    "C": [np.nan, 2, 1, 2],
                },
                index=pd.Index(["a", "b", "c", "d"]),
                dtype="Int64",
            ),
        ),
    ],
)
def test_concat_aggregate_function(objs, aggregate_function, expected):
    obj = audformat.utils.concat(
        objs,
        aggregate_function=aggregate_function,
    )
    if isinstance(obj, pd.Series):
        pd.testing.assert_series_equal(obj, expected)
    else:
        pd.testing.assert_frame_equal(obj, expected)


@pytest.mark.parametrize(
    "objs, aggregate_function, aggregate_strategy, expected",
    [
        (
            [
                pd.Series([1, 2], pd.Index(["a", "b"]), dtype="float"),
                pd.Series([1, 2], pd.Index(["a", "b"]), dtype="float"),
            ],
            tuple,
            "overlap",
            pd.Series([(1, 1), (2, 2)], pd.Index(["a", "b"]), dtype="object"),
        ),
        (
            [
                pd.Series([1, 2], pd.Index(["a", "b"]), dtype="float"),
                pd.Series([1, 2], pd.Index(["a", "b"]), dtype="float"),
            ],
            tuple,
            "mismatch",
            pd.Series([1, 2], pd.Index(["a", "b"]), dtype="float"),
        ),
        (
            [
                pd.Series([1, 2], pd.Index(["a", "b"]), dtype="float"),
                pd.Series([1, 2, 3], pd.Index(["a", "b", "c"]), dtype="float"),
            ],
            tuple,
            "overlap",
            pd.Series(
                [(1.0, 1.0), (2.0, 2.0), 3.0],
                pd.Index(["a", "b", "c"]),
                dtype="object",
            ),
        ),
        (
            [
                pd.Series([1, 2], pd.Index(["a", "b"]), dtype="float"),
                pd.Series([1, 2, 3], pd.Index(["a", "b", "c"]), dtype="float"),
            ],
            tuple,
            "mismatch",
            pd.Series([1, 2, 3], pd.Index(["a", "b", "c"]), dtype="float"),
        ),
        (
            [
                pd.Series([1, 2, 3], pd.Index(["a", "b", "c"]), dtype="float"),
                pd.Series([1, 2], pd.Index(["a", "b"]), dtype="float"),
            ],
            tuple,
            "overlap",
            pd.Series(
                [(1.0, 1.0), (2.0, 2.0), 3.0],
                pd.Index(["a", "b", "c"]),
                dtype="object",
            ),
        ),
        (
            [
                pd.Series([1, 2, 3], pd.Index(["a", "b", "c"]), dtype="float"),
                pd.Series([1, 2], pd.Index(["a", "b"]), dtype="float"),
            ],
            tuple,
            "mismatch",
            pd.Series([1, 2, 3], pd.Index(["a", "b", "c"]), dtype="float"),
        ),
        (
            [
                pd.Series([2, 3], pd.Index(["b", "c"]), dtype="float"),
                pd.Series([1, 2, 3], pd.Index(["a", "b", "c"]), dtype="float"),
            ],
            tuple,
            "overlap",
            pd.Series(
                [(2.0, 2.0), (3.0, 3.0), 1.0],
                pd.Index(["b", "c", "a"]),
                dtype="object",
            ),
        ),
        (
            [
                pd.Series([2, 3], pd.Index(["b", "c"]), dtype="float"),
                pd.Series([1, 2, 3], pd.Index(["a", "b", "c"]), dtype="float"),
            ],
            tuple,
            "mismatch",
            pd.Series([2, 3, 1], pd.Index(["b", "c", "a"]), dtype="float"),
        ),
        (
            [
                pd.Series([1, 2], pd.Index(["a", "b"]), dtype="float"),
                pd.Series([1, 2], pd.Index(["a", "b"]), dtype="float"),
            ],
            np.sum,
            "overlap",
            pd.Series([2, 4], pd.Index(["a", "b"]), dtype="float"),
        ),
        (
            [
                pd.Series([1, 2], pd.Index(["a", "b"]), dtype="float"),
                pd.Series([1, 2], pd.Index(["a", "b"]), dtype="float"),
            ],
            np.sum,
            "mismatch",
            pd.Series([1, 2], pd.Index(["a", "b"]), dtype="float"),
        ),
        (
            [
                pd.Series([1, 2], pd.Index(["a", "b"]), dtype="float"),
                pd.Series([1, 2], pd.Index(["a", "b"]), dtype="float"),
            ],
            np.var,
            "overlap",
            pd.Series([0, 0], pd.Index(["a", "b"]), dtype="float"),
        ),
        (
            [
                pd.Series([1, 2], pd.Index(["a", "b"]), dtype="float"),
                pd.Series([1, 2], pd.Index(["a", "b"]), dtype="float"),
            ],
            lambda y: "a",
            "overlap",
            pd.Series(
                ["a", "a"],
                pd.Index(["a", "b"]),
                dtype="object",
            ),
        ),
        (
            [
                pd.Series([1, 2], pd.Index(["a", "b"]), dtype="float"),
                pd.Series([1, 2], pd.Index(["a", "b"]), dtype="float"),
            ],
            lambda y: 0,
            "overlap",
            pd.Series(
                [0, 0],
                pd.Index(["a", "b"]),
                dtype="float",
            ),
        ),
        (
            [
                pd.Series([1, 2], pd.Index(["a", "b"]), dtype="int"),
                pd.Series([1, 2], pd.Index(["a", "b"]), dtype="int"),
            ],
            lambda y: 0,
            "overlap",
            pd.Series(
                [0, 0],
                pd.Index(["a", "b"]),
                dtype="Int64",
            ),
        ),
        (
            [
                pd.Series([1, 2], pd.Index(["a", "b"]), dtype="int"),
                pd.Series([1, 2], pd.Index(["a", "b"]), dtype="int"),
            ],
            lambda y: 0.5,
            "overlap",
            pd.Series(
                [0.5, 0.5],
                pd.Index(["a", "b"]),
                dtype="float",
            ),
        ),
        (
            [
                pd.Series([1, 2], pd.Index(["a", "b"]), dtype="float"),
                pd.Series([1, 2], pd.Index(["a", "b"]), dtype="float"),
            ],
            lambda y: ("a", "b"),
            "overlap",
            pd.Series(
                [("a", "b"), ("a", "b")],
                pd.Index(["a", "b"]),
                dtype="object",
            ),
        ),
        (
            [
                pd.Series([1, 2], pd.Index(["a", "b"]), dtype="float"),
                pd.Series([1, 2], pd.Index(["a", "b"]), dtype="float"),
                pd.Series([1, 2], pd.Index(["a", "b"]), dtype="float"),
            ],
            np.sum,
            "overlap",
            pd.Series([3, 6], pd.Index(["a", "b"]), dtype="float"),
        ),
        (
            [
                pd.Series(
                    [1, 2],
                    audformat.filewise_index(["a", "b"]),
                    dtype="float",
                ),
                pd.Series(
                    [1, 2],
                    audformat.filewise_index(["a", "b"]),
                    dtype="float",
                ),
            ],
            np.sum,
            "overlap",
            pd.Series(
                [2, 4],
                audformat.filewise_index(["a", "b"]),
                dtype="float",
            ),
        ),
        (
            [
                pd.Series(["a", "b"], pd.Index(["a", "b"]), dtype="string"),
                pd.Series(["a", "b"], pd.Index(["a", "b"]), dtype="string"),
            ],
            lambda y: np.char.add(y[0], y[1]),
            "overlap",
            pd.Series(["aa", "bb"], pd.Index(["a", "b"]), dtype="string"),
        ),
        (
            [
                pd.DataFrame(
                    {
                        "A": [1, 3],
                        "B": [2, 4],
                    },
                    pd.Index(["a", "b"]),
                    dtype="float",
                ),
                pd.DataFrame(
                    {
                        "A": [1, 3],
                        "B": [2, 4],
                    },
                    pd.Index(["a", "b"]),
                    dtype="float",
                ),
            ],
            None,
            "overlap",
            pd.DataFrame(
                {
                    "A": [1, 3],
                    "B": [2, 4],
                },
                pd.Index(["a", "b"]),
                dtype="float",
            ),
        ),
        (
            [
                pd.DataFrame(
                    {
                        "A": [1, 3],
                        "B": [2, 4],
                    },
                    pd.Index(["a", "b"]),
                    dtype="float",
                ),
                pd.DataFrame(
                    {
                        "A": [1, 3],
                        "B": [2, 4],
                    },
                    pd.Index(["a", "b"]),
                    dtype="float",
                ),
            ],
            np.sum,
            "overlap",
            pd.DataFrame(
                {
                    "A": [2, 6],
                    "B": [4, 8],
                },
                pd.Index(["a", "b"]),
                dtype="float",
            ),
        ),
        (
            [
                pd.DataFrame(
                    {
                        "A": [1, 3],
                        "B": [2, 4],
                    },
                    pd.Index(["a", "b"]),
                    dtype="float",
                ),
                pd.DataFrame(
                    {
                        "A": [1, 3],
                    },
                    pd.Index(["a", "b"]),
                    dtype="float",
                ),
            ],
            np.sum,
            "overlap",
            pd.DataFrame(
                {
                    "A": [2, 6],
                    "B": [2, 4],
                },
                pd.Index(["a", "b"]),
                dtype="float",
            ),
        ),
        (
            [
                pd.DataFrame(
                    {
                        "A": [1, 3],
                        "B": [2, 4],
                    },
                    pd.Index(["a", "b"]),
                    dtype="float",
                ),
                pd.DataFrame(
                    {
                        "B": [2, 4],
                    },
                    pd.Index(["a", "b"]),
                    dtype="float",
                ),
            ],
            np.sum,
            "overlap",
            pd.DataFrame(
                {
                    "A": [1, 3],
                    "B": [4, 8],
                },
                pd.Index(["a", "b"]),
                dtype="float",
            ),
        ),
        (
            [
                pd.DataFrame(
                    {
                        "A": [1, 3],
                    },
                    pd.Index(["a", "b"]),
                    dtype="float",
                ),
                pd.DataFrame(
                    {
                        "A": [1, 3],
                        "B": [2, 4],
                    },
                    pd.Index(["a", "b"]),
                    dtype="float",
                ),
            ],
            np.sum,
            "overlap",
            pd.DataFrame(
                {
                    "A": [2, 6],
                    "B": [2, 4],
                },
                pd.Index(["a", "b"]),
                dtype="float",
            ),
        ),
        (
            [
                pd.DataFrame(
                    {
                        "B": [2, 4],
                    },
                    pd.Index(["a", "b"]),
                    dtype="float",
                ),
                pd.DataFrame(
                    {
                        "A": [1, 3],
                        "B": [2, 4],
                    },
                    pd.Index(["a", "b"]),
                    dtype="float",
                ),
            ],
            np.sum,
            "overlap",
            pd.DataFrame(
                {
                    "B": [4, 8],
                    "A": [1, 3],
                },
                pd.Index(["a", "b"]),
                dtype="float",
            ),
        ),
    ],
)
def test_concat_aggregate_function_aggregate(
    objs,
    aggregate_function,
    aggregate_strategy,
    expected,
):
    obj = audformat.utils.concat(
        objs,
        aggregate_function=aggregate_function,
        aggregate_strategy=aggregate_strategy,
    )
    if isinstance(obj, pd.Series):
        pd.testing.assert_series_equal(obj, expected)
    else:
        pd.testing.assert_frame_equal(obj, expected)


@pytest.mark.parametrize(
    "objs, aggregate_function, expected",
    [
        # empty
        (
            [],
            None,
            pd.Series([], pd.Index([]), dtype="object"),
        ),
        # identical values
        (
            [
                pd.Series([1, 2], pd.Index(["a", "b"]), dtype="float"),
                pd.Series([1, 2], pd.Index(["a", "b"]), dtype="float"),
            ],
            None,
            pd.Series([1, 2], pd.Index(["a", "b"]), dtype="float"),
        ),
        (
            [
                pd.Series([1, 2], pd.Index(["a", "b"]), dtype="float"),
                pd.Series([1, 2], pd.Index(["a", "b"]), dtype="float"),
            ],
            np.mean,
            pd.Series([1, 2], pd.Index(["a", "b"]), dtype="float"),
        ),
        # different values
        (
            [
                pd.Series([1, 2], pd.Index(["a", "b"]), dtype="float"),
                pd.Series([2, 3], pd.Index(["a", "b"]), dtype="float"),
            ],
            None,
            pd.Series([2, 3], pd.Index(["a", "b"]), dtype="float"),
        ),
        (
            [
                pd.Series([1, 2], pd.Index(["a", "b"]), dtype="float"),
                pd.Series([2, 3], pd.Index(["a", "b"]), dtype="float"),
            ],
            np.mean,
            pd.Series([2, 3], pd.Index(["a", "b"]), dtype="float"),
        ),
    ],
)
def test_concat_overwrite_aggregate_function(
    objs,
    aggregate_function,
    expected,
):
    obj = audformat.utils.concat(
        objs,
        overwrite=True,
        aggregate_function=aggregate_function,
    )
    if isinstance(obj, pd.Series):
        pd.testing.assert_series_equal(obj, expected)
    else:
        pd.testing.assert_frame_equal(obj, expected)


@pytest.mark.parametrize(
    "objs, aggregate_function, aggregate_strategy, "
    "expected_error, expected_error_msg",
    [
        # wrong aggregate_strategy argument
        (
            [],
            None,
            "non-existent",
            ValueError,
            "aggregate_strategy needs to be one of: overlap, mismatch",
        ),
        # dtypes do not match
        (
            [
                pd.Series([1], audformat.filewise_index("f1")),
                pd.Series([1.0], audformat.filewise_index("f1")),
            ],
            None,
            "overlap",
            ValueError,
            (
                "Found two columns with name 'None' but different dtypes:\n"
                "Int64 != float64."
            ),
        ),
        (
            [
                pd.Series(
                    [1, 2, 3],
                    index=audformat.filewise_index(["f1", "f2", "f3"]),
                ),
                pd.Series(
                    ["a", "b", "a"],
                    index=audformat.filewise_index(["f1", "f2", "f3"]),
                    dtype="category",
                ),
            ],
            None,
            "overlap",
            ValueError,
            re.escape(
                "Found two columns with name 'None' but different dtypes:\n"
                "Int64 != CategoricalDtype(categories=['a', 'b']"
            ),
        ),
        (
            [
                pd.Series(
                    ["a", "b", "a"],
                    index=audformat.filewise_index(["f1", "f2", "f3"]),
                    dtype="string",
                ),
                pd.Series(
                    ["a", "b", "a"],
                    index=audformat.filewise_index(["f1", "f2", "f3"]),
                    dtype="category",
                ),
            ],
            None,
            "overlap",
            ValueError,
            re.escape(
                "Found two columns with name 'None' but different dtypes:\n"
                "string != CategoricalDtype(categories=['a', 'b']"
            ),
        ),
        (
            [
                pd.Series(
                    ["a", "b", "a"],
                    index=audformat.filewise_index(["f1", "f2", "f3"]),
                    dtype="category",
                ),
                pd.Series(
                    ["a", "b", "c"],
                    index=audformat.filewise_index(["f1", "f2", "f3"]),
                    dtype="category",
                ),
            ],
            None,
            "overlap",
            ValueError,
            (
                "Found two columns with name 'None' but different dtypes:\n"
                r"CategoricalDtype\(categories=\['a', 'b'\],"
                ".*"
                r"!= CategoricalDtype\(categories=\['a', 'b', 'c'\]"
            ),
        ),
        # values do not match
        (
            [
                pd.DataFrame(
                    {
                        "A": [1, 3],
                        "B": [2, 4],
                    },
                    pd.Index(["a", "b"]),
                    dtype="float",
                ),
                pd.DataFrame(
                    {
                        "A": [2, 4],
                        "B": [3, 5],
                    },
                    pd.Index(["a", "b"]),
                    dtype="float",
                ),
            ],
            None,
            "mismatch",
            ValueError,
            (
                "Found overlapping data in column 'A':\n   "
                "left  right\n"
                "a   1.0    2.0\n"
                "b   3.0    4.0"
            ),
        ),
        (
            [
                pd.Series([1, 2], pd.Index(["a", "b"]), dtype="float"),
                pd.Series([2, 3], pd.Index(["a", "b"]), dtype="float"),
            ],
            None,
            "mismatch",
            ValueError,
            (
                "Found overlapping data in column 'None':\n   "
                "left  right\n"
                "a   1.0    2.0\n"
                "b   2.0    3.0"
            ),
        ),
        (
            [
                pd.Series([1.0], audformat.filewise_index("f1")),
                pd.Series([2.0], audformat.filewise_index("f1")),
            ],
            None,
            "overlap",
            ValueError,
            (
                "Found overlapping data in column 'None':\n      "
                "left  right\n"
                "file             \n"
                "f1     1.0    2.0"
            ),
        ),
        (
            [
                pd.Series([1.0], pd.Index(["f1"], name="idx")),
                pd.Series(
                    [2.0],
                    pd.MultiIndex.from_arrays([["f1"]], names=["idx"]),
                ),
            ],
            None,
            "overlap",
            ValueError,
            (
                "Found overlapping data in column 'None':\n     "
                "left  right\n"
                "idx             \n"
                "f1    1.0    2.0"
            ),
        ),
        # index names do not match
        (
            [
                pd.Series(
                    [1.0],
                    pd.Index(["f1"], name="idx", dtype="string"),
                ),
                pd.Series(  # default dtype is object
                    [2.0],
                    pd.MultiIndex.from_arrays([["f1"]], names=["idx"]),
                ),
            ],
            None,
            "overlap",
            ValueError,
            re.escape(
                "Levels and dtypes of all objects must match. "
                "Found different level dtypes: ['str', 'object']."
            ),
        ),
        (
            [
                pd.Series([], index=pd.Index([], name="idx1"), dtype="object"),
                pd.Series([], index=pd.Index([], name="idx2"), dtype="object"),
            ],
            None,
            "overlap",
            ValueError,
            re.escape(
                "Levels and dtypes of all objects must match. "
                "Found different level names: ['idx1', 'idx2']."
            ),
        ),
        (
            [
                pd.Series([1.0], pd.Index(["f1"], name="idx1")),
                pd.Series(
                    [2.0],
                    pd.MultiIndex.from_arrays([["f2"]], names=["idx2"]),
                ),
            ],
            None,
            "overlap",
            ValueError,
            re.escape(
                "Levels and dtypes of all objects must match. "
                "Found different level names: ['idx1', 'idx2']."
            ),
        ),
    ],
)
def test_concat_errors(
    objs,
    aggregate_function,
    aggregate_strategy,
    expected_error,
    expected_error_msg,
):
    with pytest.raises(expected_error, match=expected_error_msg):
        audformat.utils.concat(
            objs,
            aggregate_function=aggregate_function,
            aggregate_strategy=aggregate_strategy,
        )
