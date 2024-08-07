from io import StringIO
import os
import re

import numpy as np
import pandas as pd
import pytest

import audeer

import audformat
from audformat import define
from audformat import utils


@pytest.mark.parametrize(
    "objs, expected",
    [
        # empty
        (
            [],
            pd.Index([]),
        ),
        # conform to audformat
        (
            [
                audformat.filewise_index(),
            ],
            audformat.filewise_index(),
        ),
        (
            [
                audformat.filewise_index(),
                audformat.filewise_index(),
            ],
            audformat.filewise_index(),
        ),
        (
            [
                audformat.filewise_index(["f1", "f2"]),
                audformat.filewise_index(["f1", "f2"]),
            ],
            audformat.filewise_index(),
        ),
        (
            [
                audformat.filewise_index(["f1", "f2"]),
                audformat.filewise_index(["f1", "f2"]),
                audformat.filewise_index(["f2", "f3"]),
            ],
            audformat.filewise_index("f3"),
        ),
        (
            [
                audformat.filewise_index(["f2", "f3"]),
                audformat.filewise_index(["f1", "f2"]),
                audformat.filewise_index("f2"),
            ],
            audformat.filewise_index(["f3", "f1"]),
        ),
        (
            [
                audformat.filewise_index(["f1", "f2"]),
                audformat.filewise_index(["f1", "f2"]),
                audformat.filewise_index("f3"),
            ],
            audformat.filewise_index("f3"),
        ),
        (
            [
                audformat.segmented_index(),
            ],
            audformat.segmented_index(),
        ),
        (
            [
                audformat.segmented_index(),
                audformat.segmented_index(),
            ],
            audformat.segmented_index(),
        ),
        (
            [
                audformat.segmented_index(["f1", "f2"]),
                audformat.segmented_index(["f1", "f2"]),
            ],
            audformat.segmented_index(),
        ),
        (
            [
                audformat.segmented_index(["f1", "f2"]),
                audformat.segmented_index(["f3", "f4"]),
            ],
            audformat.segmented_index(["f1", "f2", "f3", "f4"]),
        ),
        (
            [
                audformat.segmented_index(["f1", "f2"], [0, 0], [1, 1]),
                audformat.segmented_index(["f2", "f1"], [0, 0], [1, 1]),
                audformat.segmented_index(["f2", "f3"], [0, 0], [1, 1]),
            ],
            audformat.segmented_index("f3", 0, 1),
        ),
        (
            [
                audformat.segmented_index(["f1", "f2"], [0, 0], [1, 1]),
                audformat.segmented_index(["f2", "f1"], [0, 0], [1, 1]),
                audformat.segmented_index(["f2", "f3"], [1, 1], [2, 2]),
            ],
            audformat.segmented_index(["f2", "f3"], [1, 1], [2, 2]),
        ),
        (
            [
                audformat.filewise_index(),
                audformat.segmented_index(),
            ],
            audformat.segmented_index(),
        ),
        (
            [
                audformat.filewise_index(["f1", "f2"]),
                audformat.segmented_index(),
            ],
            audformat.segmented_index(["f1", "f2"]),
        ),
        (
            [
                audformat.filewise_index(),
                audformat.segmented_index(["f1", "f2"]),
            ],
            audformat.segmented_index(["f1", "f2"]),
        ),
        (
            [
                audformat.segmented_index(["f1", "f2"], [0, 0], [1, 1]),
                audformat.segmented_index(["f2", "f3"], [0, 0], [1, 1]),
                audformat.filewise_index(["f1", "f2"]),
            ],
            audformat.segmented_index(
                ["f1", "f3", "f1", "f2"],
                [0, 0, 0, 0],
                [1, 1, pd.NaT, pd.NaT],
            ),
        ),
        (
            [
                audformat.segmented_index(["f1", "f2"], [0, 0], [1, pd.NaT]),
                audformat.segmented_index(["f2", "f3"], [0, 0], [pd.NaT, 1]),
                audformat.filewise_index(["f1", "f2"]),
            ],
            audformat.segmented_index(
                ["f1", "f3", "f1"],
                [0, 0, 0],
                [1, 1, pd.NaT],
            ),
        ),
        (
            [
                audformat.segmented_index(["f1", "f2"], [0, 0], [1, pd.NaT]),
                audformat.segmented_index(["f2", "f3"], [0, 0], [1, 1]),
                audformat.filewise_index(["f1", "f2"]),
            ],
            audformat.segmented_index(
                ["f1", "f2", "f3", "f1"],
                [0, 0, 0, 0],
                [1, 1, 1, pd.NaT],
            ),
        ),
        (
            [
                audformat.segmented_index(["f1", "f2"], [0, 0], [1, 1]),
                audformat.filewise_index(["f1", "f2"]),
                audformat.filewise_index(["f2", "f3"]),
            ],
            audformat.segmented_index(
                ["f1", "f2", "f1", "f3"],
                [0, 0, 0, 0],
                [1, 1, pd.NaT, pd.NaT],
            ),
        ),
        # other
        (
            [
                pd.Index([]),
            ],
            pd.Index([]),
        ),
        (
            [
                pd.Index([]),
                pd.Index([]),
            ],
            pd.Index([]),
        ),
        (
            [
                pd.Index([0, 1], name="idx"),
                pd.Index([1, 2], name="idx"),
            ],
            pd.Index([0, 2], dtype="Int64", name="idx"),
        ),
        (
            [
                pd.Index([0, 1], dtype="Int64", name="idx"),
                pd.Index([1, 2], dtype="Int64", name="idx"),
            ],
            pd.Index([0, 2], dtype="Int64", name="idx"),
        ),
        (
            [
                pd.Index([0, 1], dtype="int64", name="idx"),
                pd.Index([1, 2], dtype="Int64", name="idx"),
            ],
            pd.Index([0, 2], dtype="Int64", name="idx"),
        ),
        (
            [
                pd.Index([1, 2], dtype="Int64", name="idx"),
                pd.Index([0, 1], dtype="int64", name="idx"),
            ],
            pd.Index([2, 0], dtype="Int64", name="idx"),
        ),
        (
            [
                pd.Index([0, 1], dtype="int64", name="idx"),
                pd.Index([0, 1, np.nan], dtype="Int64", name="idx"),
            ],
            pd.Index([np.nan], dtype="Int64", name="idx"),
        ),
        (
            [
                pd.Index([0, 1], name="idx"),
                pd.MultiIndex.from_arrays([[1, 2]], names=["idx"]),
            ],
            pd.Index([0, 2], dtype="Int64", name="idx"),
        ),
        (
            [
                pd.MultiIndex.from_arrays([[0, 1]], names=["idx"]),
                pd.MultiIndex.from_arrays([[1, 2]], names=["idx"]),
            ],
            audformat.utils.set_index_dtypes(
                pd.MultiIndex.from_arrays([[0, 2]], names=["idx"]),
                "Int64",
            ),
        ),
        (
            [
                pd.MultiIndex.from_arrays(
                    [["a", "b", "c"], [0, 1, 2]],
                    names=["idx1", "idx2"],
                ),
                pd.MultiIndex.from_arrays(
                    [["b", "c"], [1, 3]],
                    names=["idx1", "idx2"],
                ),
            ],
            audformat.utils.set_index_dtypes(
                pd.MultiIndex.from_arrays(
                    [["a", "c", "c"], [0, 2, 3]],
                    names=["idx1", "idx2"],
                ),
                {"idx2": "Int64"},
            ),
        ),
        pytest.param(
            [
                pd.Index([], name="idx1"),
                pd.Index([], name="idx2"),
            ],
            None,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
    ],
)
def test_difference(objs, expected):
    pd.testing.assert_index_equal(
        audformat.utils.difference(objs),
        expected,
    )
    # Ensure (A Δ B) Δ C = A Δ (B Δ C)
    if len(objs) > 2:
        pd.testing.assert_index_equal(
            audformat.utils.difference(
                [
                    objs[0],
                    audformat.utils.difference(objs[1:]),
                ]
            ).sortlevel()[0],
            audformat.utils.difference(
                [
                    audformat.utils.difference(objs[:-1]),
                    objs[-1],
                ]
            ).sortlevel()[0],
        )


@pytest.mark.parametrize(
    "obj, expected_duration",
    [
        (
            audformat.segmented_index(),
            pd.Timedelta(0, unit="s"),
        ),
        (
            audformat.segmented_index(["f1"], [0], [2]),
            pd.Timedelta(2, unit="s"),
        ),
        (
            audformat.segmented_index(["f1"], [0.1], [2]),
            pd.Timedelta(1.9, unit="s"),
        ),
        (
            audformat.segmented_index(["f1", "f2"], [0, 1], [2, 2]),
            pd.Timedelta(3, unit="s"),
        ),
        (
            pd.Series(
                index=audformat.segmented_index(["f1"], [1], [2]),
                dtype="category",
            ),
            pd.Timedelta(1, unit="s"),
        ),
        (
            pd.DataFrame(index=audformat.segmented_index(["f1"], [1], [2])),
            pd.Timedelta(1, unit="s"),
        ),
        # filewise index, but file is missing
        pytest.param(
            audformat.filewise_index(["f1"]),
            None,
            marks=pytest.mark.xfail(raises=FileNotFoundError),
        ),
        # segmented index with NaT, but file is missing
        pytest.param(
            audformat.segmented_index(["f1"], [0]),
            None,
            marks=pytest.mark.xfail(raises=FileNotFoundError),
        ),
    ],
)
def test_duration(obj, expected_duration):
    duration = audformat.utils.duration(obj)
    if pd.isnull(expected_duration):
        assert pd.isnull(duration)
    else:
        assert duration == expected_duration


@pytest.mark.parametrize(
    "index, root, expected",
    [
        (
            audformat.filewise_index(),
            None,
            audformat.filewise_index(),
        ),
        (
            audformat.segmented_index(),
            None,
            audformat.segmented_index(),
        ),
        (
            audformat.filewise_index(["f1", "f2"]),
            audeer.path("/some", "where"),
            audformat.filewise_index(
                [
                    audeer.path("/some", "where", "f1"),
                    audeer.path("/some", "where", "f2"),
                ]
            ),
        ),
        (
            audformat.filewise_index(["f1", "f2"]),
            audeer.path("some", "where"),
            audformat.filewise_index(
                [
                    audeer.path("some", "where", "f1"),
                    audeer.path("some", "where", "f2"),
                ]
            ),
        ),
        (
            audformat.filewise_index(["f1", "f2"]),
            os.path.join("some", "where"),
            audformat.filewise_index(
                [
                    os.path.join("some", "where", "f1"),
                    os.path.join("some", "where", "f2"),
                ]
            ),
        ),
        (
            audformat.filewise_index(["f1", "f2"]),
            os.path.join("some", "where") + os.path.sep,
            audformat.filewise_index(
                [
                    os.path.join("some", "where", "f1"),
                    os.path.join("some", "where", "f2"),
                ]
            ),
        ),
        (
            audformat.filewise_index(
                [
                    audeer.path("f1"),
                    audeer.path("f2"),
                ]
            ),
            audeer.path("some", "where"),
            audformat.filewise_index(
                [
                    audeer.path("some", "where") + os.path.sep + audeer.path("f1"),
                    audeer.path("some", "where") + os.path.sep + audeer.path("f2"),
                ]
            ),
        ),
        (
            audformat.segmented_index(
                ["f1", "f2"],
                ["1s", "3s"],
                ["2s", "4s"],
            ),
            audeer.path("/some", "where"),
            audformat.segmented_index(
                [
                    audeer.path("/some", "where", "f1"),
                    audeer.path("/some", "where", "f2"),
                ],
                ["1s", "3s"],
                ["2s", "4s"],
            ),
        ),
    ],
)
def test_expand_file_path(tmpdir, index, root, expected):
    expanded_index = audformat.utils.expand_file_path(index, root)
    pd.testing.assert_index_equal(expanded_index, expected)


@pytest.mark.parametrize(
    "obj, strict, mutable, expected",
    [
        (
            audformat.filewise_index(),
            False,
            True,
            "0",
        ),
        (
            audformat.filewise_index(),
            True,
            True,
            "890fa7e5864779b7c3bd85c9ede31657",
        ),
        (
            audformat.segmented_index(),
            False,
            True,
            "0",
        ),
        (
            audformat.segmented_index(),
            True,
            True,
            "10d5e40fca4e40c6c70ff64495916059",
        ),
        (
            audformat.filewise_index(["f1", "f2"]),
            False,
            True,
            "-4231615416436839963",
        ),
        (
            audformat.filewise_index(["f1", "f2"]),
            True,
            False,
            "0741235e2250e0fcd9ab7b64972f5047",
        ),
        (
            audformat.segmented_index(["f1", "f2"]),
            False,
            True,
            "-2363261461673824215",
        ),
        (
            audformat.segmented_index(["f1", "f2"]),
            True,
            False,
            "0e99d54165e6cc2dad2737982853f8c7",
        ),
        (
            audformat.segmented_index(["f1", "f2"]),
            False,
            True,
            "-2363261461673824215",
        ),
        (
            audformat.segmented_index(["f1", "f2"], [0, 0], [1, 1]),
            False,
            True,
            "-3831446135233514455",
        ),
        (
            audformat.segmented_index(["f1", "f2"], [0, 0], [1, 1]),
            True,
            False,
            "396fda484a46686b2b5c41b0ae9c94bd",
        ),
        (
            pd.Series([0, 1], audformat.filewise_index(["f1", "f2"])),
            False,
            True,
            "-8245754232361677810",
        ),
        (
            pd.Series([0, 1], audformat.filewise_index(["f1", "f2"])),
            True,
            False,
            "28c5f6feb0682079b127d8ce8debebe9",
        ),
        (
            pd.DataFrame(
                {"a": [0, 1], "b": [2, 3]},
                audformat.segmented_index(["f1", "f2"], [0, 0], [1, 1]),
            ),
            False,
            True,
            "-103439349488189352",
        ),
        (
            pd.DataFrame(
                {"a": [0, 1], "b": [2, 3]},
                audformat.segmented_index(["f1", "f2"], [0, 0], [1, 1]),
            ),
            True,
            False,
            "69785e94447fab79f2b65b1dcb4a2122",
        ),
        (
            pd.Index([0, 1], name="idx"),
            False,
            True,
            "6238072747940578789",
        ),
        (
            pd.Index([0, 1], name="idx"),
            True,
            False,
            "a02406270880cde74e66c07278b765a0",
        ),
        (
            pd.Index([0, 1], name="name"),
            False,
            True,
            "6238072747940578789",
        ),
        (
            pd.Index([0, 1], name="name"),
            True,
            False,
            "7a8303866a35ececb4ae76a4aa050209",
        ),
        (
            pd.MultiIndex.from_arrays(
                [[0, 1], ["a", "b"]],
                names=["idx1", "idx2"],
            ),
            False,
            True,
            "8378370490910668918",
        ),
        (
            pd.MultiIndex.from_arrays(
                [[0, 1], ["a", "b"]],
                names=["idx1", "idx2"],
            ),
            True,
            False,
            "be5373f6d8f801b902d8cf0e2f2a1914",
        ),
        (
            pd.MultiIndex.from_arrays(
                [[0, 1], ["a", "b"]],
                names=["name1", "name2"],
            ),
            False,
            True,
            "8378370490910668918",
        ),
        (
            pd.MultiIndex.from_arrays(
                [[0, 1], ["a", "b"]],
                names=["name1", "name2"],
            ),
            True,
            False,
            "3726b3c39fc2c1453bb45a0460630ff7",
        ),
        (
            pd.Series([0, 1], name="series"),
            False,
            True,
            "-7179254265801896228",
        ),
        (
            pd.Series([0, 1], name="series"),
            True,
            False,
            "ad0f9900c0e2f3954bde3abb6f4a9b61",
        ),
        (
            pd.Series([0, 1], name="name"),
            False,
            True,
            "-7179254265801896228",
        ),
        (
            pd.Series([0, 1], name="name"),
            True,
            False,
            "648314808f0b27e5c04479ba8509fc25",
        ),
        (
            pd.DataFrame([0, 1], columns=["frame"]),
            False,
            True,
            "-7179254265801896228",
        ),
        (
            pd.DataFrame([0, 1], columns=["frame"]),
            True,
            False,
            "36d9779e257319ff69515af6b4ade8ad",
        ),
        (
            pd.DataFrame([0, 1], columns=["name"]),
            False,
            True,
            "-7179254265801896228",
        ),
        (
            pd.DataFrame([0, 1], columns=["name"]),
            True,
            False,
            "648314808f0b27e5c04479ba8509fc25",
        ),
        pytest.param(
            pd.DataFrame(
                [0, 1, 2],
                pd.Index([0, 1, 2], dtype="Int64"),
            ),
            False,
            True,
            "5440931770055407318",
            marks=pytest.mark.skipif(
                pd.__version__ >= "2.2.0",
                reason="Changed in pandas 2.2.0",
            ),
        ),
        pytest.param(
            pd.DataFrame(
                [0, 1, 2],
                pd.Index([0, 1, 2], dtype="Int64"),
            ),
            False,
            True,
            "-5491649331962632325",
            marks=pytest.mark.skipif(
                pd.__version__ < "2.2.0",
                reason="Changed in pandas 2.2.0",
            ),
        ),
        pytest.param(
            pd.Series(
                [0, 1, 2],
                pd.Index([0, 1, 2], dtype="Int64"),
            ),
            False,
            True,
            "5440931770055407318",
            marks=pytest.mark.skipif(
                pd.__version__ >= "2.2.0",
                reason="Changed in pandas 2.2.0",
            ),
        ),
        pytest.param(
            pd.Series(
                [0, 1, 2],
                pd.Index([0, 1, 2], dtype="Int64"),
            ),
            False,
            True,
            "-5491649331962632325",
            marks=pytest.mark.skipif(
                pd.__version__ < "2.2.0",
                reason="Changed in pandas 2.2.0",
            ),
        ),
    ],
)
def test_hash(obj, strict, mutable, expected):
    md5 = utils.hash(obj, strict=strict)
    reverse_md5 = utils.hash(obj[::-1], strict=strict)
    assert md5 == expected
    if mutable:
        assert reverse_md5 == md5
    else:
        assert reverse_md5 != md5


@pytest.mark.parametrize(
    "obj, expected",
    [
        (
            audformat.filewise_index(),
            False,
        ),
        (
            audformat.segmented_index(),
            False,
        ),
        (
            audformat.filewise_index(["f1"] * 2),
            False,
        ),
        (
            audformat.segmented_index(
                ["f1"] * 2,
                [0, 2],
                [1, pd.NaT],
            ),
            False,
        ),
        (
            audformat.segmented_index(
                ["f1"] * 2,
                [0, 2],
                [pd.NaT, 3],
            ),
            True,
        ),
        (
            audformat.segmented_index(
                ["f1"] * 2,
                [0, 2],
                [pd.NaT, pd.NaT],
            ),
            True,
        ),
        (
            audformat.segmented_index(
                ["f1"] * 2,
                [0, 1],
                [2, 3],
            ),
            True,
        ),
        (
            audformat.segmented_index(
                ["f1", "f2"],
                [0, 1],
                [2, 3],
            ),
            False,
        ),
        (
            pd.Series(
                index=audformat.segmented_index(
                    ["f1"] * 2,
                    [0, 2],
                    [2, 3],
                ),
                dtype="object",
            ),
            False,
        ),
        (
            pd.DataFrame(index=audformat.filewise_index(["f1", "f2"])),
            False,
        ),
    ],
)
def test_index_has_overlap(obj, expected):
    has_overlap = audformat.utils.index_has_overlap(obj)
    assert has_overlap == expected


@pytest.mark.parametrize(
    "objs, expected",
    [
        (
            [],
            pd.Index([]),
        ),
        (
            [
                audformat.filewise_index(),
            ],
            audformat.filewise_index(),
        ),
        (
            [
                audformat.filewise_index(["f1", "f2"]),
            ],
            audformat.filewise_index(),
        ),
        (
            [
                audformat.filewise_index(),
                audformat.filewise_index(),
            ],
            audformat.filewise_index(),
        ),
        (
            [
                audformat.filewise_index(["f1", "f2"]),
                audformat.filewise_index(["f1", "f2"]),
            ],
            audformat.filewise_index(["f1", "f2"]),
        ),
        (
            [
                audformat.filewise_index(["f1", "f2"]),
                audformat.filewise_index(["f1", "f2"]),
                audformat.filewise_index(["f2", "f3"]),
            ],
            audformat.filewise_index("f2"),
        ),
        (
            [
                audformat.filewise_index(["f3", "f2", "f1"]),
                audformat.filewise_index(["f1", "f2"]),
            ],
            audformat.filewise_index(["f2", "f1"]),
        ),
        (
            [
                audformat.filewise_index(["f1", "f2"]),
                audformat.filewise_index(["f1", "f2"]),
                audformat.filewise_index("f3"),
            ],
            audformat.filewise_index(),
        ),
        (
            [
                audformat.segmented_index(),
            ],
            audformat.segmented_index(),
        ),
        (
            [
                audformat.segmented_index(["f1", "f2"]),
            ],
            audformat.segmented_index(),
        ),
        (
            [
                audformat.segmented_index(),
                audformat.segmented_index(),
            ],
            audformat.segmented_index(),
        ),
        (
            [
                audformat.segmented_index(["f1", "f2"]),
                audformat.segmented_index(["f1", "f2"]),
            ],
            audformat.segmented_index(["f1", "f2"]),
        ),
        (
            [
                audformat.segmented_index(["f1", "f2"]),
                audformat.segmented_index(["f3", "f4"]),
            ],
            audformat.segmented_index(),
        ),
        (
            [
                audformat.segmented_index(["f1", "f2"], [0, 0], [1, 1]),
                audformat.segmented_index(["f2", "f1"], [0, 0], [1, 1]),
                audformat.segmented_index(["f2", "f3"], [0, 0], [1, 1]),
            ],
            audformat.segmented_index("f2", 0, 1),
        ),
        (
            [
                audformat.segmented_index(["f1", "f2"], [0, 0], [1, 1]),
                audformat.segmented_index(["f2", "f1"], [0, 0], [1, 1]),
                audformat.segmented_index(["f2", "f3"], [1, 1], [2, 2]),
            ],
            audformat.segmented_index(),
        ),
        (
            [
                audformat.filewise_index(),
                audformat.segmented_index(),
            ],
            audformat.segmented_index(),
        ),
        (
            [
                audformat.filewise_index(["f1", "f2"]),
                audformat.segmented_index(),
            ],
            audformat.segmented_index(),
        ),
        (
            [
                audformat.filewise_index(),
                audformat.segmented_index(["f1", "f2"]),
            ],
            audformat.segmented_index(),
        ),
        (
            [
                audformat.segmented_index("f1"),
                audformat.segmented_index("f1", 0, 1),
            ],
            audformat.segmented_index(),
        ),
        (
            [
                audformat.segmented_index(["f1", "f2"], [0, 0], [1, 1]),
                audformat.segmented_index(["f2", "f3"], [0, 0], [1, 1]),
                audformat.filewise_index(["f1", "f2"]),
            ],
            audformat.segmented_index(),
        ),
        (
            [
                audformat.segmented_index(["f1", "f2"], [0, 0], [1, pd.NaT]),
                audformat.segmented_index(["f2", "f3"], [0, 0], [pd.NaT, 1]),
                audformat.filewise_index(["f1", "f2"]),
            ],
            audformat.segmented_index("f2", 0, pd.NaT),
        ),
        (
            [
                audformat.segmented_index(["f1", "f2"], [0, 0], [1, 1]),
                audformat.segmented_index(["f2", "f3"], [0, 0], [1, 1]),
                audformat.filewise_index("f1"),
            ],
            audformat.segmented_index(),
        ),
        (
            [
                audformat.segmented_index(["f1", "f2"], [0, 0], [1, pd.NaT]),
                audformat.filewise_index(["f1", "f2"]),
                audformat.filewise_index(["f2", "f3"]),
            ],
            audformat.segmented_index("f2", 0, pd.NaT),
        ),
        (
            [
                pd.Index([]),
            ],
            pd.Index([]),
        ),
        (
            [
                pd.Index([0, 1], name="idx"),
            ],
            pd.Index([], dtype="Int64", name="idx"),
        ),
        (
            [
                pd.MultiIndex.from_arrays(
                    [[0, 1], ["a", "b"]],
                    names=["idx1", "idx2"],
                ),
            ],
            audformat.utils.set_index_dtypes(
                pd.MultiIndex.from_arrays(
                    [[], []],
                    names=["idx1", "idx2"],
                ),
                {
                    "idx1": "Int64",
                    "idx2": "object",
                },
            ),
        ),
        (
            [
                pd.Index([]),
                pd.Index([]),
            ],
            pd.Index([]),
        ),
        (
            [
                pd.Index([], dtype="int64"),
                pd.Index([0, 1], dtype="int64"),
            ],
            pd.Index([], dtype="Int64"),
        ),
        (
            [
                pd.Index([0, 1], name="idx"),
                pd.Index([1, 2], name="idx"),
            ],
            pd.Index([1], dtype="Int64", name="idx"),
        ),
        (
            [
                pd.Index([1, 2, 3], name="idx"),
                pd.Index([1, np.nan], dtype="Int64", name="idx"),
            ],
            pd.Index([1], dtype="Int64", name="idx"),
        ),
        (
            [
                pd.Index([1, np.nan], dtype="Int64", name="idx"),
                pd.Index([1, 2, 3], name="idx"),
            ],
            pd.Index([1], dtype="Int64", name="idx"),
        ),
        (
            [
                pd.Index([1, np.nan], dtype="Int64", name="idx"),
                pd.Index([np.nan, 2, 3], dtype="Int64", name="idx"),
            ],
            pd.Index([np.nan], dtype="Int64", name="idx"),
        ),
        (
            [
                pd.Index([0, 1], name="idx"),
                pd.MultiIndex.from_arrays([[1, 2]], names=["idx"]),
            ],
            pd.Index([1], dtype="Int64", name="idx"),
        ),
        (
            [
                pd.MultiIndex.from_arrays([[0, 1]], names=["idx"]),
                pd.MultiIndex.from_arrays([[1, 2]], names=["idx"]),
            ],
            audformat.utils.set_index_dtypes(
                pd.MultiIndex.from_arrays([[1]], names=["idx"]),
                "Int64",
            ),
        ),
        (
            [
                pd.MultiIndex.from_arrays(
                    [["a", "b", "c"], [0, 1, 2]],
                    names=["idx1", "idx2"],
                ),
                pd.MultiIndex.from_arrays(
                    [["b", "c"], [1, 3]],
                    names=["idx1", "idx2"],
                ),
            ],
            audformat.utils.set_index_dtypes(
                pd.MultiIndex.from_arrays(
                    [["b"], [1]],
                    names=["idx1", "idx2"],
                ),
                {"idx2": "Int64"},
            ),
        ),
        pytest.param(
            [
                pd.Index([], name="idx1"),
                pd.Index([], name="idx2"),
            ],
            None,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
    ],
)
def test_intersect(objs, expected):
    pd.testing.assert_index_equal(
        audformat.utils.intersect(objs),
        expected,
    )
    # Ensure A ∩ (B ∩ C) == (A ∩ B) ∩ C
    if len(objs) > 2:
        pd.testing.assert_index_equal(
            audformat.utils.intersect(
                [
                    objs[0],
                    audformat.utils.intersect(objs[1:]),
                ]
            ),
            audformat.utils.intersect(
                [
                    audformat.utils.intersect(objs[:-1]),
                    objs[-1],
                ]
            ),
        )


@pytest.mark.parametrize(
    "objs, error_msg",
    [
        (
            [
                pd.Index([]),
            ],
            None,
        ),
        (
            [
                pd.Index([]),
                pd.Index([]),
            ],
            None,
        ),
        (
            [
                pd.Index([], name="l"),
                pd.Index([], name="L"),
            ],
            "Found different level names: ['l', 'L']",
        ),
        (
            [
                pd.Index([]),
                pd.MultiIndex([[]], [[]]),
            ],
            None,
        ),
        (
            [pd.Index([]), pd.Index([], name="None")],
            "Found different level names: [None, 'None']",
        ),
        (
            [
                pd.Index([1, 2, 3]),
                pd.Index([10, 20], name="l"),
            ],
            "Found different level names: [None, 'l']",
        ),
        (
            [
                pd.Index([1, 2, 3], name="l"),
                pd.MultiIndex.from_arrays([[10, 20]], names=["l"]),
            ],
            None,
        ),
        (
            [
                pd.Index(
                    [1, 2, 3],
                    dtype="Int64",
                    name="l",
                ),
                pd.MultiIndex.from_arrays(
                    [[10, 20]],
                    names=["l"],
                ),
            ],
            None,
        ),
        (
            [
                pd.Index([1, 2, 3], name="l"),
                pd.MultiIndex.from_arrays([[10, 20]], names=["L"]),
            ],
            "Found different level names: ['l', 'L']",
        ),
        (
            [
                pd.Index(["a", "b", "c"], name="l"),
                pd.MultiIndex.from_arrays([[10, 20]], names=["l"]),
            ],
            "Found different level dtypes: ['object', 'int']",
        ),
        (
            [
                pd.MultiIndex.from_arrays(
                    [
                        [10],
                        [20],
                    ],
                    names=["l1", "l2"],
                ),
                pd.Index([1, 2, 3], name="l"),
            ],
            "Found different number of levels: [2, 1]",
        ),
        (
            [
                pd.MultiIndex.from_arrays(
                    [
                        [1, 2, 3],
                        ["a", "b", "c"],
                    ],
                    names=["l1", "l2"],
                ),
                pd.MultiIndex.from_arrays(
                    [
                        [10],
                        ["10"],
                    ],
                    names=["l1", "l2"],
                ),
            ],
            None,
        ),
        (
            [
                pd.MultiIndex.from_arrays(
                    [
                        ["a", "b", "c"],
                        [1, 2, 3],
                    ],
                    names=["l1", "l2"],
                ),
                pd.MultiIndex.from_arrays(
                    [
                        [10],
                        ["10"],
                    ],
                    names=["l1", "l2"],
                ),
            ],
            "Found different level dtypes: " "[('object', 'int'), ('int', 'object')]",
        ),
        (
            [
                pd.MultiIndex.from_arrays(
                    [
                        ["a", "b", "c"],
                        [1, 2, 3],
                    ],
                    names=["l1", "l2"],
                ),
                pd.MultiIndex.from_arrays(
                    [
                        [],
                        [],
                    ],
                    names=["l1", "l2"],
                ),
            ],
            "Found different level dtypes: "
            "[('object', 'int'), ('object', 'object')]",
        ),
        (
            [
                pd.MultiIndex.from_arrays([[], []], names=["l1", "l2"]),
                pd.MultiIndex.from_arrays([[], []]),
            ],
            "Found different level names: " "[('l1', 'l2'), (None, None)]",
        ),
        (
            [
                pd.MultiIndex.from_arrays(
                    [
                        ["a", "b", "c"],
                        [1, 2, 3],
                    ],
                    names=["l1", "l2"],
                ),
                pd.MultiIndex.from_arrays(
                    [
                        ["a", "b", "c"],
                        [1, 2, 3],
                    ],
                    names=["L1", "L2"],
                ),
            ],
            "Found different level names: " "[('l1', 'l2'), ('L1', 'L2')]",
        ),
        (
            [
                audformat.filewise_index(["f1", "f2"]),
                audformat.filewise_index(),
            ],
            None,
        ),
        (
            [
                audformat.segmented_index(["f1", "f2"], [0, 1], [1, 2]),
                audformat.segmented_index(),
            ],
            None,
        ),
        (
            [
                audformat.filewise_index(["f1", "f2"]),
                audformat.segmented_index(["f1", "f2"], [0, 1], [1, 2]),
            ],
            "Found different number of levels: [1, 3]",
        ),
    ],
)
def test_is_index_alike(objs, error_msg):
    if error_msg is None:
        assert audformat.utils.is_index_alike(objs)
    else:
        assert not audformat.utils.is_index_alike(objs)
        with pytest.raises(
            ValueError,
            match=re.escape(error_msg),
        ):
            audformat.core.utils._assert_index_alike(objs)


@pytest.mark.parametrize(
    "obj, expected",
    [
        (
            audformat.filewise_index(),
            [],
        ),
        (
            pd.Series(dtype="object"),
            [],
        ),
        (
            pd.DataFrame(),
            [],
        ),
        (
            audformat.segmented_index(
                ["f1", "f1", "f2"],
                [0, 1, 0],
                [2, 3, 1],
            ),
            [
                (
                    "f1",
                    audformat.segmented_index(
                        ["f1", "f1"],
                        [0, 1],
                        [2, 3],
                    ),
                ),
                (
                    "f2",
                    audformat.segmented_index(["f2"], [0], [1]),
                ),
            ],
        ),
        (
            pd.Series(
                index=audformat.segmented_index(
                    ["f1", "f1", "f2"],
                    [0, 1, 0],
                    [2, 3, 1],
                ),
                dtype="object",
            ),
            [
                (
                    "f1",
                    pd.Series(
                        index=audformat.segmented_index(
                            ["f1", "f1"],
                            [0, 1],
                            [2, 3],
                        ),
                        dtype="object",
                    ),
                ),
                (
                    "f2",
                    pd.Series(
                        index=audformat.segmented_index(
                            ["f2"],
                            [0],
                            [1],
                        ),
                        dtype="object",
                    ),
                ),
            ],
        ),
        (
            pd.DataFrame(
                index=audformat.filewise_index(["f1", "f1", "f2"]),
            ),
            [
                (
                    "f1",
                    pd.DataFrame(
                        index=audformat.filewise_index(["f1", "f1"]),
                    ),
                ),
                (
                    "f2",
                    pd.DataFrame(
                        index=audformat.filewise_index(["f2"]),
                    ),
                ),
            ],
        ),
    ],
)
def test_iter_by_file(obj, expected):
    result = list(audformat.utils.iter_by_file(obj))
    if not expected:
        assert not result
    for iteration, iteration_expected in zip(result, expected):
        assert iteration[0] == iteration_expected[0]
        if isinstance(obj, pd.Index):
            pd.testing.assert_index_equal(iteration[1], iteration_expected[1])
        elif isinstance(obj, pd.Series):
            pd.testing.assert_series_equal(iteration[1], iteration_expected[1])
        elif isinstance(obj, pd.DataFrame):
            pd.testing.assert_frame_equal(iteration[1], iteration_expected[1])


@pytest.mark.parametrize(
    "labels, expected",
    [
        (
            [],
            [],
        ),
        (
            [[]],
            [],
        ),
        (
            [{}],
            {},
        ),
        (
            [[], []],
            [],
        ),
        (
            [{}, {}],
            {},
        ),
        (
            (["a"], ["b"]),
            ["a", "b"],
        ),
        (
            (["a"], ["b", "c"]),
            ["a", "b", "c"],
        ),
        (
            (["a"], ["a"]),
            ["a"],
        ),
        (
            [{"a": 0}],
            {"a": 0},
        ),
        (
            [{"a": 0}, {"b": 1}],
            {"a": 0, "b": 1},
        ),
        (
            [{"a": 0}, {"b": 1, "c": 2}],
            {"a": 0, "b": 1, "c": 2},
        ),
        (
            [{"a": 0, "b": 1}, {"b": 1, "c": 2}],
            {"a": 0, "b": 1, "c": 2},
        ),
        (
            [{"a": 0, "b": 1}, {"b": 2, "c": 2}],
            {"a": 0, "b": 2, "c": 2},
        ),
        (
            [{"a": 0}, {"a": 1}, {"a": 2}],
            {"a": 2},
        ),
        pytest.param(
            ["a", "b", "c"],
            [],
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(
            ("a", "b", "c"),
            [],
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(
            ["a", "b", 0],
            [],
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(
            [{"a": 0, "b": 1}, ["c"]],
            [],
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(
            [{"a": 0, "b": 1}, {0: "c"}],
            [],
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(
            [["a", "b"], ["b", "c"], "d"],
            [],
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(
            [{0: {"age": 20}}, {"0": {"age": 30}}],
            [],
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(
            [["a", "b"], "misc_id"],
            [],
            marks=pytest.mark.xfail(raises=ValueError),
        ),
    ],
)
def test_join_labels(labels, expected):
    assert utils.join_labels(labels) == expected


def test_join_schemes():
    # Empty list
    audformat.utils.join_schemes([], "scheme_id")

    # One database

    db1 = audformat.Database("db1")
    scheme1 = audformat.Scheme(labels={"a": [1, 2]})
    db1.schemes["scheme_id"] = scheme1
    audformat.utils.join_schemes([db1], "scheme_id")
    assert db1.schemes["scheme_id"] == scheme1

    # Two databases

    db2 = audformat.Database("db2")
    scheme2 = audformat.Scheme(labels={"b": [3]})
    db2.schemes["scheme_id"] = scheme2
    expected = audformat.Scheme(labels={"a": [1, 2], "b": [3]})
    audformat.utils.join_schemes([db1, db2], "scheme_id")
    assert db1.schemes["scheme_id"] == expected
    assert db2.schemes["scheme_id"] == expected

    # Three database

    db3 = audformat.Database("db3")
    scheme3 = audformat.Scheme(labels={"a": [4]})
    db3.schemes["scheme_id"] = scheme3
    expected = audformat.Scheme(labels={"a": [4], "b": [3]})
    audformat.utils.join_schemes([db1, db2, db3], "scheme_id")
    assert db1.schemes["scheme_id"] == expected
    assert db2.schemes["scheme_id"] == expected
    assert db3.schemes["scheme_id"] == expected

    # Fail for schemes without labels
    db4 = audformat.Database("db")
    db4.schemes["scheme_id"] = audformat.Scheme("str")
    error_msg = "All labels must be either of type 'list' or 'dict'"
    with pytest.raises(ValueError, match=error_msg):
        audformat.utils.join_schemes([db4], "scheme_id")

    # Fail for schemes with different label type
    db5 = audformat.Database("db")
    db5.schemes["scheme_id"] = audformat.Scheme("int", labels={0: "a"})
    error_msg = "Elements or keys must have the same dtype"
    with pytest.raises(ValueError, match=error_msg):
        audformat.utils.join_schemes([db1, db5], "scheme_id")

    # Fail if some schemes use labels from misc tables
    db6 = audformat.Database("db")
    db6["misc"] = audformat.MiscTable(pd.Index([0, 1, 2], name="idx"))
    db6.schemes["scheme_id"] = audformat.Scheme(dtype="int", labels="misc")
    error_msg = "The following string values were provided"
    with pytest.raises(ValueError, match=error_msg):
        audformat.utils.join_schemes([db1, db6], "scheme_id")


@pytest.mark.parametrize(
    "country, expected",
    [
        ("uy", "URY"),
        ("uy", "URY"),
        ("uruguay", "URY"),
        ("Uruguay", "URY"),
        pytest.param("xx", None, marks=pytest.mark.xfail(raises=ValueError)),
        pytest.param("xxx", None, marks=pytest.mark.xfail(raises=ValueError)),
        pytest.param("Bad country", None, marks=pytest.mark.xfail(raises=ValueError)),
    ],
)
def test_map_country(country, expected):
    assert utils.map_country(country) == expected


@pytest.mark.parametrize(
    "index, func, expected_index, expected_index_windows",
    [
        (
            audformat.filewise_index(),
            os.path.normpath,
            audformat.filewise_index(),
            audformat.filewise_index(),
        ),
        (
            audformat.filewise_index(["a/f1", "a/f2"]),
            os.path.normpath,
            audformat.filewise_index(["a/f1", "a/f2"]),
            audformat.filewise_index(["a\\f1", "a\\f2"]),
        ),
        (
            audformat.segmented_index(["a/f1"], [0], [1]),
            os.path.normpath,
            audformat.segmented_index(["a/f1"], [0], [1]),
            audformat.segmented_index(["a\\f1"], [0], [1]),
        ),
    ],
)
def test_map_file_path(index, func, expected_index, expected_index_windows):
    mapped_index = audformat.utils.map_file_path(index, func)
    if os.name == "nt":
        expected_index = expected_index_windows
    pd.testing.assert_index_equal(mapped_index, expected_index)


@pytest.mark.parametrize(
    "language, expected",
    [
        ("en", "eng"),
        ("en", "eng"),
        ("english", "eng"),
        ("English", "eng"),
        pytest.param("xx", None, marks=pytest.mark.xfail(raises=ValueError)),
        pytest.param("xxx", None, marks=pytest.mark.xfail(raises=ValueError)),
        pytest.param("Bad language", None, marks=pytest.mark.xfail(raises=ValueError)),
    ],
)
def test_map_language(language, expected):
    assert utils.map_language(language) == expected


@pytest.mark.parametrize(
    "csv,result",
    [
        (
            StringIO(
                """file
f1
f2
f3"""
            ),
            audformat.filewise_index(["f1", "f2", "f3"]),
        ),
        (
            StringIO(
                """file,value
f1,0.0
f2,1.0
f3,2.0"""
            ),
            pd.Series(
                [0.0, 1.0, 2.0],
                index=audformat.filewise_index(["f1", "f2", "f3"]),
                name="value",
            ),
        ),
        (
            StringIO(
                """file,value1,value2
f1,0.0,a
f2,1.0,b
f3,2.0,c"""
            ),
            pd.DataFrame(
                {
                    "value1": [0.0, 1.0, 2.0],
                    "value2": ["a", "b", "c"],
                },
                index=audformat.filewise_index(["f1", "f2", "f3"]),
                columns=["value1", "value2"],
            ),
        ),
        (
            StringIO(
                """file,start,value
f1,00:00:00,0.0
f1,00:00:01,1.0
f2,00:00:02,2.0"""
            ),
            pd.Series(
                [0.0, 1.0, 2.0],
                index=audformat.segmented_index(
                    ["f1", "f1", "f2"],
                    starts=["0s", "1s", "2s"],
                    ends=pd.to_timedelta([pd.NaT, pd.NaT, pd.NaT]),
                ),
                name="value",
            ),
        ),
        (
            StringIO(
                """file,start,value
f1,0,0.0
f1,1,1.0
f2,2,2.0"""
            ),
            pd.Series(
                [0.0, 1.0, 2.0],
                index=audformat.segmented_index(
                    ["f1", "f1", "f2"],
                    starts=["0s", "1s", "2s"],
                    ends=pd.to_timedelta([pd.NaT, pd.NaT, pd.NaT]),
                ),
                name="value",
            ),
        ),
        (
            StringIO(
                """file,end,value
f1,00:00:01,0.0
f1,00:00:02,1.0
f2,00:00:03,2.0"""
            ),
            pd.Series(
                [0.0, 1.0, 2.0],
                index=audformat.segmented_index(
                    ["f1", "f1", "f2"],
                    starts=["0s", "0s", "0s"],
                    ends=["1s", "2s", "3s"],
                ),
                name="value",
            ),
        ),
        (
            StringIO(
                """file,start,end
f1,00:00:00,00:00:01
f1,00:00:01,00:00:02
f2,00:00:02,00:00:03"""
            ),
            audformat.segmented_index(
                ["f1", "f1", "f2"],
                ["0s", "1s", "2s"],
                ["1s", "2s", "3s"],
            ),
        ),
        (
            StringIO(
                """file,start,end
f1,0,1
f1,1,2
f2,2,3"""
            ),
            audformat.segmented_index(
                ["f1", "f1", "f2"],
                ["0s", "1s", "2s"],
                ["1s", "2s", "3s"],
            ),
        ),
        (
            StringIO(
                """file,start,end,value
f1,00:00:00,00:00:01,0.0
f1,00:00:01,00:00:02,1.0
f2,00:00:02,00:00:03,2.0"""
            ),
            pd.Series(
                [0.0, 1.0, 2.0],
                index=audformat.segmented_index(
                    ["f1", "f1", "f2"],
                    starts=["0s", "1s", "2s"],
                    ends=["1s", "2s", "3s"],
                ),
                name="value",
            ),
        ),
        (
            StringIO(
                """file,start,end,value
f1,0,1,0.0
f1,1,2,1.0
f2,2,3,2.0"""
            ),
            pd.Series(
                [0.0, 1.0, 2.0],
                index=audformat.segmented_index(
                    ["f1", "f1", "f2"],
                    starts=["0s", "1s", "2s"],
                    ends=["1s", "2s", "3s"],
                ),
                name="value",
            ),
        ),
        (
            StringIO(
                """file,start,end,value
f1,0.0,1.0,0.0
f1,1.0,2.0,1.0
f2,2.0,3.0,2.0"""
            ),
            pd.Series(
                [0.0, 1.0, 2.0],
                index=audformat.segmented_index(
                    ["f1", "f1", "f2"],
                    starts=["0s", "1s", "2s"],
                    ends=["1s", "2s", "3s"],
                ),
                name="value",
            ),
        ),
        (
            StringIO(
                """file,start,end,value1,value2
f1,00:00:00,00:00:01,0.0,a
f1,00:00:01,00:00:02,1.0,b
f2,00:00:02,00:00:03,2.0,c"""
            ),
            pd.DataFrame(
                {
                    "value1": [0.0, 1.0, 2.0],
                    "value2": ["a", "b", "c"],
                },
                index=audformat.segmented_index(
                    ["f1", "f1", "f2"],
                    starts=["0s", "1s", "2s"],
                    ends=["1s", "2s", "3s"],
                ),
                columns=["value1", "value2"],
            ),
        ),
        pytest.param(
            StringIO(
                """value
0.0
1.0
2.0"""
            ),
            None,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
    ],
)
def test_read_csv(csv, result):
    obj = audformat.utils.read_csv(csv)
    if isinstance(result, pd.Index):
        pd.testing.assert_index_equal(obj, result)
    elif isinstance(result, pd.Series):
        pd.testing.assert_series_equal(obj, result)
    else:
        pd.testing.assert_frame_equal(obj, result)
    # Request dataframe as return type
    csv.seek(0)  # rewind string file object
    obj = audformat.utils.read_csv(csv, as_dataframe=True)
    if isinstance(result, pd.Index):
        result = pd.DataFrame([], columns=[], index=result)
    elif isinstance(result, pd.Series):
        result = result.to_frame()
    pd.testing.assert_frame_equal(obj, result)


@pytest.mark.parametrize(
    "index, extension, pattern, expected_index",
    [
        (
            audformat.filewise_index(),
            "mp3",
            None,
            audformat.filewise_index(),
        ),
        (
            audformat.segmented_index(),
            "mp3",
            None,
            audformat.segmented_index(),
        ),
        (
            audformat.filewise_index(["f1.wav", "f2.wav"]),
            "mp3",
            None,
            audformat.filewise_index(["f1.mp3", "f2.mp3"]),
        ),
        (
            audformat.segmented_index(["f1.wav", "f2.wav"]),
            "mp3",
            None,
            audformat.segmented_index(["f1.mp3", "f2.mp3"]),
        ),
        (
            audformat.filewise_index(["f1.WAV", "f2.WAV"]),
            "MP3",
            None,
            audformat.filewise_index(["f1.MP3", "f2.MP3"]),
        ),
        (
            audformat.filewise_index(["f1", "f2.wv"]),
            "mp3",
            None,
            audformat.filewise_index(["f1", "f2.mp3"]),
        ),
        (
            audformat.filewise_index(["f1.wav", "f2.wav"]),
            "",
            None,
            audformat.filewise_index(["f1", "f2"]),
        ),
        (
            audformat.filewise_index(["f1.ogg", "f2.wav"]),
            "mp3",
            ".ogg",
            audformat.filewise_index(["f1.mp3", "f2.wav"]),
        ),
    ],
)
def test_replace_file_extension(index, extension, pattern, expected_index):
    index = audformat.utils.replace_file_extension(
        index,
        extension,
        pattern=pattern,
    )
    pd.testing.assert_index_equal(index, expected_index)


@pytest.mark.parametrize(
    "index, dtypes, expected",
    [
        (
            pd.Index([]),
            "string",
            pd.Index([], dtype="string"),
        ),
        (
            pd.Index([]),
            {},
            pd.Index([]),
        ),
        (
            pd.Index(["a", "b"]),
            "string",
            pd.Index(["a", "b"], dtype="string"),
        ),
        (
            pd.Index(["a", "b"], dtype="string"),
            "string",
            pd.Index(["a", "b"], dtype="string"),
        ),
        (
            pd.Index(["a", "b"], name="idx"),
            {"idx": "string"},
            pd.Index(["a", "b"], name="idx", dtype="string"),
        ),
        (
            pd.MultiIndex.from_arrays(
                [
                    [0, 1],
                    [2, 3],
                ],
                names=["idx1", "idx2"],
            ),
            "str",
            pd.MultiIndex.from_arrays(
                [
                    ["0", "1"],
                    ["2", "3"],
                ],
                names=["idx1", "idx2"],
            ),
        ),
        (
            pd.MultiIndex.from_arrays(
                [
                    [0, 1],
                    [2, 3],
                ],
                names=["idx1", "idx2"],
            ),
            {
                "idx2": "str",
                "idx1": "str",
            },
            pd.MultiIndex.from_arrays(
                [
                    ["0", "1"],
                    ["2", "3"],
                ],
                names=["idx1", "idx2"],
            ),
        ),
        (
            pd.MultiIndex.from_arrays(
                [
                    [0, 1],
                    [2, 3],
                ],
                names=["idx1", "idx2"],
            ),
            {
                "idx2": "str",
            },
            pd.MultiIndex.from_arrays(
                [
                    [0, 1],
                    ["2", "3"],
                ],
                names=["idx1", "idx2"],
            ),
        ),
        (
            audformat.filewise_index([]),
            "int",
            pd.Index([], dtype="int", name="file"),
        ),
        (
            audformat.filewise_index(["1", "2"]),
            "int64",
            pd.Index([1, 2], name="file"),
        ),
        (
            audformat.segmented_index(["1", "2"], [0, 0], [1, 1]),
            {
                "file": "int64",
                "start": "str",
                "end": "str",
            },
            pd.MultiIndex.from_arrays(
                [
                    [1, 2],
                    ["0 days", "0 days"],
                    ["0 days 00:00:01", "0 days 00:00:01"],
                ],
                names=["file", "start", "end"],
            ),
        ),
        (
            pd.MultiIndex.from_arrays(
                [
                    ["1", "2"],
                    [0, int(1e9)],
                ],
                names=["idx", "time"],
            ),
            {
                "idx": "int64",
                "time": "timedelta64[ns]",
            },
            pd.MultiIndex.from_arrays(
                [
                    [1, 2],
                    pd.to_timedelta([0, 1], unit="s"),
                ],
                names=["idx", "time"],
            ),
        ),
        (
            pd.MultiIndex.from_arrays(
                [
                    ["1", "2"],
                    [None, None],
                ],
                names=["idx", "date"],
            ),
            {
                "idx": "int64",
                "date": "datetime64[ns]",
            },
            pd.MultiIndex.from_arrays(
                [
                    [1, 2],
                    [pd.NaT, pd.NaT],
                ],
                names=["idx", "date"],
            ),
        ),
        (
            pd.MultiIndex.from_arrays(
                [
                    ["f1", "f2"],
                    [0, int(1e9)],
                    [pd.NaT, pd.NaT],
                ],
                names=["file", "start", "end"],
            ),
            {
                "file": "string",
                "start": "timedelta64[ns]",
                "end": "timedelta64[ns]",
            },
            audformat.segmented_index(["f1", "f2"], [0, 1]),
        ),
        pytest.param(
            pd.MultiIndex.from_arrays(
                [
                    [0, 1],
                    [2, 3],
                ],
                names=["idx", "idx"],
            ),
            "str",
            None,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(
            pd.MultiIndex.from_arrays(
                [
                    [0, 1],
                    [2, 3],
                ],
                names=["idx1", "idx2"],
            ),
            {
                "idx1": "string",
                "bad": "string",
            },
            None,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(
            pd.Index([0], name="idx"),
            {"bad": "string"},
            None,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
    ],
)
def test_set_index_dtypes(index, dtypes, expected):
    index = audformat.utils.set_index_dtypes(index, dtypes)
    pd.testing.assert_index_equal(index, expected)


@pytest.mark.parametrize(
    "obj, allow_nat, files_duration, root, expected",
    [
        # empty
        (
            audformat.filewise_index(),
            True,
            None,
            None,
            audformat.segmented_index(),
        ),
        (
            audformat.filewise_index(),
            False,
            None,
            None,
            audformat.segmented_index(),
        ),
        (
            audformat.segmented_index(),
            True,
            None,
            None,
            audformat.segmented_index(),
        ),
        (
            audformat.segmented_index(),
            False,
            None,
            None,
            audformat.segmented_index(),
        ),
        # allow nat
        (
            audformat.filewise_index(pytest.DB.files[:2]),
            True,
            None,
            None,
            audformat.segmented_index(pytest.DB.files[:2]),
        ),
        (
            audformat.segmented_index(pytest.DB.files[:2]),
            True,
            None,
            None,
            audformat.segmented_index(pytest.DB.files[:2]),
        ),
        (
            audformat.segmented_index(
                pytest.DB.files[:2],
                [0.1, 0.5],
                [0.2, pd.NaT],
            ),
            True,
            None,
            None,
            audformat.segmented_index(
                pytest.DB.files[:2],
                [0.1, 0.5],
                [0.2, pd.NaT],
            ),
        ),
        # forbid nat
        (
            audformat.filewise_index(pytest.DB.files[:2]),
            False,
            None,
            pytest.DB_ROOT,
            audformat.segmented_index(
                pytest.DB.files[:2], [0, 0], [pytest.FILE_DUR, pytest.FILE_DUR]
            ),
        ),
        (
            audformat.segmented_index(pytest.DB.files[:2]),
            False,
            None,
            pytest.DB_ROOT,
            audformat.segmented_index(
                pytest.DB.files[:2], [0, 0], [pytest.FILE_DUR, pytest.FILE_DUR]
            ),
        ),
        (
            audformat.segmented_index(
                pytest.DB.files[:2],
                [0.1, 0.5],
                [0.2, pd.NaT],
            ),
            False,
            None,
            pytest.DB_ROOT,
            audformat.segmented_index(
                pytest.DB.files[:2],
                [0.1, 0.5],
                [0.2, pytest.FILE_DUR],
            ),
        ),
        # provide file durations
        (
            audformat.filewise_index(pytest.DB.files[:2]),
            False,
            {
                os.path.join(pytest.DB_ROOT, pytest.DB.files[1]): pytest.FILE_DUR * 2,
            },
            pytest.DB_ROOT,
            audformat.segmented_index(
                pytest.DB.files[:2],
                [0.0, 0.0],
                [pytest.FILE_DUR, pytest.FILE_DUR * 2],
            ),
        ),
        (
            audformat.segmented_index(
                pytest.DB.files[:2],
                [0.1, 0.5],
                [pd.NaT, pd.NaT],
            ),
            False,
            {
                os.path.join(pytest.DB_ROOT, pytest.DB.files[1]): pytest.FILE_DUR * 2,
            },
            pytest.DB_ROOT,
            audformat.segmented_index(
                pytest.DB.files[:2],
                [0.1, 0.5],
                [pytest.FILE_DUR, pytest.FILE_DUR * 2],
            ),
        ),
        # file duration with high precision
        # covering https://github.com/audeering/audformat/issues/379
        (
            audformat.segmented_index(
                [pytest.DB.files[0]],
                [0],
                [pd.NaT],
            ),
            False,
            {
                os.path.join(pytest.DB_ROOT, pytest.DB.files[0]): pd.to_timedelta(
                    2.5225, unit="s"
                ),
            },
            pytest.DB_ROOT,
            audformat.segmented_index(
                [pytest.DB.files[0]],
                [0],
                [2.5225],
            ),
        ),
        (
            audformat.segmented_index(
                pytest.DB.files[:3],
                [1, 0, 0],
                [pd.NaT, 2.37485, pd.NaT],
            ),
            False,
            {
                os.path.join(pytest.DB_ROOT, pytest.DB.files[0]): pd.to_timedelta(
                    2.5225, unit="s"
                ),
                os.path.join(pytest.DB_ROOT, pytest.DB.files[1]): pd.to_timedelta(
                    2.37485, unit="s"
                ),
                os.path.join(pytest.DB_ROOT, pytest.DB.files[2]): pd.to_timedelta(
                    3.458697083, unit="s"
                ),
            },
            pytest.DB_ROOT,
            audformat.segmented_index(
                pytest.DB.files[:3],
                [1, 0, 0],
                [2.5225, 2.37485, 3.458697083],
            ),
        ),
        # file not found
        pytest.param(
            audformat.filewise_index(pytest.DB.files[:2]),
            False,
            None,
            None,
            None,
            marks=pytest.mark.xfail(raises=FileNotFoundError),
        ),
        # series and frame
        (
            pd.Series(
                [1, 2],
                index=audformat.filewise_index(pytest.DB.files[:2]),
            ),
            True,
            None,
            None,
            audformat.segmented_index(pytest.DB.files[:2]),
        ),
        (
            pd.DataFrame(
                {"int": [1, 2], "str": ["a", "b"]},
                index=audformat.filewise_index(pytest.DB.files[:2]),
            ),
            True,
            None,
            None,
            audformat.segmented_index(pytest.DB.files[:2]),
        ),
    ],
)
def test_to_segmented_index(obj, allow_nat, files_duration, root, expected):
    result = audformat.utils.to_segmented_index(
        obj,
        allow_nat=allow_nat,
        files_duration=files_duration,
        root=root,
    )
    if not isinstance(result, pd.Index):
        result = result.index

    pd.testing.assert_index_equal(result, expected)

    if files_duration and not allow_nat:
        # for filewise tables we expect a duration for every file
        # for segmented only where end == NaT
        files = result.get_level_values(audformat.define.IndexField.FILE)
        if audformat.index_type(obj) == audformat.define.IndexType.SEGMENTED:
            mask = result.get_level_values(audformat.define.IndexField.END) == pd.NaT
            files = files[mask]
        for file in files:
            file = os.path.join(root, file)
            assert file in files_duration


@pytest.mark.parametrize(
    "obj, expected_file_names",
    [
        # empty
        (
            audformat.filewise_index(),
            [],
        ),
        (
            audformat.segmented_index(),
            [],
        ),
        (
            pd.Series(index=audformat.filewise_index(), dtype="object"),
            [],
        ),
        (
            pd.Series(index=audformat.segmented_index(), dtype="object"),
            [],
        ),
        (
            pd.DataFrame(index=audformat.filewise_index(), dtype="object"),
            [],
        ),
        (
            pd.DataFrame(index=audformat.segmented_index(), dtype="object"),
            [],
        ),
        # frame
        (
            pytest.DB["segments"].get(),
            [
                f"{i + 1:03d}_{j}"  # 001_0, 001_1, ...
                for i in range(len(pytest.DB["segments"].files.unique()))
                for j in range(
                    len(pytest.DB["segments"].files)
                    // len(pytest.DB["segments"].files.unique())
                )
            ],
        ),
        (
            pytest.DB["files"].get(),
            [f"{i + 1:03d}" for i in range(len(pytest.DB["files"]))],
        ),
        # series
        (
            pytest.DB["segments"]["string"].get(),
            [
                f"{i + 1:03d}_{j}"
                for i in range(len(pytest.DB["segments"].files.unique()))
                for j in range(
                    len(pytest.DB["segments"].files)
                    // len(pytest.DB["segments"].files.unique())
                )
            ],
        ),
        (
            pytest.DB["files"]["string"].get(),
            [f"{i + 1:03d}" for i in range(len(pytest.DB["files"]))],
        ),
        # index
        (
            pytest.DB["segments"].index,
            [
                f"{i + 1:03d}_{j}"
                for i in range(len(pytest.DB["segments"].files.unique()))
                for j in range(
                    len(pytest.DB["segments"].files)
                    // len(pytest.DB["segments"].files.unique())
                )
            ],
        ),
        (
            pytest.DB["files"].index,
            [f"{i + 1:03d}" for i in range(len(pytest.DB["files"]))],
        ),
    ],
)
def test_to_filewise_index(tmpdir, obj, expected_file_names):
    output_folder = tmpdir

    new_obj = utils.to_filewise_index(
        obj=obj,
        root=pytest.DB_ROOT,
        output_folder=output_folder,
        num_workers=3,
    )
    new_index = new_obj if isinstance(new_obj, pd.Index) else new_obj.index
    new_files = new_index.get_level_values(define.IndexField.FILE).values

    assert audformat.is_filewise_index(new_obj)

    if isinstance(obj, pd.DataFrame):
        pd.testing.assert_frame_equal(
            obj.reset_index(drop=True),
            new_obj.reset_index(drop=True),
        )
    elif isinstance(obj, pd.Series):
        pd.testing.assert_series_equal(
            obj.reset_index(drop=True),
            new_obj.reset_index(drop=True),
        )

    if audformat.is_segmented_index(obj):
        # already `framewise` frame is unprocessed
        if len(new_files) > 0:
            assert os.path.isabs(output_folder) == os.path.isabs(new_files[0])

    if audformat.is_filewise_index(obj):
        # files of unprocessed frame are relative to `root`
        new_files = [os.path.join(pytest.DB_ROOT, f) for f in new_files]

    assert all(os.path.exists(f) for f in new_files)

    file_names = [f.split("/")[-1].rsplit(".", 1)[0] for f in new_files]
    assert file_names == expected_file_names

    if len(obj) > 0 and not audformat.is_filewise_index(obj):
        error_msg = "``output_folder`` may not be contained in path to files"
        with pytest.raises(ValueError, match=error_msg):
            utils.to_filewise_index(
                obj=obj,
                root=pytest.DB_ROOT,
                output_folder=".",
                num_workers=3,
            )


@pytest.mark.parametrize(
    "max_num_seg_thres",
    [
        None,  # default
        0,  # force pd.concat() solution
    ],
)
@pytest.mark.parametrize(
    "objs, expected",
    [
        # empty
        (
            [],
            pd.Index([]),
        ),
        # conform to audformat
        (
            [
                audformat.filewise_index(),
            ],
            audformat.filewise_index(),
        ),
        (
            [
                audformat.filewise_index(),
                audformat.filewise_index(),
            ],
            audformat.filewise_index(),
        ),
        (
            [
                audformat.filewise_index(["f1", "f2"]),
                audformat.filewise_index(["f1", "f2"]),
            ],
            audformat.filewise_index(["f1", "f2"]),
        ),
        (
            [
                audformat.filewise_index(["f2", "f1"]),
                audformat.filewise_index(["f1", "f2"]),
            ],
            audformat.filewise_index(["f2", "f1"]),
        ),
        (
            [
                audformat.filewise_index(["f1", "f2"]),
                audformat.filewise_index(["f1", "f2"]),
                audformat.filewise_index(["f2", "f3"]),
            ],
            audformat.filewise_index(["f1", "f2", "f3"]),
        ),
        (
            [
                audformat.filewise_index(["f1", "f2"]),
                audformat.filewise_index(["f1", "f2"]),
                audformat.filewise_index("f3"),
            ],
            audformat.filewise_index(["f1", "f2", "f3"]),
        ),
        (
            [
                audformat.segmented_index(),
            ],
            audformat.segmented_index(),
        ),
        (
            [
                audformat.segmented_index(),
                audformat.segmented_index(),
            ],
            audformat.segmented_index(),
        ),
        (
            [
                audformat.segmented_index(["f1", "f2"]),
                audformat.segmented_index(["f1", "f2"]),
            ],
            audformat.segmented_index(["f1", "f2"]),
        ),
        (
            [
                audformat.segmented_index(["f1", "f2"]),
                audformat.segmented_index(["f3", "f4"]),
            ],
            audformat.segmented_index(["f1", "f2", "f3", "f4"]),
        ),
        (
            [
                audformat.segmented_index(["f1", "f2"], [0, 0], [1, 1]),
                audformat.segmented_index(["f2", "f1"], [0, 0], [1, 1]),
                audformat.segmented_index(["f2", "f3"], [0, 0], [1, 1]),
            ],
            audformat.segmented_index(
                ["f1", "f2", "f3"],
                [0, 0, 0],
                [1, 1, 1],
            ),
        ),
        (
            [
                audformat.segmented_index(["f1", "f2"], [0, 0], [1, 1]),
                audformat.segmented_index(["f2", "f1"], [0, 0], [1, 1]),
                audformat.segmented_index(["f2", "f3"], [1, 1], [2, 2]),
            ],
            audformat.segmented_index(
                ["f1", "f2", "f2", "f3"],
                [0, 0, 1, 1],
                [1, 1, 2, 2],
            ),
        ),
        (
            [
                audformat.filewise_index(),
                audformat.segmented_index(),
            ],
            audformat.segmented_index(),
        ),
        (
            [
                audformat.filewise_index(["f1", "f2"]),
                audformat.segmented_index(),
            ],
            audformat.segmented_index(["f1", "f2"]),
        ),
        (
            [
                audformat.filewise_index(),
                audformat.segmented_index(["f1", "f2"]),
            ],
            audformat.segmented_index(["f1", "f2"]),
        ),
        (
            [
                audformat.segmented_index(["f1", "f2"], [0, 0], [1, 1]),
                audformat.segmented_index(["f2", "f3"], [0, 0], [1, 1]),
                audformat.filewise_index(["f1", "f2"]),
            ],
            audformat.segmented_index(
                ["f1", "f2", "f3", "f1", "f2"],
                [0, 0, 0, 0, 0],
                [1, 1, 1, pd.NaT, pd.NaT],
            ),
        ),
        (
            [
                audformat.segmented_index(["f1", "f2"], [0, 0], [1, 1]),
                audformat.segmented_index(["f2", "f3"], [0, 0], [1, 1]),
                audformat.filewise_index("f1"),
            ],
            audformat.segmented_index(
                ["f1", "f2", "f3", "f1"],
                [0, 0, 0, 0],
                [1, 1, 1, pd.NaT],
            ),
        ),
        (
            [
                audformat.segmented_index(["f1", "f2"], [0, 0], [1, 1]),
                audformat.filewise_index(["f1", "f2"]),
                audformat.filewise_index(["f2", "f3"]),
            ],
            audformat.segmented_index(
                ["f1", "f2", "f1", "f2", "f3"],
                [0, 0, 0, 0, 0],
                [1, 1, pd.NaT, pd.NaT, pd.NaT],
            ),
        ),
        (
            [
                audformat.filewise_index(["f1", "f2"]),
                audformat.segmented_index(["f1", "f2"], [0, 0], [1, 1]),
                audformat.filewise_index(["f2", "f3"]),
            ],
            audformat.segmented_index(
                ["f1", "f2", "f1", "f2", "f3"],
                [0, 0, 0, 0, 0],
                [pd.NaT, pd.NaT, 1, 1, pd.NaT],
            ),
        ),
        # other
        (
            [
                pd.Index([]),
            ],
            pd.Index([]),
        ),
        (
            [
                pd.Index([]),
                pd.Index([]),
            ],
            pd.Index([]),
        ),
        (
            [
                pd.Index([0, 1], name="idx"),
                pd.Index([1, 2], name="idx"),
            ],
            pd.Index([0, 1, 2], dtype="Int64", name="idx"),
        ),
        (
            [
                pd.Index([0, 1], name="idx"),
                pd.Index([1, 2], dtype="Int64", name="idx"),
            ],
            pd.Index([0, 1, 2], dtype="Int64", name="idx"),
        ),
        (
            [
                pd.Index([1, 2], dtype="Int64", name="idx"),
                pd.Index([0, 1], name="idx"),
            ],
            pd.Index([1, 2, 0], dtype="Int64", name="idx"),
        ),
        (
            [
                pd.Index([0, 1], name="idx"),
                pd.MultiIndex.from_arrays([[1, 2]], names=["idx"]),
            ],
            pd.Index([0, 1, 2], dtype="Int64", name="idx"),
        ),
        (
            [
                pd.MultiIndex.from_arrays([[0, 1]], names=["idx"]),
                pd.MultiIndex.from_arrays([[1, 2]], names=["idx"]),
            ],
            audformat.utils.set_index_dtypes(
                pd.MultiIndex.from_arrays([[0, 1, 2]], names=["idx"]),
                "Int64",
            ),
        ),
        (
            [
                pd.MultiIndex.from_arrays(
                    [["a", "b", "c"], [0, 1, 2]],
                    names=["idx1", "idx2"],
                ),
                pd.MultiIndex.from_arrays(
                    [["b", "c"], [1, 3]],
                    names=["idx1", "idx2"],
                ),
            ],
            audformat.utils.set_index_dtypes(
                pd.MultiIndex.from_arrays(
                    [["a", "b", "c", "c"], [0, 1, 2, 3]],
                    names=["idx1", "idx2"],
                ),
                {"idx2": "Int64"},
            ),
        ),
        pytest.param(
            [
                pd.Index([], name="idx1"),
                pd.Index([], name="idx2"),
            ],
            None,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
    ],
)
def test_union(max_num_seg_thres, objs, expected):
    max_num_seg_thres_default = audformat.core.utils.UNION_MAX_INDEX_LEN_THRES
    if max_num_seg_thres is not None:
        audformat.core.utils.UNION_MAX_INDEX_LEN_THRES = max_num_seg_thres

    pd.testing.assert_index_equal(
        audformat.utils.union(objs),
        expected,
    )
    # Ensure A ∪ (B ∪ C) == (A ∪ B) ∪ C
    if len(objs) > 2:
        pd.testing.assert_index_equal(
            audformat.utils.union(
                [
                    objs[0],
                    audformat.utils.union(objs[1:]),
                ]
            ),
            audformat.utils.union(
                [
                    audformat.utils.union(objs[:-1]),
                    objs[-1],
                ]
            ),
        )

    audformat.core.utils.UNION_MAX_INDEX_LEN_THRES = max_num_seg_thres_default
