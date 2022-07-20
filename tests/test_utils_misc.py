import pytest

import pandas as pd

import audformat


@pytest.mark.parametrize(
    'objs, expected',
    [
        (
            [
                pd.Index([]),
            ],
            True,
        ),
        (
            [
                pd.Index([]),
                pd.Index([]),
            ],
            True,
        ),
        (
            [
                pd.Index([], name='l'),
                pd.Index([], name='L'),
            ],
            False,
        ),
        (
            [
                pd.Index([]),
                pd.MultiIndex([[]], [[]]),
            ],
            True,
        ),
        (
            [
                pd.Index([1, 2, 3], name='l'),
                pd.MultiIndex.from_arrays([[10, 20]], names=['l']),
            ],
            True,
        ),
        (
            [
                pd.Index(
                    [1, 2, 3],
                    dtype='Int64',
                    name='l',
                ),
                pd.MultiIndex.from_arrays(
                    [[10, 20]],
                    names=['l'],
                ),
            ],
            True,
        ),
        (
            [
                pd.Index([1, 2, 3], name='l'),
                pd.MultiIndex.from_arrays([[10, 20]], names=['L']),
            ],
            False,
        ),
        (
            [
                pd.Index(['a', 'b', 'c'], name='l'),
                pd.MultiIndex.from_arrays([[10, 20]], names=['l']),
            ],
            False,
        ),
        (
            [
                pd.Index([1, 2, 3], name='l'),
                pd.MultiIndex.from_arrays(
                    [
                        [10],
                        [20],
                    ],
                    names=['l1', 'l2'],
                ),
            ],
            False,
        ),
        (
            [
                pd.MultiIndex.from_arrays(
                    [
                        [1, 2, 3],
                        ['a', 'b', 'c'],
                    ],
                    names=['l1', 'l2'],
                ),
                pd.MultiIndex.from_arrays(
                    [
                        [10],
                        ['10'],
                    ],
                    names=['l1', 'l2'],
                ),
            ],
            True,
        ),
        (
            [
                pd.MultiIndex.from_arrays(
                    [
                        ['a', 'b', 'c'],
                        [1, 2, 3],
                    ],
                    names=['l1', 'l2'],
                ),
                pd.MultiIndex.from_arrays(
                    [
                        [10],
                        ['10'],
                    ],
                    names=['l1', 'l2'],
                ),
            ],
            False,
        ),
        (
            [
                pd.MultiIndex.from_arrays(
                    [
                        ['a', 'b', 'c'],
                        [1, 2, 3],
                    ],
                    names=['l1', 'l2'],
                ),
                pd.MultiIndex.from_arrays(
                    [
                        [],
                        [],
                    ],
                    names=['l1', 'l2'],
                ),
            ],
            False,
        ),
        (
            [
                pd.MultiIndex.from_arrays(
                    [
                        ['a', 'b', 'c'],
                        [1, 2, 3],
                    ],
                    names=['l1', 'l2'],
                ),
                pd.MultiIndex.from_arrays(
                    [
                        ['a', 'b', 'c'],
                        [1, 2, 3],
                    ],
                    names=['L1', 'L2'],
                ),
            ],
            False,
        ),
    ]
)
def test_compatible_index(objs, expected):
    assert audformat.utils.misc.compatible_index(objs) == expected
