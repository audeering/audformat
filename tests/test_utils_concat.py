import numpy as np
import pandas as pd
import pytest

import audformat


@pytest.mark.parametrize(
    'objs, overwrite, expected',
    [
        # empty
        (
            [],
            False,
            pd.Series([], pd.Index([]), dtype='object'),
        ),
        (
            [pd.Series([], pd.Index([]), dtype='object')],
            False,
            pd.Series([], pd.Index([]), dtype='object')
        ),
        (
            [pd.Series([], audformat.filewise_index(), dtype='object')],
            False,
            pd.Series([], audformat.filewise_index(), dtype='object')
        ),
        (
            [pd.Series([], audformat.segmented_index(), dtype='object')],
            False,
            pd.Series([], audformat.segmented_index(), dtype='object')
        ),
        (
            [pd.DataFrame([], audformat.segmented_index(), dtype='object')],
            False,
            pd.DataFrame([], audformat.segmented_index(), dtype='object')
        ),
        # combine series with same name
        (
            [
                pd.Series([], audformat.filewise_index(), dtype=float),
                pd.Series([1., 2.], audformat.filewise_index(['f1', 'f2'])),
            ],
            False,
            pd.Series([1., 2.], audformat.filewise_index(['f1', 'f2'])),
        ),
        (
            [
                pd.Series([1., 2.], pd.Index(['f1', 'f2'])),
                pd.Series([1., 2.], pd.Index(['f1', 'f2'])),
            ],
            False,
            pd.Series([1., 2.], pd.Index(['f1', 'f2'])),
        ),
        (
            [
                pd.Series([1.], audformat.filewise_index('f1')),
                pd.Series([2.], audformat.filewise_index('f2')),
            ],
            False,
            pd.Series([1., 2.], audformat.filewise_index(['f1', 'f2'])),
        ),
        (
            [
                pd.Series([1.], audformat.segmented_index('f1')),
                pd.Series([2.], audformat.segmented_index('f2')),
            ],
            False,
            pd.Series([1., 2.], audformat.segmented_index(['f1', 'f2'])),
        ),
        (
            [
                pd.Series([1.], audformat.filewise_index('f1')),
                pd.Series([2.], audformat.segmented_index('f2')),
            ],
            False,
            pd.Series([1., 2.], audformat.segmented_index(['f1', 'f2'])),
        ),
        (
            [
                pd.Series([1.], pd.Index(['f1'])),
                pd.Series(
                    [2.],
                    pd.MultiIndex.from_arrays([['f2']]),
                ),
            ],
            False,
            pd.Series(
                [1., 2.],
                pd.Index(['f1', 'f2']),
            ),
        ),
        (
            [
                pd.Series([1.], pd.Index(['f1'], name='idx')),
                pd.Series(
                    [2.],
                    pd.MultiIndex.from_arrays([['f2']], names=['idx']),
                ),
            ],
            False,
            pd.Series(
                [1., 2.],
                pd.Index(['f1', 'f2'], name='idx'),
            ),
        ),
        # combine values in same location
        (
            [
                pd.Series([np.nan], audformat.filewise_index('f1')),
                pd.Series([np.nan], audformat.filewise_index('f1')),
            ],
            False,
            pd.Series([np.nan], audformat.filewise_index('f1')),
        ),
        (
            [
                pd.Series([1.], audformat.filewise_index('f1')),
                pd.Series([np.nan], audformat.filewise_index('f1')),
            ],
            False,
            pd.Series([1.], audformat.filewise_index('f1')),
        ),
        (
            [
                pd.Series([1.], audformat.filewise_index('f1')),
                pd.Series([1.], audformat.filewise_index('f1')),
            ],
            False,
            pd.Series([1.], audformat.filewise_index('f1')),
        ),
        # combine series and overwrite values
        (
            [
                pd.Series([1.], audformat.filewise_index('f1')),
                pd.Series([np.nan], audformat.filewise_index('f1')),
            ],
            True,
            pd.Series([1.], audformat.filewise_index('f1')),
        ),
        (
            [
                pd.Series([1.], audformat.filewise_index('f1')),
                pd.Series([2.], audformat.filewise_index('f1')),
            ],
            True,
            pd.Series([2.], audformat.filewise_index('f1')),
        ),
        # combine values with matching dtype
        (
            [
                pd.Series(
                    [1, 2],
                    audformat.filewise_index(['f1', 'f2']),
                    dtype='int64',
                ),
                pd.Series(
                    [1, 2],
                    audformat.filewise_index(['f1', 'f2']),
                    dtype='Int64',
                ),
            ],
            False,
            pd.Series(
                [1, 2],
                audformat.filewise_index(['f1', 'f2']),
                dtype='Int64',
            ),
        ),
        (
            [
                pd.Series(
                    [1., 2.],
                    audformat.filewise_index(['f1', 'f2']),
                    dtype='float32',
                ),
                pd.Series(
                    [1., 2.],
                    audformat.filewise_index(['f1', 'f2']),
                    dtype='float64',
                ),
            ],
            False,
            pd.Series(
                [1., 2.],
                audformat.filewise_index(['f1', 'f2']),
                dtype='float64',
            ),
        ),
        (
            [
                pd.Series(
                    [1., 2.],
                    audformat.filewise_index(['f1', 'f2']),
                    dtype='float32',
                ),
                pd.Series(
                    [1., 2.],
                    audformat.filewise_index(['f1', 'f2']),
                    dtype='float64',
                ),
            ],
            False,
            pd.Series(
                [1., 2.],
                audformat.filewise_index(['f1', 'f2']),
                dtype='float64',
            ),
        ),
        (
            [
                pd.Series(
                    ['a', 'b', 'a'],
                    index=audformat.filewise_index(['f1', 'f2', 'f3']),
                ),
                pd.Series(
                    ['a', 'b', 'a'],
                    index=audformat.filewise_index(['f1', 'f2', 'f3']),
                ),
            ],
            False,
            pd.Series(
                ['a', 'b', 'a'],
                index=audformat.filewise_index(['f1', 'f2', 'f3']),
            )
        ),
        (
            [
                pd.Series(
                    ['a', 'b', 'a'],
                    index=audformat.filewise_index(['f1', 'f2', 'f3']),
                    dtype='category',
                ),
                pd.Series(
                    ['a', 'b', 'a'],
                    index=audformat.filewise_index(['f1', 'f2', 'f3']),
                    dtype='category',
                ),
            ],
            False,
            pd.Series(
                ['a', 'b', 'a'],
                index=audformat.filewise_index(['f1', 'f2', 'f3']),
                dtype='category',
            )
        ),
        # combine series with non-nullable dtype
        (
            [
                pd.Series([1, 2], audformat.filewise_index(['f1', 'f2'])),
                pd.Series([1, 2], audformat.filewise_index(['f1', 'f2'])),
            ],
            False,
            pd.Series(
                [1, 2],
                audformat.filewise_index(['f1', 'f2']),
                dtype='Int64'
            ),
        ),
        (
            [
                pd.Series(
                    True,
                    audformat.filewise_index('f1'),
                    dtype='bool',
                ),
                pd.Series(
                    True,
                    audformat.filewise_index('f2'),
                    dtype='bool',
                ),
            ],
            False,
            pd.Series(
                True,
                audformat.filewise_index(['f1', 'f2']),
                dtype='boolean',
            ),
        ),
        (
            [
                pd.Series(
                    1,
                    audformat.filewise_index('f1'),
                    dtype='int64',
                ),
                pd.Series(
                    2,
                    audformat.filewise_index('f2'),
                    dtype='int64',
                ),
            ],
            False,
            pd.Series(
                [1, 2],
                audformat.filewise_index(['f1', 'f2']),
                dtype='Int64',
            ),
        ),
        # combine series with different names
        (
            [
                pd.Series([1.], audformat.filewise_index('f1'), name='c1'),
                pd.Series([2.], audformat.filewise_index('f1'), name='c2'),
            ],
            False,
            pd.DataFrame(
                {
                    'c1': [1.],
                    'c2': [2.],
                },
                audformat.filewise_index('f1'),
            ),
        ),
        (
            [
                pd.Series([1.], audformat.filewise_index('f1'), name='c1'),
                pd.Series([2.], audformat.filewise_index('f2'), name='c2'),
            ],
            False,
            pd.DataFrame(
                {
                    'c1': [1., np.nan],
                    'c2': [np.nan, 2.],
                },
                audformat.filewise_index(['f1', 'f2']),
            ),
        ),
        (
            [
                pd.Series(
                    [1., 2.],
                    audformat.filewise_index(['f1', 'f2']),
                    name='c1',
                ),
                pd.Series(
                    [2.],
                    audformat.filewise_index('f2'),
                    name='c2',
                ),
            ],
            False,
            pd.DataFrame(
                {
                    'c1': [1., 2.],
                    'c2': [np.nan, 2.],
                },
                audformat.filewise_index(['f1', 'f2']),
            ),
        ),
        (
            [
                pd.Series(
                    [1.],
                    audformat.filewise_index('f1'),
                    name='c1'),
                pd.Series(
                    [2.],
                    audformat.segmented_index('f1', 0, 1),
                    name='c2',
                ),
            ],
            False,
            pd.DataFrame(
                {
                    'c1': [1., np.nan],
                    'c2': [np.nan, 2.],
                },
                audformat.segmented_index(
                    ['f1', 'f1'],
                    [0, 0],
                    [None, 1],
                ),
            ),
        ),
        # combine series and data frame
        (
            [
                pd.Series(
                    [1., 2.],
                    audformat.filewise_index(['f1', 'f2']),
                    name='c',
                ),
                pd.DataFrame(
                    {
                        'c': [2., 3.]
                    },
                    audformat.filewise_index(['f2', 'f3']),
                ),
            ],
            False,
            pd.DataFrame(
                {
                    'c': [1., 2., 3.],
                },
                audformat.filewise_index(['f1', 'f2', 'f3']),
            ),
        ),
        (
            [
                pd.Series(
                    [1., 2.],
                    audformat.filewise_index(['f1', 'f2']),
                    name='c1',
                ),
                pd.Series(
                    ['a', np.nan, 'c'],
                    audformat.filewise_index(['f1', 'f2', 'f3']),
                    name='c2',
                ),
                pd.DataFrame(
                    {
                        'c1': [np.nan, 4.],
                        'c2': ['b', 'd'],
                    },
                    audformat.segmented_index(['f2', 'f4']),
                ),
            ],
            False,
            pd.DataFrame(
                {
                    'c1': [1., 2., np.nan, 4.],
                    'c2': ['a', 'b', 'c', 'd']
                },
                audformat.segmented_index(['f1', 'f2', 'f3', 'f4']),
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
    'objs, aggregate_function, expected',
    [
        # empty
        (
            [],
            None,
            pd.Series([], pd.Index([]), dtype='object'),
        ),
        # identical values
        (
            [
                pd.Series([1, 2], pd.Index(['a', 'b']), dtype='float'),
                pd.Series([1, 2], pd.Index(['a', 'b']), dtype='float'),
            ],
            None,
            pd.Series([1, 2], pd.Index(['a', 'b']), dtype='float'),
        ),
        (
            [
                pd.Series([1, 2], pd.Index(['a', 'b']), dtype='float'),
                pd.Series([1, 2], pd.Index(['a', 'b']), dtype='float'),
            ],
            np.mean,
            pd.Series([1, 2], pd.Index(['a', 'b']), dtype='float'),
        ),
        (
            [
                pd.Series([1, 2], pd.Index(['a', 'b']), dtype='float'),
                pd.Series([1, 2], pd.Index(['a', 'b']), dtype='float'),
            ],
            tuple,
            pd.Series([1, 2], pd.Index(['a', 'b']), dtype='float'),
        ),
        (
            [
                pd.Series([1, 2], pd.Index(['a', 'b']), dtype='float'),
                pd.Series([1, 2], pd.Index(['a', 'b']), dtype='float'),
            ],
            np.sum,
            pd.Series([1, 2], pd.Index(['a', 'b']), dtype='float'),
        ),
        (
            [
                pd.Series([1, 2], pd.Index(['a', 'b']), dtype='float'),
                pd.Series([1, 2], pd.Index(['a', 'b']), dtype='float'),
            ],
            np.var,
            pd.Series([1, 2], pd.Index(['a', 'b']), dtype='float'),
        ),
        # one value different
        (
            [
                pd.Series([1, 2], pd.Index(['a', 'b']), dtype='float'),
                pd.Series([1, 3], pd.Index(['a', 'b']), dtype='float'),
            ],
            np.mean,
            pd.Series([1, 2.5], pd.Index(['a', 'b']), dtype='float'),
        ),
        (
            [
                pd.Series([1, 2], pd.Index(['a', 'b']), dtype='float'),
                pd.Series([1, 3], pd.Index(['a', 'b']), dtype='float'),
            ],
            np.sum,
            pd.Series([2, 5], pd.Index(['a', 'b']), dtype='float'),
        ),
        (
            [
                pd.Series([1, 2], pd.Index(['a', 'b']), dtype='float'),
                pd.Series([1, 3], pd.Index(['a', 'b']), dtype='float'),
                pd.Series([1, 4], pd.Index(['a', 'b']), dtype='float'),
            ],
            np.mean,
            pd.Series([1, 3], pd.Index(['a', 'b']), dtype='float'),
        ),
        (
            [
                pd.Series([1, 2], pd.Index(['a', 'b']), dtype='float'),
                pd.Series([1, 3], pd.Index(['a', 'b']), dtype='float'),
                pd.Series([1, 4], pd.Index(['a', 'b']), dtype='float'),
            ],
            np.sum,
            pd.Series([3, 9], pd.Index(['a', 'b']), dtype='float'),
        ),
        # different values
        (
            [
                pd.Series([1, 2], pd.Index(['a', 'b']), dtype='float'),
                pd.Series([2, 3], pd.Index(['a', 'b']), dtype='float'),
            ],
            np.mean,
            pd.Series([1.5, 2.5], pd.Index(['a', 'b']), dtype='float'),
        ),
        (
            [
                pd.Series([1, 2], pd.Index(['a', 'b']), dtype='float'),
                pd.Series([2, 3], pd.Index(['a', 'b']), dtype='float'),
            ],
            np.sum,
            pd.Series([3, 5], pd.Index(['a', 'b']), dtype='float'),
        ),
        (
            [
                pd.Series([1, 2], pd.Index(['a', 'b']), dtype='float'),
                pd.Series([2, 3], pd.Index(['a', 'b']), dtype='float'),
            ],
            lambda y: ('a', 'b'),
            pd.Series(
                [('a', 'b'), ('a', 'b')],
                pd.Index(['a', 'b']),
                dtype='object',
            ),
        ),
        (
            [
                pd.Series([1, 2], pd.Index(['a', 'b']), dtype='float'),
                pd.Series([2, 3], pd.Index(['a', 'b']), dtype='float'),
            ],
            lambda y: ('a', 'b'),
            pd.Series(
                [('a', 'b'), ('a', 'b')],
                pd.Index(['a', 'b']),
                dtype='object',
            ),
        ),
        (
            [
                pd.Series([1, 2], pd.Index(['a', 'b']), dtype='float'),
                pd.Series([2, 3], pd.Index(['a', 'b']), dtype='float'),
                pd.Series([3, 4], pd.Index(['a', 'b']), dtype='float'),
            ],
            np.mean,
            pd.Series([2, 3], pd.Index(['a', 'b']), dtype='float'),
        ),
        (
            [
                pd.Series([1, 2], pd.Index(['a', 'b']), dtype='float'),
                pd.Series([2, 3], pd.Index(['a', 'b']), dtype='float'),
                pd.Series([3, 4], pd.Index(['a', 'b']), dtype='float'),
            ],
            np.sum,
            pd.Series([6, 9], pd.Index(['a', 'b']), dtype='float'),
        ),
        (
            [
                pd.Series(
                    [1, 2],
                    audformat.filewise_index(['a', 'b']),
                    dtype='float',
                ),
                pd.Series(
                    [2, 3],
                    audformat.filewise_index(['a', 'b']),
                    dtype='float',
                ),
            ],
            np.sum,
            pd.Series(
                [3, 5],
                audformat.filewise_index(['a', 'b']),
                dtype='float',
            ),
        ),
        (
            [
                pd.Series(['a', 'b'], pd.Index(['a', 'b']), dtype='string'),
                pd.Series(['b', 'a'], pd.Index(['a', 'b']), dtype='string'),
            ],
            lambda y: np.char.add(y[0], y[1]),
            pd.Series(['ab', 'ba'], pd.Index(['a', 'b']), dtype='string'),
        ),
        (
            [
                pd.Series([1, 2], pd.Index(['a', 'b']), dtype='float'),
                pd.Series([2, 3], pd.Index(['a', 'b']), dtype='float'),
                pd.Series([3, 4], pd.Index(['a', 'b']), dtype='float'),
            ],
            lambda y: y[2],
            pd.Series([3, 4], pd.Index(['a', 'b']), dtype='float'),
        ),
        (
            [
                pd.DataFrame(
                    {
                        'A': [1, 3],
                        'B': [2, 4],
                    },
                    pd.Index(['a', 'b']),
                    dtype='float',
                ),
                pd.DataFrame(
                    {
                        'A': [2, 4],
                        'B': [3, 5],
                    },
                    pd.Index(['a', 'b']),
                    dtype='float',
                ),
            ],
            np.sum,
            pd.DataFrame(
                {
                    'A': [3, 7],
                    'B': [5, 9],
                },
                pd.Index(['a', 'b']),
                dtype='float',
            ),
        ),
        (
            [
                pd.DataFrame(
                    {
                        'A': [1, 3],
                        'B': [2, 4],
                    },
                    pd.Index(['a', 'b']),
                    dtype='float',
                ),
                pd.DataFrame(
                    {
                        'A': [2, 4],
                        'B': [3, 5],
                    },
                    pd.Index(['a', 'b']),
                    dtype='float',
                ),
            ],
            lambda y: y[1],
            pd.DataFrame(
                {
                    'A': [2, 4],
                    'B': [3, 5],
                },
                pd.Index(['a', 'b']),
                dtype='float',
            ),
        ),
        # different index
        (
            [
                pd.DataFrame(
                    {
                        'A': [1, 1, 1],
                        'B': [1, 1, 1],
                    },
                    index=pd.Index(['a', 'b', 'c']),
                    dtype='float',
                ),
                pd.DataFrame(
                    {
                        'A': [2, 2, 2],
                        'B': [2, 2, 2],
                    },
                    index=pd.Index(['b', 'c', 'd']),
                    dtype='float',
                ),
            ],
            np.sum,
            pd.DataFrame(
                {
                    'A': [1, 3, 3, 2],
                    'B': [1, 3, 3, 2],
                },
                index=pd.Index(['a', 'b', 'c', 'd']),
                dtype='float',
            ),
        ),
        (
            [
                pd.DataFrame(
                    {
                        'A': [1, 1, 1],
                        'B': [1, 1, 1],
                    },
                    index=pd.Index(['b', 'c', 'd']),
                    dtype='float',
                ),
                pd.DataFrame(
                    {
                        'A': [2, 2, 2],
                        'B': [2, 2, 2],
                    },
                    index=pd.Index(['a', 'b', 'c']),
                    dtype='float',
                ),
            ],
            np.sum,
            pd.DataFrame(
                {
                    'A': [3, 3, 1, 2],
                    'B': [3, 3, 1, 2],
                },
                index=pd.Index(['b', 'c', 'd', 'a']),
                dtype='float',
            ),
        ),
        (
            [
                pd.DataFrame(
                    {
                        'A': [1., 1., 1.],
                        'B': ['A', 'A', 'A'],
                    },
                    index=pd.Index(['a', 'b', 'c']),
                ),
                pd.DataFrame(
                    {
                        'A': [2., 2., 2.],
                        'B': ['B', 'B', 'B'],
                    },
                    index=pd.Index(['b', 'c', 'd']),
                ),
            ],
            lambda y: y[0],
            pd.DataFrame(
                {
                    'A': [1., 1., 1., 2.],
                    'B': ['A', 'A', 'A', 'B'],
                },
                index=pd.Index(['a', 'b', 'c', 'd']),
            ),
        ),
        (
            [
                pd.DataFrame(
                    {
                        'A': [1, 1, 1],
                        'B': [1, 1, 1],
                    },
                    index=pd.Index(['a', 'b', 'c']),
                ),
                pd.DataFrame(
                    {
                        'A': [2, 2, 2],
                        'B': [2, 2, 2],
                    },
                    index=pd.Index(['b', 'c', 'd']),
                ),
                pd.DataFrame(
                    {
                        'A': [3],
                        'B': [3],
                    },
                    index=pd.Index(['a']),
                ),
            ],
            np.sum,
            pd.DataFrame(
                {
                    'A': [4, 3, 3, 2],
                    'B': [4, 3, 3, 2],
                },
                index=pd.Index(['a', 'b', 'c', 'd']),
                dtype='Int64',
            ),
        ),
        (
            [
                pd.DataFrame(
                    {
                        'A': [1, 1, 1],
                        'B': [1, 1, 1],
                    },
                    index=pd.Index(['a', 'b', 'c']),
                ),
                pd.DataFrame(
                    {
                        'A': [2, 2, 2],
                        'B': [2, 2, 2],
                    },
                    index=pd.Index(['b', 'c', 'd']),
                ),
                pd.DataFrame(
                    {
                        'A': [3, 3],
                        'B': [3, 3],
                    },
                    index=pd.Index(['a', 'd']),
                ),
            ],
            np.sum,
            pd.DataFrame(
                {
                    'A': [4, 3, 3, 5],
                    'B': [4, 3, 3, 5],
                },
                index=pd.Index(['a', 'b', 'c', 'd']),
                dtype='Int64',
            ),
        ),
        (
            [
                pd.DataFrame(
                    {
                        'A': [1, 1, 1],
                        'B': [1, 1, 1],
                        'C': [1, 1, 1],
                    },
                    index=pd.Index(['a', 'b', 'c']),
                ),
                pd.DataFrame(
                    {
                        'A': [2, 2, 2],
                        'B': [2, 2, 2],
                    },
                    index=pd.Index(['b', 'c', 'd']),
                ),
                pd.DataFrame(
                    {
                        'A': [3, 3],
                        'B': [3, 3],
                    },
                    index=pd.Index(['a', 'd']),
                ),
            ],
            np.sum,
            pd.DataFrame(
                {
                    'A': [4, 3, 3, 5],
                    'B': [4, 3, 3, 5],
                    'C': [1, 1, 1, np.NaN],
                },
                index=pd.Index(['a', 'b', 'c', 'd']),
                dtype='Int64',
            ),
        ),
        (
            [
                pd.DataFrame(
                    {
                        'A': [1, 1, 1],
                        'B': [1, 1, 1],
                        'C': [1, 1, 1],
                    },
                    index=pd.Index(['a', 'b', 'c']),
                    dtype='Int64',
                ),
                pd.DataFrame(
                    {
                        'A': [2, 2, 2],
                        'B': [2, 2, 2],
                    },
                    index=pd.Index(['b', 'c', 'd']),
                    dtype='Int64',
                ),
                pd.DataFrame(
                    {
                        'A': [3, 3, 4],
                        'B': [3, 3, np.NaN],
                    },
                    index=pd.Index(['a', 'd', 'e']),
                    dtype='Int64',
                ),
            ],
            lambda y: y[0],
            pd.DataFrame(
                {
                    'A': [1, 1, 1, 2, 4],
                    'B': [1, 1, 1, 2, np.NaN],
                    'C': [1, 1, 1, np.NaN, np.NaN],
                },
                index=pd.Index(['a', 'b', 'c', 'd', 'e']),
                dtype='Int64',
            ),
        ),
        (
            [
                pd.DataFrame(
                    {
                        'A': [1, 1, 1],
                        'B': [1, 1, 1],
                    },
                    index=pd.Index(['a', 'b', 'c']),
                ),
                pd.DataFrame(
                    {
                        'A': [2, 1, 2],
                        'B': [2, 1, 2],
                        'C': [2, 1, 2],
                    },
                    index=pd.Index(['b', 'c', 'd']),
                ),
                pd.DataFrame(
                    {
                        'A': [3, 3],
                        'B': [3, 3],
                    },
                    index=pd.Index(['a', 'd']),
                ),
            ],
            np.sum,
            pd.DataFrame(
                {
                    'A': [4, 3, 2, 5],
                    'B': [4, 3, 2, 5],
                    'C': [np.NaN, 2, 1, 2],
                },
                index=pd.Index(['a', 'b', 'c', 'd']),
                dtype='Int64',
            ),
        ),
    ]
)
def test_concat_aggregate_function_non_matching(
        objs,
        aggregate_function,
        expected,
):
    obj = audformat.utils.concat(
        objs,
        aggregate_function=aggregate_function,
        aggregate='non-matching',
    )
    if isinstance(obj, pd.Series):
        pd.testing.assert_series_equal(obj, expected)
    else:
        pd.testing.assert_frame_equal(obj, expected)


@pytest.mark.parametrize(
    'objs, aggregate_function, expected',
    [
        # empty
        (
            [],
            None,
            pd.Series([], pd.Index([]), dtype='object'),
        ),
        # identical values
        (
            [
                pd.Series([1, 2], pd.Index(['a', 'b']), dtype='float'),
                pd.Series([1, 2], pd.Index(['a', 'b']), dtype='float'),
            ],
            None,
            pd.Series([1, 2], pd.Index(['a', 'b']), dtype='float'),
        ),
        (
            [
                pd.Series([1, 2], pd.Index(['a', 'b']), dtype='float'),
                pd.Series([1, 2], pd.Index(['a', 'b']), dtype='float'),
            ],
            np.mean,
            pd.Series([1, 2], pd.Index(['a', 'b']), dtype='float'),
        ),
        # different values
        (
            [
                pd.Series([1, 2], pd.Index(['a', 'b']), dtype='float'),
                pd.Series([2, 3], pd.Index(['a', 'b']), dtype='float'),
            ],
            None,
            pd.Series([2, 3], pd.Index(['a', 'b']), dtype='float'),
        ),
        (
            [
                pd.Series([1, 2], pd.Index(['a', 'b']), dtype='float'),
                pd.Series([2, 3], pd.Index(['a', 'b']), dtype='float'),
            ],
            np.mean,
            pd.Series([2, 3], pd.Index(['a', 'b']), dtype='float'),
        ),
    ]
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
    'objs, aggregate_function, aggregate, expected_error, expected_error_msg',
    [
        (
            [
                pd.DataFrame(
                    {
                        'A': [1, 3],
                        'B': [2, 4],
                    },
                    pd.Index(['a', 'b']),
                    dtype='float',
                ),
                pd.DataFrame(
                    {
                        'A': [2, 4],
                        'B': [3, 5],
                    },
                    pd.Index(['a', 'b']),
                    dtype='float',
                ),
            ],
            None,
            'non-matching',
            ValueError,
            '',
        ),
        (
            [],
            None,
            'non-existent',
            ValueError,
            "aggregate needs to be one of 'always', 'non-matching'",
        ),
        (
            [
                pd.Series([1, 2], pd.Index(['a', 'b']), dtype='float'),
                pd.Series([2, 3], pd.Index(['a', 'b']), dtype='float'),
            ],
            None,
            'non-matching',
            ValueError,
            '',
        ),
        # error: dtypes do not match
        (
            [
                pd.Series([1], audformat.filewise_index('f1')),
                pd.Series([1.], audformat.filewise_index('f1')),
            ],
            None,
            'always',
            ValueError,
            '',
        ),
        (
            [
                pd.Series(
                    [1, 2, 3],
                    index=audformat.filewise_index(['f1', 'f2', 'f3']),
                ),
                pd.Series(
                    ['a', 'b', 'a'],
                    index=audformat.filewise_index(['f1', 'f2', 'f3']),
                    dtype='category',
                ),
            ],
            None,
            'always',
            ValueError,
            '',
        ),
        (
            [
                pd.Series(
                    ['a', 'b', 'a'],
                    index=audformat.filewise_index(['f1', 'f2', 'f3']),
                ),
                pd.Series(
                    ['a', 'b', 'a'],
                    index=audformat.filewise_index(['f1', 'f2', 'f3']),
                    dtype='category',
                ),
            ],
            None,
            'always',
            ValueError,
            '',
        ),
        (
            [
                pd.Series(
                    ['a', 'b', 'a'],
                    index=audformat.filewise_index(['f1', 'f2', 'f3']),
                    dtype='category',
                ),
                pd.Series(
                    ['a', 'b', 'c'],
                    index=audformat.filewise_index(['f1', 'f2', 'f3']),
                    dtype='category',
                ),
            ],
            None,
            'always',
            ValueError,
            '',
        ),
        (
            [
                pd.Series(
                    [1.],
                    pd.Index(['f1'], name='idx', dtype='string'),
                ),
                pd.Series(  # default dtype is object
                    [2.],
                    pd.MultiIndex.from_arrays([['f1']], names=['idx']),
                ),
            ],
            None,
            'always',
            ValueError,
            '',
        ),
        # error: values do not match
        pytest.param(
            [
                pd.Series([1.], audformat.filewise_index('f1')),
                pd.Series([2.], audformat.filewise_index('f1')),
            ],
            None,
            'always',
            ValueError,
            '',
        ),
        (
            [
                pd.Series([1.], pd.Index(['f1'], name='idx')),
                pd.Series(
                    [2.],
                    pd.MultiIndex.from_arrays([['f1']], names=['idx']),
                ),
            ],
            None,
            'always',
            ValueError,
            '',
        ),
        # error: index names do not match
        pytest.param(
            [
                pd.Series([], index=pd.Index([], name='idx1'), dtype='object'),
                pd.Series([], index=pd.Index([], name='idx2'), dtype='object'),
            ],
            None,
            'always',
            ValueError,
            '',
        ),
        (
            [
                pd.Series([1.], pd.Index(['f1'], name='idx1')),
                pd.Series(
                    [2.],
                    pd.MultiIndex.from_arrays([['f2']], names=['idx2']),
                ),
            ],
            None,
            'always',
            ValueError,
            '',
        ),
    ],
)
def test_concat_errors(
        objs,
        aggregate_function,
        aggregate,
        expected_error,
        expected_error_msg,
):
    with pytest.raises(expected_error, match=expected_error_msg):
        audformat.utils.concat(
            objs,
            aggregate_function=aggregate_function,
            aggregate=aggregate,
        )
