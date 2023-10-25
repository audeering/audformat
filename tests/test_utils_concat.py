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
        # error: dtypes do not match
        pytest.param(
            [
                pd.Series([1], audformat.filewise_index('f1')),
                pd.Series([1.], audformat.filewise_index('f1')),
            ],
            False,
            None,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(
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
            False,
            None,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(
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
            False,
            None,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(
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
            False,
            None,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(
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
            False,
            None,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        # error: values do not match
        pytest.param(
            [
                pd.Series([1.], audformat.filewise_index('f1')),
                pd.Series([2.], audformat.filewise_index('f1')),
            ],
            False,
            None,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(
            [
                pd.Series([1.], pd.Index(['f1'], name='idx')),
                pd.Series(
                    [2.],
                    pd.MultiIndex.from_arrays([['f1']], names=['idx']),
                ),
            ],
            False,
            None,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        # error: index names do not match
        pytest.param(
            [
                pd.Series([], index=pd.Index([], name='idx1'), dtype='object'),
                pd.Series([], index=pd.Index([], name='idx2'), dtype='object'),
            ],
            False,
            None,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(
            [
                pd.Series([1.], pd.Index(['f1'], name='idx1')),
                pd.Series(
                    [2.],
                    pd.MultiIndex.from_arrays([['f2']], names=['idx2']),
                ),
            ],
            False,
            None,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
    ],
)
def test_concat(objs, overwrite, expected):
    obj = audformat.utils.concat(objs, overwrite=overwrite)
    if isinstance(obj, pd.Series):
        pd.testing.assert_series_equal(obj, expected)
    else:
        pd.testing.assert_frame_equal(obj, expected)
