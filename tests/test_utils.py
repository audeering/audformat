from io import StringIO
import os
import shutil

import numpy as np
import pandas as pd
import pytest

import audeer
import audformat
from audformat import utils
from audformat import define


@pytest.mark.parametrize(
    'objs, overwrite, expected',
    [
        # empty
        (
            [],
            False,
            pd.Series([], audformat.filewise_index(), dtype='object'),
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
                pd.Series([1., 2.], audformat.filewise_index(['f1', 'f2'])),
                pd.Series([1., 2.], audformat.filewise_index(['f1', 'f2'])),
            ],
            False,
            pd.Series([1., 2.], audformat.filewise_index(['f1', 'f2'])),
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
                    ['a', np.nan, 'd'],
                    audformat.filewise_index(['f1', 'f2', 'f4']),
                    name='c2',
                ),
                pd.DataFrame(
                    {
                        'c1': [np.nan, 3.],
                        'c2': ['b', 'c'],
                    },
                    audformat.segmented_index(['f2', 'f3']),
                ),
            ],
            False,
            pd.DataFrame(
                {
                    'c1': [1., 2., 3., np.nan],
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
    ],
)
def test_concat(objs, overwrite, expected):
    obj = utils.concat(objs, overwrite=overwrite)
    if isinstance(obj, pd.Series):
        pd.testing.assert_series_equal(obj, expected)
    else:
        pd.testing.assert_frame_equal(obj, expected)


@pytest.mark.parametrize(
    'obj, expected_duration',
    [
        (
            audformat.segmented_index(),
            pd.Timedelta(0, unit='s'),
        ),
        (
            audformat.segmented_index(['f1'], [0], [2]),
            pd.Timedelta(2, unit='s'),
        ),
        (
            audformat.segmented_index(['f1'], [0.1], [2]),
            pd.Timedelta(1.9, unit='s'),
        ),
        (
            audformat.segmented_index(['f1', 'f2'], [0, 1], [2, 2]),
            pd.Timedelta(3, unit='s'),
        ),
        (
            pd.Series(
                index=audformat.segmented_index(['f1'], [1], [2]),
                dtype='category',
            ),
            pd.Timedelta(1, unit='s'),
        ),
        (
            pd.DataFrame(index=audformat.segmented_index(['f1'], [1], [2])),
            pd.Timedelta(1, unit='s'),
        ),
        # filewise index, but file is missing
        pytest.param(
            audformat.filewise_index(['f1']),
            None,
            marks=pytest.mark.xfail(raises=FileNotFoundError),
        ),
        # segmented index with NaT, but file is missing
        pytest.param(
            audformat.segmented_index(['f1'], [0]),
            None,
            marks=pytest.mark.xfail(raises=FileNotFoundError),
        ),
    ]
)
def test_duration(obj, expected_duration):
    duration = audformat.utils.duration(obj)
    if pd.isnull(expected_duration):
        assert pd.isnull(duration)
    else:
        assert duration == expected_duration


@pytest.mark.parametrize(
    'index, root, expected',
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
            audformat.filewise_index(['f1', 'f2']),
            '.',
            audformat.filewise_index(
                [
                    audeer.safe_path('f1'),
                    audeer.safe_path('f2'),
                ]
            ),
        ),
        (
            audformat.filewise_index(['f1', 'f2']),
            os.path.join('some', 'where'),
            audformat.filewise_index(
                [
                    audeer.safe_path(os.path.join('some', 'where', 'f1')),
                    audeer.safe_path(os.path.join('some', 'where', 'f2')),
                ]
            ),
        ),
        (
            audformat.filewise_index(['f1', 'f2']),
            os.path.join('some', 'where') + os.path.sep,
            audformat.filewise_index(
                [
                    audeer.safe_path(os.path.join('some', 'where', 'f1')),
                    audeer.safe_path(os.path.join('some', 'where', 'f2')),
                ]
            ),
        ),
        (
            audformat.filewise_index(['f1', 'f2']),
            audeer.safe_path(os.path.join('some', 'where')),
            audformat.filewise_index(
                [
                    audeer.safe_path(os.path.join('some', 'where', 'f1')),
                    audeer.safe_path(os.path.join('some', 'where', 'f2')),
                ]
            ),
        ),
        (
            audformat.filewise_index(
                [
                    audeer.safe_path('f1'),
                    audeer.safe_path('f2'),
                ]
            ),
            audeer.safe_path(os.path.join('some', 'where')),
            audformat.filewise_index(
                [
                    audeer.safe_path(os.path.join('some', 'where'))
                    + os.path.sep
                    + audeer.safe_path('f1'),
                    audeer.safe_path(os.path.join('some', 'where'))
                    + os.path.sep
                    + audeer.safe_path('f2'),
                ]
            ),
        ),
        (
            audformat.segmented_index(
                ['f1', 'f2'],
                ['1s', '3s'],
                ['2s', '4s'],
            ),
            '.',
            audformat.segmented_index(
                [
                    audeer.safe_path('f1'),
                    audeer.safe_path('f2'),
                ],
                ['1s', '3s'],
                ['2s', '4s'],
            ),
        )
    ]
)
def test_expand_file_path(tmpdir, index, root, expected):
    expanded_index = audformat.utils.expand_file_path(index, root)
    pd.testing.assert_index_equal(expanded_index, expected)


@pytest.mark.parametrize(
    'obj, expected',
    [
        (
            audformat.filewise_index(),
            '0',
        ),
        (
            audformat.segmented_index(),
            '0',
        ),
        (
            audformat.filewise_index(['f1', 'f2']),
            '-4231615416436839963',
        ),
        (
            audformat.segmented_index(['f1', 'f2']),
            '-2363261461673824215',
        ),
        (
            audformat.segmented_index(['f1', 'f2']),
            '-2363261461673824215',
        ),
        (
            audformat.segmented_index(['f1', 'f2'], [0, 0], [1, 1]),
            '-3831446135233514455',
        ),
        (
            pd.Series([0, 1], audformat.filewise_index(['f1', 'f2'])),
            '-8245754232361677810',
        ),
        (
            pd.DataFrame(
                {'a': [0, 1], 'b': [2, 3]},
                audformat.segmented_index(['f1', 'f2'], [0, 0], [1, 1]),
            ),
            '-103439349488189352',
        ),
    ]
)
def test_hash(obj, expected):
    assert utils.hash(obj) == expected
    assert utils.hash(obj[::-1]) == expected


@pytest.mark.parametrize(
    'objs, expected',
    [
        (
            [],
            audformat.filewise_index(),
        ),
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
                audformat.filewise_index(['f1', 'f2']),
                audformat.filewise_index(['f1', 'f2']),
            ],
            audformat.filewise_index(['f1', 'f2']),
        ),
        (
            [
                audformat.filewise_index(['f1', 'f2']),
                audformat.filewise_index(['f1', 'f2']),
                audformat.filewise_index(['f2', 'f3']),
            ],
            audformat.filewise_index('f2'),
        ),
        (
            [
                audformat.filewise_index(['f1', 'f2']),
                audformat.filewise_index(['f1', 'f2']),
                audformat.filewise_index('f3'),
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
                audformat.segmented_index(),
                audformat.segmented_index(),
            ],
            audformat.segmented_index(),
        ),
        (
            [
                audformat.segmented_index(['f1', 'f2']),
                audformat.segmented_index(['f1', 'f2']),
            ],
            audformat.segmented_index(['f1', 'f2']),
        ),
        (
            [
                audformat.segmented_index(['f1', 'f2']),
                audformat.segmented_index(['f3', 'f4']),
            ],
            audformat.segmented_index(),
        ),
        (
            [
                audformat.segmented_index(['f1', 'f2'], [0, 0], [1, 1]),
                audformat.segmented_index(['f2', 'f1'], [0, 0], [1, 1]),
                audformat.segmented_index(['f2', 'f3'], [0, 0], [1, 1]),
            ],
            audformat.segmented_index('f2', 0, 1),
        ),
        (
            [
                audformat.segmented_index(['f1', 'f2'], [0, 0], [1, 1]),
                audformat.segmented_index(['f2', 'f1'], [0, 0], [1, 1]),
                audformat.segmented_index(['f2', 'f3'], [1, 1], [2, 2]),
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
                audformat.filewise_index(['f1', 'f2']),
                audformat.segmented_index(),
            ],
            audformat.segmented_index(),
        ),
        (
            [
                audformat.filewise_index(),
                audformat.segmented_index(['f1', 'f2']),
            ],
            audformat.segmented_index(),
        ),
        (
            [
                audformat.segmented_index(['f1', 'f2'], [0, 0], [1, 1]),
                audformat.segmented_index(['f2', 'f3'], [0, 0], [1, 1]),
                audformat.filewise_index(['f1', 'f2']),
            ],
            audformat.segmented_index('f2', 0, 1),
        ),
        (
            [
                audformat.segmented_index(['f1', 'f2'], [0, 0], [1, 1]),
                audformat.segmented_index(['f2', 'f3'], [0, 0], [1, 1]),
                audformat.filewise_index('f1'),
            ],
            audformat.segmented_index(),
        ),
        (
            [
                audformat.segmented_index(['f1', 'f2'], [0, 0], [1, 1]),
                audformat.filewise_index(['f1', 'f2']),
                audformat.filewise_index(['f2', 'f3']),
            ],
            audformat.segmented_index('f2', 0, 1),
        ),
    ]
)
def test_intersect(objs, expected):
    pd.testing.assert_index_equal(
        audformat.utils.intersect(objs),
        expected,
    )


@pytest.mark.parametrize(
    'labels, expected',
    [
        (
            [],
            [],
        ),
        (
            (['a'], ['b']),
            ['a', 'b'],
        ),
        (
            (['a'], ['b', 'c']),
            ['a', 'b', 'c'],
        ),
        (
            (['a'], ['a']),
            ['a'],
        ),
        (
            [{'a': 0}],
            {'a': 0},
        ),
        (
            [{'a': 0}, {'b': 1}],
            {'a': 0, 'b': 1},
        ),
        (
            [{'a': 0}, {'b': 1, 'c': 2}],
            {'a': 0, 'b': 1, 'c': 2},
        ),
        (
            [{'a': 0, 'b': 1}, {'b': 1, 'c': 2}],
            {'a': 0, 'b': 1, 'c': 2},
        ),
        (
            [{'a': 0, 'b': 1}, {'b': 2, 'c': 2}],
            {'a': 0, 'b': 2, 'c': 2},
        ),
        (
            [{'a': 0}, {'a': 1}, {'a': 2}],
            {'a': 2},
        ),
        pytest.param(
            ['a', 'b', 'c'],
            [],
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(
            ('a', 'b', 'c'),
            [],
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(
            [{'a': 0, 'b': 1}, ['c']],
            [],
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(
            [['a', 'b'], ['b', 'c'], 'd'],
            [],
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(
            [{0: {'age': 20}}, {'0': {'age': 30}}],
            [],
            marks=pytest.mark.xfail(raises=ValueError),
        ),
    ]
)
def test_join_labels(labels, expected):
    assert utils.join_labels(labels) == expected


def test_join_schemes():
    # Empty list
    audformat.utils.join_schemes([], 'scheme_id')
    # One database
    db1 = audformat.Database('db1')
    scheme1 = audformat.Scheme(labels={'a': [1, 2]})
    db1.schemes['scheme_id'] = scheme1
    audformat.utils.join_schemes([db1], 'scheme_id')
    assert db1.schemes['scheme_id'] == scheme1
    # Two databases
    db2 = audformat.Database('db2')
    scheme2 = audformat.Scheme(labels={'b': [3]})
    db2.schemes['scheme_id'] = scheme2
    expected = audformat.Scheme(labels={'a': [1, 2], 'b': [3]})
    audformat.utils.join_schemes([db1, db2], 'scheme_id')
    assert db1.schemes['scheme_id'] == expected
    assert db2.schemes['scheme_id'] == expected
    # Three database
    db3 = audformat.Database('db3')
    scheme3 = audformat.Scheme(labels={'a': [4]})
    db3.schemes['scheme_id'] = scheme3
    expected = audformat.Scheme(labels={'a': [4], 'b': [3]})
    audformat.utils.join_schemes([db1, db2, db3], 'scheme_id')
    # Fail for schemes without labels
    with pytest.raises(ValueError):
        db = audformat.Database('db')
        db.schemes['scheme_id'] = audformat.Scheme('str')
        audformat.utils.join_schemes([db], 'scheme_id')


@pytest.mark.parametrize(
    'language, expected',
    [
        ('en', 'eng'),
        ('en', 'eng'),
        ('english', 'eng'),
        ('English', 'eng'),
        pytest.param(
            'xx', None,
            marks=pytest.mark.xfail(raises=ValueError)
        ),
        pytest.param(
            'xxx', None,
            marks=pytest.mark.xfail(raises=ValueError)
        ),
        pytest.param(
            'Bad language', None,
            marks=pytest.mark.xfail(raises=ValueError)
        )
    ]
)
def test_map_language(language, expected):
    assert utils.map_language(language) == expected


@pytest.mark.parametrize('csv,result', [
    (
        StringIO('''file
f1
f2
f3'''),
        pd.Index(
            ['f1', 'f2', 'f3'],
            name='file',
        ),
    ),
    (
        StringIO('''file,value
f1,0.0
f2,1.0
f3,2.0'''),
        pd.Series(
            [0.0, 1.0, 2.0],
            index=audformat.filewise_index(['f1', 'f2', 'f3']),
            name='value',
        ),
    ),
    (
        StringIO('''file,value1,value2
f1,0.0,a
f2,1.0,b
f3,2.0,c'''),
        pd.DataFrame(
            {
                'value1': [0.0, 1.0, 2.0],
                'value2': ['a', 'b', 'c'],
            },
            index=audformat.filewise_index(['f1', 'f2', 'f3']),
            columns=['value1', 'value2'],
        ),
    ),
    (
        StringIO('''file,start,value
f1,00:00:00,0.0
f1,00:00:01,1.0
f2,00:00:02,2.0'''),
        pd.Series(
            [0.0, 1.0, 2.0],
            index=audformat.segmented_index(
                ['f1', 'f1', 'f2'],
                starts=['0s', '1s', '2s'],
                ends=pd.to_timedelta([pd.NaT, pd.NaT, pd.NaT]),
            ),
            name='value',
        ),
    ),
    (
        StringIO('''file,end,value
f1,00:00:01,0.0
f1,00:00:02,1.0
f2,00:00:03,2.0'''),
        pd.Series(
            [0.0, 1.0, 2.0],
            index=audformat.segmented_index(
                ['f1', 'f1', 'f2'],
                starts=['0s', '0s', '0s'],
                ends=['1s', '2s', '3s'],
            ),
            name='value',
        ),
    ),
    (
        StringIO('''file,start,end
f1,00:00:00,00:00:01
f1,00:00:01,00:00:02
f2,00:00:02,00:00:03'''),
        pd.MultiIndex.from_arrays(
            [
                ['f1', 'f1', 'f2'],
                pd.to_timedelta(['0s', '1s', '2s']),
                pd.to_timedelta(['1s', '2s', '3s']),
            ],
            names=['file', 'start', 'end'],
        ),
    ),
    (
        StringIO('''file,start,end,value
f1,00:00:00,00:00:01,0.0
f1,00:00:01,00:00:02,1.0
f2,00:00:02,00:00:03,2.0'''),
        pd.Series(
            [0.0, 1.0, 2.0],
            index=audformat.segmented_index(
                ['f1', 'f1', 'f2'],
                starts=['0s', '1s', '2s'],
                ends=['1s', '2s', '3s'],
            ),
            name='value',
        ),
    ),
    (
        StringIO('''file,start,end,value1,value2
f1,00:00:00,00:00:01,0.0,a
f1,00:00:01,00:00:02,1.0,b
f2,00:00:02,00:00:03,2.0,c'''),
        pd.DataFrame(
            {
                'value1': [0.0, 1.0, 2.0],
                'value2': ['a', 'b', 'c'],
            },
            index=audformat.segmented_index(
                ['f1', 'f1', 'f2'],
                starts=['0s', '1s', '2s'],
                ends=['1s', '2s', '3s'],
            ),
            columns=['value1', 'value2'],
        ),

    ),
    pytest.param(
        StringIO('''value
0.0
1.0
2.0'''),
        None,
        marks=pytest.mark.xfail(raises=ValueError)
    )
])
def test_read_csv(csv, result):
    obj = audformat.utils.read_csv(csv)
    if isinstance(result, pd.Index):
        pd.testing.assert_index_equal(obj, result)
    elif isinstance(result, pd.Series):
        pd.testing.assert_series_equal(obj, result)
    else:
        pd.testing.assert_frame_equal(obj, result)


@pytest.mark.parametrize(
    'index, extension, pattern, expected_index',
    [
        (
            audformat.filewise_index(),
            'mp3',
            None,
            audformat.filewise_index(),
        ),
        (
            audformat.segmented_index(),
            'mp3',
            None,
            audformat.segmented_index(),
        ),
        (
            audformat.filewise_index(['f1.wav', 'f2.wav']),
            'mp3',
            None,
            audformat.filewise_index(['f1.mp3', 'f2.mp3']),
        ),
        (
            audformat.segmented_index(['f1.wav', 'f2.wav']),
            'mp3',
            None,
            audformat.segmented_index(['f1.mp3', 'f2.mp3']),
        ),
        (
            audformat.filewise_index(['f1.WAV', 'f2.WAV']),
            'MP3',
            None,
            audformat.filewise_index(['f1.MP3', 'f2.MP3']),
        ),
        (
            audformat.filewise_index(['f1', 'f2.wv']),
            'mp3',
            None,
            audformat.filewise_index(['f1', 'f2.mp3']),
        ),
        (
            audformat.filewise_index(['f1.wav', 'f2.wav']),
            '',
            None,
            audformat.filewise_index(['f1', 'f2']),
        ),
        (
            audformat.filewise_index(['f1.ogg', 'f2.wav']),
            'mp3',
            '.ogg',
            audformat.filewise_index(['f1.mp3', 'f2.wav']),
        ),
    ]
)
def test_replace_file_extension(index, extension, pattern, expected_index):
    index = audformat.utils.replace_file_extension(
        index,
        extension,
        pattern=pattern,
    )
    pd.testing.assert_index_equal(index, expected_index)


@pytest.mark.parametrize(
    'obj, allow_nat, files_duration, root, expected',
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
                pytest.DB.files[:2],
                [0, 0],
                [pytest.FILE_DUR, pytest.FILE_DUR]
            ),
        ),
        (
            audformat.segmented_index(pytest.DB.files[:2]),
            False,
            None,
            pytest.DB_ROOT,
            audformat.segmented_index(
                pytest.DB.files[:2],
                [0, 0],
                [pytest.FILE_DUR, pytest.FILE_DUR]
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
                os.path.join(pytest.DB_ROOT, pytest.DB.files[1]):
                    pytest.FILE_DUR * 2,
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
                os.path.join(pytest.DB_ROOT, pytest.DB.files[1]):
                    pytest.FILE_DUR * 2,
            },
            pytest.DB_ROOT,
            audformat.segmented_index(
                pytest.DB.files[:2],
                [0.1, 0.5],
                [pytest.FILE_DUR, pytest.FILE_DUR * 2],
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
                {'int': [1, 2], 'str': ['a', 'b']},
                index=audformat.filewise_index(pytest.DB.files[:2]),
            ),
            True,
            None,
            None,
            audformat.segmented_index(pytest.DB.files[:2]),
        ),
    ]
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
            mask = result.get_level_values(
                audformat.define.IndexField.END
            ) == pd.NaT
            files = files[mask]
        for file in files:
            file = os.path.join(root, file)
            assert file in files_duration


@pytest.mark.parametrize(
    'output_folder,table_id,expected_file_names',
    [
        pytest.param(
            '.',
            'segments',
            None,
            marks=pytest.mark.xfail(raises=ValueError)
        ),
        pytest.param(
            os.path.abspath(''),
            'segments',
            None,
            marks=pytest.mark.xfail(raises=ValueError)
        ),
        (
            'tmp',
            'segments',
            [
                str(i).zfill(3) + f'_{j}'
                for i in range(1, 11)
                for j in range(10)
            ]
        ),
        (
            'tmp',
            'files',
            [str(i).zfill(3) for i in range(1, 101)]
        )
    ]
)
def test_to_filewise(output_folder, table_id, expected_file_names):

    has_existed = os.path.exists(output_folder)

    frame = utils.to_filewise_index(
        obj=pytest.DB[table_id].get(),
        root=pytest.DB_ROOT,
        output_folder=output_folder,
        num_workers=3,
    )

    assert audformat.index_type(frame) == define.IndexType.FILEWISE
    pd.testing.assert_frame_equal(
        pytest.DB[table_id].get().reset_index(drop=True),
        frame.reset_index(drop=True),
    )
    files = frame.index.get_level_values(define.IndexField.FILE).values

    if table_id == 'segmented':  # already `framewise` frame is unprocessed
        assert os.path.isabs(output_folder) == os.path.isabs(files[0])

    if table_id == 'files':
        # files of unprocessed frame are relative to `root`
        files = [os.path.join(pytest.DB_ROOT, f) for f in files]
    assert all(os.path.exists(f) for f in files)

    file_names = [f.split(os.path.sep)[-1].rsplit('.', 1)[0] for f in files]
    assert file_names == expected_file_names

    # clean-up
    if not has_existed:  # output folder was created and can be removed
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
    else:
        if table_id == 'segments':
            for f in frame.index.get_level_values(
                    define.IndexField.FILE):
                if os.path.exists(f):
                    os.remove(f)


@pytest.mark.parametrize(
    'objs, expected',
    [
        (
            [],
            audformat.filewise_index(),
        ),
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
                audformat.filewise_index(['f1', 'f2']),
                audformat.filewise_index(['f1', 'f2']),
            ],
            audformat.filewise_index(['f1', 'f2']),
        ),
        (
            [
                audformat.filewise_index(['f1', 'f2']),
                audformat.filewise_index(['f1', 'f2']),
                audformat.filewise_index(['f2', 'f3']),
            ],
            audformat.filewise_index(['f1', 'f2', 'f3']),
        ),
        (
            [
                audformat.filewise_index(['f1', 'f2']),
                audformat.filewise_index(['f1', 'f2']),
                audformat.filewise_index('f3'),
            ],
            audformat.filewise_index(['f1', 'f2', 'f3']),
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
                audformat.segmented_index(['f1', 'f2']),
                audformat.segmented_index(['f1', 'f2']),
            ],
            audformat.segmented_index(['f1', 'f2']),
        ),
        (
            [
                audformat.segmented_index(['f1', 'f2']),
                audformat.segmented_index(['f3', 'f4']),
            ],
            audformat.segmented_index(['f1', 'f2', 'f3', 'f4']),
        ),
        (
            [
                audformat.segmented_index(['f1', 'f2'], [0, 0], [1, 1]),
                audformat.segmented_index(['f2', 'f1'], [0, 0], [1, 1]),
                audformat.segmented_index(['f2', 'f3'], [0, 0], [1, 1]),
            ],
            audformat.segmented_index(
                ['f1', 'f2', 'f3'],
                [0, 0, 0],
                [1, 1, 1],
            ),
        ),
        (
            [
                audformat.segmented_index(['f1', 'f2'], [0, 0], [1, 1]),
                audformat.segmented_index(['f2', 'f1'], [0, 0], [1, 1]),
                audformat.segmented_index(['f2', 'f3'], [1, 1], [2, 2]),
            ],
            audformat.segmented_index(
                ['f1', 'f2', 'f2', 'f3'],
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
                audformat.filewise_index(['f1', 'f2']),
                audformat.segmented_index(),
            ],
            audformat.segmented_index(['f1', 'f2']),
        ),
        (
            [
                audformat.filewise_index(),
                audformat.segmented_index(['f1', 'f2']),
            ],
            audformat.segmented_index(['f1', 'f2']),
        ),
        (
            [
                audformat.segmented_index(['f1', 'f2'], [0, 0], [1, 1]),
                audformat.segmented_index(['f2', 'f3'], [0, 0], [1, 1]),
                audformat.filewise_index(['f1', 'f2']),
            ],
            audformat.segmented_index(
                ['f1', 'f1', 'f2', 'f2', 'f3'],
                [0, 0, 0, 0, 0],
                [pd.NaT, 1, pd.NaT, 1, 1],
            ),
        ),
        (
            [
                audformat.segmented_index(['f1', 'f2'], [0, 0], [1, 1]),
                audformat.segmented_index(['f2', 'f3'], [0, 0], [1, 1]),
                audformat.filewise_index('f1'),
            ],
            audformat.segmented_index(
                ['f1', 'f1', 'f2', 'f3'],
                [0, 0, 0, 0],
                [pd.NaT, 1, 1, 1],
            ),
        ),
        (
            [
                audformat.segmented_index(['f1', 'f2'], [0, 0], [1, 1]),
                audformat.filewise_index(['f1', 'f2']),
                audformat.filewise_index(['f2', 'f3']),
            ],
            audformat.segmented_index(
                ['f1', 'f1', 'f2', 'f2', 'f3'],
                [0, 0, 0, 0, 0],
                [pd.NaT, 1, pd.NaT, 1, pd.NaT],
            ),
        ),
    ]
)
def test_union(objs, expected):
    pd.testing.assert_index_equal(
        audformat.utils.union(objs),
        expected,
    )
