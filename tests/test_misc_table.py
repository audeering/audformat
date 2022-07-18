import numpy as np
import pandas as pd
import pytest

import audformat
import audformat.testing


@pytest.mark.parametrize(
    'table',
    [
        audformat.MiscTable(pd.Index([], name='idx')),
        audformat.testing.create_db(
            data={
                'misc': pd.Series(
                    [0., 1., 2.],
                    pd.MultiIndex.from_tuples(
                        [
                            ('a', 0),
                            ('b', 1),
                            ('c', 2),
                        ],
                        names=['idx1', 'idx2'],
                    ),
                ),
            },
        )['misc'],
        pytest.DB['misc'],
    ]
)
def test_copy(table):
    table_copy = table.copy()
    assert str(table_copy) == str(table)
    pd.testing.assert_frame_equal(table_copy.df, table.df)


@pytest.mark.parametrize(
    'index_object, index_values, index_dtype, '
    'expected_pandas_dtype, expected_audformat_dtype',
    [
        (
            pd.Index,
            [],
            None,
            'object',
            audformat.define.DataType.STRING,
        ),
        (
            pd.DatetimeIndex,
            [],
            'datetime64[ns]',
            'datetime64[ns]',
            audformat.define.DataType.DATE,
        ),
        (
            pd.Index,
            [],
            float,
            'float64',
            audformat.define.DataType.FLOAT,
        ),
        (
            pd.Index,
            [],
            int,
            'int64',
            audformat.define.DataType.INTEGER,
        ),
        (
            pd.Index,
            [],
            str,
            'object',
            audformat.define.DataType.STRING,
        ),
        (
            pd.TimedeltaIndex,
            [],
            'timedelta64[ns]',
            'timedelta64[ns]',
            audformat.define.DataType.TIME,
        ),
        (
            pd.DatetimeIndex,
            [0],
            'datetime64[ns]',
            'datetime64[ns]',
            audformat.define.DataType.DATE,
        ),
        (
            pd.Index,
            [0.0],
            None,
            'float64',
            audformat.define.DataType.FLOAT,
        ),
        (
            pd.Index,
            [0],
            None,
            'Int64',
            audformat.define.DataType.INTEGER,
        ),
        # The following test does not work under Python 3.7
        # as the index has dtype object
        # instead of Int64
        # (
        #     pd.Index,
        #     [np.NaN],
        #     'Int64',
        #     [audformat.define.DataType.INTEGER],
        # ),
        (
            pd.Index,
            ['0'],
            None,
            'object',
            audformat.define.DataType.STRING,
        ),
        (
            pd.TimedeltaIndex,
            [0],
            'timedelta64[ns]',
            'timedelta64[ns]',
            audformat.define.DataType.TIME,
        ),
    ]
)
def test_dtype(
        tmpdir,
        index_object,
        index_values,
        index_dtype,
        expected_pandas_dtype,
        expected_audformat_dtype,
):

    name = 'idx'
    index = index_object(index_values, dtype=index_dtype, name=name)
    table = audformat.MiscTable(index)

    assert table.levels[name] == expected_audformat_dtype
    assert table.index.dtype == expected_pandas_dtype

    # Store and load table
    db = audformat.testing.create_db(minimal=True)
    db['misc'] = table
    assert db['misc'].levels[name] == expected_audformat_dtype
    assert db['misc'].index.dtype == expected_pandas_dtype

    db_root = tmpdir.join('db')
    db.save(db_root, storage_format='csv')
    db_new = audformat.Database.load(db_root)
    assert db_new['misc'].levels[name] == expected_audformat_dtype
    assert db_new['misc'].index.dtype == expected_pandas_dtype


@pytest.mark.parametrize(
    'table, column, expected',
    [
        (
            pytest.DB['misc'],
            'int',
            pytest.DB['misc'].df['int'],
        ),
    ]
)
def test_get_column(table, column, expected):
    pd.testing.assert_series_equal(table[column].get(), expected)


@pytest.mark.parametrize(
    'index',
    [
        pd.Index([], name='idx'),
        pd.MultiIndex.from_tuples(
            [
                ('a', 0),
                ('b', 1),
                ('c', 2),
            ],
            names=['idx1', 'idx2'],
        ),
        # invalid level names
        pytest.param(
            pd.Index([]),
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(
            pd.Index([], name=''),
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(
            pd.MultiIndex.from_tuples(
                [
                    ('a', 0),
                    ('b', 1),
                    ('c', 2),
                ],
                names=['idx', 'idx'],
            ),
            marks=pytest.mark.xfail(raises=ValueError),
        ),
    ]
)
def test_level_names(index):
    audformat.MiscTable(index)
