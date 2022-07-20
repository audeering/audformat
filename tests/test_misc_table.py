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
    'column_values, column_dtype, '
    'expected_pandas_dtype, expected_audformat_dtype',
    [
        (
            [],
            None,
            'object',
            audformat.define.DataType.STRING,
        ),
        (
            [],
            'datetime64[ns]',
            'datetime64[ns]',
            audformat.define.DataType.DATE,
        ),
        (
            [],
            float,
            'float64',
            audformat.define.DataType.FLOAT,
        ),
        (
            [],
            int,
            'Int64',
            audformat.define.DataType.INTEGER,
        ),
        (
            [],
            'int64',
            'Int64',
            audformat.define.DataType.INTEGER,
        ),
        (
            [],
            'Int64',
            'Int64',
            audformat.define.DataType.INTEGER,
        ),
        (
            [],
            str,
            'object',
            audformat.define.DataType.STRING,
        ),
        (
            [],
            'timedelta64[ns]',
            'timedelta64[ns]',
            audformat.define.DataType.TIME,
        ),
        (
            [0],
            'datetime64[ns]',
            'datetime64[ns]',
            audformat.define.DataType.DATE,
        ),
        (
            [0.0],
            None,
            'float64',
            audformat.define.DataType.FLOAT,
        ),
        (
            [0],
            None,
            'Int64',
            audformat.define.DataType.INTEGER,
        ),
        (
            [np.NaN],
            'Int64',
            'Int64',
            audformat.define.DataType.INTEGER,
        ),
        (
            [0, np.NaN],
            'Int64',
            'Int64',
            audformat.define.DataType.INTEGER,
        ),
        (
            [np.NaN],
            'Int64',
            'Int64',
            audformat.define.DataType.INTEGER,
        ),
        (
            ['0'],
            None,
            'object',
            audformat.define.DataType.STRING,
        ),
        (
            [0],
            'timedelta64[ns]',
            'timedelta64[ns]',
            audformat.define.DataType.TIME,
        ),
    ]
)
def test_dtype_column(
        tmpdir,
        column_values,
        column_dtype,
        expected_pandas_dtype,
        expected_audformat_dtype,
):

    name = 'column'
    y = pd.Series(column_values, dtype=column_dtype, name=name)

    name = 'idx'
    index_values = [f'f{n}' for n in range(len(column_values))]
    index = pd.Index(index_values, dtype='str', name=name)

    db = audformat.testing.create_db(minimal=True)
    db['misc'] = audformat.MiscTable(index)
    db.schemes['column'] = audformat.Scheme(expected_audformat_dtype)
    db['misc']['column'] = audformat.Column(scheme_id='column')
    db['misc']['column'].set(y.values)

    assert db['misc']['column'].scheme.dtype == expected_audformat_dtype
    assert db['misc'].df['column'].dtype == expected_pandas_dtype

    # Store and load table
    db_root = tmpdir.join('db')
    db.save(db_root, storage_format='csv')
    db_new = audformat.Database.load(db_root)

    assert db_new['misc']['column'].scheme.dtype == expected_audformat_dtype
    assert db_new['misc'].df['column'].dtype == expected_pandas_dtype


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
            'Int64',
            audformat.define.DataType.INTEGER,
        ),
        (
            pd.Index,
            [],
            'int64',
            'Int64',
            audformat.define.DataType.INTEGER,
        ),
        (
            pd.Index,
            [],
            'Int64',
            'Int64',
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
        (
            pd.Index,
            [np.NaN],
            'Int64',
            'Int64',
            audformat.define.DataType.INTEGER,
        ),
        (
            pd.Index,
            [0, np.NaN],
            'Int64',
            'Int64',
            audformat.define.DataType.INTEGER,
        ),
        (
            pd.Index,
            [np.NaN],
            'Int64',
            'Int64',
            audformat.define.DataType.INTEGER,
        ),
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
        (  # list as index -> converted to str
            pd.Index,
            [[0]],
            'object',
            'object',
            audformat.define.DataType.STRING,
        ),
    ]
)
def test_dtype_index(
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
    'index_values, index_dtype, '
    'expected_pandas_dtype, expected_audformat_dtype',
    [
        (
            [],
            'datetime64[ns]',
            'datetime64[ns]',
            audformat.define.DataType.DATE,
        ),
        (
            [],
            float,
            'float64',
            audformat.define.DataType.FLOAT,
        ),
        (
            [],
            int,
            'Int64',
            audformat.define.DataType.INTEGER,
        ),
        (
            [],
            'int64',
            'Int64',
            audformat.define.DataType.INTEGER,
        ),
        (
            [],
            'Int64',
            'Int64',
            audformat.define.DataType.INTEGER,
        ),
        (
            [],
            str,
            'object',
            audformat.define.DataType.STRING,
        ),
        (
            [],
            'timedelta64[ns]',
            'timedelta64[ns]',
            audformat.define.DataType.TIME,
        ),
        (
            [0],
            'datetime64[ns]',
            'datetime64[ns]',
            audformat.define.DataType.DATE,
        ),
        (
            [0.0],
            None,
            'float64',
            audformat.define.DataType.FLOAT,
        ),
        (
            [0],
            None,
            'Int64',
            audformat.define.DataType.INTEGER,
        ),
        (
            [np.NaN],
            'Int64',
            'Int64',
            audformat.define.DataType.INTEGER,
        ),
        (
            [0, np.NaN],
            'Int64',
            'Int64',
            audformat.define.DataType.INTEGER,
        ),
        (
            [np.NaN],
            'Int64',
            'Int64',
            audformat.define.DataType.INTEGER,
        ),
        (
            ['0'],
            None,
            'object',
            audformat.define.DataType.STRING,
        ),
        (
            [0],
            'timedelta64[ns]',
            'timedelta64[ns]',
            audformat.define.DataType.TIME,
        ),
    ]
)
def test_dtype_multiindex(
        tmpdir,
        index_values,
        index_dtype,
        expected_pandas_dtype,
        expected_audformat_dtype,
):
    expected_audformat_dtypes = [expected_audformat_dtype] * 2
    expected_pandas_dtypes = [expected_pandas_dtype] * 2
    index = pd.MultiIndex.from_arrays(
        [
            pd.Series(index_values, dtype=index_dtype),
            pd.Series(index_values, dtype=index_dtype),
        ],
        names=['idx1', 'idx2'],

    )
    table = audformat.MiscTable(index)
    assert list(table.levels.values()) == expected_audformat_dtypes
    assert list(table.index.dtypes) == expected_pandas_dtypes

    # Store and load table
    db = audformat.testing.create_db(minimal=True)
    db['misc'] = table
    assert list(db['misc'].levels.values()) == expected_audformat_dtypes
    assert list(db['misc'].index.dtypes) == expected_pandas_dtypes

    db_root = tmpdir.join('db')
    db.save(db_root, storage_format='csv')
    db_new = audformat.Database.load(db_root)
    assert list(db_new['misc'].levels.values()) == expected_audformat_dtypes
    assert list(db_new['misc'].index.dtypes) == expected_pandas_dtypes


@pytest.mark.parametrize(
    'index_values, index_dtype, '
    'expected_pandas_dtype, expected_audformat_dtype',
    [
        (
            [],
            'datetime64[ns]',
            'datetime64[ns]',
            audformat.define.DataType.DATE,
        ),
        (
            [],
            float,
            'float64',
            audformat.define.DataType.FLOAT,
        ),
        (
            [],
            int,
            'Int64',
            audformat.define.DataType.INTEGER,
        ),
        (
            [],
            'int64',
            'Int64',
            audformat.define.DataType.INTEGER,
        ),
        (
            [],
            'Int64',
            'Int64',
            audformat.define.DataType.INTEGER,
        ),
        (
            [],
            str,
            'object',
            audformat.define.DataType.STRING,
        ),
        (
            [],
            'timedelta64[ns]',
            'timedelta64[ns]',
            audformat.define.DataType.TIME,
        ),
        (
            [0],
            'datetime64[ns]',
            'datetime64[ns]',
            audformat.define.DataType.DATE,
        ),
        (
            [0.0],
            None,
            'float64',
            audformat.define.DataType.FLOAT,
        ),
        (
            [0],
            None,
            'Int64',
            audformat.define.DataType.INTEGER,
        ),
        (
            [np.NaN],
            'Int64',
            'Int64',
            audformat.define.DataType.INTEGER,
        ),
        (
            [0, np.NaN],
            'Int64',
            'Int64',
            audformat.define.DataType.INTEGER,
        ),
        (
            [np.NaN],
            'Int64',
            'Int64',
            audformat.define.DataType.INTEGER,
        ),
        (
            ['0'],
            None,
            'object',
            audformat.define.DataType.STRING,
        ),
        (
            [0],
            'timedelta64[ns]',
            'timedelta64[ns]',
            audformat.define.DataType.TIME,
        ),
    ]
)
def test_dtype_multiindex_single_level(
        tmpdir,
        index_values,
        index_dtype,
        expected_pandas_dtype,
        expected_audformat_dtype,
):
    name = 'idx'
    index = pd.MultiIndex.from_arrays(
        [
            pd.Series(index_values, dtype=index_dtype),
        ],
        names=[name],

    )
    table = audformat.MiscTable(index)
    assert table.levels[name] == expected_audformat_dtype
    assert table.index.dtypes[name] == expected_pandas_dtype

    # Store and load table
    db = audformat.testing.create_db(minimal=True)
    db['misc'] = table
    assert db['misc'].levels[name] == expected_audformat_dtype
    assert db['misc'].index.dtypes[name] == expected_pandas_dtype

    db_root = tmpdir.join('db')
    db.save(db_root, storage_format='csv')
    db_new = audformat.Database.load(db_root)
    # After loading we now longer have a MultiIndex
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
