import numpy as np
import pandas as pd
import pytest

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
            audformat.define.DataType.OBJECT,
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
            audformat.define.DataType.OBJECT,
        ),
        (
            [],
            'string',
            'string',
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
            audformat.define.DataType.OBJECT,
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
    y = pd.Series(column_values, dtype=column_dtype or 'object', name=name)

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
            audformat.define.DataType.OBJECT,
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
            audformat.define.DataType.OBJECT,
        ),
        (
            pd.Index,
            [],
            'string',
            'string',
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
            audformat.define.DataType.OBJECT,
        ),
        (
            pd.TimedeltaIndex,
            [0],
            'timedelta64[ns]',
            'timedelta64[ns]',
            audformat.define.DataType.TIME,
        ),
        (
            pd.Index,
            [[0]],
            'object',
            'object',
            audformat.define.DataType.OBJECT,
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
            audformat.define.DataType.OBJECT,
        ),
        (
            [],
            'string',
            'string',
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
            audformat.define.DataType.OBJECT,
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
            audformat.define.DataType.OBJECT,
        ),
        (
            [],
            'string',
            'string',
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
            audformat.define.DataType.OBJECT,
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


def test_drop_and_pick_index():

    table_id = 'misc'
    index = pytest.DB[table_id].index[:5]
    df_pick = pytest.DB[table_id].pick_index(index).get()
    index = pytest.DB[table_id].index[5:]
    df_drop = pytest.DB[table_id].drop_index(index).get()
    pd.testing.assert_frame_equal(df_pick, df_drop)

    index = pytest.DB['segments'].index[:5]
    with pytest.raises(
            ValueError,
            match='Levels and dtypes of all objects must match',
    ):
        pytest.DB[table_id].drop_index(index).get()
    with pytest.raises(
            ValueError,
            match='Levels and dtypes of all objects must match',
    ):
        pytest.DB[table_id].pick_index(index).get()


def test_extend_index():

    db = audformat.testing.create_db(minimal=True)
    db.schemes['scheme'] = audformat.Scheme()

    # empty and invalid

    db['misc'] = audformat.MiscTable(pd.Index([], name='idx'))
    db['misc'].extend_index(pd.Index([], name='idx'))
    assert db['misc'].get().empty
    with pytest.raises(
            ValueError,
            match='Levels and dtypes of all objects must match',
    ):
        db['misc'].extend_index(pd.Index([], name='other'))

    db.drop_tables('misc')

    # not empty

    db['misc'] = audformat.MiscTable(pd.Index([], name='idx'))
    db['misc']['columns'] = audformat.Column(scheme_id='scheme')
    db['misc'].extend_index(
        pd.Index(['1', '2'], name='idx'),
        fill_values='a',
        inplace=True,
    )
    np.testing.assert_equal(
        db['misc']['columns'].get().values,
        np.array(['a', 'a']),
    )
    index = pd.Index(['1', '3'], name='idx')
    db['misc'].extend_index(
        index,
        fill_values='b',
        inplace=True,
    )
    np.testing.assert_equal(
        db['misc']['columns'].get().values,
        np.array(['a', 'a', 'b']),
    )

    db.drop_tables('misc')


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


@pytest.mark.parametrize(
    'index, columns',
    [
        (
            pd.Index([], name='idx'),
            ['column'],
        ),
        (
            pd.MultiIndex([[], []], [[], []], names=['idx1', 'idx2']),
            ['column'],
        ),
        pytest.param(
            pd.Index([], name='idx'),
            ['idx'],
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(
            pd.MultiIndex([[], []], [[], []], names=['idx1', 'idx2']),
            ['column', 'idx2'],
            marks=pytest.mark.xfail(raises=ValueError),
        ),
    ]
)
def test_level_and_column_names(index, columns):
    misc = audformat.MiscTable(index)
    for column in columns:
        misc[column] = audformat.Column()


def test_load_old_pickle(tmpdir):
    # We have stored string dtype as object dtype before
    # and have to fix this when loading old PKL files from cache.
    # This does only affect columns
    # as there was no MiscTable available.

    # Create PKL file containing strings as object
    y = pd.Series(['c'], dtype='object', name='column')
    index = pd.Index(['i'], dtype='object', name='idx')

    db = audformat.testing.create_db(minimal=True)
    db['misc'] = audformat.MiscTable(index)
    db.schemes['column'] = audformat.Scheme(audformat.define.DataType.OBJECT)
    db['misc']['column'] = audformat.Column(scheme_id='column')
    db['misc']['column'].set(y.values)
    db_root = tmpdir.join('db')
    db.save(db_root, storage_format='pkl')

    # Change scheme dtype to string and store header again
    db.schemes['column'] = audformat.Scheme(audformat.define.DataType.STRING)
    db.save(db_root, header_only=True)

    # Load and check that dtype is string
    db_new = audformat.Database.load(db_root)
    assert db_new.schemes['column'].dtype == audformat.define.DataType.STRING
    assert db_new['misc'].df['column'].dtype == 'string'
