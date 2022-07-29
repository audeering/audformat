import typing

import audeer
import numpy as np
import pandas as pd
import pytest

import audformat.testing


def create_db_misc_table(
    obj: typing.Union[pd.Series, pd.DataFrame] = None,
    *,
    rater: audformat.Rater = None,
    media: audformat.Media = None,
    split: audformat.Split = None,
    scheme_id: str = None,  # overwrite id of first scheme
) -> audformat.MiscTable:
    if obj is None:
        obj = pd.Series(
            index=pd.Index([], name='idx'),
            dtype=float,
        )
    db = audformat.testing.create_db(
        data={'misc': obj}
    )
    table = db['misc']
    if rater is not None:
        db.raters['rater'] = rater
        for column in table.columns.values():
            column.rater_id = 'rater'
    if media is not None:
        db.media['media'] = media
        table.media_id = 'media'
    if split is not None:
        db.splits['split'] = split
        table.split_id = 'split'
    if scheme_id is not None:
        old_scheme_id = list(db.schemes)[0]
        db.schemes[scheme_id] = db.schemes.pop(old_scheme_id)
        for column in table.columns.values():
            if column.scheme_id == old_scheme_id:
                column.scheme_id = scheme_id
    return table


def create_misc_table(
        obj: typing.Union[pd.Series, pd.DataFrame],
) -> audformat.MiscTable:
    r"""Helper function to create Table."""
    table = audformat.MiscTable(obj.index)
    if isinstance(obj, pd.Series):
        obj = obj.to_frame()
    for name in obj:
        table[name] = audformat.Column()
        table[name].set(obj[name].values)
    return table


@pytest.mark.parametrize(
    'tables, expected',
    [
        # empty
        (
            [
                create_misc_table(
                    pd.Series(
                        index=pd.Index([], name='idx'),
                        dtype='object',
                    ),
                ),
            ],
            create_misc_table(
                pd.Series(
                    index=pd.Index([], name='idx'),
                    dtype='object',
                ),
            ),
        ),
        (
            [
                create_misc_table(
                    pd.Series(
                        index=pd.Index([], name='idx'),
                        dtype='object',
                    ),
                ),
            ] * 3,
            create_misc_table(
                pd.Series(
                    index=pd.Index([], name='idx'),
                    dtype='object',
                ),
            ),
        ),
        # content + empty
        (
            [
                create_misc_table(
                    pd.Series(
                        [1.],
                        index=pd.Index(['a'], name='idx'),
                    ),
                ),
                create_misc_table(
                    pd.Series(
                        index=pd.Index([], name='idx'),
                        dtype='object',
                    ),
                ),
            ],
            create_misc_table(
                pd.Series(
                    [1.],
                    index=pd.Index(['a'], name='idx'),
                ),
            ),
        ),
        # empty + content
        (
            [
                create_misc_table(
                    pd.Series(
                        index=pd.Index([], name='idx'),
                        dtype='object',
                    )
                ),
                create_misc_table(
                    pd.Series(
                        [1.],
                        index=pd.Index(['a'], name='idx'),
                    ),
                ),
            ],
            create_misc_table(
                pd.Series(
                    [1.],
                    index=pd.Index(['a'], name='idx'),
                ),
            ),
        ),
        # content + content
        (
            [
                create_misc_table(
                    pd.Series(
                        [1., 2.],
                        index=pd.Index(['a', 'b'], name='idx'),
                    ),
                ),
                create_misc_table(
                    pd.Series(
                        [2., 3.],
                        index=pd.Index(['b', 'c'], name='idx'),
                    ),
                ),
            ],
            create_misc_table(
                pd.Series(
                    [1., 2., 3.],
                    index=pd.Index(['a', 'b', 'c'], name='idx'),
                ),
            ),
        ),
        # different columns
        (
            [
                create_misc_table(
                    pd.Series(
                        [1., 1.],
                        index=pd.Index(['a', 'b'], name='idx'),
                        name='c1',
                    ),
                ),
                create_misc_table(
                    pd.Series(
                        [2., 2.],
                        index=pd.Index(['b', 'c'], name='idx'),
                        name='c2',
                    ),
                ),
            ],
            create_misc_table(
                pd.DataFrame(
                    {
                        'c1': [1., 1., np.nan],
                        'c2': [np.nan, 2., 2.],
                    },
                    index=pd.Index(['a', 'b', 'c'], name='idx'),
                ),
            ),
        ),
        pytest.param(  # value mismatch
            [
                create_misc_table(
                    pd.Series(
                        [1., 1.],
                        index=pd.Index(['a', 'b'], name='idx'),
                    ),
                ),
                create_misc_table(
                    pd.Series(
                        [2., 2.],
                        index=pd.Index(['b', 'c'], name='idx'),
                    ),
                ),
            ],
            None,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(  # level dimension mismatch
            [
                create_misc_table(
                    pd.Series(
                        [],
                        index=pd.MultiIndex.from_arrays(
                            [[], []],
                            names=['idx1', 'idx2'],
                        ),
                        dtype='object',
                    ),
                ),
                create_misc_table(
                    pd.Series(
                        [],
                        index=pd.Index([], name='idx1'),
                        dtype='object',
                    ),
                )
            ],
            None,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(  # level name mismatch
            [
                create_misc_table(
                    pd.Series(
                        [],
                        index=pd.Index([], name='idx1'),
                        dtype='object',
                    ),
                ),
                create_misc_table(
                    pd.Series(
                        [],
                        index=pd.Index([], name='idx2'),
                        dtype='object',
                    ),
                )
            ],
            None,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(  # level dtype mismatch
            [
                create_misc_table(
                    pd.Series(
                        [],
                        index=pd.Index([], name='idx1'),
                        dtype='string',
                    ),
                ),
                create_misc_table(
                    pd.Series(
                        [],
                        index=pd.Index([], name='idx2'),
                        dtype='object',
                    ),
                )
            ],
            None,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
    ]
)
def test_add(tables, expected):
    table = tables[0]
    for other in tables[1:]:
        table += other
    assert table.media_id is None
    assert table.split_id is None
    for column in table.columns.values():
        assert column.scheme_id is None
        assert column.rater_id is None
    assert table == expected


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

    # drop and pick with pd.Index

    index = pytest.DB[table_id].index[:2]
    df_pick = pytest.DB[table_id].pick_index(index).get()
    index = pytest.DB[table_id].index[2:]
    df_drop = pytest.DB[table_id].drop_index(index).get()

    assert len(df_pick) == len(df_drop) == 2
    pd.testing.assert_frame_equal(df_pick, df_drop)

    # drop and pick with pd.MultiIndex

    index = pd.MultiIndex.from_arrays(
        [pytest.DB[table_id].index[:2].to_list()],
        names=[pytest.DB[table_id].index.name],
    )
    index = audformat.utils.set_index_dtypes(index, 'string')
    df_pick = pytest.DB[table_id].pick_index(index).get()
    index = pd.MultiIndex.from_arrays(
        [pytest.DB[table_id].index[2:].to_list()],
        names=[pytest.DB[table_id].index.name],
    )
    index = audformat.utils.set_index_dtypes(index, 'string')
    df_drop = pytest.DB[table_id].drop_index(index).get()

    assert len(df_pick) == len(df_drop) == 2
    pd.testing.assert_frame_equal(df_pick, df_drop)

    # invalid index

    index = pytest.DB['segments'].index[:2]
    with pytest.raises(
        ValueError,
        match='Cannot drop',
    ):
        pytest.DB[table_id].drop_index(index).get()
    with pytest.raises(
        ValueError,
        match='Cannot pick',
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
        match='Cannot extend',
    ):
        db['misc'].extend_index(pd.Index([], name='other'))

    db.drop_tables('misc')

    # extend with pd.Index

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

    # extend with pd.MultiIndex

    index = pd.MultiIndex.from_arrays([['1', '4']], names=['idx'])
    db['misc'].extend_index(
        index,
        fill_values='b',
        inplace=True,
    )
    np.testing.assert_equal(
        db['misc']['columns'].get().values,
        np.array(['a', 'a', 'b', 'b']),
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


@pytest.mark.parametrize(
    'table, overwrite, others',
    [
        # empty
        (
            create_db_misc_table(),
            False,
            [],
        ),
        (
            create_db_misc_table(),
            False,
            create_db_misc_table(),
        ),
        # same column, with overlap
        (
            create_db_misc_table(
                pd.Series(
                    [1., 2.],
                    index=pd.Index(['a', 'b'], name='idx'),
                )
            ),
            False,
            create_db_misc_table(
                pd.Series(
                    [2., 3.],  # ok, value do match
                    index=pd.Index(['b', 'c'], name='idx'),
                )
            ),
        ),
        (
            create_db_misc_table(
                pd.Series(
                    [1., 2.],
                    index=pd.Index(['a', 'b'], name='idx'),
                )
            ),
            False,
            create_db_misc_table(
                pd.Series(
                    [np.nan, 3.],  # ok, value do match
                    index=pd.Index(['b', 'c'], name='idx'),
                )
            ),
        ),
        pytest.param(
            create_db_misc_table(
                pd.Series(
                    [1., 2.],
                    index=pd.Index(['a', 'b'], name='idx'),
                )
            ),
            False,
            create_db_misc_table(
                pd.Series(
                    [99., 3.],  # error, value do not match
                    index=pd.Index(['b', 'c'], name='idx'),
                )
            ),
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        (
            create_db_misc_table(
                pd.Series(
                    [1., 2.],
                    index=pd.Index(['a', 'b'], name='idx'),
                )
            ),
            True,
            create_db_misc_table(
                pd.Series(
                    [99., 3.],  # ok, will be overwritten
                    index=pd.Index(['a', 'b'], name='idx'),
                )
            ),
        ),
        # columns with new schemes
        (
            create_db_misc_table(
                pd.Series(
                    [1., 2.],
                    index=pd.Index(['a', 'b'], name='idx'),
                    name='c1',
                )
            ),
            False,
            [
                create_db_misc_table(
                    pd.Series(
                        ['a', 'b'],
                        index=pd.Index(['b', 'c'], name='idx'),
                        name='c2',
                    )
                ),
                create_db_misc_table(
                    pd.Series(
                        [1, 2],
                        index=pd.Index(['b', 'c'], name='idx'),
                        name='c3',
                    )
                ),
            ],
        ),
        # error: scheme mismatch
        pytest.param(
            create_db_misc_table(
                pd.Series(
                    [1., 2.],
                    index=pd.Index(['a', 'b'], name='idx'),
                )
            ),
            False,
            create_db_misc_table(  # same column, different scheme
                pd.Series(
                    ['a', 'b'],
                    index=pd.Index(['a', 'b'], name='idx'),
                )
            ),
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(
            create_db_misc_table(
                pd.Series(
                    [1., 2.],
                    index=pd.Index(['a', 'b'], name='idx'),
                )
            ),
            False,
            create_misc_table(  # no scheme
                pd.Series(
                    [1., 2.],
                    index=pd.Index(['a', 'b'], name='idx'),
                )
            ),
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(
            create_db_misc_table(
                pd.Series(
                    [1., 2.],
                    index=pd.Index(['a', 'b'], name='idx'),
                    name='c1',
                ),
                scheme_id='scheme',
            ),
            False,
            create_db_misc_table(
                pd.Series(  # different scheme with same id
                    ['a', 'b'],
                    index=pd.Index(['a', 'b'], name='idx'),
                    name='c2',
                ),
                scheme_id='scheme',
            ),
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        # column with new rater
        (
            create_db_misc_table(
                pd.Series(
                    index=pd.Index([], name='idx'),
                    dtype='object',
                    name='c1',
                ),
            ),
            False,
            create_db_misc_table(
                pd.Series(
                    index=pd.Index([], name='idx'),
                    dtype='object',
                    name='c2',
                ),
                rater=audformat.Rater(
                    audformat.define.RaterType.HUMAN),
            ),
        ),
        # error: rater mismatch
        pytest.param(
            create_db_misc_table(
                rater=audformat.Rater(audformat.define.RaterType.HUMAN),
            ),
            False,
            create_db_misc_table(
                rater=audformat.Rater(audformat.define.RaterType.MACHINE),
            ),
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(
            create_db_misc_table(
                rater=audformat.Rater(audformat.define.RaterType.HUMAN),
            ),
            False,
            create_db_misc_table(),
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(
            create_db_misc_table(),
            False,
            create_db_misc_table(
                rater=audformat.Rater(audformat.define.RaterType.MACHINE),
            ),
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(
            create_db_misc_table(
                pd.Series(
                    index=pd.Index([], name='idx'),
                    dtype='object',
                    name='c1',
                ),
                rater=audformat.Rater(audformat.define.RaterType.HUMAN),
            ),
            False,
            create_db_misc_table(
                pd.Series(
                    index=pd.Index([], name='idx'),
                    dtype='object',
                    name='c2',
                ),
                rater=audformat.Rater(audformat.define.RaterType.MACHINE),
            ),
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        # media and split match
        (
            create_db_misc_table(
                media=audformat.Media(audformat.define.MediaType.AUDIO),
                split=audformat.Split(audformat.define.SplitType.TEST),
            ),
            False,
            create_db_misc_table(
                media=audformat.Media(audformat.define.MediaType.AUDIO),
                split=audformat.Split(audformat.define.SplitType.TEST),
            ),
        ),
        # error: media mismatch
        pytest.param(
            create_db_misc_table(
                media=audformat.Media(audformat.define.MediaType.AUDIO),
            ),
            False,
            create_db_misc_table(
                media=audformat.Media(audformat.define.MediaType.VIDEO),
            ),
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(
            create_db_misc_table(
                media=audformat.Media(audformat.define.MediaType.AUDIO),
            ),
            False,
            create_db_misc_table(),
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(
            create_db_misc_table(),
            False,
            create_db_misc_table(
                media=audformat.Media(audformat.define.MediaType.AUDIO),
            ),
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        # error: split mismatch
        pytest.param(
            create_db_misc_table(
                split=audformat.Split(audformat.define.SplitType.TEST),
            ),
            False,
            create_db_misc_table(
                split=audformat.Split(audformat.define.SplitType.TRAIN),
            ),
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(
            create_db_misc_table(
                split=audformat.Split(audformat.define.SplitType.TEST),
            ),
            False,
            create_db_misc_table(),
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(
            create_db_misc_table(),
            False,
            create_db_misc_table(
                split=audformat.Split(audformat.define.SplitType.TRAIN),
            ),
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        # error: not assigned to db
        pytest.param(
            audformat.MiscTable(pd.Index([], name='idx')),
            False,
            [],
            marks=pytest.mark.xfail(raises=RuntimeError),
        ),
        # error: different level dimensions
        pytest.param(
            create_db_misc_table(
                pd.Series(
                    index=pd.Index([], name='idx'),
                    dtype='object',
                ),
            ),
            False,
            create_db_misc_table(
                pd.Series(
                    index=pd.MultiIndex.from_arrays(
                        [[], []],
                        names=['idx1', 'idx2'],
                    ),
                    dtype='object',
                )
            ),
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        # error: different level names
        pytest.param(
            create_db_misc_table(
                pd.Series(
                    index=pd.Index([], name='idx1'),
                    dtype='object',
                ),
            ),
            False,
            create_db_misc_table(
                pd.Series(
                    index=pd.Index([], name='idx2'),
                    dtype='object',
                ),
            ),
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        # error: different level dtype
        pytest.param(
            create_db_misc_table(
                pd.Series(
                    index=pd.Index([], dtype='int', name='idx'),
                    dtype='object',
                ),
            ),
            False,
            create_db_misc_table(
                pd.Series(
                    index=pd.Index([], dtype='string', name='idx'),
                    dtype='object',
                ),
            ),
            marks=pytest.mark.xfail(raises=ValueError),
        ),
    ]
)
def test_update(table, overwrite, others):
    df = table.get()
    table.update(others, overwrite=overwrite)
    others = audeer.to_list(others)
    df = audformat.utils.concat(
        [df] + [other.df for other in others],
        overwrite=overwrite,
    )
    assert audformat.utils.is_index_alike([table.index, df.index])
    if isinstance(df, pd.Series):
        df = df.to_frame()
    pd.testing.assert_frame_equal(table.df, df)
    for other in others:
        for column_id, column in other.columns.items():
            assert column.scheme == table[column_id].scheme
            assert column.rater == table[column_id].rater
