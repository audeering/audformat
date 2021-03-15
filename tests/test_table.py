import typing

import pytest
import numpy as np
import pandas as pd

import audeer

import audformat
import audformat.testing


def create_db_table(
    obj: typing.Union[pd.Series, pd.DataFrame] = None,
    *,
    rater: audformat.Rater = None,
    media: audformat.Media = None,
    split: audformat.Split = None,
    scheme_id: str = None,  # overwrite id of first scheme
) -> audformat.Table:
    if obj is None:
        obj = pd.Series(
            index=audformat.filewise_index(),
            dtype=float,
        )
    db = audformat.testing.create_db(
        data={'table': obj}
    )
    table = db['table']
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


def create_table(
        obj: typing.Union[pd.Series, pd.DataFrame],
) -> audformat.Table:
    r"""Helper function to create Table."""
    table = audformat.Table(obj.index)
    if isinstance(obj, pd.Series):
        obj = obj.to_frame()
    for name in obj:
        table[name] = audformat.Column()
        table[name].set(obj[name].values)
    table._df = table.df.astype(obj.dtypes)
    return table


def test_access():
    db = audformat.testing.create_db()
    for table_id, table in db.tables.items():
        assert db.tables[table_id] == db[table_id]
        assert str(db.tables[table_id]) == str(db[table_id])
        if table.media_id is not None:
            assert table.media == db.media[table.media_id]
        else:
            assert table.media is None
        if table.split_id is not None:
            assert table.split == db.splits[table.split_id]
        else:
            assert table.split is None


@pytest.mark.parametrize(
    'tables, expected',
    [
        # empty
        (
            [
                create_table(
                    pd.Series(
                        index=audformat.filewise_index(),
                        dtype=float,
                    )
                ),
            ],
            create_table(
                pd.Series(
                    index=audformat.filewise_index(),
                    dtype=float,
                )
            ),
        ),
        (
            [
                create_table(
                    pd.Series(
                        index=audformat.filewise_index(),
                        dtype=float,
                    )
                ),
            ] * 3,
            create_table(
                pd.Series(
                    index=audformat.filewise_index(),
                    dtype=float,
                )
            ),
        ),
        # content + empty
        (
            [
                create_table(
                    pd.Series(
                        [1.],
                        index=audformat.filewise_index('f1'),
                    )
                ),
                create_table(
                    pd.Series(
                        index=audformat.filewise_index(),
                        dtype=float,
                    )
                ),
            ],
            create_table(
                pd.Series(
                    [1.],
                    index=audformat.filewise_index('f1'),
                )
            ),
        ),
        # empty + content
        (
            [
                create_table(
                    pd.Series(
                        index=audformat.filewise_index(),
                        dtype=float,
                    )
                ),
                create_table(
                    pd.Series(
                        [1.],
                        index=audformat.filewise_index('f1'),
                    )
                ),
            ],
            create_table(
                pd.Series(
                    [1.],
                    index=audformat.filewise_index('f1'),
                )
            ),
        ),
        # filewise + segmented
        (
            [
                create_table(
                    pd.Series(
                        index=audformat.filewise_index(),
                        dtype=float,
                        name='c1',
                    )
                ),
                create_table(
                    pd.Series(
                        index=audformat.segmented_index(),
                        dtype=float,
                        name='c2',
                    )
                ),
            ],
            create_table(
                pd.DataFrame(
                    {
                        'c1': pd.Series(
                            index=audformat.segmented_index(),
                            dtype=float,
                            name='c1',
                        ),
                        'c2': pd.Series(
                            index=audformat.segmented_index(),
                            dtype=float,
                            name='c2',
                        )
                    },
                )
            ),
        ),
        # same column
        (
            [
                create_table(
                    pd.Series(
                        [1., np.nan],
                        index=audformat.filewise_index(['f1', 'f2']),
                    )
                ),
                create_table(
                    pd.Series(
                        [2., 3.],
                        index=audformat.filewise_index(['f2', 'f3']),
                    )
                ),
                create_table(
                    pd.Series(
                        [3., 4.],
                        index=audformat.filewise_index(['f3', 'f4']),
                    )
                ),
            ],
            create_table(
                pd.Series(
                    [1., 2., 3., 4.],
                    index=audformat.filewise_index(['f1', 'f2', 'f3', 'f4']),
                )
            ),
        ),
        pytest.param(  # value mismatch
            [
                create_table(
                    pd.Series(
                        [1.],
                        index=audformat.filewise_index('f1'),
                    )
                ),
                create_table(
                    pd.Series(
                        [-1.],
                        index=audformat.filewise_index('f1'),
                    )
                ),
            ],
            None,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        # different columns
        (
            [
                create_table(
                    pd.Series(
                        [1., 2.],
                        index=audformat.filewise_index(['f1', 'f2']),
                        name='c1',
                    )
                ),
                create_table(
                    pd.Series(
                        [2., 3.],
                        index=audformat.filewise_index(['f2', 'f3']),
                        name='c2',
                    )
                ),
            ],
            create_table(
                pd.DataFrame(
                    {
                        'c1': [1., 2., np.nan],
                        'c2': [np.nan, 2., 3.],
                    },
                    index=audformat.filewise_index(['f1', 'f2', 'f3']),
                )
            ),
        ),
        # filewise + segmented
        (
            [
                create_table(
                    pd.Series(
                        [1.],
                        index=audformat.filewise_index('f1'),
                    )
                ),
                create_table(
                    pd.Series(
                        [1.],
                        index=audformat.segmented_index('f1', 0, 1),
                    )
                ),
            ],
            create_table(
                pd.Series(
                    [1., 1.],
                    index=audformat.segmented_index(
                        ['f1', 'f1'],
                        [0, 0],
                        [None, 1],
                    ),
                )
            ),
        ),
        (
            [
                pytest.DB['files'],
                pytest.DB['segments'],
            ],
            create_table(
                audformat.utils.concat(
                    [
                        pytest.DB['files'].df,
                        pytest.DB['segments'].df,
                    ]
                )
            )
        )
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


def test_add_2():

    db = audformat.testing.create_db(minimal=True)
    db.media['media'] = audformat.Media()
    db.splits['split'] = audformat.Split()

    # empty tables

    audformat.testing.add_table(
        db, 'table1', audformat.define.IndexType.FILEWISE,
        p_none=0.25, num_files=5, media_id='media', split_id='split',
    )
    audformat.testing.add_table(
        db, 'table2', audformat.define.IndexType.FILEWISE,
        p_none=0.25, num_files=[1, 6, 7, 8, 9], media_id='media',
    )
    db['table'] = db['table1'] + db['table2']
    pd.testing.assert_index_equal(
        db['table'].files,
        db['table1'].files.union(db['table2'].files)
    )
    assert db['table'].media_id is None
    assert db['table'].split_id is None
    for column in db['table'].columns:
        assert column.scheme_id is None
        assert column.rater_id is None

    # add table to itself

    pd.testing.assert_frame_equal(
        (db['table1'] + db['table1']).df,
        db['table1'].df,
    )

    # add two schemes

    db.schemes['scheme1'] = audformat.Scheme()
    db.schemes['scheme2'] = audformat.Scheme()

    # tables of same type without overlap

    for table_type in [
        audformat.define.IndexType.FILEWISE,
        audformat.define.IndexType.SEGMENTED
    ]:
        db.drop_tables(list(db.tables))
        audformat.testing.add_table(
            db,
            'table1',
            table_type,
            num_files=5,
            columns=['scheme1'],
        )
        audformat.testing.add_table(
            db,
            'table2',
            table_type,
            num_files=(6, 7, 8, 9, 10),
            columns='scheme1',
        )
        db['table'] = db['table1'] + db['table2']
        pd.testing.assert_frame_equal(
            db['table'].get(),
            pd.concat([db['table1'].get(), db['table2'].get()])
        )

    # tables of same type with overlap

    db.drop_tables(list(db.tables))
    audformat.testing.add_table(
        db,
        'table1',
        audformat.define.IndexType.FILEWISE,
        num_files=(1, 2),
        columns='scheme1',
    )
    audformat.testing.add_table(
        db,
        'table2',
        audformat.define.IndexType.FILEWISE,
        num_files=(1,),
        columns='scheme1',
    )
    db['table2'].df.iloc[0] = np.nan  # ok if other value is nan
    db['table'] = db['table1'] + db['table2']
    pd.testing.assert_series_equal(
        db['table']['scheme1'].get(),
        db['table1']['scheme1'].get()
    )
    with pytest.raises(ValueError):
        db['table2'].df.iloc[0] = 'do not match'  # values do not match
        db['table'] = db['table1'] + db['table2']

    # filewise with segmented table

    for num_files_1, num_files_2 in (
            (5, 5),
            (5, 4),
            (4, 5),
    ):
        db.drop_tables(list(db.tables))
        audformat.testing.add_table(
            db,
            'table1',
            audformat.define.IndexType.FILEWISE,
            columns='scheme1',
            num_files=num_files_1,
        )
        audformat.testing.add_table(
            db,
            'table2',
            audformat.define.IndexType.SEGMENTED,
            columns='scheme2',
            num_files=num_files_2,
        )
        db['table'] = db['table1'] + db['table2']
        assert db['table'].type == audformat.define.IndexType.SEGMENTED
        np.testing.assert_equal(
            db['table']['scheme1'].get().dropna().unique(),
            db['table1']['scheme1'].get().values)
        pd.testing.assert_series_equal(
            db['table']['scheme2'].get().dropna(),
            db['table2']['scheme2'].get().dropna())

    # segmented with filewise table

    for num_files_1, num_files_2 in (
            (5, 5),
            (5, 4),
            (4, 5),
    ):
        db.drop_tables(list(db.tables))
        audformat.testing.add_table(
            db,
            'table1',
            audformat.define.IndexType.SEGMENTED,
            columns='scheme1',
            num_files=num_files_1,
        )
        audformat.testing.add_table(
            db,
            'table2',
            audformat.define.IndexType.FILEWISE,
            columns='scheme2',
            num_files=num_files_2,
        )
        db['table'] = db['table1'] + db['table2']
        assert db['table'].type == audformat.define.IndexType.SEGMENTED
        np.testing.assert_equal(
            db['table']['scheme2'].get().dropna().unique(),
            db['table2']['scheme2'].get().values)
        pd.testing.assert_series_equal(
            db['table']['scheme1'].get().dropna(),
            db['table1']['scheme1'].get().dropna())


@pytest.mark.parametrize(
    'table',
    [
        audformat.Table(),
        pytest.DB['files'],
        pytest.DB['segments'],
    ]
)
def test_copy(table):
    table_copy = table.copy()
    assert str(table_copy) == str(table)
    pd.testing.assert_frame_equal(table_copy.df, table.df)


@pytest.mark.parametrize(
    'inplace',
    [
        False,
        True,
    ]
)
def test_drop_and_pick_columns(inplace):

    db = audformat.testing.create_db()

    assert 'float' in db['files'].columns
    table = db['files'].drop_columns('float', inplace=inplace)
    if not inplace:
        assert 'float' in db['files'].columns
    assert 'float' not in table.columns
    assert 'string' in table.columns
    table = db['files'].pick_columns('time', inplace=inplace)
    assert 'time' in table.columns
    assert 'string' not in table.columns
    if not inplace:
        assert 'string' in db['files'].columns


@pytest.mark.parametrize(
    'files',
    [
        pytest.DB.files,
        pytest.DB.files[0],
        [pytest.DB.files[0], 'does-not-exist.wav'],
        lambda x: '1' in x,
    ]
)
@pytest.mark.parametrize(
    'table',
    [
        pytest.DB['files'],
        pytest.DB['segments'],
    ]
)
def test_drop_files(files, table):

    table = table.drop_files(files, inplace=False)
    if callable(files):
        files = table.files.to_series().apply(files)
    elif isinstance(files, str):
        files = [files]
    assert table.files.intersection(files).empty


@pytest.mark.parametrize(
    'files',
    [
        pytest.DB.files,
        pytest.DB.files[0],
        [pytest.DB.files[0], 'does-not-exist.wav'],
        lambda x: '1' in x,
    ]
)
@pytest.mark.parametrize(
    'table',
    [
        pytest.DB['files'],
        pytest.DB['segments'],
    ]
)
def test_pick_files(files, table):

    table = table.pick_files(files, inplace=False)
    if callable(files):
        files = table.files[table.files.to_series().apply(files)]
        seen = set()
        seen_add = seen.add
        files = [x for x in files if not (x in seen or seen_add(x))]
    elif isinstance(files, str):
        files = [files]
    pd.testing.assert_index_equal(
        table.files.unique(),
        audformat.filewise_index(files).intersection(table.files),
    )


def test_and_pick_index():

    for table in ['files', 'segments']:
        index = pytest.DB[table].index[:5]
        df_pick = pytest.DB[table].pick_index(index).get()
        index = pytest.DB[table].index[5:]
        df_drop = pytest.DB[table].drop_index(index).get()
        pd.testing.assert_frame_equal(df_pick, df_drop)

    index = pytest.DB['segments'].index[:5]
    with pytest.raises(ValueError):
        pytest.DB['files'].drop_index(index).get()
    with pytest.raises(ValueError):
        pytest.DB['files'].pick_index(index).get()


def test_empty():

    db = audformat.testing.create_db(minimal=True)
    db['table'] = audformat.Table()

    assert db['table'].type == audformat.define.IndexType.FILEWISE
    assert db['table'].files.empty
    assert len(db['table']) == 0

    db['table']['column'] = audformat.Column()

    assert db['table']['column'].get().dtype == object


def test_exceptions():
    db = audformat.testing.create_db(minimal=True)

    # invalid segments

    with pytest.raises(ValueError):
        db['table'] = audformat.Table(
            audformat.segmented_index(
                files=['f1', 'f2'],
                starts=['0s'],
                ends=['1s']
            ),
        )
    with pytest.raises(ValueError):
        db['table'] = audformat.Table(
            audformat.segmented_index(
                files=['f1', 'f2'],
                starts=['0s', '1s'],
                ends=['1s'],
            )
        )
    with pytest.raises(ValueError):
        db['table'] = audformat.Table(
            audformat.segmented_index(
                files=['f1', 'f2'],
                starts=['0s'],
                ends=['1s', '2s'],
            )
        )

    # bad scheme or rater

    with pytest.raises(audformat.errors.BadIdError):
        db['table'] = audformat.Table(
            audformat.filewise_index(['f1', 'f2']),
        )
        db['table']['column'] = audformat.Column(scheme_id='invalid')
    with pytest.raises(audformat.errors.BadIdError):
        db['table'] = audformat.Table(
            audformat.filewise_index(['f1', 'f2']),
        )
        db['table']['column'] = audformat.Column(rater_id='invalid')


def test_extend_index():

    db = audformat.testing.create_db(minimal=True)
    db.schemes['scheme'] = audformat.Scheme()

    # empty and invalid

    db['table'] = audformat.Table()
    db['table'].extend_index(audformat.filewise_index())
    assert db['table'].get().empty
    with pytest.raises(ValueError):
        db['table'].extend_index(
            audformat.segmented_index(
                files=['1.wav', '2.wav'],
                starts=['0s', '1s'],
                ends=['1s', '2s'],
            ),
            fill_values='a',
        )

    db.drop_tables('table')

    # filewise

    db['table'] = audformat.Table()
    db['table']['columns'] = audformat.Column(scheme_id='scheme')
    db['table'].extend_index(
        audformat.filewise_index(['1.wav', '2.wav']),
        fill_values='a',
        inplace=True,
    )
    np.testing.assert_equal(
        db['table']['columns'].get().values,
        np.array(['a', 'a']),
    )
    index = pd.Index(['1.wav', '3.wav'],
                     name=audformat.define.IndexField.FILE)
    db['table'].extend_index(
        index,
        fill_values='b',
        inplace=True,
    )
    np.testing.assert_equal(
        db['table']['columns'].get().values,
        np.array(['a', 'a', 'b']),
    )

    db.drop_tables('table')

    # segmented

    db['table'] = audformat.Table(
        audformat.segmented_index()
    )
    db['table']['columns'] = audformat.Column(scheme_id='scheme')
    db['table'].extend_index(
        audformat.segmented_index(
            files=['1.wav', '2.wav'],
            starts=['0s', '1s'],
            ends=['1s', '2s'],
        ),
        fill_values='a',
        inplace=True,
    )
    np.testing.assert_equal(
        db['table']['columns'].get().values,
        np.array(['a', 'a']),
    )
    index = audformat.segmented_index(
        files=['1.wav', '3.wav'],
        starts=['0s', '2s'],
        ends=['1s', '3s'],
    )
    db['table'].extend_index(
        index,
        fill_values={'columns': 'b'},
        inplace=True,
    )
    np.testing.assert_equal(
        db['table']['columns'].get().values,
        np.array(['a', 'a', 'b']),
    )

    db.drop_tables('table')


@pytest.mark.parametrize('num_files,values', [
    (6, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
    (6, np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])),
    (6, pd.Series(range(6), dtype='float').values,),
])
def test_filewise(num_files, values):

    db = audformat.testing.create_db(minimal=True)
    audformat.testing.add_table(
        db, 'table', audformat.define.IndexType.FILEWISE, num_files=num_files,
    )
    db.schemes['scheme'] = audformat.Scheme(
        dtype=audformat.define.DataType.FLOAT,
    )
    table = db['table']

    # empty table
    df = pd.DataFrame(index=audformat.filewise_index(table.files))
    pd.testing.assert_frame_equal(table.get(), df)

    # no values
    df['column'] = np.nan
    table['column'] = audformat.Column(scheme_id='scheme')
    pd.testing.assert_frame_equal(table.get(), df)

    # single
    df['column'] = np.nan
    table.df['column'] = np.nan
    df.iloc[0] = values[0]
    index = audformat.filewise_index(table.files[0])
    table.set({'column': values[0]}, index=index)
    pd.testing.assert_frame_equal(
        table.get(index),
        df.loc[[table.files[0]]],
    )
    pd.testing.assert_frame_equal(table.get(), df)

    # slice
    df['column'] = np.nan
    table.df['column'] = np.nan
    df['column'][1:-1] = values[1:-1]
    index = audformat.filewise_index(table.files[1:-1])
    table.set({'column': values[1:-1]}, index=index)
    pd.testing.assert_frame_equal(
        table.get(index),
        df.loc[table.files[1:-1]],
    )
    pd.testing.assert_frame_equal(table.get(), df)

    # all
    df['column'] = np.nan
    table.df['column'] = np.nan
    df.iloc[:, 0] = values
    table.set({'column': values})
    pd.testing.assert_frame_equal(table.get(), df)

    # scalar
    df['column'] = np.nan
    table.df['column'] = np.nan
    df.iloc[:, 0] = values[0]
    table.set({'column': values[0]})
    pd.testing.assert_frame_equal(table.get(), df)

    # data frame
    df['column'] = np.nan
    table.df['column'] = np.nan
    df.iloc[:, 0] = values
    table.set(df)
    pd.testing.assert_frame_equal(table.get(), df)


@pytest.mark.parametrize(
    'table, map',
    [
        (pytest.DB['files'], {'label_map_int': 'int'}),
        (pytest.DB['files'], {'label_map_int': 'label_map_int'}),
        (pytest.DB['files'], {'label_map_str': 'prop1'}),
        (pytest.DB['segments'], {'label_map_str': ['prop1', 'prop2']}),
        (pytest.DB['segments'], {  # duplicates will be ignored
            'label_map_str': ['prop1', 'prop2', 'prop1', 'prop2']
        }),
        (pytest.DB['segments'], {
            'label_map_str': ['label_map_str', 'prop1', 'prop2']
        }),
        pytest.param(  # no database
            audformat.Table(), 'map',
            marks=pytest.mark.xfail(raises=RuntimeError),
        ),
    ]
)
def test_map(table, map):
    result = table.get(map=map)
    expected = table.get()
    for column, mapped_columns in map.items():
        mapped_columns = audeer.to_list(mapped_columns)
        if len(mapped_columns) == 1:
            expected[mapped_columns[0]] = table.columns[column].get(
                map=mapped_columns[0],
            )
        else:
            for mapped_column in mapped_columns:
                if mapped_column != column:
                    expected[mapped_column] = table.columns[column].get(
                        map=mapped_column,
                    )
        if column not in mapped_columns:
            expected.drop(columns=column, inplace=True)
    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize('num_files,num_segments_per_file,values', [
    (3, 2, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
    (3, 2, np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])),
    (3, 2, pd.Series(range(6), dtype='float').values),
])
def test_segmented(num_files, num_segments_per_file, values):

    db = audformat.testing.create_db(minimal=True)
    audformat.testing.add_table(
        db, 'table', audformat.define.IndexType.SEGMENTED,
        num_files=num_files, num_segments_per_file=num_segments_per_file,
    )
    db.schemes['scheme'] = audformat.Scheme(
        dtype=audformat.define.DataType.FLOAT,
    )
    table = db['table']

    # empty table
    df = pd.DataFrame(index=table._df.index)
    pd.testing.assert_frame_equal(table.get(), df)

    # no values
    df['column'] = np.nan
    table['column'] = audformat.Column(scheme_id='scheme')
    pd.testing.assert_frame_equal(table.get(), df)

    # single
    df['column'] = np.nan
    table.df['column'] = np.nan
    index = audformat.segmented_index(
        table.files[0],
        starts=table.starts[0],
        ends=table.ends[0],
    )
    df.loc[index] = values[0]
    table.set({'column': values[0]}, index=index)
    pd.testing.assert_frame_equal(
        table.get(index),
        df.loc[index],
    )

    # slice
    df['column'] = np.nan
    table.df['column'] = np.nan
    index = audformat.segmented_index(
        table.files[1:-1],
        starts=table.starts[1:-1],
        ends=table.ends[1:-1],
    )
    df.loc[index, :] = values[1:-1]
    table.set({'column': values[1:-1]}, index=index)
    pd.testing.assert_frame_equal(
        table.get(index),
        df.loc[index],
    )

    # all
    df['column'] = np.nan
    table.df['column'] = np.nan
    df.iloc[:, 0] = values
    table.set({'column': values})
    pd.testing.assert_frame_equal(table.get(), df)

    # scalar
    df['column'] = np.nan
    table.df['column'] = np.nan
    df.iloc[:, 0] = values[0]
    table.set({'column': values[0]})
    pd.testing.assert_frame_equal(table.get(), df)

    # data frame
    df['column'] = np.nan
    table.df['column'] = np.nan
    table.set(df)
    pd.testing.assert_frame_equal(table.get(), df)


def test_type():

    db = audformat.testing.create_db()

    assert db['files'].is_filewise
    assert db['segments'].is_segmented

    assert sorted(set(db['files'].index)) == sorted(db.files)
    pd.testing.assert_index_equal(db.segments, db['segments'].index)
    pd.testing.assert_index_equal(db.files, db['files'].files)
    pd.testing.assert_index_equal(
        db['files'].starts.unique(),
        pd.TimedeltaIndex(
            [0],
            name=audformat.define.IndexField.START,
        )
    )
    pd.testing.assert_index_equal(
        db['files'].ends.unique(),
        pd.TimedeltaIndex(
            [pd.NaT],
            name=audformat.define.IndexField.END,
        )
    )
    pd.testing.assert_index_equal(db['files'].index, db['files'].index)


@pytest.mark.parametrize(
    'table, overwrite, others',
    [
        # empty
        (
            create_db_table(),
            False,
            [],
        ),
        (
            create_db_table(),
            False,
            create_db_table(),
        ),
        # same column, with overlap
        (
            create_db_table(
                pd.Series(
                    [1., 2.],
                    index=audformat.filewise_index(['f1', 'f2']),
                )
            ),
            False,
            create_db_table(
                pd.Series(
                    [2., 3.],  # ok, value do match
                    index=audformat.filewise_index(['f2', 'f3']),
                )
            ),
        ),
        (
            create_db_table(
                pd.Series(
                    [1., 2.],
                    index=audformat.filewise_index(['f1', 'f2']),
                )
            ),
            False,
            create_db_table(
                pd.Series(
                    [np.nan, 3.],  # ok, value is nan
                    index=audformat.filewise_index(['f2', 'f3']),
                )
            ),
        ),
        pytest.param(
            create_db_table(
                pd.Series(
                    [1., 2.],
                    index=audformat.filewise_index(['f1', 'f2']),
                )
            ),
            False,
            create_db_table(
                pd.Series(
                    [99., 3.],  # error, value do not match
                    index=audformat.filewise_index(['f2', 'f3']),
                )
            ),
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        (
            create_db_table(
                pd.Series(
                    [1., 2.],
                    index=audformat.filewise_index(['f1', 'f2']),
                )
            ),
            True,
            create_db_table(
                pd.Series(
                    [99., 3.],  # ok, will be overwritten
                    index=audformat.filewise_index(['f2', 'f3']),
                )
            ),
        ),
        # columns with new schemes
        (
            create_db_table(
                pd.Series(
                    [1., 2.],
                    index=audformat.filewise_index(['f1', 'f2']),
                    name='c1',
                )
            ),
            False,
            [
                create_db_table(
                    pd.Series(
                        ['a', 'b'],
                        index=audformat.filewise_index(['f2', 'f3']),
                        name='c2',
                    )
                ),
                create_db_table(
                    pd.Series(
                        [1, 2],
                        index=audformat.filewise_index(['f2', 'f3']),
                        name='c3',
                    )
                ),
            ],
        ),
        # error: scheme mismatch
        pytest.param(
            create_db_table(
                pd.Series(
                    [1., 2.],
                    index=audformat.filewise_index(['f1', 'f2']),
                )
            ),
            False,
            create_db_table(  # same column, different scheme
                pd.Series(
                    ['a', 'b'],
                    index=audformat.filewise_index(['f2', 'f3']),
                )
            ),
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(
            create_db_table(
                pd.Series(
                    [1., 2.],
                    index=audformat.filewise_index(['f1', 'f2']),
                )
            ),
            False,
            create_table(  # no scheme
                pd.Series(
                    [1., 2.],
                    index=audformat.filewise_index(['f1', 'f2']),
                )
            ),
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(
            create_db_table(
                pd.Series(
                    [1., 2.],
                    index=audformat.filewise_index(['f1', 'f2']),
                    name='c1',
                ),
                scheme_id='scheme',
            ),
            False,
            create_db_table(
                pd.Series(  # different scheme with same id
                    ['a', 'b'],
                    index=audformat.filewise_index(['f1', 'f2']),
                    name='c2',
                ),
                scheme_id='scheme',
            ),
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        # column with new rater
        (
            create_db_table(
                pd.Series(
                    index=audformat.filewise_index(),
                    dtype=float,
                    name='c1',
                ),
            ),
            False,
            create_db_table(
                pd.Series(
                    index=audformat.filewise_index(),
                    dtype=float,
                    name='c2',
                ),
                rater=audformat.Rater(
                    audformat.define.RaterType.HUMAN),
            ),
        ),
        # error: rater mismatch
        pytest.param(
            create_db_table(
                rater=audformat.Rater(audformat.define.RaterType.HUMAN),
            ),
            False,
            create_db_table(
                rater=audformat.Rater(audformat.define.RaterType.MACHINE),
            ),
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(
            create_db_table(
                rater=audformat.Rater(audformat.define.RaterType.HUMAN),
            ),
            False,
            create_db_table(),
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(
            create_db_table(),
            False,
            create_db_table(
                rater=audformat.Rater(audformat.define.RaterType.MACHINE),
            ),
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(
            create_db_table(
                pd.Series(
                    index=audformat.filewise_index(),
                    dtype=float,
                    name='c1',
                ),
                rater=audformat.Rater(audformat.define.RaterType.HUMAN),
            ),
            False,
            create_db_table(
                pd.Series(
                    index=audformat.filewise_index(),
                    dtype=float,
                    name='c2',
                ),
                rater=audformat.Rater(audformat.define.RaterType.MACHINE),
            ),
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        # media and split match
        (
            create_db_table(
                media=audformat.Media(audformat.define.MediaType.AUDIO),
                split=audformat.Split(audformat.define.SplitType.TEST),
            ),
            False,
            create_db_table(
                media=audformat.Media(audformat.define.MediaType.AUDIO),
                split=audformat.Split(audformat.define.SplitType.TEST),
            ),
        ),
        # error: media mismatch
        pytest.param(
            create_db_table(
                media=audformat.Media(audformat.define.MediaType.AUDIO),
            ),
            False,
            create_db_table(
                media=audformat.Media(audformat.define.MediaType.VIDEO),
            ),
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(
            create_db_table(
                media=audformat.Media(audformat.define.MediaType.AUDIO),
            ),
            False,
            create_db_table(),
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(
            create_db_table(),
            False,
            create_db_table(
                media=audformat.Media(audformat.define.MediaType.AUDIO),
            ),
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        # error: split mismatch
        pytest.param(
            create_db_table(
                split=audformat.Split(audformat.define.SplitType.TEST),
            ),
            False,
            create_db_table(
                split=audformat.Split(audformat.define.SplitType.TRAIN),
            ),
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(
            create_db_table(
                split=audformat.Split(audformat.define.SplitType.TEST),
            ),
            False,
            create_db_table(),
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(
            create_db_table(),
            False,
            create_db_table(
                split=audformat.Split(audformat.define.SplitType.TRAIN),
            ),
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        # error: not assigned to db
        pytest.param(
            audformat.Table(),
            False,
            [],
            marks=pytest.mark.xfail(raises=RuntimeError),
        ),
        # error: different index type
        pytest.param(
            create_db_table(
                pd.Series(
                    [1., 2.],
                    index=audformat.filewise_index(['f1', 'f2']),
                )
            ),
            False,
            create_db_table(
                pd.Series(
                    [2., 3.],
                    index=audformat.segmented_index(['f2', 'f3']),
                )
            ),
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(
            create_db_table(
                pd.Series(
                    [1., 2.],
                    index=audformat.segmented_index(['f1', 'f2']),
                )
            ),
            False,
            create_db_table(
                pd.Series(
                    [2., 3.],
                    index=audformat.filewise_index(['f2', 'f3']),
                )
            ),
            marks=pytest.mark.xfail(raises=ValueError),
        ),
    ]
)
def test_update(table, overwrite, others):
    df = table.get()
    table.update(others, overwrite=overwrite)
    if isinstance(others, audformat.Table):
        others = [others]
    df = audformat.utils.concat(
        [df] + [other.df for other in others],
        overwrite=overwrite,
    )
    assert table.type == audformat.index_type(df)
    if isinstance(df, pd.Series):
        df = df.to_frame()
    pd.testing.assert_frame_equal(table.df, df)
    for other in others:
        for column_id, column in other.columns.items():
            assert column.scheme == table[column_id].scheme
            assert column.rater == table[column_id].rater
