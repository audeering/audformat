import pytest
import numpy as np
import pandas as pd

import audeer

import audformat
import audformat.testing


def test_table_access():
    db = audformat.testing.create_db()
    for table_id in db.tables.keys():
        assert db.tables[table_id] == db[table_id]
        assert str(db.tables[table_id]) == str(db[table_id])


def test_add():

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
    pd.testing.assert_index_equal(db['table'].files,
                                  db['table1'].files.union(
                                      db['table2'].files))
    assert db['table'].media_id == 'media'
    assert db['table'].split_id is None

    # add table to itself

    assert db['table1'] + db['table1'] == db['table1']

    # add two schemes

    db.schemes['scheme1'] = audformat.Scheme()
    db.schemes['scheme2'] = audformat.Scheme()

    # tables of same type without overlap

    for table_type in (audformat.define.IndexType.FILEWISE,
                       audformat.define.IndexType.SEGMENTED):
        db.drop_tables(list(db.tables))
        audformat.testing.add_table(
            db, 'table1', table_type,
            num_files=5, columns=['scheme1'],
        )
        audformat.testing.add_table(
            db, 'table2', table_type,
            num_files=(6, 7, 8, 9, 10), columns='scheme1',
        )
        db['table'] = db['table1'] + db['table2']
        pd.testing.assert_frame_equal(db['table'].get(),
                                      pd.concat([db['table1'].get(),
                                                 db['table2'].get()]))

    # tables of same type with overlap

    db.drop_tables(list(db.tables))
    audformat.testing.add_table(
        db, 'table1', audformat.define.IndexType.FILEWISE,
        num_files=(1, 2), columns='scheme1',
    )
    audformat.testing.add_table(
        db, 'table2', audformat.define.IndexType.FILEWISE,
        num_files=(1,), columns='scheme1',
    )
    db['table2'].df.iloc[0] = np.nan
    db['table'] = db['table1'] + db['table2']
    pd.testing.assert_series_equal(db['table']['scheme1'].get(),
                                   db['table1']['scheme1'].get())
    with pytest.raises(ValueError):
        db['table2'].df.iloc[0] = db['table1'].df.iloc[0]
        db['table'] = db['table1'] + db['table2']

    # filewise with segmented table

    for num_files_1, num_files_2 in (
            (5, 5),
            (5, 4),
            (4, 5),
    ):
        db.drop_tables(list(db.tables))
        audformat.testing.add_table(
            db, 'table1', audformat.define.IndexType.FILEWISE,
            columns='scheme1', num_files=num_files_1,
        )
        audformat.testing.add_table(
            db, 'table2', audformat.define.IndexType.SEGMENTED,
            columns='scheme2', num_files=num_files_2,
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
            db, 'table1', audformat.define.IndexType.SEGMENTED,
            columns='scheme1', num_files=num_files_1,
        )
        audformat.testing.add_table(
            db, 'table2', audformat.define.IndexType.FILEWISE,
            columns='scheme2', num_files=num_files_2,
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
