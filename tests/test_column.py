import re

import pytest
import numpy as np
import pandas as pd

import audformat
import audformat.testing


def test_access():
    db = audformat.testing.create_db()
    for table in db.tables.values():
        for column_id, column in table.columns.items():
            assert table.columns[column_id] == table[column_id]
            assert str(table.columns[column_id]) == str(table[column_id])
            if column.scheme_id is not None:
                assert column.scheme == db.schemes[column.scheme_id]
            else:
                assert column.scheme is None
            if column.rater_id is not None:
                assert column.rater == db.raters[column.rater_id]
            else:
                assert column.rater is None


def test_exceptions():
    column = audformat.Column()
    with pytest.raises(RuntimeError):
        column.set([])
    with pytest.raises(RuntimeError):
        column.get()


@pytest.mark.parametrize(
    'num_files, values', [
        (6, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
        (6, np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])),
        (6, pd.Series(range(6), dtype='float').values,),
    ]
)
def test_filewise(num_files, values):

    db = audformat.testing.create_db(minimal=True)
    audformat.testing.add_table(
        db, 'table', audformat.define.IndexType.FILEWISE, num_files=num_files
    )
    db.schemes['scheme'] = audformat.Scheme(
        dtype=audformat.define.DataType.FLOAT,
    )

    db['table']['column'] = audformat.Column(scheme_id='scheme')

    table = db['table']
    column = db['table']['column']
    column_id = 'column'

    # empty
    series = pd.Series(
        [np.nan] * num_files,
        audformat.filewise_index(table.files),
        name=column_id,
    )
    pd.testing.assert_series_equal(column.get(), series)

    # set single
    series[:] = np.nan
    table.df['column'] = np.nan
    series[0] = values[0]
    index = audformat.filewise_index(table.files[0])
    column.set(values[0], index=index)
    pd.testing.assert_series_equal(column.get(index), series[[0]])
    pd.testing.assert_series_equal(column.get(), series)

    # set slice
    series[:] = np.nan
    table.df['column'] = np.nan
    series[1:-1] = values[1:-1]
    index = audformat.filewise_index(table.files[1:-1])
    column.set(values[1:-1], index=index)
    pd.testing.assert_series_equal(column.get(index), series[1:-1])
    pd.testing.assert_series_equal(column.get(), series)

    # set all
    series[:] = np.nan
    table.df['column'] = np.nan
    series[:] = values
    column.set(values)
    pd.testing.assert_series_equal(column.get(), series)

    # set scalar
    series[:] = np.nan
    table.df['column'] = np.nan
    series[:] = values[0]
    column.set(values[0])
    pd.testing.assert_series_equal(column.get(), series)

    # get segments
    table.df['column'] = values
    index = audformat.segmented_index(
        [db.files[0], db.files[0], db.files[1]],
        starts=['0s', '1s', '0s'],
        ends=['1s', '2s', pd.NaT],
    )
    pd.testing.assert_series_equal(
        column.get(index),
        pd.Series(
            [values[0], values[0], values[1]],
            index=index,
            name='column',
        )
    )

    # try to use segmented index
    with pytest.raises(ValueError):
        index = audformat.segmented_index()
        column.set([], index=index)


def test_get_as_segmented():

    db = pytest.DB

    y = db['files']['bool'].get()
    assert audformat.index_type(y) == audformat.define.IndexType.FILEWISE
    assert not db._files_duration

    # convert to segmented index

    y = db['files']['bool'].get(
        as_segmented=True,
        allow_nat=True,
    )
    assert audformat.index_type(y) == audformat.define.IndexType.SEGMENTED
    assert not db._files_duration
    assert y.index.get_level_values(
        audformat.define.IndexField.END
    ).isna().all()

    # replace NaT with file duration

    y = db['files']['bool'].get(
        as_segmented=True,
        allow_nat=False,
    )
    assert audformat.index_type(y) == audformat.define.IndexType.SEGMENTED
    assert db._files_duration
    assert not y.index.get_level_values(
        audformat.define.IndexField.END
    ).isna().any()

    # reset db

    db._files_duration = {}


@pytest.mark.parametrize(
    'column, map',
    [
        (pytest.DB['files']['label_map_int'], 'int'),
        (pytest.DB['files']['label_map_int'], 'label_map_int'),
        (pytest.DB['files']['label_map_str'], 'prop1'),
        (pytest.DB['segments']['label_map_str'], 'prop2'),
        pytest.param(  # no mappings
            pytest.DB['files']['label'], 'label1',
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(  # no schemes
            pytest.DB['files']['no_scheme'], 'map',
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(  # no labels
            pytest.DB['files']['string'], 'map',
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(  # no labels
            pytest.DB['files']['string'], 'map',
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(  # no labels
            pytest.DB['files']['label_map_str'], 'bad',
            marks=pytest.mark.xfail(raises=ValueError),
        ),
    ]
)
def test_map(column, map):
    result = column.get(map=map)
    expected = column.get()
    mapping = {}
    for key, value in pytest.DB.schemes[column.scheme_id].labels.items():
        if isinstance(value, dict):
            value = value[map]
        mapping[key] = value
    expected = expected.map(mapping)
    expected.name = map
    pd.testing.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    'num_files, num_segments_per_file, values', [
        (3, 2, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
        (3, 2, np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])),
        (3, 2, pd.Series(range(6), dtype='float').values),
    ]
)
def test_segmented(num_files, num_segments_per_file, values):

    db = audformat.testing.create_db(minimal=True)
    audformat.testing.add_table(
        db, 'table', audformat.define.IndexType.SEGMENTED,
        num_files=num_files,
        num_segments_per_file=num_segments_per_file,
    )
    db.schemes['scheme'] = audformat.Scheme(
        dtype=audformat.define.DataType.FLOAT,
    )

    db['table']['column'] = audformat.Column(scheme_id='scheme')

    table = db['table']
    column = db['table']['column']
    column_id = 'column'

    # empty
    series = pd.Series(
        [np.nan] * num_files * num_segments_per_file,
        db.segments,
        name=column_id,
    )
    pd.testing.assert_series_equal(column.get(), series)

    # set single
    series[:] = np.nan
    table.df['column'] = np.nan
    series[0] = values[0]
    index = audformat.segmented_index(
        table.files[0],
        starts=table.starts[0],
        ends=table.ends[0],
    )
    column.set(values[0], index=index)
    pd.testing.assert_series_equal(column.get(index), series[[0]])
    pd.testing.assert_series_equal(column.get(), series)

    # set slice
    series[:] = np.nan
    table.df['column'] = np.nan
    series[1:-1] = values[1:-1]
    index = audformat.segmented_index(
        table.files[1:-1],
        starts=table.starts[1:-1],
        ends=table.ends[1:-1],
    )
    column.set(values[1:-1], index=index)
    pd.testing.assert_series_equal(column.get(index), series[1:-1])
    pd.testing.assert_series_equal(column.get(), series)

    # set all
    series[:] = np.nan
    table.df['column'] = np.nan
    series[:] = values
    column.set(values)
    pd.testing.assert_series_equal(column.get(), series)

    # set scalar
    series[:] = np.nan
    table.df['column'] = np.nan
    series[:] = values[0]
    column.set(values[0])
    pd.testing.assert_series_equal(column.get(), series)

    # set files
    series[:] = np.nan
    table.df['column'] = np.nan
    series[0:num_segments_per_file * 2] = values[-1]
    index = audformat.filewise_index(db.files[:2])
    column.set(values[-1], index=index)
    pd.testing.assert_series_equal(
        column.get(index),
        series[0:num_segments_per_file * 2],
    )

    # set series
    series[:] = np.nan
    table.df['column'] = np.nan
    column.set(series)
    pd.testing.assert_series_equal(column.get(), series)

    # test df
    series[:] = np.nan
    table.df['column'] = np.nan
    pd.testing.assert_series_equal(table.df[column_id], series)


def test_set_labels():
    # Ensure we handle NaN when setting labels
    # from Series and ndarray,
    # see https://github.com/audeering/audformat/issues/89
    db = pytest.DB
    scheme_id = 'label_test'
    db.schemes[scheme_id] = audformat.Scheme(
        labels={
            'h': {'category': 'happy'},
            'n': {'category': 'neutral'},
            'b': {'category': 'bored'},
            'a': {'category': 'angry'},
        }
    )
    db[scheme_id] = audformat.Table(
        audformat.filewise_index(['a', 'b', 'c', 'd', 'e'])
    )
    db[scheme_id][scheme_id] = audformat.Column(scheme_id=scheme_id)
    y = pd.Series([np.NaN, 'b', 'h', 'a', 'n'])
    db[scheme_id][scheme_id].set(y)  # Series
    db[scheme_id][scheme_id].set(y.values)  # ndarray
    # Clean up database
    db.drop_tables(scheme_id)
    db.schemes.pop(scheme_id, None)


def test_set_invalid_values():

    db = pytest.DB
    num = len(db['files'])

    # Values that do match scheme are converted if possible
    table = db['files'].copy()  # use copy as we change values here
    # string -> float
    table['float'].set(['-1.0'] * num)
    error_msg = re.escape("could not convert string to float: 'a'")
    with pytest.raises(ValueError, match=error_msg):
        table['float'].set('a')
    # float -> string
    table['string'].set(1.0)
    error_msg = re.escape('Some value(s) do not match scheme')
    with pytest.raises(ValueError, match=error_msg):
        db['files']['label'].set(1.0)

    with pytest.raises(ValueError):
        db['files']['float'].set(-2.0)

    with pytest.raises(ValueError):
        db['files']['float'].set([-2.0] * num)

    with pytest.raises(ValueError):
        db['files']['float'].set(np.array([-2.0] * num))

    with pytest.raises(ValueError):
        db['files']['float'].set(pd.Series([-2.0] * num))

    with pytest.raises(ValueError):
        db['files']['float'].set(2.0)

    with pytest.raises(ValueError):
        db['files']['float'].set([2.0] * num)

    with pytest.raises(ValueError):
        db['files']['float'].set(np.array([2.0] * num))

    with pytest.raises(ValueError):
        db['files']['float'].set(pd.Series([-2.0] * num))

    with pytest.raises(ValueError):
        db['files']['label'].set('bad')

    with pytest.raises(ValueError):
        db['files']['label'].set(['bad'] * len(db['files']))

    with pytest.raises(ValueError):
        db['files']['label'].set(np.array(['bad'] * len(db['files'])))

    with pytest.raises(ValueError):
        db['files']['label'].set(pd.Series(['bad'] * len(db['files'])))
