import numpy as np
import pandas as pd
import pytest

import audeer

import audformat
import audformat.testing


@pytest.fixture(scope='function')
def mono_db(tmpdir):
    r"""Database with ..."""
    name = 'mono-db'
    path = audeer.mkdir(audeer.path(tmpdir, name))
    db = audformat.Database(name)

    # --- Splits
    db.splits['train'] = audformat.Split('train')
    db.splits['test'] = audformat.Split('test')

    # --- Schemes
    db.schemes['age'] = audformat.Scheme('int', minimum=0)
    db.schemes['height'] = audformat.Scheme('float')
    db.schemes['rating'] = audformat.Scheme('int', labels=[0, 1, 2])
    db.schemes['regression'] = audformat.Scheme('float')
    db.schemes['speaker.weight'] = audformat.Scheme(
        'str',
        labels=['low', 'normal', 'high'],
    )
    db.schemes['winner'] = audformat.Scheme(
        'str',
        labels={
            'w1': {'year': 1995},
            'w2': {'year': 1996},
            'w3': {'year': 1997},
        },
    )
    db.schemes['weather'] = audformat.Scheme(
        'str',
        labels=['cloudy', 'rainy', 'sunny'],
    )

    # --- Misc tables
    index = pd.Index(['s1', 's2', 's3'], name='speaker', dtype='string')
    db['speaker'] = audformat.MiscTable(index)
    db['speaker']['age'] = audformat.Column(scheme_id='age')
    db['speaker']['gender'] = audformat.Column()
    db['speaker']['age'].set([23, np.NaN, 59])
    db['speaker']['gender'].set(['female', '', 'male'])
    db['speaker']['height-with-10y'] = audformat.Column(scheme_id='height')
    db['speaker']['height-with-10y'].set([1.12, 1.45, 1.01])
    db['speaker']['current-height'] = audformat.Column(scheme_id='height')
    db['speaker']['current-height'].set([1.76, 1.95, 1.80])
    db['speaker']['weight'] = audformat.Column(scheme_id='speaker.weight')
    db['speaker']['weight'].set(['normal', 'high', 'low'])

    index = pd.Index(['today', 'yesterday'], name='day', dtype='string')
    db['weather'] = audformat.MiscTable(index)
    db['weather']['weather'] = audformat.Column(scheme_id='weather')
    db['weather']['weather'].set(['cloudy', 'sunny'])

    # --- Schemes with misc tables
    db.schemes['speaker'] = audformat.Scheme('str', labels='speaker')

    # --- Filewise tables
    index = audformat.filewise_index(['f1.wav', 'f2.wav', 'f3.wav'])
    db['files'] = audformat.Table(index)
    db['files']['channel0'] = audformat.Column(scheme_id='speaker')
    db['files']['channel0'].set(['s1', 's2', 's3'])
    db['files']['winner'] = audformat.Column(scheme_id='winner')
    db['files']['winner'].set(['w1', 'w1', 'w2'])
    db['files']['perceived-age'] = audformat.Column(scheme_id='age')
    db['files']['perceived-age'].set([25, 34, 45])

    index = audformat.filewise_index(['f1.wav'])
    db['files.sub'] = audformat.Table(index)
    db['files.sub']['speaker'] = audformat.Column(scheme_id='speaker')
    db['files.sub']['speaker'].set('s1')

    index = audformat.filewise_index(['f1.wav', 'f3.wav'])
    db['gender'] = audformat.Table(index)
    db['gender']['sex'] = audformat.Column()
    db['gender']['sex'].set(['female', 'male'])

    index = audformat.filewise_index(['f1.wav', 'f2.wav'])
    db['rating.train'] = audformat.Table(index, split_id='train')
    db['rating.train']['rating'] = audformat.Column(scheme_id='rating')
    db['rating.train']['rating'].set([0, 1])

    index = audformat.filewise_index(['f3.wav'])
    db['rating.test'] = audformat.Table(index, split_id='test')
    db['rating.test']['rating'] = audformat.Column(scheme_id='rating')
    db['rating.test']['rating'].set([1])

    # --- Segmented tables
    index = audformat.segmented_index(
        ['f1.wav', 'f1.wav', 'f1.wav', 'f2.wav'],
        [0, 0.1, 0.3, 0],
        [0.2, 0.2, 0.5, 0.7],
    )
    db['segments'] = audformat.Table(index)
    db['segments']['rating'] = audformat.Column(scheme_id='rating')
    db['segments']['rating'].set([1, 1, 2, 2])
    db['segments']['winner'] = audformat.Column(scheme_id='winner')
    db['segments']['winner'].set(['w1', 'w1', 'w1', 'w1'])
    db['segments']['regression'] = audformat.Column(scheme_id='regression')
    db['segments']['regression'].set([0.3, 0.2, 0.6, 0.4])

    db.save(path)
    audformat.testing.create_audio_files(db, channels=1, file_duration='1s')

    return db


@pytest.fixture(scope='function')
def stereo_db(tmpdir):
    r"""Database with stereo files and same scheme for both channels.

    It contains two tables,
    in one the same labels are used for both channels,
    and in the other different labels are used.

    """
    name = 'stereo-db'
    path = audeer.mkdir(audeer.path(tmpdir, name))
    db = audformat.Database(name)

    # --- Schemes
    db.schemes['age'] = audformat.Scheme('int', minimum=0)

    # --- Misc tables
    index = pd.Index(['s1', 's2', 's3'], name='speaker', dtype='string')
    db['speaker'] = audformat.MiscTable(index)
    db['speaker']['age'] = audformat.Column(scheme_id='age')
    db['speaker']['gender'] = audformat.Column()
    db['speaker']['age'].set([23, np.NaN, 59])
    db['speaker']['gender'].set(['female', '', 'male'])

    # --- Schemes with misc tables
    db.schemes['speaker'] = audformat.Scheme('str', labels='speaker')

    # --- Filewise tables
    index = audformat.filewise_index(['f1.wav', 'f2.wav', 'f3.wav'])
    db['run1'] = audformat.Table(index)
    db['run1']['channel0'] = audformat.Column(scheme_id='speaker')
    db['run1']['channel1'] = audformat.Column(scheme_id='speaker')
    db['run1']['channel0'].set(['s1', 's2', 's3'])
    db['run1']['channel1'].set(['s3', 's1', 's2'])

    db['run2'] = audformat.Table(index)
    db['run2']['channel0'] = audformat.Column(scheme_id='speaker')
    db['run2']['channel1'] = audformat.Column(scheme_id='speaker')
    db['run2']['channel0'].set(['s1', 's2', 's3'])
    db['run2']['channel1'].set(['s1', 's2', 's3'])

    db['run3'] = audformat.Table(index)
    db['run3']['channel0'] = audformat.Column(scheme_id='speaker')
    db['run3']['channel1'] = audformat.Column(scheme_id='speaker')
    db['run3']['channel0'].set(['s2', 's1', 's3'])
    db['run3']['channel1'].set(['s2', 's1', 's3'])

    db.save(path)
    audformat.testing.create_audio_files(db, channels=2, file_duration='1s')

    return db


@pytest.mark.parametrize(
    'db, schemes, expected',
    [
        (
            'mono_db',
            'non-existing',
            pd.DataFrame([]),
        ),
        (
            'mono_db',
            'weather',
            pd.DataFrame([]),
        ),
        (
            'mono_db',
            'gender',
            pd.concat(
                [
                    pd.Series(
                        ['female', '', 'male'],
                        index=audformat.filewise_index(
                            ['f1.wav', 'f2.wav', 'f3.wav']
                        ),
                        dtype=pd.CategoricalDtype(
                            categories=['female', '', 'male'],
                            ordered=False,
                        ),
                        name='gender',
                    ),
                ],
                axis=1,
            ),
        ),
        (
            'mono_db',
            ['sex', 'gender'],
            pd.concat(
                [
                    pd.Series(
                        ['female', 'male', np.NaN],
                        index=audformat.filewise_index(
                            ['f1.wav', 'f3.wav', 'f2.wav']
                        ),
                        dtype='object',
                        name='sex',
                    ),
                    pd.Series(
                        ['female', '', 'male'],
                        index=audformat.filewise_index(
                            ['f1.wav', 'f2.wav', 'f3.wav']
                        ),
                        dtype=pd.CategoricalDtype(
                            categories=['female', '', 'male'],
                            ordered=False,
                        ),
                        name='gender',
                    ),
                ],
                axis=1,
            )
        ),
        (
            'mono_db',
            ['gender', 'sex', 'non-existing'],
            pd.concat(
                [
                    pd.Series(
                        ['female', '', 'male'],
                        index=audformat.filewise_index(
                            ['f1.wav', 'f2.wav', 'f3.wav']
                        ),
                        dtype=pd.CategoricalDtype(
                            categories=['female', '', 'male'],
                            ordered=False,
                        ),
                        name='gender',
                    ),
                    pd.Series(
                        ['female', np.NaN, 'male'],
                        index=audformat.filewise_index(
                            ['f1.wav', 'f2.wav', 'f3.wav']
                        ),
                        dtype='object',
                        name='sex',
                    ),
                ],
                axis=1,
            )
        ),
        (
            'mono_db',
            'age',
            pd.concat(
                [
                    pd.Series(
                        [23, np.NaN, 59],
                        index=audformat.filewise_index(
                            ['f1.wav', 'f2.wav', 'f3.wav']
                        ),
                        dtype=pd.CategoricalDtype(
                            categories=[23.0, 59.0, 25.0, 34.0, 45.0],
                            ordered=False,
                        ),
                        name='age',
                    ),
                    pd.Series(
                        [25.0, 34.0, 45.0],
                        index=audformat.filewise_index(
                            ['f1.wav', 'f2.wav', 'f3.wav']
                        ),
                        dtype=pd.CategoricalDtype(
                            categories=[23.0, 59.0, 25.0, 34.0, 45.0],
                            ordered=False,
                        ),
                        name='perceived-age',
                    ),
                ],
                axis=1,
            ),
        ),
        (
            'mono_db',
            'height',
            pd.concat(
                [
                    pd.Series(
                        [1.12, 1.45, 1.01],
                        index=audformat.filewise_index(
                            ['f1.wav', 'f2.wav', 'f3.wav']
                        ),
                        dtype=pd.CategoricalDtype(
                            categories=[1.12, 1.45, 1.01, 1.76, 1.95, 1.8],
                            ordered=False,
                        ),
                        name='height-with-10y',
                    ),
                    pd.Series(
                        [1.76, 1.95, 1.80],
                        index=audformat.filewise_index(
                            ['f1.wav', 'f2.wav', 'f3.wav']
                        ),
                        dtype=pd.CategoricalDtype(
                            categories=[1.12, 1.45, 1.01, 1.76, 1.95, 1.8],
                            ordered=False,
                        ),
                        name='current-height',
                    ),
                ],
                axis=1,
            ),
        ),
        (
            'mono_db',
            'winner',
            pd.concat(
                [
                    pd.Series(
                        ['w1', 'w1', 'w2', 'w1', 'w1', 'w1', 'w1'],
                        index=audformat.utils.union(
                            [
                                audformat.filewise_index(
                                    ['f1.wav', 'f2.wav', 'f3.wav']
                                ),
                                audformat.segmented_index(
                                    ['f1.wav', 'f1.wav', 'f1.wav', 'f2.wav'],
                                    [0, 0.1, 0.3, 0],
                                    [0.2, 0.2, 0.5, 0.7],
                                ),
                            ]
                        ),
                        dtype=pd.CategoricalDtype(
                            ['w1', 'w2', 'w3'],
                            ordered=False,
                        ),
                        name='winner',
                    ),
                ],
                axis=1,
            ),
        ),
        (
            'mono_db',
            'year',
            pd.concat(
                [
                    pd.Series(
                        [1995, 1995, 1996, 1995, 1995, 1995, 1995],
                        index=audformat.utils.union(
                            [
                                audformat.filewise_index(
                                    ['f1.wav', 'f2.wav', 'f3.wav']
                                ),
                                audformat.segmented_index(
                                    ['f1.wav', 'f1.wav', 'f1.wav', 'f2.wav'],
                                    [0, 0.1, 0.3, 0],
                                    [0.2, 0.2, 0.5, 0.7],
                                ),
                            ]
                        ),
                        dtype=pd.CategoricalDtype(
                            [1995, 1996, 1997],
                            ordered=False,
                        ),
                        name='year',
                    ),
                ],
                axis=1,
            ),
        ),
        (
            'mono_db',
            'rating',
            pd.concat(
                [
                    pd.Series(
                        [1, 0, 1, 1, 1, 2, 2],
                        index=audformat.utils.union(
                            [
                                audformat.filewise_index(
                                    ['f3.wav', 'f1.wav', 'f2.wav']
                                ),
                                audformat.segmented_index(
                                    ['f1.wav', 'f1.wav', 'f1.wav', 'f2.wav'],
                                    [0, 0.1, 0.3, 0],
                                    [0.2, 0.2, 0.5, 0.7],
                                ),
                            ]
                        ),
                        dtype=pd.CategoricalDtype(
                            [0, 1, 2],
                            ordered=False,
                        ),
                        name='rating',
                    ),
                ],
                axis=1,
            ),
        ),
        (
            'mono_db',
            'regression',
            pd.concat(
                [
                    pd.Series(
                        [0.3, 0.2, 0.6, 0.4],
                        index=audformat.segmented_index(
                            ['f1.wav', 'f1.wav', 'f1.wav', 'f2.wav'],
                            [0, 0.1, 0.3, 0],
                            [0.2, 0.2, 0.5, 0.7],
                        ),
                        dtype='float',
                        name='regression',
                    ),
                ],
                axis=1,
            ),
        ),
    ]
)
def test_database_get(request, db, schemes, expected):
    db = request.getfixturevalue(db)
    df = db.get(schemes)
    print(f'{df=}')
    print(f'{expected=}')
    pd.testing.assert_frame_equal(df, expected)


# Aggregate functions
# that define how to store the labels in the data frame

def rename_column(y, db, table_id, column_id):
    y.name = f'{y.name}-{table_id}-{column_id}'
    return y

def select_run1(y, db, table_id, column_id):
    if table_id != 'run1':
        if audformat.is_filewise_index(y.index):
            index = audformat.filewise_index()
        else:
            index = audformat.segmented_index()
        y = pd.Series(index=index, name=y.name, dtype=y.dtype)
    return y

def select_run2(y, db, table_id, column_id):
    if table_id != 'run2':
        if audformat.is_filewise_index(y.index):
            index = audformat.filewise_index()
        else:
            index = audformat.segmented_index()
        y = pd.Series(index=index, name=y.name, dtype=y.dtype)
    return y

def average_rating_segments(y, db, table_id, column_id):
    name = y.name
    dtype = y.dtype
    if table_id == 'segments':
        files = y.index.get_level_values('file').unique()
        data = [y.loc[file].mode().values[0] for file in files]
        index = audformat.filewise_index(files)
    else:
        data = []
        index = audformat.filewise_index()
    return pd.Series(data, index=index, name=name, dtype=dtype)

def add_db_name(y, db, table_id, column_id):
    y.name = f'{y.name}-{db.name}'
    return y

def rename_sex_to_gender_without_dtype_adjustment(y, db, table_id, column_id):
    if y.name == 'sex':
        y.name = 'gender'
    return y

def rename_sex_to_gender(y, db, table_id, column_id):
    if y.name == 'sex':
        y.name = 'gender'
        # Make sure it uses the correct dtype as well
        dtype = pd.CategoricalDtype(
            categories=['female', '', 'male'],
            ordered=False,
        )
        y = y.astype(dtype)
    return y

@pytest.mark.parametrize(
    'db, schemes, aggregate_function, expected',
    [
        (
            # Choose different names for each run and channel:
            # run1, channel0: ['female', '', 'male']
            # run1, channel1: ['male', 'female', '']
            # run2, channel0: ['female', '', 'male']
            # run2, channel1: ['female', '', 'male']
            # run3, channel0: ['', 'female', 'male']
            # run3, channel1: ['', 'female', 'male']
            'stereo_db',
            'gender',
            rename_column,
            pd.concat(
                [
                    pd.Series(
                        ['female', '', 'male'],
                        index=audformat.filewise_index(
                            ['f1.wav', 'f2.wav', 'f3.wav']
                        ),
                        dtype=pd.CategoricalDtype(
                            categories=['female', '', 'male'],
                            ordered=False,
                        ),
                        name='gender-run1-channel0',
                    ),
                    pd.Series(
                        ['male', 'female', ''],
                        index=audformat.filewise_index(
                            ['f1.wav', 'f2.wav', 'f3.wav']
                        ),
                        dtype=pd.CategoricalDtype(
                            categories=['female', '', 'male'],
                            ordered=False,
                        ),
                        name='gender-run1-channel1',
                    ),
                    pd.Series(
                        ['female', '', 'male'],
                        index=audformat.filewise_index(
                            ['f1.wav', 'f2.wav', 'f3.wav']
                        ),
                        dtype=pd.CategoricalDtype(
                            categories=['female', '', 'male'],
                            ordered=False,
                        ),
                        name='gender-run2-channel0',
                    ),
                    pd.Series(
                        ['female', '', 'male'],
                        index=audformat.filewise_index(
                            ['f1.wav', 'f2.wav', 'f3.wav']
                        ),
                        dtype=pd.CategoricalDtype(
                            categories=['female', '', 'male'],
                            ordered=False,
                        ),
                        name='gender-run2-channel1',
                    ),
                    pd.Series(
                        ['', 'female', 'male'],
                        index=audformat.filewise_index(
                            ['f1.wav', 'f2.wav', 'f3.wav']
                        ),
                        dtype=pd.CategoricalDtype(
                            categories=['female', '', 'male'],
                            ordered=False,
                        ),
                        name='gender-run3-channel0',
                    ),
                    pd.Series(
                        ['', 'female', 'male'],
                        index=audformat.filewise_index(
                            ['f1.wav', 'f2.wav', 'f3.wav']
                        ),
                        dtype=pd.CategoricalDtype(
                            categories=['female', '', 'male'],
                            ordered=False,
                        ),
                        name='gender-run3-channel1',
                    ),
                ],
                axis=1,
            ),
        ),
        pytest.param(
            # Fail as we select runs with different labels:
            # run1, channel0: ['female', '', 'male']
            # run1, channel1: ['male', 'female', '']
            'stereo_db',
            'gender',
            select_run1,
            None,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        (
            # Select run with identical labels:
            # run2, channel0: ['female', '', 'male']
            # run2, channel1: ['female', '', 'male']
            'stereo_db',
            'gender',
            select_run2,
            pd.concat(
                [
                    pd.Series(
                        ['female', '', 'male'],
                        index=audformat.filewise_index(
                            ['f1.wav', 'f2.wav', 'f3.wav']
                        ),
                        dtype=pd.CategoricalDtype(
                            categories=['female', '', 'male'],
                            ordered=False,
                        ),
                        name='gender',
                    ),
                ],
                axis=1,
            ),
        ),
        (
            # Select segments table
            # and return maxvote for each file
            # f1: 1, 1, 2
            # f2: 2
            'mono_db',
            'rating',
            average_rating_segments,
            pd.concat(
                [
                    pd.Series(
                        [1, 2],
                        index=audformat.filewise_index(
                            ['f1.wav', 'f2.wav']
                        ),
                        dtype=pd.CategoricalDtype(
                            categories=[0, 1, 2],
                            ordered=False,
                        ),
                        name='rating',
                    ),
                ],
                axis=1,
            ),
        ),
        (
            # Add name of database to column name
            'mono_db',
            'gender',
            add_db_name,
            pd.concat(
                [
                    pd.Series(
                        ['female', '', 'male'],
                        index=audformat.filewise_index(
                            ['f1.wav', 'f2.wav', 'f3.wav']
                        ),
                        dtype=pd.CategoricalDtype(
                            categories=['female', '', 'male'],
                            ordered=False,
                        ),
                        name='gender-mono-db',
                    ),
                ],
                axis=1,
            ),
        ),
        pytest.param(
            # Rename sex scheme to gender,
            # however as we don't fix the dtype
            # it will raise an error
            'mono_db',
            ['sex', 'gender'],
            rename_sex_to_gender_without_dtype_adjustment,
            None,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        (
            # Rename sex scheme to gender
            'mono_db',
            ['gender', 'sex'],
            rename_sex_to_gender,
            pd.concat(
                [
                    pd.Series(
                        ['female', '', 'male'],
                        index=audformat.filewise_index(
                            ['f1.wav', 'f2.wav', 'f3.wav']
                        ),
                        dtype=pd.CategoricalDtype(
                            categories=['female', '', 'male'],
                            ordered=False,
                        ),
                        name='gender',
                    ),
                ],
                axis=1,
            ),
        ),
        (
            # Rename sex scheme to gender,
            # the order how we collect schemes
            # influences the order of the inex
            'mono_db',
            ['sex', 'gender'],
            rename_sex_to_gender,
            pd.concat(
                [
                    pd.Series(
                        ['female', 'male', ''],
                        index=audformat.filewise_index(
                            ['f1.wav', 'f3.wav', 'f2.wav']
                        ),
                        dtype=pd.CategoricalDtype(
                            categories=['female', '', 'male'],
                            ordered=False,
                        ),
                        name='gender',
                    ),
                ],
                axis=1,
            ),
        ),
    ]
)
def test_database_get_aggregate_function(
        request,
        db,
        schemes,
        aggregate_function,
        expected,
):
    db = request.getfixturevalue(db)
    df = db.get(schemes, aggregate_function=aggregate_function)
    print(f'{df=}')
    print(f'{expected=}')
    pd.testing.assert_frame_equal(df, expected)


@pytest.mark.parametrize(
    'db, schemes, tables, splits, expected',
    [
        (
            'mono_db',
            'gender',
            [],
            None,
            pd.DataFrame(
                [],
                columns=['gender'],
                index=pd.Index([], dtype='string'),
                dtype=pd.CategoricalDtype(
                    categories=['female', '', 'male'],
                    ordered=False,
                ),
            ),
        ),
        (
            'mono_db',
            'gender',
            None,
            [],
            pd.DataFrame(
                [],
                columns=['gender'],
                index=pd.Index([], dtype='string'),
                dtype=pd.CategoricalDtype(
                    categories=['female', '', 'male'],
                    ordered=False,
                ),
            ),
        ),
        (
            'mono_db',
            'gender',
            [],
            [],
            pd.DataFrame(
                [],
                columns=['gender'],
                index=pd.Index([], dtype='string'),
                dtype=pd.CategoricalDtype(
                    categories=['female', '', 'male'],
                    ordered=False,
                ),
            ),
        ),
        (
            'mono_db',
            'gender',
            'non-existing',
            None,
            pd.DataFrame(
                [],
                columns=['gender'],
                index=pd.Index([], dtype='string'),
                dtype=pd.CategoricalDtype(
                    categories=['female', '', 'male'],
                    ordered=False,
                ),
            ),
        ),
        (
            'mono_db',
            'gender',
            None,
            'non-existing',
            pd.DataFrame(
                [],
                columns=['gender'],
                index=pd.Index([], dtype='string'),
                dtype=pd.CategoricalDtype(
                    categories=['female', '', 'male'],
                    ordered=False,
                ),
            ),
        ),
        (
            'mono_db',
            'gender',
            'non-existing',
            'non-existing',
            pd.DataFrame(
                [],
                columns=['gender'],
                index=pd.Index([], dtype='string'),
                dtype=pd.CategoricalDtype(
                    categories=['female', '', 'male'],
                    ordered=False,
                ),
            ),
        ),
        (
            'mono_db',
            'gender',
            ['gender'],
            ['train'],
            pd.DataFrame(
                [],
                columns=['gender'],
                index=pd.Index([], dtype='string'),
                dtype=pd.CategoricalDtype(
                    categories=['female', '', 'male'],
                    ordered=False,
                ),
            ),
        ),
        (
            'mono_db',
            'gender',
            ['gender'],
            None,
            pd.DataFrame(
                ['female', 'male'],
                columns=['gender'],
                index=audformat.filewise_index(['f1.wav', 'f3.wav']),
                dtype=pd.CategoricalDtype(
                    categories=['female', '', 'male'],
                    ordered=False,
                ),
            ),
        ),
        (
            'mono_db',
            'rating',
            'segments',
            None,
            pd.concat(
                [
                    pd.Series(
                        [1, 1, 2, 2],
                        index=audformat.segmented_index(
                            ['f1.wav', 'f1.wav', 'f1.wav', 'f2.wav'],
                            [0, 0.1, 0.3, 0],
                            [0.2, 0.2, 0.5, 0.7],
                        ),
                        dtype=pd.CategoricalDtype(
                            [0, 1, 2],
                            ordered=False,
                        ),
                        name='rating',
                    ),
                ],
                axis=1,
            ),
        ),
        (
            'mono_db',
            'rating',
            None,
            'train',
            pd.concat(
                [
                    pd.Series(
                        [0, 1],
                        index=audformat.filewise_index(
                            ['f1.wav', 'f2.wav']
                        ),
                        dtype=pd.CategoricalDtype(
                            [0, 1, 2],
                            ordered=False,
                        ),
                        name='rating',
                    ),
                ],
                axis=1,
            ),
        ),
        (
            'mono_db',
            'rating',
            ['rating.train', 'rating.test'],
            'train',
            pd.concat(
                [
                    pd.Series(
                        [0, 1],
                        index=audformat.filewise_index(
                            ['f1.wav', 'f2.wav']
                        ),
                        dtype=pd.CategoricalDtype(
                            [0, 1, 2],
                            ordered=False,
                        ),
                        name='rating',
                    ),
                ],
                axis=1,
            ),
        ),
        (
            'mono_db',
            'rating',
            None,
            ['train', 'test'],
            pd.concat(
                [
                    pd.Series(
                        [1, 0, 1],
                        index=audformat.filewise_index(
                            ['f3.wav', 'f1.wav', 'f2.wav']
                        ),
                        dtype=pd.CategoricalDtype(
                            [0, 1, 2],
                            ordered=False,
                        ),
                        name='rating',
                    ),
                ],
                axis=1,
            ),
        ),
        (
            'mono_db',
            'rating',
            ['rating.train', 'rating.test'],
            None,
            pd.concat(
                [
                    pd.Series(
                        [1, 0, 1],
                        index=audformat.filewise_index(
                            ['f3.wav', 'f1.wav', 'f2.wav']
                        ),
                        dtype=pd.CategoricalDtype(
                            [0, 1, 2],
                            ordered=False,
                        ),
                        name='rating',
                    ),
                ],
                axis=1,
            ),
        ),
        (
            'mono_db',
            'rating',
            ['rating.train', 'rating.test'],
            ['train', 'test'],
            pd.concat(
                [
                    pd.Series(
                        [1, 0, 1],
                        index=audformat.filewise_index(
                            ['f3.wav', 'f1.wav', 'f2.wav']
                        ),
                        dtype=pd.CategoricalDtype(
                            [0, 1, 2],
                            ordered=False,
                        ),
                        name='rating',
                    ),
                ],
                axis=1,
            ),
        ),
    ]
)
def test_database_get_limit_search(
        request,
        db,
        schemes,
        tables,
        splits,
        expected,
):
    db = request.getfixturevalue(db)
    df = db.get(schemes, tables=tables, splits=splits)
    pd.testing.assert_frame_equal(df, expected)


def test_database_get_errors():

    # Scheme with different categorical dtypes
    db = audformat.Database('db')
    db.schemes['label'] = audformat.Scheme('int', labels=[0, 1])
    db['speaker'] = audformat.MiscTable(
        pd.Index(['s1', 's2'], dtype='string', name='speaker')
    )
    db['speaker']['label'] = audformat.Column()
    db['speaker']['label'].set([1.0, 1.0])
    db['files'] = audformat.Table(audformat.filewise_index(['f1', 'f2']))
    db['files']['label'] = audformat.Column(scheme_id='label')
    db['files']['label'].set([0, 1])
    db['other'] = audformat.Table(audformat.filewise_index(['f1', 'f2']))
    db.schemes['speaker'] = audformat.Scheme('str', labels='speaker')
    db['other']['speaker'] = audformat.Column(scheme_id='speaker')
    db['other']['speaker'].set(['s1', 's2'])
    error_msg = 'All categorical data must have the same dtype.'
    with pytest.raises(ValueError, match=error_msg):
        db.get('label')
