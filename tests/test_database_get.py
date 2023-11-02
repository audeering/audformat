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
    db.schemes['selection'] = audformat.Scheme('int', labels=[0, 1])
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
    db['speaker']['age'].set([23, np.NaN, 59])
    db['speaker']['gender'] = audformat.Column()
    db['speaker']['gender'].set(['female', '', 'male'])
    db['speaker']['height-with-10y'] = audformat.Column(scheme_id='height')
    db['speaker']['height-with-10y'].set([1.12, 1.45, 1.01])
    db['speaker']['current-height'] = audformat.Column(scheme_id='height')
    db['speaker']['current-height'].set([1.76, 1.95, 1.80])
    db['speaker']['weight'] = audformat.Column(scheme_id='speaker.weight')
    db['speaker']['weight'].set(['normal', 'high', 'low'])
    db['speaker']['selection'] = audformat.Column()
    db['speaker']['selection'].set([1.0, 1.0, 1.0])

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
    db['other'] = audformat.Table(index)
    db['other']['sex'] = audformat.Column()
    db['other']['sex'].set(['female', 'male'])
    db['other']['weight'] = audformat.Column()
    db['other']['weight'].set([87, 86])
    db['other']['selection'] = audformat.Column(scheme_id='selection')
    db['other']['selection'].set([1, 1])

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
    db['speaker']['age'].set([23, np.NaN, 59])
    db['speaker']['gender'] = audformat.Column()
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
    db['run3']['channel1'].set(['s2', 's2', 's3'])

    db.save(path)
    audformat.testing.create_audio_files(db, channels=2, file_duration='1s')

    return db


@pytest.mark.parametrize(
    'db, scheme, additional_schemes, expected',
    [
        (
            'mono_db',
            'non-existing',
            [],
            pd.DataFrame(
                {
                    'non-existing': [],
                },
                index=audformat.filewise_index(),
                dtype='object',
            ),
        ),
        (
            'mono_db',
            'weather',
            [],
            pd.DataFrame(
                {
                    'weather': [],
                },
                index=audformat.filewise_index(),
                dtype='object',
            ),
        ),
        (
            'mono_db',
            'gender',
            [],
            pd.DataFrame(
                {
                    'gender': ['female', '', 'male'],
                },
                index=audformat.filewise_index(
                    ['f1.wav', 'f2.wav', 'f3.wav']
                ),
                dtype='string',
            ),
        ),
        (
            'mono_db',
            'sex',
            ['gender'],
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
                        dtype='string',
                        name='gender',
                    ),
                ],
                axis=1,
            ),
        ),
        (
            'mono_db',
            'gender',
            ['sex', 'non-existing'],
            pd.concat(
                [
                    pd.Series(
                        ['female', '', 'male'],
                        index=audformat.filewise_index(
                            ['f1.wav', 'f2.wav', 'f3.wav']
                        ),
                        dtype='string',
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
                    pd.Series(
                        [],
                        index=audformat.filewise_index(),
                        dtype='object',
                        name='non-existing',
                    ),
                ],
                axis=1,
            ),
        ),
        (
            'mono_db',
            'winner',
            [],
            pd.DataFrame(
                {
                    'winner': ['w1', 'w1', 'w2', 'w1', 'w1', 'w1', 'w1'],
                },
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
            ),
        ),
        (
            'mono_db',
            'year',
            [],
            pd.DataFrame(
                {
                    'year': [1995, 1995, 1996, 1995, 1995, 1995, 1995],
                },
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
                dtype='Int64',
            ),
        ),
        (
            'mono_db',
            'rating',
            [],
            pd.DataFrame(
                {
                    'rating': [1, 0, 1, 1, 1, 2, 2],
                },
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
            ),
        ),
        (
            'mono_db',
            'regression',
            [],
            pd.DataFrame(
                {
                    'regression': [0.3, 0.2, 0.6, 0.4],
                },
                index=audformat.segmented_index(
                    ['f1.wav', 'f1.wav', 'f1.wav', 'f2.wav'],
                    [0, 0.1, 0.3, 0],
                    [0.2, 0.2, 0.5, 0.7],
                ),
                dtype='float',
            ),
        ),
        (
            'mono_db',
            'selection',
            [],
            pd.DataFrame(
                {
                    'selection': [1, 1, 1],
                },
                index=audformat.filewise_index(
                    ['f1.wav', 'f2.wav', 'f3.wav']
                ),
                dtype=pd.CategoricalDtype(
                    [1, 0],
                    ordered=False,
                ),
            ),
        ),
    ]
)
def test_database_get(request, db, scheme, additional_schemes, expected):
    db = request.getfixturevalue(db)
    df = db.get(scheme, additional_schemes)
    pd.testing.assert_frame_equal(df, expected)


# Aggregate functions
# that define how to store the labels in the data frame

def add_db_name(y, db, table_id, column_id, label_id):
    y.name = f'{y.name}-{db.name}'
    return y

def average_rating_segments(y, db, table_id, column_id, label_id):
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

def column_names(y, db, table_id, column_id, label_id):
    y.name = column_id
    return y

def rename_column(y, db, table_id, column_id, label_id):
    y.name = f'{y.name}-{table_id}-{column_id}'
    return y

def rename_sex_to_gender(y, *args):
    if y.name == 'sex':
        y.name = 'gender'
        # Make sure it uses the correct dtype as well
        y = y.astype('string')
    return y

def rename_sex_to_gender_without_dtype_adjustment(y, *args):
    if y.name == 'sex':
        y.name = 'gender'
    return y

def select_column(y, db, table_id, column_id, label_id, column):
    if column_id != column:
        y = y[0:0]
    return y

def select_label(y, db, table_id, column_id, label_id, label):
    if label_id != label:
        y = y[0:0]
    return y

def select_table(y, db, table_id, column_id, label_id, table):
    if table_id != table:
        y = y[0:0]
    return y

@pytest.mark.parametrize(
    'db, schemes, aggregate_function, modify_function, expected',
    [

        # Tests based on `mono_db`,
        # with the following tables, columns
        # matching the scheme `age`.
        #
        # `files`
        # | file   | channel0 (age) | perceived-age |
        # | ------ | -------------- | ------------- |
        # | f1.wav |             23 |            25 |
        # | f2.wav |                |            34 |
        # | f3.wav |             59 |            45 |
        #
        # `files.sub`
        # | file   | speaker (age) |
        # | ------ | ------------- |
        # | f1.wav |            23 |
        #
        (
            # Select first value (based on table/column order)
            #
            # files, age: 23, NaN, 59
            'mono_db',
            'age',
            lambda y: y[0],
            None,
            pd.DataFrame(
                {
                    # NOTE: 34 as second value might not be
                    # what a user expects.
                    # Using a modifier to select a column
                    # is the safer choice.
                    'age': [23, 34, 59],
                },
                index=audformat.filewise_index(
                    ['f1.wav', 'f2.wav', 'f3.wav']
                ),
                dtype='Int64',
            ),
        ),
        (
            # Select second value (based on table/column order)
            #
            # files, perceived age: 25, 34, 45
            'mono_db',
            'age',
            lambda y: y[1],
            None,
            pd.DataFrame(
                {
                    'age': [25, 34, 45],
                },
                index=audformat.filewise_index(
                    ['f1.wav', 'f2.wav', 'f3.wav']
                ),
                dtype='Int64',
            ),
        ),
        (
            # Return all columns using column name
            #
            # files, channel0:      23, NaN, 59
            # files, perceived-age: 25, 34, 45
            # files.sub, speaker:   23, NaN, NaN
            'mono_db',
            'age',
            None,
            column_names,
            pd.DataFrame(
                {
                    'channel0': [23, np.NaN, 59],
                    'perceived-age': [25, 34, 45],
                    'speaker': [23, np.NaN, np.NaN],
                },
                index=audformat.filewise_index(
                    ['f1.wav', 'f2.wav', 'f3.wav']
                ),
                dtype='Int64',
            ),
        ),

        # Tests based on `mono_db`,
        # with the following tables, columns
        # matching the scheme `height`.
        #
        # `files`
        # | file   | channel0 (height-with-10y) | channel0 (current-height) |
        # | ------ | -------------------------- | ------------------------- |
        # | f1.wav |                       1.12 |                      1.76 |
        # | f2.wav |                       1.45 |                      1.95 |
        # | f3.wav |                       1.01 |                      1.80 |
        #
        (
            'mono_db',
            'height',
            None,
            lambda *args: select_label(*args, 'current-height'),
            pd.DataFrame(
                {
                    'height': [1.76, 1.95, 1.80],
                },
                index=audformat.filewise_index(
                    ['f1.wav', 'f2.wav', 'f3.wav']
                ),
                dtype='float',
            ),
        ),

        # Tests based on `mono_db`,
        # with the following tables, columns
        # matching the scheme `rating`.
        #
        # `rating.train`
        # | file   | rating |
        # | ------ | ------ |
        # | f1.wav |      0 |
        # | f2.wav |      1 |
        #
        # `rating.test`
        # | file   | rating |
        # | ------ | ------ |
        # | f3.wav |      1 |
        #
        # `segments`
        # | file   | start |   end | rating |
        # | ------ | ----- | ----- | ------ |
        # | f1.wav |   0.0 |   0.2 |      1 |
        # | f1.wav |   0.1 |   0.2 |      1 |
        # | f1.wav |   0.3 |   0.5 |      2 |
        # | f2.wav |   0.0 |   0.7 |      2 |
        #
        (
            # Select `segments` table
            # and return maxvote for each file
            #
            # rating: [1, 2]
            #
            'mono_db',
            'rating',
            None,
            average_rating_segments,
            pd.DataFrame(
                {
                    'rating': [1, 2],
                },
                index=audformat.filewise_index(
                    ['f1.wav', 'f2.wav']
                ),
                dtype=pd.CategoricalDtype(
                    categories=[0, 1, 2],
                    ordered=False,
                ),
            ),
        ),

        # Tests based on `mono_db`,
        # with the following tables, columns
        # matching the schemes `gender` and `sex`
        # (considering scheme mappings and column names).
        #
        # `files`
        # | file   | channel0 (gender) |
        # | ------ | ----------------- |
        # | f1.wav | female            |
        # | f2.wav |                   |
        # | f3.wav | male              |
        #
        # `files.sub`
        # | file   | speaker (gender) |
        # | ------ | ---------------- |
        # | f1.wav | female           |
        #
        # `other`
        # | file   | sex    |
        # | ------ | ------ |
        # | f1.wav | female |
        # | f3.wav | male   |
        #
        (
            # Add name of database to column name
            #
            # gender-mono-db: ['female', '', 'male']
            #
            'mono_db',
            'gender',
            None,
            add_db_name,
            pd.DataFrame(
                {
                    'gender-mono-db': ['female', '', 'male'],
                },
                index=audformat.filewise_index(
                    ['f1.wav', 'f2.wav', 'f3.wav']
                ),
                dtype='string',
            ),
        ),
        pytest.param(
            # Rename sex scheme to gender
            #
            # As we don't fix the dtype
            # it will raise an error
            #
            'mono_db',
            ['sex', 'gender'],
            None,
            rename_sex_to_gender_without_dtype_adjustment,
            None,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        (
            # Rename sex scheme to gender
            #
            # gender: ['female', '', 'male']
            #
            'mono_db',
            ['gender', 'sex'],
            None,
            rename_sex_to_gender,
            pd.DataFrame(
                {
                    'gender': ['female', '', 'male'],
                },
                index=audformat.filewise_index(
                    ['f1.wav', 'f2.wav', 'f3.wav']
                ),
                dtype='string',
            ),
        ),
        (
            # Rename sex scheme to gender
            #
            # The order how we collect schemes
            # influences the order of the index.
            #
            # gender: ['female', 'male', '']
            #
            'mono_db',
            ['sex', 'gender'],
            None,
            rename_sex_to_gender,
            pd.DataFrame(
                {
                    'gender': ['female', 'male', ''],
                },
                index=audformat.filewise_index(
                    ['f1.wav', 'f3.wav', 'f2.wav']
                ),
                dtype='string',
            ),
        ),

        # Tests based on `stereo_db,
        # with the following tables, columns
        # matching the scheme `gender`.
        # All have [f1.wav, f2.wav, f3.wav] as index
        #
        # | table | column   | values                       |
        # | ----- | -------- | ---------------------------- |
        # | run1  | channel0 | ['female', '',       'male'] |
        # | run1  | channel1 | ['male',   'female', ''    ] |
        # | run2  | channel0 | ['female', '',       'male'] |
        # | run2  | channel1 | ['female', '',       'male'] |
        # | run3  | channel0 | ['',       'female', 'male'] |
        # | run3  | channel1 | ['',       '',       'male'] |
        #
        (
            # maxvote
            #
            # gender: ['female', '', 'male']
            #
            'stereo_db',
            'gender',
            lambda y: y.mode()[0],
            None,
            pd.DataFrame(
                {
                    'gender': ['female', '', 'male'],
                },
                index=audformat.filewise_index(
                    ['f1.wav', 'f2.wav', 'f3.wav']
                ),
                dtype='string',
            ),
        ),
        (
            # Rename scheme column based on table and column
            #
            # gender-run1-channel0: ['female', '', 'male']
            # gender-run1-channel1: ['male', 'female', '']
            # gender-run2-channel0: ['female', '', 'male']
            # gender-run2-channel1: ['female', '', 'male']
            # gender-run3-channel0: ['', 'female', 'male']
            # gender-run3-channel1: ['', '', 'male']
            #
            'stereo_db',
            'gender',
            None,
            rename_column,
            pd.DataFrame(
                {
                    'gender-run1-channel0': ['female', '', 'male'],
                    'gender-run1-channel1': ['male', 'female', ''],
                    'gender-run2-channel0': ['female', '', 'male'],
                    'gender-run2-channel1': ['female', '', 'male'],
                    'gender-run3-channel0': ['', 'female', 'male'],
                    'gender-run3-channel1': ['', '', 'male'],
                },
                index=audformat.filewise_index(
                    ['f1.wav', 'f2.wav', 'f3.wav']
                ),
                dtype='string',
            ),
        ),
        pytest.param(
            # Fail as we select runs with different labels
            #
            # run1, channel0: ['female', '', 'male']
            # run1, channel1: ['male', 'female', '']
            #
            # gender: ValueError
            #
            'stereo_db',
            'gender',
            None,
            lambda *args: select_table(*args, 'run1'),
            None,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        (
            # Select run with identical labels
            #
            # run2, channel0: ['female', '', 'male']
            # run2, channel1: ['female', '', 'male']
            #
            # gender: ['female', '', 'male']
            #
            'stereo_db',
            'gender',
            None,
            lambda *args: select_table(*args, 'run2'),
            pd.DataFrame(
                {
                    'gender': ['female', '', 'male'],
                },
                index=audformat.filewise_index(
                    ['f1.wav', 'f2.wav', 'f3.wav']
                ),
                dtype='string',
            ),
        ),
        (
            # Select channel0 and return maxvote over runs
            #
            # run1, channel0: ['female', '',       'male']
            # run2, channel0: ['female', '',       'male']
            # run3, channel0: ['',       'female', 'male']
            #
            # gender: ['female', '', 'male']
            #
            'stereo_db',
            'gender',
            lambda y: y.mode()[0],
            lambda *args: select_column(*args, 'channel0'),
            pd.DataFrame(
                {
                    'gender': ['female', '', 'male'],
                },
                index=audformat.filewise_index(
                    ['f1.wav', 'f2.wav', 'f3.wav']
                ),
                dtype='string',
            ),
        ),

    ]
)
def test_database_get_aggregate_and_modify_function(
        request,
        db,
        schemes,
        aggregate_function,
        modify_function,
        expected,
):
    db = request.getfixturevalue(db)
    df = db.get(
        schemes,
        aggregate_function=aggregate_function,
        modify_function=modify_function,
    )
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
                {
                    'gender': [],
                },
                index=audformat.filewise_index(),
                dtype='string',
            ),
        ),
        (
            'mono_db',
            'gender',
            None,
            [],
            pd.DataFrame(
                {
                    'gender': [],
                },
                index=audformat.filewise_index(),
                dtype='string',
            ),
        ),
        (
            'mono_db',
            'gender',
            [],
            [],
            pd.DataFrame(
                {
                    'gender': [],
                },
                index=audformat.filewise_index(),
                dtype='string',
            ),
        ),
        (
            'mono_db',
            'gender',
            'non-existing',
            None,
            pd.DataFrame(
                {
                    'gender': [],
                },
                index=audformat.filewise_index(),
                dtype='string',
            ),
        ),
        (
            'mono_db',
            'gender',
            None,
            'non-existing',
            pd.DataFrame(
                {
                    'gender': [],
                },
                index=audformat.filewise_index(),
                dtype='string',
            ),
        ),
        (
            'mono_db',
            ['gender', 'sex'],
            None,
            'non-existing',
            pd.concat(
                [
                    pd.Series(
                        [],
                        index=audformat.filewise_index(),
                        dtype='string',
                        name='gender',
                    ),
                    pd.Series(
                        [],
                        index=audformat.filewise_index(),
                        dtype='object',
                        name='sex',
                    ),
                ],
                axis=1,
            ),
        ),
        (
            'mono_db',
            'gender',
            'non-existing',
            'non-existing',
            pd.DataFrame(
                {
                    'gender': [],
                },
                index=audformat.filewise_index(),
                dtype='string',
            ),
        ),
        (
            'mono_db',
            'gender',
            ['other'],
            ['train'],
            pd.DataFrame(
                {
                    'gender': [],
                },
                index=audformat.filewise_index(),
                dtype='string',
            ),
        ),
        (
            'mono_db',
            'gender',
            ['other'],
            None,
            pd.DataFrame(
                {
                    'gender': ['female', 'male'],
                },
                index=audformat.filewise_index(['f1.wav', 'f3.wav']),
                dtype='string',
            ),
        ),
        (
            'mono_db',
            'rating',
            'segments',
            None,
            pd.DataFrame(
                {
                    'rating': [1, 1, 2, 2],
                },
                index=audformat.segmented_index(
                    ['f1.wav', 'f1.wav', 'f1.wav', 'f2.wav'],
                    [0, 0.1, 0.3, 0],
                    [0.2, 0.2, 0.5, 0.7],
                ),
                dtype=pd.CategoricalDtype(
                    [0, 1, 2],
                    ordered=False,
                ),
            ),
        ),
        (
            'mono_db',
            'rating',
            None,
            'train',
            pd.DataFrame(
                {
                    'rating': [0, 1],
                },
                index=audformat.filewise_index(
                    ['f1.wav', 'f2.wav']
                ),
                dtype=pd.CategoricalDtype(
                    [0, 1, 2],
                    ordered=False,
                ),
            ),
        ),
        (
            'mono_db',
            'rating',
            ['rating.train', 'rating.test'],
            'train',
            pd.DataFrame(
                {
                    'rating': [0, 1],
                },
                index=audformat.filewise_index(
                    ['f1.wav', 'f2.wav']
                ),
                dtype=pd.CategoricalDtype(
                    [0, 1, 2],
                    ordered=False,
                ),
            ),
        ),
        (
            'mono_db',
            'rating',
            None,
            ['train', 'test'],
            pd.DataFrame(
                {
                    'rating': [1, 0, 1],
                },
                index=audformat.filewise_index(
                    ['f3.wav', 'f1.wav', 'f2.wav']
                ),
                dtype=pd.CategoricalDtype(
                    [0, 1, 2],
                    ordered=False,
                ),
            ),
        ),
        (
            'mono_db',
            'rating',
            ['rating.train', 'rating.test'],
            None,
            pd.DataFrame(
                {
                    'rating': [1, 0, 1],
                },
                index=audformat.filewise_index(
                    ['f3.wav', 'f1.wav', 'f2.wav']
                ),
                dtype=pd.CategoricalDtype(
                    [0, 1, 2],
                    ordered=False,
                ),
            ),
        ),
        (
            'mono_db',
            'rating',
            ['rating.train', 'rating.test'],
            ['train', 'test'],
            pd.DataFrame(
                {
                    'rating': [1, 0, 1],
                },
                index=audformat.filewise_index(
                    ['f3.wav', 'f1.wav', 'f2.wav']
                ),
                dtype=pd.CategoricalDtype(
                    [0, 1, 2],
                    ordered=False,
                ),
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


@pytest.mark.parametrize(
    'db, schemes, strict, expected',
    [
        (
            'mono_db',
            'sex',
            False,
            pd.DataFrame(
                {
                    'sex': ['female', 'male'],
                },
                index=audformat.filewise_index(
                    ['f1.wav', 'f3.wav'],
                ),
                dtype='object',
            ),
        ),
        (
            'mono_db',
            'sex',
            True,
            pd.DataFrame(
                {
                    'sex': [],
                },
                index=audformat.filewise_index(),
                dtype='object',
            ),
        ),
        (
            'mono_db',
            'year',
            False,
            pd.DataFrame(
                {
                    'year': [1995, 1995, 1996, 1995, 1995, 1995, 1995],
                },
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
                dtype='Int64',
            ),
        ),
        (
            'mono_db',
            'year',
            True,
            pd.DataFrame(
                {
                    'year': [],
                },
                index=audformat.filewise_index(),
                dtype='object',
            ),
        ),
        (
            'mono_db',
            ['sex', 'year'],
            True,
            pd.DataFrame(
                {
                    'sex': [],
                    'year': [],
                },
                index=audformat.filewise_index(),
                dtype='object',
            ),
        ),
        (
            'mono_db',
            ['gender', 'sex', 'year'],
            True,
            pd.DataFrame(
                {
                    'gender': [],
                    'sex': [],
                    'year': [],
                },
                index=audformat.filewise_index(),
                dtype='object',
            ),
        ),
    ]
)
def test_database_get_strict(request, db, schemes, strict, expected):
    db = request.getfixturevalue(db)
    df = db.get(schemes, strict=strict)
    pd.testing.assert_frame_equal(df, expected)


@pytest.mark.parametrize(
    'db, schemes, expected_error, expected_error_msg',
    [
        (
            'mono_db',
            'age',
            ValueError,
            (
                "Found overlapping data in column 'age':\n"
                "        left  right\n"
                "file               \n"
                "f1.wav    23     25\n"
                "f3.wav    59     45"
            ),
        ),
        (
            'mono_db',
            'height',
            ValueError,
            (
                "Found overlapping data in column 'height':\n"
                "        left  right\n"
                "file               \n"
                "f1.wav  1.12   1.76\n"
                "f2.wav  1.45   1.95\n"
                "f3.wav  1.01   1.80"
            ),
        ),
        (
            'mono_db',
            'weight',
            TypeError,
            "dtype of categories must be the same",
        ),
    ]
)
def test_database_get_errors(
        request,
        db,
        schemes,
        expected_error,
        expected_error_msg,
):
    db = request.getfixturevalue(db)
    with pytest.raises(expected_error, match=expected_error_msg):
        db.get(schemes)
