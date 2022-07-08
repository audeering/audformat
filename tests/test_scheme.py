import numpy as np
import pandas as pd
import pytest

import audformat
import audformat.testing


def test_scheme_assign_values():

    db = audformat.testing.create_db(minimal=True)
    speakers = ['spk1', 'spk2', 'spk3']
    ages = [33, 44, 55]
    index = pd.Index(speakers, name='speaker')
    db['misc'] = audformat.MiscTable(index)
    db['misc']['age'] = audformat.Column()
    db['misc']['age'].set(ages)
    db.schemes['scheme'] = audformat.Scheme(labels='misc', dtype='str')
    db['table'] = audformat.Table(audformat.filewise_index(['f1', 'f2', 'f3']))
    db['table']['speaker'] = audformat.Column(scheme_id='scheme')
    db['table']['speaker'].set(speakers)

    assert list(db['table']['speaker'].get()) == speakers
    assert list(db['table']['speaker'].get(map='age')) == ages
    assert list(db['table'].get(map={'speaker': 'age'})['age']) == ages


def test_scheme_contains():

    db = pytest.DB

    assert 'tests' in db.schemes['string']
    assert 0.0 in db.schemes['float']
    assert 1.0 in db.schemes['float']
    assert -1.0 in db.schemes['float']
    assert 2.0 not in db.schemes['float']
    assert -2.0 not in db.schemes['float']
    assert pd.Timedelta(0.5, unit='s') in db.schemes['time']
    assert 'label1' in db.schemes['label']
    assert 'label1' in db.schemes['label_map_str']
    assert 1 in db.schemes['label_map_int']

    # Misc table
    # assigned scheme
    assert 'label1' in db.schemes['label_map_misc']
    # remove table
    # db.drop_tables(['misc'])
    db.misc_tables.pop('misc')
    assert 'label1' not in db.schemes['label_map_misc']
    # unassigned scheme
    scheme = audformat.Scheme(labels='misc', dtype='str')
    assert 'label1' not in scheme


@pytest.mark.parametrize(
    'dtype, values',
    [
        pytest.param(  # bool not supported
            audformat.define.DataType.BOOL,
            [True, False],
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        (
            audformat.define.DataType.DATE,
            pd.to_datetime(['3/11/2000', '3/12/2000', '3/13/2000']),
        ),
        (
            audformat.define.DataType.INTEGER,
            [1, 2, 3],
        ),
        (
            audformat.define.DataType.FLOAT,
            [1.0, 2.0, 3.0],
        ),
        (
            audformat.define.DataType.STRING,
            ['a', 'b', 'c'],
        ),
        (
            audformat.define.DataType.TIME,
            pd.to_timedelta(['1s', '2s', '3s']),
        ),
    ]
)
def test_scheme_dtypes(dtype, values):
    db = audformat.Database('test')
    index = pd.Index(values, name='labels')
    db['misc'] = audformat.MiscTable(index)
    index = audformat.filewise_index([f'f{idx}' for idx in range(len(values))])
    db.schemes['scheme'] = audformat.Scheme(dtype=dtype, labels='misc')
    db['table'] = audformat.Table(index)
    db['table']['labels'] = audformat.Column(scheme_id='scheme')
    db['table']['labels'].set(values)

    assert set(db['table']['labels'].get()) == set(values)


def test_scheme_errors():

    db = audformat.Database('test')

    # dtype mismatch
    error_msg = (
        "Data type is set to 'str', "
        "but data type of labels is 'int'."
    )
    with pytest.raises(ValueError, match=error_msg):
        audformat.Scheme(
            audformat.define.DataType.STRING,
            labels=[1, 2, 3],
        )

    # unknown type
    error_msg = (
        "Bad value 'bad', "
        "expected one of \\['bool', 'date', 'float', 'int', 'str', 'time'\\]"
    )
    with pytest.raises(ValueError, match=error_msg):
        audformat.Scheme('bad')

    # labels not list or dictionary
    error_msg = (
        'Labels must be passed '
        'as a dictionary, list or ID of a misc table.'
    )
    with pytest.raises(ValueError, match=error_msg):
        audformat.Scheme(labels=set([1, 2, 3]))

    # labels do not have the same type
    error_msg = (
        'All labels must be of the same data type.'
    )
    with pytest.raises(ValueError, match=error_msg):
        audformat.Scheme(labels=[1, '2', 3])

    # update labels when scheme has no label
    error_msg = (
        'Cannot replace labels when '
        'scheme does not define labels.'
    )
    with pytest.raises(ValueError, match=error_msg):
        scheme = audformat.Scheme(audformat.define.DataType.INTEGER)
        scheme.replace_labels(['a', 'b'])

    # misc table needs to define data type
    error_msg = (
        "'dtype' has to be provided "
        "when using a misc table as labels."
    )
    with pytest.raises(ValueError, match=error_msg):
        audformat.Scheme(labels='misc')

    # misc table not assigned to a database
    error_msg = (
        "The misc table 'misc' used as scheme labels "
        "needs to be assigned to the database"
    )
    scheme = audformat.Scheme(labels='misc', dtype='str')
    with pytest.raises(ValueError, match=error_msg):
        db.schemes['misc'] = scheme

    # filewise table used instead of misc table
    error_msg = (
        "The table 'table' used as scheme labels "
        "needs to be a misc table."
    )
    db['table'] = audformat.Table(audformat.filewise_index(['f1']))
    scheme = audformat.Scheme(labels='table', dtype='str')
    with pytest.raises(ValueError, match=error_msg):
        db.schemes['misc'] = scheme

    # misc table has different dtype
    error_msg = (
        "Data type is set to 'str', "
        "but data type of labels in misc table is 'int'."
    )
    db['misc'] = audformat.MiscTable(pd.Index([0], name='misc'))
    scheme = audformat.Scheme(labels='misc', dtype='str')
    with pytest.raises(ValueError, match=error_msg):
        db.schemes['misc'] = scheme

    # misc table should only contain an one-dimensional index
    error_msg = (
        "Index of misc table 'misc' used as scheme labels "
        "is only allowed to have a single level."
    )
    db['misc'] = audformat.MiscTable(
        pd.MultiIndex.from_arrays(
            [
                [1, 2],
                ['a', 'b'],
            ],
            names=['misc-1', 'misc-2'],
        )
    )
    scheme = audformat.Scheme(labels='misc', dtype='str')
    with pytest.raises(ValueError, match=error_msg):
        db.schemes['misc'] = scheme

    # misc table should not contain duplicates
    error_msg = (
        "Index of misc table 'misc' used as scheme labels "
        "is not allowed to contain duplicates."
    )
    db['misc'] = audformat.MiscTable(pd.Index([0, 0], name='misc'))
    scheme = audformat.Scheme(labels='misc', dtype='int')
    with pytest.raises(ValueError, match=error_msg):
        db.schemes['misc'] = scheme


@pytest.mark.parametrize(
    'values, labels, new_labels, expected',
    [
        (
            pd.Series(
                index=audformat.filewise_index(),
                dtype=pd.CategoricalDtype(),
            ),
            [],
            [],
            pd.Series(
                index=audformat.filewise_index(),
                dtype=pd.CategoricalDtype(),
            ),
        ),
        (
            pd.Series(
                index=audformat.filewise_index(),
                dtype=pd.CategoricalDtype(['a', 'b']),
            ),
            ['a', 'b'],
            ['b', 'c', 'd'],
            pd.Series(
                index=audformat.filewise_index(),
                dtype=pd.CategoricalDtype(['b', 'c', 'd']),
            ),
        ),
        (
            pd.Series(
                ['a', 'b'],
                index=audformat.filewise_index(['f1', 'f2']),
                dtype=pd.CategoricalDtype(['a', 'b']),
            ),
            ['a', 'b'],
            ['a'],
            pd.Series(
                ['a', None],
                index=audformat.filewise_index(['f1', 'f2']),
                dtype=pd.CategoricalDtype(['a']),
            ),
        ),
        (
            pd.Series(
                ['a', 'b'],
                index=audformat.filewise_index(['f1', 'f2']),
                dtype=pd.CategoricalDtype(['a', 'b']),
            ),
            ['a', 'b'],
            ['a', 'b', 'c'],
            pd.Series(
                ['a', 'b'],
                index=audformat.filewise_index(['f1', 'f2']),
                dtype=pd.CategoricalDtype(['a', 'b', 'c']),
            ),
        ),
        (
            pd.Series(
                ['a', 'b'],
                index=audformat.filewise_index(['f1', 'f2']),
                dtype=pd.CategoricalDtype(['a', 'b']),
            ),
            ['a', 'b'],
            ['c'],
            pd.Series(
                [None, None],
                index=audformat.filewise_index(['f1', 'f2']),
                dtype=pd.CategoricalDtype(['c']),
            ),
        ),
        (
            pd.Series(
                ['a', 'b'],
                index=audformat.filewise_index(['f1', 'f2']),
                dtype=pd.CategoricalDtype(['a', 'b']),
            ),
            ['a', 'b'],
            [],
            pd.Series(
                [None, None],
                index=audformat.filewise_index(['f1', 'f2']),
                dtype=pd.CategoricalDtype([]),
            ),
        ),
        (
            pd.Series(
                [0, 1],
                index=audformat.filewise_index(['f1', 'f2']),
                dtype=pd.CategoricalDtype([0, 1]),
            ),
            {0: 'a', 1: 'b'},
            {1: {'b': 'B'}, 2: {'c': 'C'}},
            pd.Series(
                [None, 1],
                index=audformat.filewise_index(['f1', 'f2']),
                dtype=pd.CategoricalDtype([1, 2]),
            ),
        ),
        (
            pd.Series(
                [0, 1],
                index=audformat.filewise_index(['f1', 'f2']),
                dtype=pd.CategoricalDtype([0, 1]),
            ),
            {0: 'a', 1: 'b'},
            [1, 2],
            pd.Series(
                [None, 1],
                index=audformat.filewise_index(['f1', 'f2']),
                dtype=pd.CategoricalDtype([1, 2]),
            ),
        ),
        # error: dtype of labels does not match
        pytest.param(
            pd.Series(
                index=audformat.filewise_index(),
                dtype=pd.CategoricalDtype(),
            ),
            ['a', 'b'],
            [0, 1],
            pd.Series(
                index=audformat.filewise_index(),
                dtype=pd.CategoricalDtype(),
            ),
            marks=pytest.mark.xfail(raises=ValueError),
        )
    ]
)
def test_replace_labels(values, labels, new_labels, expected):

    db = audformat.testing.create_db(minimal=True)
    db.schemes['scheme'] = audformat.Scheme(labels=labels)
    db['table'] = audformat.Table(index=values.index)
    db['table']['columns'] = audformat.Column(scheme_id='scheme')
    db['table']['columns'].set(values)
    db.schemes['scheme'].replace_labels(new_labels)
    pd.testing.assert_series_equal(
        db['table']['columns'].get(),
        expected,
        check_names=False,
    )


def test_replace_labels_misc_table():

    db = audformat.testing.create_db(minimal=True)
    db['misc'] = audformat.MiscTable(
        pd.Index(['spk1', 'spk2'], name='speaker')
    )

    # non-assigned scheme
    scheme = audformat.Scheme('str', labels='misc')
    assert scheme.labels == 'misc'
    assert scheme._get_labels() == {}

    # replace with non-existing misc table scheme
    # and back again
    scheme.replace_labels('misc-non-existing')
    scheme.replace_labels('misc')

    # assigned scheme
    db.schemes['scheme'] = scheme
    assert scheme.labels == 'misc'
    assert scheme._get_labels() == {'spk1': {}, 'spk2': {}}

    # use scheme
    db['table'] = audformat.Table(index=audformat.filewise_index(['f1', 'f2']))
    db['table']['columns'] = audformat.Column(scheme_id='scheme')
    db['table']['columns'].set(['spk1', 'spk2'])

    # replace with new misc table scheme
    db['misc-new'] = audformat.MiscTable(
        pd.Index(['spk1', 'spk2', 'spk3'], name='speaker')
    )
    scheme.replace_labels('misc-new')
    assert scheme.labels == 'misc-new'
    assert scheme._get_labels() == {'spk1': {}, 'spk2': {}, 'spk3': {}}

    # replace with dictionary
    scheme.replace_labels({'spk1': {}})
    assert scheme.labels == {'spk1': {}}
    assert scheme._get_labels() == {'spk1': {}}
    assert list(db['table']['columns'].get().values) == ['spk1', np.NaN]

    # replace again with misc table
    scheme.replace_labels('misc')
    assert scheme.labels == 'misc'
    assert scheme._get_labels() == {'spk1': {}, 'spk2': {}}

    # replace with list
    scheme.replace_labels(['spk1', 'spk2'])
    assert scheme.labels == ['spk1', 'spk2']
    assert scheme._get_labels() == ['spk1', 'spk2']

    # replace with non-existing misc table scheme
    error_msg = (
        "The misc table 'misc-non-existing' used as scheme labels "
        "needs to be assigned to the database."
    )
    with pytest.raises(ValueError, match=error_msg):
        scheme.replace_labels('misc-non-existing')
