import pandas as pd
import pytest

import audformat
import audformat.testing


def test_scheme():

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


def test_scheme_errors():

    # dtype mismatch
    with pytest.raises(ValueError):
        audformat.Scheme(
            audformat.define.DataType.STRING,
            labels=[1, 2, 3],
        )

    # unknown type
    with pytest.raises(ValueError):
        audformat.Scheme('bad')

    # labels not list or dictionary
    with pytest.raises(ValueError):
        audformat.Scheme(labels=set([1, 2, 3]))

    # labels do not have the same type
    with pytest.raises(ValueError):
        audformat.Scheme(labels=[1, '2', 3])

    # update labels when scheme has no label
    with pytest.raises(ValueError):
        scheme = audformat.Scheme(audformat.define.DataType.INTEGER)
        scheme.replace_labels(['a', 'b'])


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
