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
