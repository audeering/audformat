import pytest

import audformat


@pytest.mark.parametrize(
    'dtype, expected',
    [
        (
            'boolean',
            audformat.define.DataType.BOOL,
        ),
        (
            'datetime64[ns]',
            audformat.define.DataType.DATE,
        ),
        (
            'float',
            audformat.define.DataType.FLOAT,
        ),
        (
            'int',
            audformat.define.DataType.INTEGER,
        ),
        (
            'int64',
            audformat.define.DataType.INTEGER,
        ),
        (
            'Int64',
            audformat.define.DataType.INTEGER,
        ),
        (
            'string',
            audformat.define.DataType.STRING,
        ),
        (
            'timedelta64[ns]',
            audformat.define.DataType.TIME,
        ),
        (
            'object',
            audformat.define.DataType.OBJECT,
        ),
    ]
)
def test_to_audformat_dtype(dtype, expected):
    dtype = audformat.core.common.to_audformat_dtype(dtype)
    assert dtype == expected


@pytest.mark.parametrize(
    'dtype, expected',
    [
        (
            audformat.define.DataType.BOOL,
            'boolean',
        ),
        (
            audformat.define.DataType.DATE,
            'datetime64[ns]',
        ),
        (
            audformat.define.DataType.FLOAT,
            'float',
        ),
        (
            audformat.define.DataType.INTEGER,
            'Int64',
        ),
        (
            audformat.define.DataType.OBJECT,
            'object',
        ),
        (
            audformat.define.DataType.STRING,
            'string',
        ),
        (
            audformat.define.DataType.TIME,
            'timedelta64[ns]',
        ),
    ]
)
def test_to_pandas_dtype(dtype, expected):
    dtype = audformat.core.common.to_pandas_dtype(dtype)
    assert dtype == expected


def test_items_order():

    d = audformat.core.common.HeaderDict(sorted_iter=False)
    d['b'] = 1
    d['a'] = 0
    assert dict(d.items()) == {'b': 1, 'a': 0}
    assert list(d.keys()) == ['b', 'a']
    assert list(d.values()) == [1, 0]

    d = audformat.core.common.HeaderDict(sorted_iter=True)
    d['b'] = 1
    d['a'] = 0
    assert dict(d.items()) == {'a': 0, 'b': 1}
    assert list(d.keys()) == ['a', 'b']
    assert list(d.values()) == [0, 1]
