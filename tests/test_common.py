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
            'str',
            audformat.define.DataType.STRING,
        ),
        (
            'timedelta64[ns]',
            audformat.define.DataType.TIME,
        ),
        (
            'object',
            audformat.define.DataType.STRING,
        ),
    ]
)
def test_to_audformat_dtype(dtype, expected):
    print(dtype)
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
            audformat.define.DataType.STRING,
            'str',
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
