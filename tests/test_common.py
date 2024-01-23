import pytest

import audformat


@pytest.mark.parametrize(
    "path, expected",
    [
        ("/root/file.txt", False),
        ("./file.txt", False),
        ("../file.txt", False),
        ("doc/./file.txt", False),
        ("doc/../file.txt", False),
        ("doc/../../file.txt", False),
        (r"C:\\doc\file.txt", False),
        (r"a\b", False),
        ("a/b", True),
        ("b", True),
        ("file.txt", True),
    ],
)
def test_is_relative_path(path, expected):
    relative = audformat.core.common.is_relative_path(path)
    assert relative == expected


def test_items_order():
    d = audformat.core.common.HeaderDict(sort_by_key=False)
    d["b"] = 1
    d["c"] = 2
    d["a"] = 0
    assert dict(d.items()) == {"b": 1, "c": 2, "a": 0}
    assert list(d.keys()) == ["b", "c", "a"]
    assert list(d.values()) == [1, 2, 0]
    assert list(d) == ["b", "c", "a"]
    for item, expected in zip(d, ["b", "c", "a"]):
        assert item == expected
    for item, expected in zip(reversed(d), ["a", "c", "b"]):
        assert item == expected
    assert d.popitem(last=True) == ("a", 0)
    assert d.popitem(last=False) == ("b", 1)
    assert list(d) == ["c"]

    d = audformat.core.common.HeaderDict(sort_by_key=True)
    d["b"] = 1
    d["c"] = 2
    d["a"] = 0
    assert dict(d.items()) == {"a": 0, "b": 1, "c": 2}
    assert list(d.keys()) == ["a", "b", "c"]
    assert list(d.values()) == [0, 1, 2]
    assert list(d) == ["a", "b", "c"]
    for item, expected in zip(d, ["a", "b", "c"]):
        assert item == expected
    for item, expected in zip(reversed(d), ["c", "b", "a"]):
        assert item == expected
    assert d.popitem(last=True) == ("c", 2)
    assert d.popitem(last=False) == ("a", 0)
    assert list(d) == ["b"]


@pytest.mark.parametrize(
    "meta",
    [
        {},
        {
            "key": "value",
            "meta": 1234,
        },
    ],
)
def test_meta_dict(meta):
    header = audformat.core.common.HeaderBase(meta=meta)
    assert header.meta == meta

    header_2 = audformat.core.common.HeaderBase()
    header_2.from_dict(header.to_dict())
    assert header_2.meta == meta


@pytest.mark.parametrize(
    "dtype, expected",
    [
        (
            "boolean",
            audformat.define.DataType.BOOL,
        ),
        (
            "datetime64[ns]",
            audformat.define.DataType.DATE,
        ),
        (
            "float",
            audformat.define.DataType.FLOAT,
        ),
        (
            "int",
            audformat.define.DataType.INTEGER,
        ),
        (
            "int64",
            audformat.define.DataType.INTEGER,
        ),
        (
            "Int64",
            audformat.define.DataType.INTEGER,
        ),
        (
            "string",
            audformat.define.DataType.STRING,
        ),
        (
            "timedelta64[ns]",
            audformat.define.DataType.TIME,
        ),
        (
            "object",
            audformat.define.DataType.OBJECT,
        ),
    ],
)
def test_to_audformat_dtype(dtype, expected):
    dtype = audformat.core.common.to_audformat_dtype(dtype)
    assert dtype == expected


@pytest.mark.parametrize(
    "dtype, expected",
    [
        (
            audformat.define.DataType.BOOL,
            "boolean",
        ),
        (
            audformat.define.DataType.DATE,
            "datetime64[ns]",
        ),
        (
            audformat.define.DataType.FLOAT,
            "float",
        ),
        (
            audformat.define.DataType.INTEGER,
            "Int64",
        ),
        (
            audformat.define.DataType.OBJECT,
            "object",
        ),
        (
            audformat.define.DataType.STRING,
            "string",
        ),
        (
            audformat.define.DataType.TIME,
            "timedelta64[ns]",
        ),
    ],
)
def test_to_pandas_dtype(dtype, expected):
    dtype = audformat.core.common.to_pandas_dtype(dtype)
    assert dtype == expected
