import re

import numpy as np
import pandas as pd
import pytest

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


@pytest.mark.parametrize(
    "column_values, column_dtype, expected_pandas_dtype, expected_audformat_dtype",
    [
        (
            [],
            None,
            "object",
            audformat.define.DataType.OBJECT,
        ),
        (
            [],
            bool,
            "boolean",
            audformat.define.DataType.BOOL,
        ),
        (
            [],
            "boolean",
            "boolean",
            audformat.define.DataType.BOOL,
        ),
        (
            [],
            "datetime64[ns]",
            "datetime64[ns]",
            audformat.define.DataType.DATE,
        ),
        (
            [],
            float,
            "float64",
            audformat.define.DataType.FLOAT,
        ),
        (
            [],
            int,
            "Int64",
            audformat.define.DataType.INTEGER,
        ),
        (
            [],
            "int64",
            "Int64",
            audformat.define.DataType.INTEGER,
        ),
        (
            [],
            "Int64",
            "Int64",
            audformat.define.DataType.INTEGER,
        ),
        (
            [],
            str,
            "object",
            audformat.define.DataType.OBJECT,
        ),
        (
            [],
            "string",
            "string",
            audformat.define.DataType.STRING,
        ),
        (
            [],
            "timedelta64[ns]",
            "timedelta64[ns]",
            audformat.define.DataType.TIME,
        ),
        (
            [0],
            "datetime64[ns]",
            "datetime64[ns]",
            audformat.define.DataType.DATE,
        ),
        (
            [0.0],
            None,
            "float64",
            audformat.define.DataType.FLOAT,
        ),
        (
            [0],
            None,
            "Int64",
            audformat.define.DataType.INTEGER,
        ),
        (
            [np.nan],
            "Int64",
            "Int64",
            audformat.define.DataType.INTEGER,
        ),
        (
            [0, np.nan],
            "Int64",
            "Int64",
            audformat.define.DataType.INTEGER,
        ),
        (
            [np.nan],
            "Int64",
            "Int64",
            audformat.define.DataType.INTEGER,
        ),
        (
            ["0"],
            None,
            "object",
            audformat.define.DataType.OBJECT,
        ),
        (
            [0],
            "timedelta64[ns]",
            "timedelta64[ns]",
            audformat.define.DataType.TIME,
        ),
        (
            [True],
            None,
            "boolean",
            audformat.define.DataType.BOOL,
        ),
        (
            [True, False],
            bool,
            "boolean",
            audformat.define.DataType.BOOL,
        ),
        (
            [True, False],
            "boolean",
            "boolean",
            audformat.define.DataType.BOOL,
        ),
    ],
)
def test_dtype(
    tmpdir,
    column_values,
    column_dtype,
    expected_pandas_dtype,
    expected_audformat_dtype,
):
    r"""Test table columns have correct dtype.

    Ensures that a dataframe column,
    associated with a table,
    has the dtype,
    which corresponds to the scheme of the column.

    Args:
        tmpdir: pytest tmpdir fixture
        column_values: values assigned to the column
        column_dtype: pandas dtype of values assigned to column
        expected_pandas_dtype: pandas dtype of column after assignment
        expected_audformat_dtype: audformat dtype corresponding
            to the expected pandas dtype.
            This is assigned to the scheme of the column

    """
    y = pd.Series(column_values, dtype=column_dtype or "object")

    index_values = [f"f{n}" for n in range(len(column_values))]
    index = audformat.filewise_index(index_values)

    db = audformat.testing.create_db(minimal=True)
    db["table"] = audformat.Table(index)
    db.schemes["column"] = audformat.Scheme(expected_audformat_dtype)
    db["table"]["column"] = audformat.Column(scheme_id="column")
    db["table"]["column"].set(y.values)

    assert db["table"]["column"].scheme.dtype == expected_audformat_dtype
    assert db["table"].df["column"].dtype == expected_pandas_dtype

    # Store and load table
    db_root = tmpdir.join("db")
    db.save(db_root, storage_format="csv")
    db_new = audformat.Database.load(db_root)

    assert db_new["table"]["column"].scheme.dtype == expected_audformat_dtype
    assert db_new["table"].df["column"].dtype == expected_pandas_dtype


def test_exceptions():
    column = audformat.Column()
    with pytest.raises(RuntimeError):
        column.set([])
    with pytest.raises(RuntimeError):
        column.get()


@pytest.mark.parametrize(
    "num_files, values",
    [
        (6, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
        (6, np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])),
        (
            6,
            pd.Series(range(6), dtype="float").values,
        ),
    ],
)
def test_filewise(num_files, values):
    db = audformat.testing.create_db(minimal=True)
    audformat.testing.add_table(
        db, "table", audformat.define.IndexType.FILEWISE, num_files=num_files
    )
    db.schemes["scheme"] = audformat.Scheme(
        dtype=audformat.define.DataType.FLOAT,
    )

    db["table"]["column"] = audformat.Column(scheme_id="scheme")

    table = db["table"]
    column = db["table"]["column"]
    column_id = "column"

    # empty
    series = pd.Series(
        [np.nan] * num_files,
        audformat.filewise_index(table.files),
        name=column_id,
    )
    pd.testing.assert_series_equal(column.get(), series)

    # set single
    series[:] = np.nan
    table.df["column"] = np.nan
    series.iloc[0] = values[0]
    index = audformat.filewise_index(table.files[0])
    column.set(values[0], index=index)
    pd.testing.assert_series_equal(column.get(index), series.iloc[[0]])
    pd.testing.assert_series_equal(column.get(), series)

    # set slice
    series[:] = np.nan
    table.df["column"] = np.nan
    series[1:-1] = values[1:-1]
    index = audformat.filewise_index(table.files[1:-1])
    column.set(values[1:-1], index=index)
    pd.testing.assert_series_equal(column.get(index), series[1:-1])
    pd.testing.assert_series_equal(column.get(), series)

    # set all
    series[:] = np.nan
    table.df["column"] = np.nan
    series[:] = values
    column.set(values)
    pd.testing.assert_series_equal(column.get(), series)

    # set scalar
    series[:] = np.nan
    table.df["column"] = np.nan
    series[:] = values[0]
    column.set(values[0])
    pd.testing.assert_series_equal(column.get(), series)

    # get segments
    table.df["column"] = values
    index = audformat.segmented_index(
        [db.files[0], db.files[0], db.files[1]],
        starts=["0s", "1s", "0s"],
        ends=["1s", "2s", pd.NaT],
    )
    pd.testing.assert_series_equal(
        column.get(index),
        pd.Series(
            [values[0], values[0], values[1]],
            index=index,
            name="column",
        ),
    )

    # try to use segmented index
    with pytest.raises(ValueError):
        index = audformat.segmented_index()
        column.set([], index=index)


def test_get_as_segmented():
    db = pytest.DB

    y = db["files"]["bool"].get()
    assert audformat.index_type(y) == audformat.define.IndexType.FILEWISE
    assert not db._files_duration

    # convert to segmented index

    y = db["files"]["bool"].get(
        as_segmented=True,
        allow_nat=True,
    )
    assert audformat.index_type(y) == audformat.define.IndexType.SEGMENTED
    assert not db._files_duration
    assert y.index.get_level_values(audformat.define.IndexField.END).isna().all()

    # replace NaT with file duration

    y = db["files"]["bool"].get(
        as_segmented=True,
        allow_nat=False,
    )
    assert audformat.index_type(y) == audformat.define.IndexType.SEGMENTED
    assert db._files_duration
    assert not y.index.get_level_values(audformat.define.IndexField.END).isna().any()

    # reset db

    db._files_duration = {}


@pytest.mark.parametrize(
    "column, map, expected_dtype",
    [
        (pytest.DB["files"]["label_map_int"], "int", "string"),
        (pytest.DB["files"]["label_map_int"], "label_map_int", "string"),
        (pytest.DB["files"]["label_map_str"], "prop1", "Int64"),
        (pytest.DB["segments"]["label_map_str"], "prop2", "string"),
        pytest.param(  # no mappings
            pytest.DB["files"]["label"],
            "label1",
            None,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(  # no schemes
            pytest.DB["files"]["no_scheme"],
            "map",
            None,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(  # no labels
            pytest.DB["files"]["string"],
            "map",
            None,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(  # no labels
            pytest.DB["files"]["string"],
            "map",
            None,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(  # no labels in dict
            pytest.DB["files"]["label_map_str"],
            "bad",
            None,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
    ],
)
def test_map(column, map, expected_dtype):
    result = column.get(map=map)
    expected = column.get()
    mapping = {}
    for key, value in pytest.DB.schemes[column.scheme_id].labels.items():
        if isinstance(value, dict):
            if map in value:
                value = value[map]
            else:
                value = np.nan
        mapping[key] = value
    expected = expected.map(mapping).astype(expected_dtype)
    expected.name = map
    pd.testing.assert_series_equal(result, expected)


@pytest.fixture(scope="function")
def db_scheme_with_labels():
    r"""Database with different scheme labels as dictionary."""
    db = audformat.testing.create_db(minimal=True)
    db.name = "db_scheme_with_labels"
    db.schemes["scheme"] = audformat.Scheme(
        labels={
            "a": {
                "bool": True,
                "date": pd.to_datetime("2018-10-26"),
                "float": 45.0,
                "int": 0,
                "object": b"abc",
                "str": "A",
                "time": pd.to_timedelta(300, unit="s"),
            },
            "b": {
                "bool": False,
                "date": pd.to_datetime("2018-10-27"),
                "float": 67.0,
                "int": 0,
                "object": 2,
                "str": "B",
                "time": pd.to_timedelta(301, unit="s"),
            },
            "c": {
                "bool": False,
                "date": pd.to_datetime("2018-10-28"),
                "float": 78.0,
                "int": 1,
                "object": "abc",
                "str": "C",
                "time": pd.to_timedelta(302, unit="s"),
            },
        },
    )
    db["table"] = audformat.Table(audformat.filewise_index(["f1", "f2", "f3"]))
    db["table"]["column"] = audformat.Column(scheme_id="scheme")
    db["table"]["column"].set(["a", "b", "c"])
    return db


@pytest.fixture(scope="function")
def db_scheme_with_misc_table():
    r"""Database with different scheme labels stored in misc table."""
    db = audformat.testing.create_db(minimal=True)
    db.name = "db_scheme_with_misc_table"
    # Schemes
    db.schemes["bool"] = audformat.Scheme("bool")
    db.schemes["date"] = audformat.Scheme("date")
    db.schemes["int-categories"] = audformat.Scheme(
        "int",
        labels={0: "Berlin", 1: "Gilching"},
    )
    db.schemes["float"] = audformat.Scheme("float", minimum=0)
    db.schemes["str"] = audformat.Scheme("str")
    db.schemes["str-categories"] = audformat.Scheme(
        "str",
        labels=["female", "male"],
    )
    db.schemes["time"] = audformat.Scheme("time")
    # Misc table
    db["misc"] = audformat.MiscTable(
        pd.Index(
            ["a", "b", "c"],
            name="index",
            dtype="string",
        ),
    )
    db["misc"]["bool"] = audformat.Column(scheme_id="bool")
    db["misc"]["bool"].set([True, False, False])
    db["misc"]["date"] = audformat.Column(scheme_id="date")
    db["misc"]["date"].set(
        [
            pd.to_datetime("2018-10-26"),
            pd.to_datetime("2018-10-27"),
            pd.to_datetime("2018-10-28"),
        ]
    )
    db["misc"]["float"] = audformat.Column(scheme_id="float")
    db["misc"]["float"].set([45.0, 67.0, 78.0])
    db["misc"]["int-categories"] = audformat.Column(scheme_id="int-categories")
    db["misc"]["int-categories"].set([0, 0, 1])
    db["misc"]["str"] = audformat.Column(scheme_id="str")
    db["misc"]["str"].set(["Jae", "Joe", "John"])
    db["misc"]["str-categories"] = audformat.Column(scheme_id="str-categories")
    db["misc"]["str-categories"].set(["female", "male", "male"])
    db["misc"]["str-without-scheme"] = audformat.Column()
    db["misc"]["str-without-scheme"].set(["A", "B", "C"])
    db["misc"]["time"] = audformat.Column(scheme_id="time")
    db["misc"]["time"].set(
        [
            pd.to_timedelta(300, unit="s"),
            pd.to_timedelta(301, unit="s"),
            pd.to_timedelta(302, unit="s"),
        ]
    )
    # Scheme using misc table
    db.schemes["scheme"] = audformat.Scheme("str", labels="misc")
    db["table"] = audformat.Table(audformat.filewise_index(["f1", "f2", "f3"]))
    db["table"]["column"] = audformat.Column(scheme_id="scheme")
    db["table"]["column"].set(["a", "b", "c"])
    return db


@pytest.mark.parametrize(
    "db, map, expected",
    [
        (
            "db_scheme_with_labels",
            None,
            pd.Series(
                ["a", "b", "c"],
                index=audformat.filewise_index(["f1", "f2", "f3"]),
                name="column",
                dtype=pd.CategoricalDtype(
                    categories=["a", "b", "c"],
                    ordered=False,
                ),
            ),
        ),
        (
            "db_scheme_with_labels",
            "str",
            pd.Series(
                ["A", "B", "C"],
                index=audformat.filewise_index(["f1", "f2", "f3"]),
                name="str",
                dtype="string",
            ),
        ),
        (
            "db_scheme_with_labels",
            "float",
            pd.Series(
                [45.0, 67.0, 78.0],
                index=audformat.filewise_index(["f1", "f2", "f3"]),
                name="float",
                dtype="float",
            ),
        ),
        (
            "db_scheme_with_labels",
            "int",
            pd.Series(
                [0, 0, 1],
                index=audformat.filewise_index(["f1", "f2", "f3"]),
                name="int",
                dtype="Int64",
            ),
        ),
        (
            "db_scheme_with_labels",
            "object",
            pd.Series(
                [b"abc", 2, "abc"],
                index=audformat.filewise_index(["f1", "f2", "f3"]),
                name="object",
                dtype="object",
            ),
        ),
        (
            "db_scheme_with_labels",
            "date",
            pd.Series(
                [
                    pd.to_datetime("2018-10-26"),
                    pd.to_datetime("2018-10-27"),
                    pd.to_datetime("2018-10-28"),
                ],
                index=audformat.filewise_index(["f1", "f2", "f3"]),
                name="date",
                dtype="datetime64[ns]",
            ),
        ),
        (
            "db_scheme_with_labels",
            "time",
            pd.Series(
                [
                    pd.to_timedelta(300, unit="s"),
                    pd.to_timedelta(301, unit="s"),
                    pd.to_timedelta(302, unit="s"),
                ],
                index=audformat.filewise_index(["f1", "f2", "f3"]),
                name="time",
                dtype="timedelta64[ns]",
            ),
        ),
        (
            "db_scheme_with_labels",
            "bool",
            pd.Series(
                [True, False, False],
                index=audformat.filewise_index(["f1", "f2", "f3"]),
                name="bool",
                dtype="boolean",
            ),
        ),
        (
            "db_scheme_with_misc_table",
            None,
            pd.Series(
                ["a", "b", "c"],
                index=audformat.filewise_index(["f1", "f2", "f3"]),
                name="column",
                dtype=pd.CategoricalDtype(
                    categories=["a", "b", "c"],
                    ordered=False,
                ),
            ),
        ),
        (
            "db_scheme_with_misc_table",
            "str-without-scheme",
            pd.Series(
                ["A", "B", "C"],
                index=audformat.filewise_index(["f1", "f2", "f3"]),
                name="str-without-scheme",
                dtype="string",
            ),
        ),
        (
            "db_scheme_with_misc_table",
            "str-categories",
            pd.Series(
                ["female", "male", "male"],
                index=audformat.filewise_index(["f1", "f2", "f3"]),
                name="str-categories",
                dtype=pd.CategoricalDtype(
                    categories=["female", "male"],
                    ordered=False,
                ),
            ),
        ),
        (
            "db_scheme_with_misc_table",
            "int-categories",
            pd.Series(
                [0, 0, 1],
                index=audformat.filewise_index(["f1", "f2", "f3"]),
                name="int-categories",
                dtype=pd.CategoricalDtype(
                    categories=[0, 1],
                    ordered=False,
                ),
            ),
        ),
        (
            "db_scheme_with_misc_table",
            "float",
            pd.Series(
                [45.0, 67.0, 78.0],
                index=audformat.filewise_index(["f1", "f2", "f3"]),
                name="float",
                dtype="float",
            ),
        ),
        (
            "db_scheme_with_misc_table",
            "str",
            pd.Series(
                ["Jae", "Joe", "John"],
                index=audformat.filewise_index(["f1", "f2", "f3"]),
                name="str",
                dtype="string",
            ),
        ),
        (
            "db_scheme_with_misc_table",
            "date",
            pd.Series(
                [
                    pd.to_datetime("2018-10-26"),
                    pd.to_datetime("2018-10-27"),
                    pd.to_datetime("2018-10-28"),
                ],
                index=audformat.filewise_index(["f1", "f2", "f3"]),
                name="date",
                dtype="datetime64[ns]",
            ),
        ),
        (
            "db_scheme_with_misc_table",
            "time",
            pd.Series(
                [
                    pd.to_timedelta(300, unit="s"),
                    pd.to_timedelta(301, unit="s"),
                    pd.to_timedelta(302, unit="s"),
                ],
                index=audformat.filewise_index(["f1", "f2", "f3"]),
                name="time",
                dtype="timedelta64[ns]",
            ),
        ),
        (
            "db_scheme_with_misc_table",
            "bool",
            pd.Series(
                [True, False, False],
                index=audformat.filewise_index(["f1", "f2", "f3"]),
                name="bool",
                dtype="boolean",
            ),
        ),
    ],
)
def test_map_dtypes(request, db, map, expected):
    db = request.getfixturevalue(db)
    y = db["table"]["column"].get(map=map)
    pd.testing.assert_series_equal(y, expected)

    # Change label entry and check that dtype stays the same
    if map == "str" and db.name == "db_scheme_with_labels":
        db.schemes["scheme"].replace_labels(
            {
                "a": {"str": "A"},
                "b": {"str": "B"},
            }
        )
        y = db["table"]["column"].get(map=map)
        expected = pd.Series(
            ["A", "B", None],
            index=y.index,
            name=map,
            dtype="string",
        )
        pd.testing.assert_series_equal(y, expected)

    elif map == "str-without-scheme" and db.name == "db_scheme_with_misc_table":
        db["misc"]["str-without-scheme"].set(["A", "B", None])
        y = db["table"]["column"].get(map=map)
        expected = pd.Series(
            ["A", "B", None],
            index=y.index,
            name=map,
            dtype="string",
        )
        pd.testing.assert_series_equal(y, expected)


@pytest.mark.parametrize(
    "num_files, num_segments_per_file, values",
    [
        (3, 2, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
        (3, 2, np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])),
        (3, 2, pd.Series(range(6), dtype="float").values),
    ],
)
def test_segmented(num_files, num_segments_per_file, values):
    db = audformat.testing.create_db(minimal=True)
    audformat.testing.add_table(
        db,
        "table",
        audformat.define.IndexType.SEGMENTED,
        num_files=num_files,
        num_segments_per_file=num_segments_per_file,
    )
    db.schemes["scheme"] = audformat.Scheme(
        dtype=audformat.define.DataType.FLOAT,
    )

    db["table"]["column"] = audformat.Column(scheme_id="scheme")

    table = db["table"]
    column = db["table"]["column"]
    column_id = "column"

    # empty
    series = pd.Series(
        [np.nan] * num_files * num_segments_per_file,
        db.segments,
        name=column_id,
    )
    pd.testing.assert_series_equal(column.get(), series)

    # set single
    series[:] = np.nan
    table.df["column"] = np.nan
    series.iloc[0] = values[0]
    index = audformat.segmented_index(
        table.files[0],
        starts=table.starts[0],
        ends=table.ends[0],
    )
    column.set(values[0], index=index)
    pd.testing.assert_series_equal(column.get(index), series.iloc[[0]])
    pd.testing.assert_series_equal(column.get(), series)

    # set slice
    series[:] = np.nan
    table.df["column"] = np.nan
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
    table.df["column"] = np.nan
    series[:] = values
    column.set(values)
    pd.testing.assert_series_equal(column.get(), series)

    # set scalar
    series[:] = np.nan
    table.df["column"] = np.nan
    series[:] = values[0]
    column.set(values[0])
    pd.testing.assert_series_equal(column.get(), series)

    # set files
    series[:] = np.nan
    table.df["column"] = np.nan
    series[0 : num_segments_per_file * 2] = values[-1]
    index = audformat.filewise_index(db.files[:2])
    column.set(values[-1], index=index)
    pd.testing.assert_series_equal(
        column.get(index),
        series[0 : num_segments_per_file * 2],
    )

    # set series
    series[:] = np.nan
    table.df["column"] = np.nan
    column.set(series)
    pd.testing.assert_series_equal(column.get(), series)

    # test df
    series[:] = np.nan
    table.df["column"] = np.nan
    pd.testing.assert_series_equal(table.df[column_id], series)


@pytest.mark.parametrize(
    "timezone, values, expected_dates",
    [
        (
            "UTC",
            [1],
            [pd.Timestamp("1970-01-01T00:00:00.000000001")],
        ),
        (
            "Europe/Berlin",
            [1],
            [pd.Timestamp("1970-01-01T01:00:00.000000001")],
        ),
    ],
)
def test_set_dates(timezone, values, expected_dates):
    # Ensure we handle dates with different time zones,
    # see https://github.com/audeering/audformat/issues/364
    db = audformat.Database("db")
    db.schemes["date"] = audformat.Scheme("date")
    index = audformat.filewise_index([f"f{n}" for n in range(len(values))])
    db["table"] = audformat.Table(index)
    db["table"]["column"] = audformat.Column(scheme_id="date")
    dates = pd.to_datetime(values, utc=True)
    dates = dates.tz_convert(timezone)
    db["table"]["column"].set(dates)
    assert list(db["table"].df.column) == expected_dates


def test_set_labels():
    # Ensure we handle NaN when setting labels
    # from Series and ndarray,
    # see https://github.com/audeering/audformat/issues/89
    db = pytest.DB
    scheme_id = "label_test"
    db.schemes[scheme_id] = audformat.Scheme(
        labels={
            "h": {"category": "happy"},
            "n": {"category": "neutral"},
            "b": {"category": "bored"},
            "a": {"category": "angry"},
        }
    )
    db[scheme_id] = audformat.Table(audformat.filewise_index(["a", "b", "c", "d", "e"]))
    db[scheme_id][scheme_id] = audformat.Column(scheme_id=scheme_id)
    y = pd.Series([np.nan, "b", "h", "a", "n"])
    db[scheme_id][scheme_id].set(y)  # Series
    db[scheme_id][scheme_id].set(y.values)  # ndarray
    # Clean up database
    db.drop_tables(scheme_id)
    db.schemes.pop(scheme_id, None)


def test_set_invalid_values():
    db = pytest.DB
    num = len(db["files"])

    # Values that do match scheme are converted if possible
    table = db["files"].copy()  # use copy as we change values here
    # string -> float
    table["float"].set(["-1.0"] * num)
    error_msg = re.escape("could not convert string to float: 'a'")
    with pytest.raises(ValueError, match=error_msg):
        table["float"].set("a")
    # float -> string
    table["string"].set(1.0)
    error_msg = re.escape("Some value(s) do not match scheme")
    with pytest.raises(ValueError, match=error_msg):
        db["files"]["label"].set(1.0)

    with pytest.raises(ValueError):
        db["files"]["float"].set(-2.0)

    with pytest.raises(ValueError):
        db["files"]["float"].set([-2.0] * num)

    with pytest.raises(ValueError):
        db["files"]["float"].set(np.array([-2.0] * num))

    with pytest.raises(ValueError):
        db["files"]["float"].set(pd.Series([-2.0] * num))

    with pytest.raises(ValueError):
        db["files"]["float"].set(2.0)

    with pytest.raises(ValueError):
        db["files"]["float"].set([2.0] * num)

    with pytest.raises(ValueError):
        db["files"]["float"].set(np.array([2.0] * num))

    with pytest.raises(ValueError):
        db["files"]["float"].set(pd.Series([-2.0] * num))

    with pytest.raises(ValueError):
        db["files"]["label"].set("bad")

    with pytest.raises(ValueError):
        db["files"]["label"].set(["bad"] * len(db["files"]))

    with pytest.raises(ValueError):
        db["files"]["label"].set(np.array(["bad"] * len(db["files"])))

    with pytest.raises(ValueError):
        db["files"]["label"].set(pd.Series(["bad"] * len(db["files"])))
