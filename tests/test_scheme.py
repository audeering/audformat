import re

import numpy as np
import pandas as pd
import pytest

import audformat
import audformat.testing


def test_scheme_assign_values():
    db = audformat.testing.create_db(minimal=True)
    speakers = ["spk1", "spk2", "spk3"]
    ages = [33, 44, 55]
    index = pd.Index(speakers, name="speaker", dtype="string")
    db.schemes["age"] = audformat.Scheme("int", minimum=0)
    db["misc"] = audformat.MiscTable(index)
    db["misc"]["age"] = audformat.Column(scheme_id="age")
    db["misc"]["age"].set(ages)
    db.schemes["scheme"] = audformat.Scheme(labels="misc", dtype="str")
    db["table"] = audformat.Table(audformat.filewise_index(["f1", "f2", "f3"]))
    db["table"]["speaker"] = audformat.Column(scheme_id="scheme")
    db["table"]["speaker"].set(speakers)

    assert list(db["table"]["speaker"].get()) == speakers
    assert list(db["table"]["speaker"].get(map="age")) == ages
    assert list(db["table"].get(map={"speaker": "age"})["age"]) == ages


@pytest.mark.parametrize(
    "scheme, values, error, error_msg_last_part",
    [
        (
            audformat.Scheme("str", labels=["spk1", "spk2"]),
            ["spk4", "spk5", "spk6"],
            ValueError,
            "'spk4', 'spk5', 'spk6'",
        ),
        (
            audformat.Scheme("str", labels=["spk1", "spk2"]),
            ["spk4", "spk5", None],
            ValueError,
            "'spk4', 'spk5'",
        ),
        (
            audformat.Scheme("int", labels=[20, 30]),
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            ValueError,
            "0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ...",
        ),
        (
            audformat.Scheme("int", minimum=0),
            [0, -1, -2],
            ValueError,
            "minimum -2 smaller than scheme minimum",
        ),
        (
            audformat.Scheme("int", maximum=0),
            [0, 1, 2],
            ValueError,
            "maximum 2 larger than scheme maximum",
        ),
        (
            audformat.Scheme("int", minimum=-1, maximum=1),
            [0, -2, 2],
            ValueError,
            (
                "minimum -2 smaller than scheme minimum\n"
                "maximum 2 larger than scheme maximum"
            ),
        ),
        (
            audformat.Scheme("int", minimum=-1, maximum=1),
            [np.nan, -2, 2],
            ValueError,
            (
                "minimum -2 smaller than scheme minimum\n"
                "maximum 2 larger than scheme maximum"
            ),
        ),
        (
            audformat.Scheme("int", minimum=0),
            [np.nan, np.nan, np.nan],
            None,
            None,
        ),
        (
            audformat.Scheme("str", labels=["spk1", "spk2"]),
            [None, None, None],
            None,
            None,
        ),
    ],
)
def test_scheme_assign_bad_values(scheme, values, error, error_msg_last_part):
    """Test setting of values not matching scheme."""
    db = audformat.testing.create_db(minimal=True)
    db["table"] = audformat.Table(audformat.filewise_index(["f1", "f2", "f3"]))
    db.schemes["scheme"] = scheme
    db["table"]["column"] = audformat.Column(scheme_id="scheme")

    error_msg = re.escape(
        "Some value(s) do not match scheme\n"
        f"{db.schemes['scheme']}\n"
        "with scheme ID 'scheme':\n"
        f"{error_msg_last_part}\n"
    )
    if error is None:
        db["table"]["column"].set(values)
    else:
        with pytest.raises(error, match=error_msg):
            db["table"]["column"].set(values)


@pytest.mark.parametrize(
    "scheme, values",
    [
        (
            audformat.Scheme(
                audformat.define.DataType.FLOAT,
            ),
            [1.0, 2.0],
        ),
        (
            audformat.Scheme(
                audformat.define.DataType.FLOAT,
                minimum=1.0,
                maximum=2.0,
            ),
            [1.0, 2.0],
        ),
        pytest.param(  # minimum too low
            audformat.Scheme(
                audformat.define.DataType.FLOAT,
                minimum=1.0,
                maximum=2.0,
            ),
            [0.0, 2.0],
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(  # maximum too high
            audformat.Scheme(
                audformat.define.DataType.FLOAT,
                minimum=1.0,
                maximum=2.0,
            ),
            [1.0, 3.0],
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        # Finding minimum and maximum in array
        # should nopt be affected by NaN, see
        # https://github.com/audeering/audformat/issues/305
        pytest.param(  # minimum too low
            audformat.Scheme(
                audformat.define.DataType.FLOAT,
                minimum=1.0,
                maximum=2.0,
            ),
            np.array([0.0, np.nan, 2.0]),
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(  # maximum too high
            audformat.Scheme(
                audformat.define.DataType.FLOAT,
                minimum=1.0,
                maximum=2.0,
            ),
            np.array([1.0, np.nan, 3.0]),
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        # Make sure we convert None to float
        # before checking minimum/maximum, see
        # https://github.com/audeering/audformat/issues/307
        (
            audformat.Scheme(
                audformat.define.DataType.FLOAT,
                minimum=1.0,
                maximum=2.0,
            ),
            np.array([1.0, None, 2.0]),
        ),
        pytest.param(  # minimum too low
            audformat.Scheme(
                audformat.define.DataType.FLOAT,
                minimum=1.0,
                maximum=2.0,
            ),
            np.array([0.0, None, 2.0]),
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(  # maximum too high
            audformat.Scheme(
                audformat.define.DataType.FLOAT,
                minimum=1.0,
                maximum=2.0,
            ),
            np.array([1.0, None, 3.0]),
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        # Using 0.0 failed before, see
        # https://github.com/audeering/audformat/issues/304
        pytest.param(  # minimum too low
            audformat.Scheme(
                audformat.define.DataType.FLOAT,
                minimum=0.0,
                maximum=2.0,
            ),
            np.array([-1.0, 2.0]),
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(  # maximum too high
            audformat.Scheme(
                audformat.define.DataType.FLOAT,
                minimum=-1.0,
                maximum=0.0,
            ),
            np.array([1.0, 2.0]),
            marks=pytest.mark.xfail(raises=ValueError),
        ),
    ],
)
def test_scheme_minimum_maximum(scheme, values):
    db = audformat.testing.create_db(minimal=True)
    audformat.testing.add_table(
        db,
        "table",
        audformat.define.IndexType.FILEWISE,
        num_files=len(values),
    )
    db.schemes["scheme"] = scheme
    db["table"]["column"] = audformat.Column(scheme_id="scheme")

    table = db["table"]
    column = db["table"]["column"]
    column_id = "column"

    expected_series = pd.Series(
        values,
        audformat.filewise_index(table.files),
        name=column_id,
        dtype="float64",
    )
    column.set(values)
    pd.testing.assert_series_equal(column.get(), expected_series)


def test_scheme_contains():
    db = pytest.DB

    assert "tests" in db.schemes["string"]
    assert 0.0 in db.schemes["float"]
    assert 1.0 in db.schemes["float"]
    assert -1.0 in db.schemes["float"]
    assert 2.0 not in db.schemes["float"]
    assert -2.0 not in db.schemes["float"]
    assert pd.Timedelta(0.5, unit="s") in db.schemes["time"]
    assert "label1" in db.schemes["label"]
    assert "label1" in db.schemes["label_map_str"]
    assert 1 in db.schemes["label_map_int"]

    # Misc table
    # assigned scheme
    assert "label1" in db.schemes["label_map_misc"]
    # remove table
    # db.drop_tables(['misc'])
    db.misc_tables.pop("misc")
    assert "label1" not in db.schemes["label_map_misc"]
    # unassigned scheme
    scheme = audformat.Scheme(labels="misc", dtype="str")
    assert "label1" not in scheme


@pytest.mark.parametrize(
    "dtype, values",
    [
        pytest.param(  # bool not supported
            audformat.define.DataType.BOOL,
            [True, False],
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        (
            audformat.define.DataType.DATE,
            pd.to_datetime(["3/11/2000", "3/12/2000", "3/13/2000"]),
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
            ["a", "b", "c"],
        ),
        (
            audformat.define.DataType.TIME,
            pd.to_timedelta(["1s", "2s", "3s"]),
        ),
    ],
)
def test_scheme_dtypes(dtype, values):
    db = audformat.Database("test")
    pandas_dtype = audformat.core.common.to_pandas_dtype(dtype)
    index = pd.Index(values, name="labels", dtype=pandas_dtype)
    db["misc"] = audformat.MiscTable(index)
    index = audformat.filewise_index([f"f{idx}" for idx in range(len(values))])
    db.schemes["scheme"] = audformat.Scheme(dtype=dtype, labels="misc")
    db["table"] = audformat.Table(index)
    db["table"]["labels"] = audformat.Column(scheme_id="scheme")
    db["table"]["labels"].set(values)

    assert set(db["table"]["labels"].get()) == set(values)


def test_scheme_errors():
    db = audformat.Database("test")

    # dtype mismatch
    error_msg = "Data type is set to 'str', " "but data type of labels is 'int'."
    with pytest.raises(ValueError, match=error_msg):
        audformat.Scheme(
            audformat.define.DataType.STRING,
            labels=[1, 2, 3],
        )

    # unknown type
    error_msg = (
        "Bad value 'bad', "
        "expected one of "
        "\\['bool', 'date', 'float', 'int', 'object', 'str', 'time'\\]"
    )
    with pytest.raises(ValueError, match=error_msg):
        audformat.Scheme("bad")

    # labels not list or dictionary
    error_msg = "Labels must be passed " "as a dictionary, list or ID of a misc table."
    with pytest.raises(ValueError, match=error_msg):
        audformat.Scheme(labels={1, 2, 3})

    # labels do not have the same type
    error_msg = "All labels must be of the same data type."
    with pytest.raises(ValueError, match=error_msg):
        audformat.Scheme(labels=[1, "2", 3])

    # update labels when scheme has no label
    error_msg = "Cannot replace labels when " "scheme does not define labels."
    with pytest.raises(ValueError, match=error_msg):
        scheme = audformat.Scheme(audformat.define.DataType.INTEGER)
        scheme.replace_labels(["a", "b"])

    # misc table needs to define data type
    error_msg = "'dtype' has to be provided " "when using a misc table as labels."
    with pytest.raises(ValueError, match=error_msg):
        audformat.Scheme(labels="misc")

    # misc table not assigned to a database
    error_msg = (
        "The misc table 'misc' used as scheme labels "
        "needs to be assigned to the database"
    )
    scheme = audformat.Scheme(labels="misc", dtype="str")
    with pytest.raises(ValueError, match=error_msg):
        db.schemes["misc"] = scheme

    # filewise table used instead of misc table
    error_msg = "The table 'table' used as scheme labels " "needs to be a misc table."
    db["table"] = audformat.Table(audformat.filewise_index(["f1"]))
    scheme = audformat.Scheme(labels="table", dtype="str")
    with pytest.raises(ValueError, match=error_msg):
        db.schemes["misc"] = scheme

    # misc table has different dtype
    error_msg = (
        "Data type is set to 'str', " "but data type of labels in misc table is 'int'."
    )
    db["misc"] = audformat.MiscTable(pd.Index([0], name="misc"))
    scheme = audformat.Scheme(labels="misc", dtype="str")
    with pytest.raises(ValueError, match=error_msg):
        db.schemes["misc"] = scheme

    # misc table should only contain a one-dimensional index
    error_msg = (
        "Index of misc table 'misc' used as scheme labels "
        "is only allowed to have a single level."
    )
    db["misc"] = audformat.MiscTable(
        pd.MultiIndex.from_arrays(
            [
                [1, 2],
                ["a", "b"],
            ],
            names=["misc-1", "misc-2"],
        )
    )
    scheme = audformat.Scheme(labels="misc", dtype="str")
    with pytest.raises(ValueError, match=error_msg):
        db.schemes["misc"] = scheme

    # misc table should not contain duplicates
    error_msg = (
        "Index of misc table 'misc' used as scheme labels "
        "is not allowed to contain duplicates."
    )
    db["misc"] = audformat.MiscTable(pd.Index([0, 0], name="misc"))
    scheme = audformat.Scheme(labels="misc", dtype="int")
    with pytest.raises(ValueError, match=error_msg):
        db.schemes["misc"] = scheme

    # cannot assign misc table scheme to itself
    db["misc"] = audformat.MiscTable(pd.Index([], name="misc"))
    db.schemes["scheme"] = audformat.Scheme(labels="misc", dtype="object")
    error_msg = (
        "Scheme 'scheme' uses misc table 'misc' as labels "
        "and cannot be used with columns of the same table."
    )
    with pytest.raises(ValueError, match=error_msg):
        db["misc"]["column"] = audformat.Column(scheme_id="scheme")

    # cannot assign column of a misc table used in a scheme
    # to a scheme that already uses a misc table
    db["misc2"] = audformat.MiscTable(pd.Index([], name="misc2"))
    db.schemes["scheme2"] = audformat.Scheme(labels="misc2", dtype="object")
    error_msg = (
        "Since the misc table 'misc2' "
        "is used as labels in scheme 'scheme2' "
        "its columns cannot be used with a scheme "
        "that also uses labels from a misc table."
    )
    with pytest.raises(ValueError, match=error_msg):
        db["misc2"]["column"] = audformat.Column(scheme_id="scheme")

    # Cannot assign misc table as scheme if one of its column
    # is linked to a scheme that already uses a misc table
    db["misc3"] = audformat.MiscTable(pd.Index([], name="misc3"))
    db["misc3"]["column"] = audformat.Column(scheme_id="scheme")
    scheme = audformat.Scheme(labels="misc3", dtype="object")
    error_msg = (
        "The misc table 'misc3' cannot be used as scheme labels "
        "when one of its columns is assigned to a scheme "
        "that uses labels from a misc table."
    )
    with pytest.raises(ValueError, match=error_msg):
        db.schemes["scheme3"] = scheme


def test_scheme_labels():
    # No labels
    s = audformat.Scheme("str")
    assert s.labels is None
    assert s.labels_as_list == []

    # List of labels
    labels = ["a", "b"]
    s = audformat.Scheme("str", labels=labels)
    assert s.labels == labels
    assert s.labels_as_list == labels

    # Dict of labels
    labels = {"a": 0, "b": 1}
    s = audformat.Scheme("str", labels=labels)
    assert s.labels == labels
    assert s.labels_as_list == list(labels)

    # Misc table as labels
    labels = ["a", "b"]
    db = audformat.Database("db")
    index = pd.Index(labels, name="labels", dtype="string")
    db["labels"] = audformat.MiscTable(index)
    db.schemes["s"] = audformat.Scheme("str", labels="labels")
    assert db.schemes["s"].labels == "labels"
    assert db.schemes["s"].labels_as_list == labels


@pytest.mark.parametrize(
    "values, labels, new_labels, expected",
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
                dtype=pd.CategoricalDtype(["a", "b"]),
            ),
            ["a", "b"],
            ["b", "c", "d"],
            pd.Series(
                index=audformat.filewise_index(),
                dtype=pd.CategoricalDtype(["b", "c", "d"]),
            ),
        ),
        (
            pd.Series(
                ["a", "b"],
                index=audformat.filewise_index(["f1", "f2"]),
                dtype=pd.CategoricalDtype(["a", "b"]),
            ),
            ["a", "b"],
            ["a"],
            pd.Series(
                ["a", None],
                index=audformat.filewise_index(["f1", "f2"]),
                dtype=pd.CategoricalDtype(["a"]),
            ),
        ),
        (
            pd.Series(
                ["a", "b"],
                index=audformat.filewise_index(["f1", "f2"]),
                dtype=pd.CategoricalDtype(["a", "b"]),
            ),
            ["a", "b"],
            ["a", "b", "c"],
            pd.Series(
                ["a", "b"],
                index=audformat.filewise_index(["f1", "f2"]),
                dtype=pd.CategoricalDtype(["a", "b", "c"]),
            ),
        ),
        (
            pd.Series(
                ["a", "b"],
                index=audformat.filewise_index(["f1", "f2"]),
                dtype=pd.CategoricalDtype(["a", "b"]),
            ),
            ["a", "b"],
            ["c"],
            pd.Series(
                [None, None],
                index=audformat.filewise_index(["f1", "f2"]),
                dtype=pd.CategoricalDtype(["c"]),
            ),
        ),
        (
            pd.Series(
                ["a", "b"],
                index=audformat.filewise_index(["f1", "f2"]),
                dtype=pd.CategoricalDtype(["a", "b"]),
            ),
            ["a", "b"],
            [],
            pd.Series(
                [None, None],
                index=audformat.filewise_index(["f1", "f2"]),
                dtype=pd.CategoricalDtype([]),
            ),
        ),
        (
            pd.Series(
                [0, 1],
                index=audformat.filewise_index(["f1", "f2"]),
                dtype=pd.CategoricalDtype([0, 1]),
            ),
            {0: "a", 1: "b"},
            {1: {"b": "B"}, 2: {"c": "C"}},
            pd.Series(
                [None, 1],
                index=audformat.filewise_index(["f1", "f2"]),
                dtype=pd.CategoricalDtype([1, 2]),
            ),
        ),
        (
            pd.Series(
                [0, 1],
                index=audformat.filewise_index(["f1", "f2"]),
                dtype=pd.CategoricalDtype([0, 1]),
            ),
            {0: "a", 1: "b"},
            [1, 2],
            pd.Series(
                [None, 1],
                index=audformat.filewise_index(["f1", "f2"]),
                dtype=pd.CategoricalDtype([1, 2]),
            ),
        ),
        # error: dtype of labels does not match
        pytest.param(
            pd.Series(
                index=audformat.filewise_index(),
                dtype=pd.CategoricalDtype(),
            ),
            ["a", "b"],
            [0, 1],
            pd.Series(
                index=audformat.filewise_index(),
                dtype=pd.CategoricalDtype(),
            ),
            marks=pytest.mark.xfail(raises=ValueError),
        ),
    ],
)
def test_replace_labels(values, labels, new_labels, expected):
    db = audformat.testing.create_db(minimal=True)
    db.schemes["scheme"] = audformat.Scheme(labels=labels)
    db["table"] = audformat.Table(index=values.index)
    db["table"]["columns"] = audformat.Column(scheme_id="scheme")
    db["table"]["columns"].set(values)
    db.schemes["scheme"].replace_labels(new_labels)
    pd.testing.assert_series_equal(
        db["table"]["columns"].get(),
        expected,
        check_names=False,
    )


def test_replace_labels_misc_table():
    db = audformat.testing.create_db(minimal=True)
    db["labels"] = audformat.MiscTable(
        pd.Index(
            ["a", "b", "c", "d"],
            dtype="string",
            name="idx",
        )
    )

    # non-assigned scheme
    scheme = audformat.Scheme(
        audformat.define.DataType.STRING,
        labels="labels",
    )
    assert scheme.labels == "labels"
    assert scheme._labels_to_dict() == {}

    # replace with non-existing misc table scheme
    # and back again
    scheme.replace_labels("misc-non-existing")
    scheme.replace_labels("labels")

    # assigned scheme
    db.schemes["scheme"] = scheme
    assert scheme.labels == "labels"
    assert scheme._labels_to_dict() == {"a": {}, "b": {}, "c": {}, "d": {}}

    # use scheme in (miscellaneous) table
    db["table"] = audformat.Table(
        index=audformat.filewise_index(["f1", "f2", "f3", "f4"]),
    )
    db["table"]["column"] = audformat.Column(scheme_id="scheme")
    db["table"]["column"].set(["a", "b", "c", "d"])
    db["misc"] = audformat.MiscTable(
        index=pd.Index(
            [0, 1, 2, 3],
            name="idx",
        ),
    )
    db["misc"]["column"] = audformat.Column(scheme_id="scheme")
    db["misc"]["column"].set(["a", "b", "c", "d"])

    # replace with new misc table scheme
    db["labels-new"] = audformat.MiscTable(
        pd.Index(
            ["b", "c", "d", "e"],
            dtype="string",
            name="idx",
        )
    )
    scheme.replace_labels("labels-new")
    assert scheme.labels == "labels-new"
    assert scheme._labels_to_dict() == {"b": {}, "c": {}, "d": {}, "e": {}}
    expected = [np.nan, "b", "c", "d"]
    assert list(db["table"]["column"].get().values) == expected
    assert list(db["misc"]["column"].get().values) == expected

    # extend labels
    db["labels-new"].extend_index(
        pd.Index(
            ["f"],
            dtype="string",
            name="idx",
        ),
        inplace=True,
    )
    expected = {"b": {}, "c": {}, "d": {}, "e": {}, "f": {}}
    assert scheme._labels_to_dict() == expected
    assert "f" in db["table"]["column"].get().dtype.categories
    assert "f" in db["misc"]["column"].get().dtype.categories

    # pick labels
    db["labels-new"].pick_index(
        pd.Index(
            ["c", "d", "e"],
            dtype="string",
            name="idx",
        ),
        inplace=True,
    )
    assert scheme._labels_to_dict() == {"c": {}, "d": {}, "e": {}}
    expected = [np.nan, np.nan, "c", "d"]
    assert list(db["table"]["column"].get().values) == expected
    assert list(db["misc"]["column"].get().values) == expected

    # drop labels
    db["labels-new"].drop_index(
        pd.Index(
            ["c"],
            dtype="string",
            name="idx",
        ),
        inplace=True,
    )
    assert scheme._labels_to_dict() == {"d": {}, "e": {}}
    expected = [np.nan, np.nan, np.nan, "d"]
    assert list(db["table"]["column"].get().values) == expected
    assert list(db["misc"]["column"].get().values) == expected

    # replace with dictionary
    scheme.replace_labels({"e": {}})
    assert scheme.labels == {"e": {}}
    assert scheme._labels_to_dict() == {"e": {}}
    expected = [np.nan, np.nan, np.nan, np.nan]
    assert list(db["table"]["column"].get().values) == expected
    assert list(db["misc"]["column"].get().values) == expected

    # replace again with misc table
    scheme.replace_labels("labels")
    assert scheme.labels == "labels"
    assert scheme._labels_to_dict() == {"a": {}, "b": {}, "c": {}, "d": {}}

    # replace with list
    scheme.replace_labels(["a", "b"])
    assert scheme.labels == ["a", "b"]
    assert scheme._labels_to_dict() == {"a": {}, "b": {}}

    # replace with non-existing misc table scheme
    error_msg = (
        "The misc table 'misc-non-existing' used as scheme labels "
        "needs to be assigned to the database."
    )
    with pytest.raises(ValueError, match=error_msg):
        scheme.replace_labels("misc-non-existing")

    # replace labels with a misc table that
    # has a column that already links a misc table
    scheme.replace_labels("labels")
    db["labels-new"]["column"] = audformat.Column(scheme_id="scheme")
    error_msg = (
        "The misc table 'labels-new' cannot be used as scheme labels "
        "when one of its columns is assigned to a scheme "
        "that uses labels from a misc table."
    )
    with pytest.raises(ValueError, match=error_msg):
        scheme.replace_labels("labels-new")
