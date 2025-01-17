from __future__ import annotations

import os
import random
import re
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as parquet
import pytest

import audeer

import audformat
import audformat.testing


def create_db_table(
    obj: pd.Series | pd.DataFrame = None,
    *,
    rater: audformat.Rater = None,
    media: audformat.Media = None,
    split: audformat.Split = None,
    scheme_id: str = None,  # overwrite id of first scheme
) -> audformat.Table:
    if obj is None:
        obj = pd.Series(
            index=audformat.filewise_index(),
            dtype=float,
        )
    db = audformat.testing.create_db(data={"table": obj})
    table = db["table"]
    if rater is not None:
        db.raters["rater"] = rater
        for column in table.columns.values():
            column.rater_id = "rater"
    if media is not None:
        db.media["media"] = media
        table.media_id = "media"
    if split is not None:
        db.splits["split"] = split
        table.split_id = "split"
    if scheme_id is not None:
        old_scheme_id = list(db.schemes)[0]
        db.schemes[scheme_id] = db.schemes.pop(old_scheme_id)
        for column in table.columns.values():
            if column.scheme_id == old_scheme_id:
                column.scheme_id = scheme_id
    return table


def create_table(
    obj: pd.Index | pd.Series | pd.DataFrame,
) -> audformat.Table:
    r"""Helper function to create Table."""
    index = obj if isinstance(obj, pd.Index) else obj.index
    table = audformat.Table(index)
    if not isinstance(obj, pd.Index):
        if isinstance(obj, pd.Series):
            obj = obj.to_frame()
        for name in obj:
            table[name] = audformat.Column()
            table[name].set(obj[name].values)
        # change 'int64' to 'Int64'
        dtypes = {
            name: "Int64" if pd.api.types.is_integer_dtype(dtype) else dtype
            for name, dtype in obj.dtypes.items()
        }
        table._df = table.df.astype(dtypes)
    return table


def test_access():
    db = audformat.testing.create_db()
    for table_id, table in db.tables.items():
        assert db.tables[table_id] == db[table_id]
        assert str(db.tables[table_id]) == str(db[table_id])
        if table.media_id is not None:
            assert table.media == db.media[table.media_id]
        else:
            assert table.media is None
        if table.split_id is not None:
            assert table.split == db.splits[table.split_id]
        else:
            assert table.split is None


@pytest.mark.parametrize(
    "tables, expected",
    [
        # empty
        (
            [
                create_table(
                    pd.Series(
                        index=audformat.filewise_index(),
                        dtype=float,
                    )
                ),
            ],
            create_table(
                pd.Series(
                    index=audformat.filewise_index(),
                    dtype=float,
                )
            ),
        ),
        (
            [
                create_table(
                    pd.Series(
                        index=audformat.filewise_index(),
                        dtype=float,
                    )
                ),
            ]
            * 3,
            create_table(
                pd.Series(
                    index=audformat.filewise_index(),
                    dtype=float,
                )
            ),
        ),
        # content + empty
        (
            [
                create_table(
                    pd.Series(
                        [1.0],
                        index=audformat.filewise_index("f1"),
                    )
                ),
                create_table(
                    pd.Series(
                        index=audformat.filewise_index(),
                        dtype=float,
                    )
                ),
            ],
            create_table(
                pd.Series(
                    [1.0],
                    index=audformat.filewise_index("f1"),
                )
            ),
        ),
        # empty + content
        (
            [
                create_table(
                    pd.Series(
                        index=audformat.filewise_index(),
                        dtype=float,
                    )
                ),
                create_table(
                    pd.Series(
                        [1.0],
                        index=audformat.filewise_index("f1"),
                    )
                ),
            ],
            create_table(
                pd.Series(
                    [1.0],
                    index=audformat.filewise_index("f1"),
                )
            ),
        ),
        # filewise + segmented
        (
            [
                create_table(
                    pd.Series(
                        index=audformat.filewise_index(),
                        dtype=float,
                        name="c1",
                    )
                ),
                create_table(
                    pd.Series(
                        index=audformat.segmented_index(),
                        dtype=float,
                        name="c2",
                    )
                ),
            ],
            create_table(
                pd.DataFrame(
                    {
                        "c1": pd.Series(
                            index=audformat.segmented_index(),
                            dtype=float,
                            name="c1",
                        ),
                        "c2": pd.Series(
                            index=audformat.segmented_index(),
                            dtype=float,
                            name="c2",
                        ),
                    },
                )
            ),
        ),
        # same column
        (
            [
                create_table(
                    pd.Series(
                        [1.0, np.nan],
                        index=audformat.filewise_index(["f1", "f2"]),
                    )
                ),
                create_table(
                    pd.Series(
                        [2.0, 3.0],
                        index=audformat.filewise_index(["f2", "f3"]),
                    )
                ),
                create_table(
                    pd.Series(
                        [3.0, 4.0],
                        index=audformat.filewise_index(["f3", "f4"]),
                    )
                ),
            ],
            create_table(
                pd.Series(
                    [1.0, 2.0, 3.0, 4.0],
                    index=audformat.filewise_index(["f1", "f2", "f3", "f4"]),
                )
            ),
        ),
        pytest.param(  # value mismatch
            [
                create_table(
                    pd.Series(
                        [1.0],
                        index=audformat.filewise_index("f1"),
                    )
                ),
                create_table(
                    pd.Series(
                        [-1.0],
                        index=audformat.filewise_index("f1"),
                    )
                ),
            ],
            None,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        # different columns
        (
            [
                create_table(
                    pd.Series(
                        [1.0, 2.0],
                        index=audformat.filewise_index(["f1", "f2"]),
                        name="c1",
                    )
                ),
                create_table(
                    pd.Series(
                        [2.0, 3.0],
                        index=audformat.filewise_index(["f2", "f3"]),
                        name="c2",
                    )
                ),
            ],
            create_table(
                pd.DataFrame(
                    {
                        "c1": [1.0, 2.0, np.nan],
                        "c2": [np.nan, 2.0, 3.0],
                    },
                    index=audformat.filewise_index(["f1", "f2", "f3"]),
                )
            ),
        ),
        # filewise + segmented
        (
            [
                create_table(
                    pd.Series(
                        [1.0],
                        index=audformat.filewise_index("f1"),
                    )
                ),
                create_table(
                    pd.Series(
                        [1.0],
                        index=audformat.segmented_index("f1", 0, 1),
                    )
                ),
            ],
            create_table(
                pd.Series(
                    [1.0, 1.0],
                    index=audformat.segmented_index(
                        ["f1", "f1"],
                        [0, 0],
                        [None, 1],
                    ),
                )
            ),
        ),
        (
            [
                pytest.DB["files"],
                pytest.DB["segments"],
            ],
            create_table(
                audformat.utils.concat(
                    [
                        pytest.DB["files"].df,
                        pytest.DB["segments"].df,
                    ]
                )
            ),
        ),
    ],
)
def test_add(tables, expected):
    table = tables[0]
    for other in tables[1:]:
        table += other
    assert table.media_id is None
    assert table.split_id is None
    for column in table.columns.values():
        assert column.scheme_id is None
        assert column.rater_id is None
    assert table == expected


def test_add_2():
    db = audformat.testing.create_db(minimal=True)
    db.media["media"] = audformat.Media()
    db.splits["split"] = audformat.Split()

    # empty tables

    audformat.testing.add_table(
        db,
        "table1",
        audformat.define.IndexType.FILEWISE,
        p_none=0.25,
        num_files=5,
        media_id="media",
        split_id="split",
    )
    audformat.testing.add_table(
        db,
        "table2",
        audformat.define.IndexType.FILEWISE,
        p_none=0.25,
        num_files=[1, 6, 7, 8, 9],
        media_id="media",
    )
    db["table"] = db["table1"] + db["table2"]
    pd.testing.assert_index_equal(
        db["table"].files, db["table1"].files.union(db["table2"].files)
    )
    assert db["table"].media_id is None
    assert db["table"].split_id is None
    for column in db["table"].columns:
        assert column.scheme_id is None
        assert column.rater_id is None

    # add table to itself

    pd.testing.assert_frame_equal(
        (db["table1"] + db["table1"]).df,
        db["table1"].df,
    )

    # add two schemes

    db.schemes["scheme1"] = audformat.Scheme()
    db.schemes["scheme2"] = audformat.Scheme()

    # tables of same type without overlap

    for table_type in [
        audformat.define.IndexType.FILEWISE,
        audformat.define.IndexType.SEGMENTED,
    ]:
        db.drop_tables(list(db.tables))
        audformat.testing.add_table(
            db,
            "table1",
            table_type,
            num_files=5,
            columns=["scheme1"],
        )
        audformat.testing.add_table(
            db,
            "table2",
            table_type,
            num_files=(6, 7, 8, 9, 10),
            columns="scheme1",
        )
        db["table"] = db["table1"] + db["table2"]
        pd.testing.assert_frame_equal(
            db["table"].get(), pd.concat([db["table1"].get(), db["table2"].get()])
        )

    # tables of same type with overlap

    db.drop_tables(list(db.tables))
    audformat.testing.add_table(
        db,
        "table1",
        audformat.define.IndexType.FILEWISE,
        num_files=(1, 2),
        columns="scheme1",
    )
    audformat.testing.add_table(
        db,
        "table2",
        audformat.define.IndexType.FILEWISE,
        num_files=(1,),
        columns="scheme1",
    )
    db["table2"].df.iloc[0] = np.nan  # ok if other value is nan
    db["table"] = db["table1"] + db["table2"]
    pd.testing.assert_series_equal(
        db["table"]["scheme1"].get(), db["table1"]["scheme1"].get()
    )
    with pytest.raises(ValueError):
        db["table2"].df.iloc[0] = "do not match"  # values do not match
        db["table"] = db["table1"] + db["table2"]

    # filewise with segmented table

    for num_files_1, num_files_2 in (
        (5, 5),
        (5, 4),
        (4, 5),
    ):
        db.drop_tables(list(db.tables))
        audformat.testing.add_table(
            db,
            "table1",
            audformat.define.IndexType.FILEWISE,
            columns="scheme1",
            num_files=num_files_1,
        )
        audformat.testing.add_table(
            db,
            "table2",
            audformat.define.IndexType.SEGMENTED,
            columns="scheme2",
            num_files=num_files_2,
        )
        db["table"] = db["table1"] + db["table2"]
        assert db["table"].type == audformat.define.IndexType.SEGMENTED
        np.testing.assert_equal(
            db["table"]["scheme1"].get().dropna().unique(),
            db["table1"]["scheme1"].get().values,
        )
        pd.testing.assert_series_equal(
            db["table"]["scheme2"].get().dropna(),
            db["table2"]["scheme2"].get().dropna(),
        )

    # segmented with filewise table

    for num_files_1, num_files_2 in (
        (5, 5),
        (5, 4),
        (4, 5),
    ):
        db.drop_tables(list(db.tables))
        audformat.testing.add_table(
            db,
            "table1",
            audformat.define.IndexType.SEGMENTED,
            columns="scheme1",
            num_files=num_files_1,
        )
        audformat.testing.add_table(
            db,
            "table2",
            audformat.define.IndexType.FILEWISE,
            columns="scheme2",
            num_files=num_files_2,
        )
        db["table"] = db["table1"] + db["table2"]
        assert db["table"].type == audformat.define.IndexType.SEGMENTED
        np.testing.assert_equal(
            db["table"]["scheme2"].get().dropna().unique(),
            db["table2"]["scheme2"].get().values,
        )
        pd.testing.assert_series_equal(
            db["table"]["scheme1"].get().dropna(),
            db["table1"]["scheme1"].get().dropna(),
        )


@pytest.mark.parametrize(
    "table",
    [
        audformat.Table(),
        pytest.DB["files"],
        pytest.DB["segments"],
    ],
)
def test_copy(table):
    table_copy = table.copy()
    assert str(table_copy) == str(table)
    pd.testing.assert_frame_equal(table_copy.df, table.df)


@pytest.mark.parametrize(
    "inplace",
    [
        False,
        True,
    ],
)
def test_drop_and_pick_columns(inplace):
    db = audformat.testing.create_db()

    assert "float" in db["files"].columns
    table = db["files"].drop_columns("float", inplace=inplace)
    if not inplace:
        assert "float" in db["files"].columns
    assert "float" not in table.columns
    assert "string" in table.columns
    table = db["files"].pick_columns("time", inplace=inplace)
    assert "time" in table.columns
    assert "string" not in table.columns
    if not inplace:
        assert "string" in db["files"].columns


def test_drop_and_pick_index():
    for table in ["files", "segments"]:
        index = pytest.DB[table].index[:5]
        df_pick = pytest.DB[table].pick_index(index).get()
        index = pytest.DB[table].index[5:]
        df_drop = pytest.DB[table].drop_index(index).get()

        assert len(df_pick) == len(df_drop) == 5
        pd.testing.assert_frame_equal(df_pick, df_drop)

    index = pytest.DB["segments"].index[:5]
    with pytest.raises(ValueError, match="Cannot drop rows"):
        pytest.DB["files"].drop_index(index).get()
    with pytest.raises(
        ValueError,
        match="Cannot pick rows",
    ):
        pytest.DB["files"].pick_index(index).get()


def test_drop_extend_and_pick_index_order():
    # Ensure order of index is preserved.
    index = audformat.filewise_index(["f4", "f3", "f2", "f1"])
    table = audformat.Table(index)
    # pick
    new_table = table.pick_index(audformat.filewise_index(["f1", "f2"]))
    pd.testing.assert_index_equal(
        new_table.index,
        audformat.filewise_index(["f2", "f1"]),
    )
    # extend
    new_table = table.extend_index(audformat.filewise_index("f5"))
    pd.testing.assert_index_equal(
        new_table.index,
        audformat.filewise_index(["f4", "f3", "f2", "f1", "f5"]),
    )
    # drop
    new_table = table.drop_index(audformat.filewise_index(["f1", "f2"]))
    pd.testing.assert_index_equal(
        new_table.index,
        audformat.filewise_index(["f4", "f3"]),
    )


@pytest.mark.parametrize(
    "table, index, expected",
    [
        # table and index empty
        (
            create_table(audformat.filewise_index()),
            audformat.filewise_index(),
            audformat.filewise_index(),
        ),
        (
            create_table(audformat.segmented_index()),
            audformat.segmented_index(),
            audformat.segmented_index(),
        ),
        # table empty
        (
            create_table(audformat.filewise_index()),
            audformat.filewise_index(["f1", "f2"]),
            audformat.filewise_index(),
        ),
        (
            create_table(audformat.segmented_index()),
            audformat.segmented_index(
                ["f1", "f1", "f2"],
                [0, 1, 0],
                [1, 2, 3],
            ),
            audformat.segmented_index(),
        ),
        # index empty
        (
            create_table(audformat.filewise_index(["f1", "f2"])),
            audformat.filewise_index(),
            audformat.filewise_index(["f1", "f2"]),
        ),
        (
            create_table(
                audformat.segmented_index(
                    ["f1", "f1", "f2"],
                    [0, 1, 0],
                    [1, 2, 3],
                ),
            ),
            audformat.segmented_index(),
            audformat.segmented_index(
                ["f1", "f1", "f2"],
                [0, 1, 0],
                [1, 2, 3],
            ),
        ),
        # index and table identical
        (
            create_table(audformat.filewise_index(["f1", "f2"])),
            audformat.filewise_index(["f2", "f1"]),
            audformat.filewise_index(),
        ),
        (
            create_table(
                audformat.segmented_index(
                    ["f1", "f1", "f2"],
                    [0, 1, 0],
                    [1, 2, 3],
                ),
            ),
            audformat.segmented_index(
                ["f2", "f1", "f1"],
                [0, 1, 0],
                [3, 2, 1],
            ),
            audformat.segmented_index(),
        ),
        # index within table
        (
            create_table(audformat.filewise_index(["f1", "f2"])),
            audformat.filewise_index(["f2"]),
            audformat.filewise_index(["f1"]),
        ),
        (
            create_table(
                audformat.segmented_index(
                    ["f1", "f1", "f2"],
                    [0, 1, 0],
                    [1, 2, 3],
                ),
            ),
            audformat.segmented_index("f1", 1, 2),
            audformat.segmented_index(
                ["f1", "f2"],
                [0, 0],
                [1, 3],
            ),
        ),
        # table within index
        (
            create_table(audformat.filewise_index(["f2"])),
            audformat.filewise_index(["f1", "f2"]),
            audformat.filewise_index(),
        ),
        (
            create_table(audformat.segmented_index("f1", 1, 2)),
            audformat.segmented_index(
                ["f1", "f1", "f2"],
                [0, 1, 0],
                [1, 2, 3],
            ),
            audformat.segmented_index(),
        ),
        # index and table overlap
        (
            create_table(audformat.filewise_index(["f1", "f2"])),
            audformat.filewise_index(["f2", "f3"]),
            audformat.filewise_index(["f1"]),
        ),
        (
            create_table(
                audformat.segmented_index(
                    ["f1", "f1", "f2"],
                    [0, 1, 0],
                    [1, 2, 3],
                ),
            ),
            audformat.segmented_index(
                ["f2", "f1", "f1"],
                [0, 1, 0],
                [3, 2, 2],
            ),
            audformat.segmented_index("f1", 0, 1),
        ),
        # dtype of file level is object
        (
            create_table(audformat.filewise_index(["f1", "f2"])),
            pd.Index(["f2", "f3"], dtype="object", name="file"),
            audformat.filewise_index(["f1"]),
        ),
        (
            create_table(pd.Index(["f1", "f2"], dtype="object", name="file")),
            audformat.filewise_index(["f2", "f3"]),
            audformat.filewise_index(["f1"]),
        ),
        (
            create_table(pd.Index(["f1", "f2"], dtype="object", name="file")),
            pd.Index(["f2", "f3"], dtype="object", name="file"),
            audformat.filewise_index(["f1"]),
        ),
        # different index type
        pytest.param(
            create_table(audformat.segmented_index()),
            audformat.filewise_index(),
            None,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(
            create_table(audformat.filewise_index()),
            audformat.segmented_index(),
            None,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
    ],
)
def test_drop_index(table, index, expected):
    index_org = table.index.copy()
    table_new = table.drop_index(index, inplace=False)
    pd.testing.assert_index_equal(table_new.index, expected)
    pd.testing.assert_index_equal(table.index, index_org)
    table.drop_index(index, inplace=True)
    pd.testing.assert_index_equal(table.index, expected)


@pytest.mark.parametrize(
    "files",
    [
        pytest.DB.files,
        pytest.DB.files[0],
        [pytest.DB.files[0], "does-not-exist.wav"],
        lambda x: "1" in x,
    ],
)
@pytest.mark.parametrize(
    "table",
    [
        pytest.DB["files"],
        pytest.DB["segments"],
    ],
)
def test_drop_files(files, table):
    table = table.drop_files(files, inplace=False)
    if callable(files):
        files = table.files.to_series().apply(files)
    elif isinstance(files, str):
        files = [files]
    assert len(table.files.intersection(files)) == 0


def test_empty():
    db = audformat.testing.create_db(minimal=True)
    db["table"] = audformat.Table()

    assert db["table"].type == audformat.define.IndexType.FILEWISE
    assert len(db["table"].files) == 0
    assert len(db["table"]) == 0

    db["table"]["column"] = audformat.Column()

    assert db["table"]["column"].get().dtype == object


def test_exceptions():
    db = audformat.testing.create_db(minimal=True)

    # invalid segments

    with pytest.raises(ValueError):
        db["table"] = audformat.Table(
            audformat.segmented_index(files=["f1", "f2"], starts=["0s"], ends=["1s"]),
        )
    with pytest.raises(ValueError):
        db["table"] = audformat.Table(
            audformat.segmented_index(
                files=["f1", "f2"],
                starts=["0s", "1s"],
                ends=["1s"],
            )
        )
    with pytest.raises(ValueError):
        db["table"] = audformat.Table(
            audformat.segmented_index(
                files=["f1", "f2"],
                starts=["0s"],
                ends=["1s", "2s"],
            )
        )

    # bad scheme or rater

    with pytest.raises(audformat.errors.BadIdError):
        db["table"] = audformat.Table(
            audformat.filewise_index(["f1", "f2"]),
        )
        db["table"]["column"] = audformat.Column(scheme_id="invalid")
    with pytest.raises(audformat.errors.BadIdError):
        db["table"] = audformat.Table(
            audformat.filewise_index(["f1", "f2"]),
        )
        db["table"]["column"] = audformat.Column(rater_id="invalid")

    # level and column names must be unique

    with pytest.raises(ValueError, match="index level with same name"):
        db["table"] = audformat.Table(
            audformat.filewise_index(),
        )
        db["table"][audformat.define.IndexField.FILE] = audformat.Column()

    with pytest.raises(ValueError, match="index level with same name"):
        db["table"] = audformat.Table(
            audformat.segmented_index(),
        )
        db["table"][audformat.define.IndexField.FILE] = audformat.Column()

    with pytest.raises(ValueError, match="index level with same name"):
        db["table"] = audformat.Table(
            audformat.segmented_index(),
        )
        db["table"][audformat.define.IndexField.START] = audformat.Column()

    with pytest.raises(ValueError, match="index level with same name"):
        db["table"] = audformat.Table(
            audformat.segmented_index(),
        )
        db["table"][audformat.define.IndexField.END] = audformat.Column()


def test_extend_index():
    db = audformat.testing.create_db(minimal=True)
    db.schemes["scheme"] = audformat.Scheme()

    # empty and invalid

    db["table"] = audformat.Table()
    db["table"].extend_index(audformat.filewise_index())
    assert db["table"].get().empty
    with pytest.raises(
        ValueError,
        match="Cannot extend",
    ):
        db["table"].extend_index(
            audformat.segmented_index(
                files=["1.wav", "2.wav"],
                starts=["0s", "1s"],
                ends=["1s", "2s"],
            ),
            fill_values="a",
        )

    db.drop_tables("table")

    # filewise

    db["table"] = audformat.Table()
    db["table"]["columns"] = audformat.Column(scheme_id="scheme")
    db["table"].extend_index(
        audformat.filewise_index(["1.wav", "2.wav"]),
        fill_values="a",
        inplace=True,
    )
    np.testing.assert_equal(
        db["table"]["columns"].get().values,
        np.array(["a", "a"]),
    )
    index = audformat.filewise_index(["1.wav", "3.wav"])
    db["table"].extend_index(
        index,
        fill_values="b",
        inplace=True,
    )
    np.testing.assert_equal(
        db["table"]["columns"].get().values,
        np.array(["a", "a", "b"]),
    )
    index = pd.Index(["4.wav"], dtype="object", name="file")
    db["table"].extend_index(
        index,
        fill_values="c",
        inplace=True,
    )
    np.testing.assert_equal(
        db["table"]["columns"].get().values,
        np.array(["a", "a", "b", "c"]),
    )

    db.drop_tables("table")

    # segmented

    db["table"] = audformat.Table(audformat.segmented_index())
    db["table"]["columns"] = audformat.Column(scheme_id="scheme")
    db["table"].extend_index(
        audformat.segmented_index(
            files=["1.wav", "2.wav"],
            starts=["0s", "1s"],
            ends=["1s", "2s"],
        ),
        fill_values="a",
        inplace=True,
    )
    np.testing.assert_equal(
        db["table"]["columns"].get().values,
        np.array(["a", "a"]),
    )
    index = audformat.segmented_index(
        files=["1.wav", "3.wav"],
        starts=["0s", "2s"],
        ends=["1s", "3s"],
    )
    db["table"].extend_index(
        index,
        fill_values={"columns": "b"},
        inplace=True,
    )
    np.testing.assert_equal(
        db["table"]["columns"].get().values,
        np.array(["a", "a", "b"]),
    )
    index = pd.MultiIndex.from_arrays(
        [
            ["3.wav"],
            [pd.Timedelta(0)],
            [pd.Timedelta(4, unit="s")],
        ],
        names=["file", "start", "end"],
    )
    db["table"].extend_index(
        index,
        fill_values={"columns": "c"},
        inplace=True,
    )
    np.testing.assert_equal(
        db["table"]["columns"].get().values,
        np.array(["a", "a", "b", "c"]),
    )

    db.drop_tables("table")


@pytest.mark.parametrize(
    "num_files,values",
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
        db,
        "table",
        audformat.define.IndexType.FILEWISE,
        num_files=num_files,
    )
    db.schemes["scheme"] = audformat.Scheme(
        dtype=audformat.define.DataType.FLOAT,
    )
    table = db["table"]

    # empty table
    df = pd.DataFrame(index=audformat.filewise_index(table.files))
    pd.testing.assert_frame_equal(table.get(), df)

    # no values
    df["column"] = np.nan
    table["column"] = audformat.Column(scheme_id="scheme")
    pd.testing.assert_frame_equal(table.get(), df)

    # single
    df["column"] = np.nan
    table.df["column"] = np.nan
    df.iloc[0] = values[0]
    index = audformat.filewise_index(table.files[0])
    table.set({"column": values[0]}, index=index)
    pd.testing.assert_frame_equal(
        table.get(index),
        df.loc[[table.files[0]]],
    )
    pd.testing.assert_frame_equal(table.get(), df)

    # slice
    df["column"] = np.nan
    table.df["column"] = np.nan
    rows = df.index[1:-1]
    df.loc[rows, "column"] = values[1:-1]
    index = audformat.filewise_index(table.files[1:-1])
    table.set({"column": values[1:-1]}, index=index)
    pd.testing.assert_frame_equal(
        table.get(index),
        df.loc[table.files[1:-1]],
    )
    pd.testing.assert_frame_equal(table.get(), df)

    # dtype of file level is object
    df["column"] = np.nan
    table.df["column"] = np.nan
    rows = df.index[1:-1]
    df.loc[rows, "column"] = values[1:-1]
    index = pd.Index(table.files[1:-1], dtype="object", name="file")
    table.set({"column": values[1:-1]}, index=index)
    pd.testing.assert_frame_equal(
        table.get(index),
        df.loc[table.files[1:-1]],
    )
    pd.testing.assert_frame_equal(table.get(), df)

    # all
    df["column"] = np.nan
    table.df["column"] = np.nan
    df.iloc[:, 0] = values
    table.set({"column": values})
    pd.testing.assert_frame_equal(table.get(), df)

    # scalar
    df["column"] = np.nan
    table.df["column"] = np.nan
    df.iloc[:, 0] = values[0]
    table.set({"column": values[0]})
    pd.testing.assert_frame_equal(table.get(), df)

    # data frame
    df["column"] = np.nan
    table.df["column"] = np.nan
    df.iloc[:, 0] = values
    table.set(df)
    pd.testing.assert_frame_equal(table.get(), df)


def test_get_as_segmented():
    db = pytest.DB

    df = db["files"].get()
    assert audformat.index_type(df) == audformat.define.IndexType.FILEWISE
    assert not db._files_duration

    # convert to segmented index

    df = db["files"].get(
        as_segmented=True,
        allow_nat=True,
    )
    assert audformat.index_type(df) == audformat.define.IndexType.SEGMENTED
    assert not db._files_duration
    assert df.index.get_level_values(audformat.define.IndexField.END).isna().all()

    # replace NaT with file duration

    df = db["files"].get(
        as_segmented=True,
        allow_nat=False,
    )
    assert audformat.index_type(df) == audformat.define.IndexType.SEGMENTED
    assert db._files_duration
    assert not df.index.get_level_values(audformat.define.IndexField.END).isna().any()

    # reset db

    db._files_duration = {}


def test_get_preserves_dtypes():
    db = pytest.DB

    for table in [db["files"], db["segments"]]:
        df = table.get()
        pd.testing.assert_series_equal(df.dtypes, table.df.dtypes)
        for index in [table.files, db.segments]:
            df = table.get(index)
            pd.testing.assert_series_equal(df.dtypes, table.df.dtypes)


def test_load(tmpdir):
    path_pkl = os.path.join(tmpdir, "db.table.pkl")
    path_no_ext = os.path.join(tmpdir, "db.table")

    # Test backward compatibility
    table = create_db_table(
        pd.Series(
            [1.0, 2.0],
            index=audformat.filewise_index(["f1", "f2"]),
            name="column",
        )
    )
    table.df.to_pickle(path_pkl, compression="xz")
    table_loaded = audformat.Table()
    table_loaded.load(path_no_ext)
    pd.testing.assert_frame_equal(table.df, table_loaded.df)

    # corrupt pickle file
    with open(path_pkl, "wb"):
        pass
    with pytest.raises(EOFError):
        table_loaded.load(path_no_ext)

    # repeat with CSV|PARQUET file as fall back
    for ext in [
        audformat.define.TableStorageFormat.CSV,
        audformat.define.TableStorageFormat.PARQUET,
    ]:
        table.save(path_no_ext, storage_format=ext)
        with open(path_pkl, "wb"):
            pass
        table_loaded = audformat.Table()
        table_loaded.columns = table.columns
        table_loaded._db = table._db
        table_loaded.load(path_no_ext)
        pd.testing.assert_frame_equal(table.df, table_loaded.df)

        # check if pickle file was recovered
        df = pd.read_pickle(path_pkl)
        pd.testing.assert_frame_equal(table.df, df)

        os.remove(f"{path_no_ext}.{ext}")


class TestLoadBrokenCsv:
    r"""Test loading of malformed csv files.

    If csv files contain a lot of special characters,
    or a different number of columns,
    than specified in the database header,
    loading of them should not fail.

    See https://github.com/audeering/audformat/issues/449

    """

    def database_with_hidden_columns(self) -> audformat.Database:
        r"""Database with hidden columns.

        Create database with hidden columns
        that are stored in csv,
        but not in the header of the table.

        Ensure:

        * it contains an empty table
        * the columns use schemes with time and date data types
        * at least one column has no scheme

        as those cases needed special care with csv files,
        before switching to use pyarrow.csv.read_csv()
        in https://github.com/audeering/audformat/pull/419.

        Returns:
            database

        """
        db = audformat.Database("mydb")
        db.schemes["date"] = audformat.Scheme("date")
        db.schemes["time"] = audformat.Scheme("time")
        db["table"] = audformat.Table(audformat.filewise_index("file.wav"))
        db["table"]["date"] = audformat.Column(scheme_id="date")
        db["table"]["date"].set([pd.to_datetime("2018-10-26")])
        db["table"]["time"] = audformat.Column(scheme_id="time")
        db["table"]["time"].set([pd.Timedelta(1)])
        db["table"]["no-scheme"] = audformat.Column()
        db["table"]["no-scheme"].set(["label"])
        db["empty-table"] = audformat.Table(audformat.filewise_index())
        db["empty-table"]["column"] = audformat.Column()
        # Add a hidden column to the table dataframes,
        # without adding it to the table header
        db["table"].df["hidden"] = ["hidden"]
        db["empty-table"].df["hidden"] = []
        return db

    def test_load_broken_csv(self, tmpdir):
        r"""Test loading a database table from broken csv files.

        Broken csv files
        refer to csv tables,
        that raise an error
        when loading with ``pyarrow.csv.read_csv()``.

        Args:
            tmpdir: tmpdir fixture

        """
        db = self.database_with_hidden_columns()
        build_dir = audeer.mkdir(tmpdir, "build")
        db.save(build_dir, storage_format="csv")
        db_loaded = audformat.Database.load(build_dir, load_data=True)
        assert "table" in db_loaded
        assert "empty-table" in db_loaded
        assert "hidden" not in db_loaded["table"].df
        assert "hidden-column" not in db_loaded["empty-table"].df


def test_load_old_pickle(tmpdir):
    # We have stored string dtype as object dtype before
    # and have to fix this when loading old PKL files from cache.

    # Create PKL file containing strings as object
    y = pd.Series(["c"], dtype="object", name="column")
    index = pd.Index(["f1"], dtype="object", name="file")

    db = audformat.testing.create_db(minimal=True)
    db["table"] = audformat.Table(index)
    db.schemes["column"] = audformat.Scheme(audformat.define.DataType.OBJECT)
    db["table"]["column"] = audformat.Column(scheme_id="column")
    db["table"]["column"].set(y.values)
    db_root = tmpdir.join("db")
    db.save(db_root, storage_format="pkl")

    # Change scheme dtype to string and store header again
    db.schemes["column"] = audformat.Scheme(audformat.define.DataType.STRING)
    db.save(db_root, header_only=True)

    # Load and check that dtype is string
    db_new = audformat.Database.load(db_root)
    assert db_new.schemes["column"].dtype == audformat.define.DataType.STRING
    assert db_new["table"].df["column"].dtype == "string"
    assert db_new["table"].index.dtype == "string"


@pytest.mark.parametrize(
    "table, map",
    [
        (pytest.DB["files"], {"label_map_int": "int"}),
        (pytest.DB["files"], {"label_map_int": "label_map_int"}),
        (pytest.DB["files"], {"label_map_str": "prop1"}),
        (pytest.DB["segments"], {"label_map_str": ["prop1", "prop2"]}),
        (
            pytest.DB["segments"],
            {  # duplicates will be ignored
                "label_map_str": ["prop1", "prop2", "prop1", "prop2"]
            },
        ),
        (pytest.DB["segments"], {"label_map_str": ["label_map_str", "prop1", "prop2"]}),
        pytest.param(  # no database
            audformat.Table(),
            "map",
            marks=pytest.mark.xfail(raises=RuntimeError),
        ),
    ],
)
def test_map(table, map):
    result = table.get(map=map)
    expected = table.get()
    for column, mapped_columns in map.items():
        mapped_columns = audeer.to_list(mapped_columns)
        if len(mapped_columns) == 1:
            expected[mapped_columns[0]] = table.columns[column].get(
                map=mapped_columns[0],
            )
        else:
            for mapped_column in mapped_columns:
                if mapped_column != column:
                    expected[mapped_column] = table.columns[column].get(
                        map=mapped_column,
                    )
        if column not in mapped_columns:
            expected.drop(columns=column, inplace=True)
    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize("storage_format", ["csv", "parquet"])
class TestHash:
    r"""Test if PARQUET file hash changes with table.

    We store a MD5 sum associated with the dataframe,
    that was used to create the file,
    in the metadata of the PARQUET file.
    Those MD5 sum is supposed to change,
    if any of the table rows, (index) columns changes,
    the data type of the entries changes,
    or the name of a column changes.

    Args:
        tmpdir: tmpdir fixture
        storage_format: storage format of table file

    """

    def db(self, tmpdir, storage_format):
        r"""Create minimal database with scheme and table."""
        self.db_root = audeer.path(tmpdir, "db")
        self.storage_format = storage_format
        self.table_file = audeer.path(self.db_root, f"db.table.{storage_format}")
        db = audformat.Database("mydb")
        db.schemes["int"] = audformat.Scheme("int")
        index = audformat.segmented_index(["f1", "f2"], [0, 1], [1, 2])
        db["table"] = audformat.Table(index)
        db["table"]["column"] = audformat.Column(scheme_id="int")
        db["table"]["column"].set([0, 1])
        db.save(self.db_root, storage_format=self.storage_format)
        return db

    def md5(self) -> str:
        r"""Get MD5 sum for table file."""
        if self.storage_format == "csv":
            return audeer.md5(self.table_file)
        elif self.storage_format == "parquet":
            return parquet.read_schema(self.table_file).metadata[b"hash"].decode()

    def test_change_index(self, tmpdir, storage_format):
        r"""Change table index."""
        db = self.db(tmpdir, storage_format)
        md5 = self.md5()
        index = audformat.segmented_index(["f1", "f1"], [0, 1], [1, 2])
        db["table"] = audformat.Table(index)
        db["table"]["column"] = audformat.Column(scheme_id="int")
        db["table"]["column"].set([0, 1])
        db.save(self.db_root, storage_format=self.storage_format)
        assert self.md5() != md5

    def test_change_column_name(self, tmpdir, storage_format):
        r"""Change table column name."""
        db = self.db(tmpdir, storage_format)
        md5 = self.md5()
        index = audformat.segmented_index(["f1", "f2"], [0, 1], [1, 2])
        db["table"] = audformat.Table(index)
        db["table"]["col"] = audformat.Column(scheme_id="int")
        db["table"]["col"].set([0, 1])
        db.save(self.db_root, storage_format=self.storage_format)
        assert self.md5() != md5

    def test_change_column_order(self, tmpdir, storage_format):
        r"""Change order of table columns."""
        db = self.db(tmpdir, storage_format)
        index = audformat.segmented_index(["f1", "f2"], [0, 1], [1, 2])
        db["table"] = audformat.Table(index)
        db["table"]["col1"] = audformat.Column(scheme_id="int")
        db["table"]["col1"].set([0, 1])
        db["table"]["col2"] = audformat.Column(scheme_id="int")
        db["table"]["col2"].set([0, 1])
        db.save(self.db_root, storage_format=self.storage_format)
        md5 = self.md5()
        db["table"] = audformat.Table(index)
        db["table"]["col2"] = audformat.Column(scheme_id="int")
        db["table"]["col2"].set([0, 1])
        db["table"]["col1"] = audformat.Column(scheme_id="int")
        db["table"]["col1"].set([0, 1])
        db.save(self.db_root, storage_format=self.storage_format)
        assert self.md5() != md5

    def test_change_row_order(self, tmpdir, storage_format):
        r"""Change order of table rows."""
        db = self.db(tmpdir, storage_format)
        md5 = self.md5()
        index = audformat.segmented_index(["f2", "f1"], [1, 0], [2, 1])
        db["table"] = audformat.Table(index)
        db["table"]["column"] = audformat.Column(scheme_id="int")
        db["table"]["column"].set([1, 0])
        db.save(self.db_root, storage_format=storage_format)
        assert self.md5() != md5

    def test_change_values(self, tmpdir, storage_format):
        r"""Change table values."""
        db = self.db(tmpdir, storage_format)
        md5 = self.md5()
        index = audformat.segmented_index(["f1", "f2"], [0, 1], [1, 2])
        db["table"] = audformat.Table(index)
        db["table"]["column"] = audformat.Column(scheme_id="int")
        db["table"]["column"].set([1, 0])
        db.save(self.db_root, storage_format=self.storage_format)
        assert self.md5() != md5

    def test_copy_table(self, tmpdir, storage_format):
        r"""Replace table with identical copy."""
        db = self.db(tmpdir, storage_format)
        md5 = self.md5()
        table = db["table"].copy()
        db["table"] = table
        db.save(self.db_root, storage_format=self.storage_format)
        assert self.md5() == md5


@pytest.mark.parametrize(
    "table_id, expected_hash",
    [
        (
            "files",
            "a66a22ee4158e0e5100f1d797155ad81",
        ),
        (
            "segments",
            "f69eb4a5d19da71e5da00a9b13beb3db",
        ),
        (
            "misc",
            "331f79758b195cb9b7d0e8889e830eb2",
        ),
    ],
)
def test_parquet_hash_reproducibility(tmpdir, table_id, expected_hash):
    r"""Test reproducibility of binary PARQUET files.

    When storing the same dataframe
    to different PARQUET files,
    the files will slightly vary
    and have different MD5 sums.

    To provide a reproducible hash,
    in order to judge if a table has changed,
    we calculate the hash of the table
    and store it in the metadata
    of the schema
    of a the table.

    """
    random.seed(1)  # ensure the same random table values are created
    db = audformat.testing.create_db()

    # Write to PARQUET file and check if correct hash is stored
    path_wo_ext = audeer.path(tmpdir, table_id)
    path = f"{path_wo_ext}.parquet"
    db[table_id].save(path_wo_ext, storage_format="parquet")
    metadata = parquet.read_schema(path).metadata
    assert metadata[b"hash"].decode() == expected_hash

    # Load table from PARQUET file, and overwrite it
    db[table_id].load(path_wo_ext)
    os.remove(path)
    db[table_id].save(path_wo_ext, storage_format="parquet")
    metadata = parquet.read_schema(path).metadata
    assert metadata[b"hash"].decode() == expected_hash


@pytest.mark.parametrize(
    "files",
    [
        pytest.DB.files,
        pytest.DB.files[0],
        [pytest.DB.files[0], "does-not-exist.wav"],
        lambda x: "1" in x,
    ],
)
@pytest.mark.parametrize(
    "table",
    [
        pytest.DB["files"],
        pytest.DB["segments"],
    ],
)
def test_pick_files(files, table):
    table = table.pick_files(files, inplace=False)
    if callable(files):
        files = table.files[table.files.to_series().apply(files)]
        seen = set()
        seen_add = seen.add
        files = [x for x in files if not (x in seen or seen_add(x))]
    elif isinstance(files, str):
        files = [files]
    pd.testing.assert_index_equal(
        table.files.unique(),
        audformat.filewise_index(files).intersection(table.files),
    )


@pytest.mark.parametrize(
    "table, index, expected",
    [
        # table and index empty
        (
            create_table(audformat.filewise_index()),
            audformat.filewise_index(),
            audformat.filewise_index(),
        ),
        (
            create_table(audformat.segmented_index()),
            audformat.segmented_index(),
            audformat.segmented_index(),
        ),
        # table empty
        (
            create_table(audformat.filewise_index()),
            audformat.filewise_index(["f1", "f2"]),
            audformat.filewise_index(),
        ),
        (
            create_table(audformat.segmented_index()),
            audformat.segmented_index(
                ["f1", "f1", "f2"],
                [0, 1, 0],
                [1, 2, 3],
            ),
            audformat.segmented_index(),
        ),
        # index empty
        (
            create_table(audformat.filewise_index(["f1", "f2"])),
            audformat.filewise_index(),
            audformat.filewise_index(),
        ),
        (
            create_table(
                audformat.segmented_index(
                    ["f1", "f1", "f2"],
                    [0, 1, 0],
                    [1, 2, 3],
                ),
            ),
            audformat.segmented_index(),
            audformat.segmented_index(),
        ),
        # index and table identical
        (
            create_table(audformat.filewise_index(["f1", "f2"])),
            audformat.filewise_index(["f2", "f1"]),
            audformat.filewise_index(["f1", "f2"]),
        ),
        (
            create_table(
                audformat.segmented_index(
                    ["f1", "f1", "f2"],
                    [0, 1, 0],
                    [1, 2, 3],
                ),
            ),
            audformat.segmented_index(
                ["f2", "f1", "f1"],
                [0, 1, 0],
                [3, 2, 1],
            ),
            audformat.segmented_index(
                ["f1", "f1", "f2"],
                [0, 1, 0],
                [1, 2, 3],
            ),
        ),
        # index within table
        (
            create_table(audformat.filewise_index(["f1", "f2"])),
            audformat.filewise_index(["f2"]),
            audformat.filewise_index(["f2"]),
        ),
        (
            create_table(
                audformat.segmented_index(
                    ["f1", "f1", "f2"],
                    [0, 1, 0],
                    [1, 2, 3],
                ),
            ),
            audformat.segmented_index("f1", 1, 2),
            audformat.segmented_index("f1", 1, 2),
        ),
        # table within index
        (
            create_table(audformat.filewise_index(["f2"])),
            audformat.filewise_index(["f1", "f2"]),
            audformat.filewise_index(["f2"]),
        ),
        (
            create_table(audformat.segmented_index("f1", 1, 2)),
            audformat.segmented_index(
                ["f1", "f1", "f2"],
                [0, 1, 0],
                [1, 2, 3],
            ),
            audformat.segmented_index("f1", 1, 2),
        ),
        # index and table overlap
        (
            create_table(audformat.filewise_index(["f1", "f2"])),
            audformat.filewise_index(["f2", "f3"]),
            audformat.filewise_index(["f2"]),
        ),
        (
            create_table(
                audformat.segmented_index(
                    ["f1", "f1", "f2"],
                    [0, 1, 0],
                    [1, 2, 3],
                ),
            ),
            audformat.segmented_index(
                ["f2", "f1", "f1"],
                [0, 1, 0],
                [3, 2, 2],
            ),
            audformat.segmented_index(
                ["f1", "f2"],
                [1, 0],
                [2, 3],
            ),
        ),
        # dtype of file level is object
        (
            create_table(audformat.filewise_index(["f1", "f2"])),
            pd.Index(["f2"], dtype="object", name="file"),
            audformat.filewise_index(["f2"]),
        ),
        (
            create_table(pd.Index(["f1", "f2"], dtype="object", name="file")),
            audformat.filewise_index(["f2"]),
            audformat.filewise_index(["f2"]),
        ),
        (
            create_table(pd.Index(["f1", "f2"], dtype="object", name="file")),
            pd.Index(["f2"], dtype="object", name="file"),
            audformat.filewise_index(["f2"]),
        ),
        # different index type
        pytest.param(
            create_table(audformat.segmented_index()),
            audformat.filewise_index(),
            None,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(
            create_table(audformat.filewise_index()),
            audformat.segmented_index(),
            None,
            marks=pytest.mark.xfail(raises=ValueError),
        ),
    ],
)
def test_pick_index(table, index, expected):
    index_org = table.index.copy()
    table_new = table.pick_index(index, inplace=False)
    pd.testing.assert_index_equal(table_new.index, expected)
    pd.testing.assert_index_equal(table.index, index_org)
    table.pick_index(index, inplace=True)
    pd.testing.assert_index_equal(table.index, expected)


@pytest.mark.parametrize(
    "storage_format",
    [
        pytest.param(
            "csv",
            marks=pytest.mark.skip(reason="CSV does not support numpy arrays"),
        ),
        "parquet",
        "pkl",
    ],
)
def test_save_and_load(tmpdir, storage_format):
    r"""Test saving and loading of a table.

    Ensures the table dataframe representation
    is identical after saving and loading a table.

    Args:
        tmpdir: tmpdir fixture
        storage_format: storage format
            the table should be written to disk.
            This will also be used as file extension

    """
    db = audformat.testing.create_db()

    # Extend database with more table/scheme combinations
    db.schemes["int-labels"] = audformat.Scheme(
        dtype=audformat.define.DataType.INTEGER,
        labels=[0, 1],
    )
    db.schemes["object"] = audformat.Scheme(audformat.define.DataType.OBJECT)
    index = pd.MultiIndex.from_arrays(
        [[0, 1], ["a", "b"]],
        names=["idx1", "idx2"],
    )
    index = audformat.utils.set_index_dtypes(
        index,
        {
            "idx1": audformat.define.DataType.INTEGER,
            "idx2": audformat.define.DataType.OBJECT,
        },
    )
    db["multi-misc"] = audformat.MiscTable(index)
    db["multi-misc"]["int"] = audformat.Column(scheme_id="int-labels")
    db["multi-misc"]["int"].set([0, pd.NA])
    db["multi-misc"]["bool"] = audformat.Column(scheme_id="bool")
    db["multi-misc"]["bool"].set([True, pd.NA])
    db["multi-misc"]["arrays"] = audformat.Column(scheme_id="object")
    db["multi-misc"]["arrays"].set([np.array([0, 1]), np.array([2, 3])])
    db["multi-misc"]["lists"] = audformat.Column(scheme_id="object")
    db["multi-misc"]["lists"].set([[0, 1], [2, 3]])
    db["multi-misc"]["no-scheme"] = audformat.Column()
    db["multi-misc"]["no-scheme"].set([0, 1])

    for table_id in list(db):
        expected_df = db[table_id].get()
        path_wo_ext = audeer.path(tmpdir, table_id)
        path = f"{path_wo_ext}.{storage_format}"
        db[table_id].save(path_wo_ext, storage_format=storage_format)
        assert os.path.exists(path)
        db[table_id].load(path_wo_ext)
        pd.testing.assert_frame_equal(db[table_id].df, expected_df)


@pytest.mark.parametrize(
    "storage_format, expected_error, expected_error_msg",
    [
        (
            "non-existing",
            audformat.errors.BadValueError,
            re.escape(
                "Bad value 'non-existing', expected one of ['csv', 'parquet', 'pkl']"
            ),
        ),
    ],
)
def test_save_errors(tmpdir, storage_format, expected_error, expected_error_msg):
    r"""Test errors when saving a table.

    Args:
        tmpdir: tmpdir fixture
        storage_format: storage format of table
        expected_error: expected error, e.g. ``ValueError``
        expected_error_msg: expected test of error message

    """
    db = audformat.testing.create_db()
    table_id = list(db)[0]
    path_wo_ext = audeer.path(tmpdir, table_id)
    with pytest.raises(expected_error, match=expected_error_msg):
        db[table_id].save(path_wo_ext, storage_format=storage_format)


@pytest.mark.parametrize(
    "num_files,num_segments_per_file,values",
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
    table = db["table"]

    # empty table
    df = pd.DataFrame(index=table.index)
    pd.testing.assert_frame_equal(table.get(), df)

    # no values
    df["column"] = np.nan
    table["column"] = audformat.Column(scheme_id="scheme")
    pd.testing.assert_frame_equal(table.get(), df)

    # single
    df["column"] = np.nan
    table.df["column"] = np.nan
    index = audformat.segmented_index(
        table.files[0],
        starts=table.starts[0],
        ends=table.ends[0],
    )
    df.loc[index] = values[0]
    table.set({"column": values[0]}, index=index)
    pd.testing.assert_frame_equal(
        table.get(index),
        df.loc[index],
    )

    # slice
    df["column"] = np.nan
    table.df["column"] = np.nan
    index = audformat.segmented_index(
        table.files[1:-1],
        starts=table.starts[1:-1],
        ends=table.ends[1:-1],
    )
    df.loc[index, :] = values[1:-1]
    table.set({"column": values[1:-1]}, index=index)
    pd.testing.assert_frame_equal(
        table.get(index),
        df.loc[index],
    )

    # dtype of file level is object
    df["column"] = np.nan
    table.df["column"] = np.nan
    index = pd.MultiIndex.from_arrays(
        [
            table.files[1:-1].astype("object"),
            table.starts[1:-1],
            table.ends[1:-1],
        ],
        names=["file", "start", "end"],
    )
    df.loc[index, :] = values[1:-1]
    table.set({"column": values[1:-1]}, index=index)
    pd.testing.assert_frame_equal(
        table.get(index),
        df.loc[index],
    )

    # all
    df["column"] = np.nan
    table.df["column"] = np.nan
    df.iloc[:, 0] = values
    table.set({"column": values})
    pd.testing.assert_frame_equal(table.get(), df)

    # scalar
    df["column"] = np.nan
    table.df["column"] = np.nan
    df.iloc[:, 0] = values[0]
    table.set({"column": values[0]})
    pd.testing.assert_frame_equal(table.get(), df)

    # data frame
    df["column"] = np.nan
    table.df["column"] = np.nan
    table.set(df)
    pd.testing.assert_frame_equal(table.get(), df)


def test_type():
    db = audformat.testing.create_db()

    assert db["files"].is_filewise
    assert db["segments"].is_segmented

    assert sorted(set(db["files"].index)) == sorted(db.files)
    pd.testing.assert_index_equal(db.segments, db["segments"].index)
    pd.testing.assert_index_equal(db.files, db["files"].files)
    pd.testing.assert_index_equal(
        db["files"].starts.unique(),
        pd.TimedeltaIndex(
            [0],
            name=audformat.define.IndexField.START,
        ),
    )
    pd.testing.assert_index_equal(
        db["files"].ends.unique(),
        pd.TimedeltaIndex(
            [pd.NaT],
            name=audformat.define.IndexField.END,
        ),
    )
    pd.testing.assert_index_equal(db["files"].index, db["files"].index)


@pytest.mark.parametrize(
    "table, overwrite, others",
    [
        # empty
        (
            create_db_table(),
            False,
            [],
        ),
        (
            create_db_table(),
            False,
            create_db_table(),
        ),
        # same column, with overlap
        (
            create_db_table(
                pd.Series(
                    [1.0, 2.0],
                    index=audformat.filewise_index(["f1", "f2"]),
                )
            ),
            False,
            create_db_table(
                pd.Series(
                    [2.0, 3.0],  # ok, value do match
                    index=audformat.filewise_index(["f2", "f3"]),
                )
            ),
        ),
        (
            create_db_table(
                pd.Series(
                    [1.0, 2.0],
                    index=audformat.filewise_index(["f1", "f2"]),
                )
            ),
            False,
            create_db_table(
                pd.Series(
                    [np.nan, 3.0],  # ok, value is nan
                    index=audformat.filewise_index(["f2", "f3"]),
                )
            ),
        ),
        pytest.param(
            create_db_table(
                pd.Series(
                    [1.0, 2.0],
                    index=audformat.filewise_index(["f1", "f2"]),
                )
            ),
            False,
            create_db_table(
                pd.Series(
                    [99.0, 3.0],  # error, value do not match
                    index=audformat.filewise_index(["f2", "f3"]),
                )
            ),
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        (
            create_db_table(
                pd.Series(
                    [1.0, 2.0],
                    index=audformat.filewise_index(["f1", "f2"]),
                )
            ),
            True,
            create_db_table(
                pd.Series(
                    [99.0, 3.0],  # ok, will be overwritten
                    index=audformat.filewise_index(["f2", "f3"]),
                )
            ),
        ),
        # columns with new schemes
        (
            create_db_table(
                pd.Series(
                    [1.0, 2.0],
                    index=audformat.filewise_index(["f1", "f2"]),
                    name="c1",
                )
            ),
            False,
            [
                create_db_table(
                    pd.Series(
                        ["a", "b"],
                        index=audformat.filewise_index(["f2", "f3"]),
                        name="c2",
                    )
                ),
                create_db_table(
                    pd.Series(
                        [1, 2],
                        index=audformat.filewise_index(["f2", "f3"]),
                        name="c3",
                    )
                ),
            ],
        ),
        # error: scheme mismatch
        pytest.param(
            create_db_table(
                pd.Series(
                    [1.0, 2.0],
                    index=audformat.filewise_index(["f1", "f2"]),
                )
            ),
            False,
            create_db_table(  # same column, different scheme
                pd.Series(
                    ["a", "b"],
                    index=audformat.filewise_index(["f2", "f3"]),
                )
            ),
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(
            create_db_table(
                pd.Series(
                    [1.0, 2.0],
                    index=audformat.filewise_index(["f1", "f2"]),
                )
            ),
            False,
            create_table(  # no scheme
                pd.Series(
                    [1.0, 2.0],
                    index=audformat.filewise_index(["f1", "f2"]),
                )
            ),
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(
            create_db_table(
                pd.Series(
                    [1.0, 2.0],
                    index=audformat.filewise_index(["f1", "f2"]),
                    name="c1",
                ),
                scheme_id="scheme",
            ),
            False,
            create_db_table(
                pd.Series(  # different scheme with same id
                    ["a", "b"],
                    index=audformat.filewise_index(["f1", "f2"]),
                    name="c2",
                ),
                scheme_id="scheme",
            ),
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        # column with new rater
        (
            create_db_table(
                pd.Series(
                    index=audformat.filewise_index(),
                    dtype=float,
                    name="c1",
                ),
            ),
            False,
            create_db_table(
                pd.Series(
                    index=audformat.filewise_index(),
                    dtype=float,
                    name="c2",
                ),
                rater=audformat.Rater(audformat.define.RaterType.HUMAN),
            ),
        ),
        # error: rater mismatch
        pytest.param(
            create_db_table(
                rater=audformat.Rater(audformat.define.RaterType.HUMAN),
            ),
            False,
            create_db_table(
                rater=audformat.Rater(audformat.define.RaterType.MACHINE),
            ),
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(
            create_db_table(
                rater=audformat.Rater(audformat.define.RaterType.HUMAN),
            ),
            False,
            create_db_table(),
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(
            create_db_table(),
            False,
            create_db_table(
                rater=audformat.Rater(audformat.define.RaterType.MACHINE),
            ),
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(
            create_db_table(
                pd.Series(
                    index=audformat.filewise_index(),
                    dtype=float,
                    name="c1",
                ),
                rater=audformat.Rater(audformat.define.RaterType.HUMAN),
            ),
            False,
            create_db_table(
                pd.Series(
                    index=audformat.filewise_index(),
                    dtype=float,
                    name="c2",
                ),
                rater=audformat.Rater(audformat.define.RaterType.MACHINE),
            ),
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        # media and split match
        (
            create_db_table(
                media=audformat.Media(audformat.define.MediaType.AUDIO),
                split=audformat.Split(audformat.define.SplitType.TEST),
            ),
            False,
            create_db_table(
                media=audformat.Media(audformat.define.MediaType.AUDIO),
                split=audformat.Split(audformat.define.SplitType.TEST),
            ),
        ),
        # error: media mismatch
        pytest.param(
            create_db_table(
                media=audformat.Media(audformat.define.MediaType.AUDIO),
            ),
            False,
            create_db_table(
                media=audformat.Media(audformat.define.MediaType.VIDEO),
            ),
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(
            create_db_table(
                media=audformat.Media(audformat.define.MediaType.AUDIO),
            ),
            False,
            create_db_table(),
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(
            create_db_table(),
            False,
            create_db_table(
                media=audformat.Media(audformat.define.MediaType.AUDIO),
            ),
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        # error: split mismatch
        pytest.param(
            create_db_table(
                split=audformat.Split(audformat.define.SplitType.TEST),
            ),
            False,
            create_db_table(
                split=audformat.Split(audformat.define.SplitType.TRAIN),
            ),
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(
            create_db_table(
                split=audformat.Split(audformat.define.SplitType.TEST),
            ),
            False,
            create_db_table(),
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(
            create_db_table(),
            False,
            create_db_table(
                split=audformat.Split(audformat.define.SplitType.TRAIN),
            ),
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        # error: not assigned to db
        pytest.param(
            audformat.Table(),
            False,
            [],
            marks=pytest.mark.xfail(raises=RuntimeError),
        ),
        # error: different index type
        pytest.param(
            create_db_table(
                pd.Series(
                    [1.0, 2.0],
                    index=audformat.filewise_index(["f1", "f2"]),
                )
            ),
            False,
            create_db_table(
                pd.Series(
                    [2.0, 3.0],
                    index=audformat.segmented_index(["f2", "f3"]),
                )
            ),
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        pytest.param(
            create_db_table(
                pd.Series(
                    [1.0, 2.0],
                    index=audformat.segmented_index(["f1", "f2"]),
                )
            ),
            False,
            create_db_table(
                pd.Series(
                    [2.0, 3.0],
                    index=audformat.filewise_index(["f2", "f3"]),
                )
            ),
            marks=pytest.mark.xfail(raises=ValueError),
        ),
    ],
)
def test_update(table, overwrite, others):
    df = table.get()
    table.update(others, overwrite=overwrite)
    others = audeer.to_list(others)
    df = audformat.utils.concat(
        [df] + [other.df for other in others],
        overwrite=overwrite,
    )
    assert table.type == audformat.index_type(df)
    if isinstance(df, pd.Series):
        df = df.to_frame()
    pd.testing.assert_frame_equal(table.df, df)
    for other in others:
        for column_id, column in other.columns.items():
            assert column.scheme == table[column_id].scheme
            assert column.rater == table[column_id].rater


@pytest.mark.parametrize("update_other_formats", [True, False])
@pytest.mark.parametrize(
    "storage_format, existing_formats",
    [
        ("csv", []),
        ("csv", []),
        ("csv", ["pkl"]),
        ("csv", ["parquet", "pkl"]),
        ("pkl", ["parquet"]),
        ("pkl", ["csv"]),
        ("pkl", ["parquet", "csv"]),
        ("parquet", ["pkl"]),
        ("parquet", ["csv"]),
        ("parquet", ["pkl", "csv"]),
    ],
)
def test_update_other_formats(
    tmpdir,
    storage_format,
    existing_formats,
    update_other_formats,
):
    r"""Tests updating of other table formats.

    When a table is stored with `audformat.Table.save()`
    as CSV, PARQUET, or PKL file,
    a user might select
    that all other existing file representations of the table
    are updated as well.
    E.g. if a PKL file of the same table exists,
    and a user saves to a CSV file
    with the argument `update_other_formats=True`,
    it should write the table to the CSV and PKL file.

    Args:
        tmpdir: tmpdir fixture
        storage_format: storage format of table
        existing_formats: formats the table should be stored in
            before saving to ``storage_format``
        update_other_formats: if tables specified in ``existing_formats``
            should be updated when saving ``storage_format``

    """
    db = audformat.testing.create_db()

    table_id = "files"
    table_file = audeer.path(tmpdir, "table")

    # Create existing table files and pause for a short time
    old_mtime = {}
    for ext in existing_formats:
        db[table_id].save(
            table_file,
            storage_format=ext,
            update_other_formats=False,
        )
        old_mtime[ext] = os.path.getmtime(f"{table_file}.{ext}")
    time.sleep(0.05)

    # Store table to requested format
    db[table_id].save(
        table_file,
        storage_format=storage_format,
        update_other_formats=update_other_formats,
    )

    # Collect mtimes of existing table files
    mtime = {}
    formats = existing_formats + [storage_format]
    for ext in formats:
        mtime[ext] = os.path.getmtime(f"{table_file}.{ext}")

    # Ensure mtimes are correct
    if update_other_formats:
        if "pickle" in formats and "csv" in formats:
            assert mtime["pickle"] >= mtime["csv"]
        if "pickle" in formats and "parquet" in formats:
            assert mtime["pickle"] >= mtime["parquet"]
        if "csv" in formats and "parquet" in formats:
            assert mtime["csv"] >= mtime["parquet"]
    else:
        for ext in existing_formats:
            assert mtime[ext] == old_mtime[ext]
            assert mtime[storage_format] > old_mtime[ext]
