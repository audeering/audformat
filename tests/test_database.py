import datetime
import filecmp
import os
import re

import pandas as pd
import pytest

import audeer
import audiofile

import audformat
import audformat.testing


def full_path(
    db: audformat.Database,
    db_root: str,
):
    # Faster solution then using db.map_files()
    root = db_root + os.path.sep
    for table in db.tables.values():
        if table.is_filewise:
            table.df.index = root + table.df.index
            table.df.index.name = "file"
        elif len(table.df.index) > 0:
            table.df.index = table.df.index.set_levels(
                root + table.df.index.levels[0],
                level="file",
            )


def test_create_db():
    db = audformat.testing.create_db()
    assert all(["\\" not in file for file in db.files])


@pytest.mark.parametrize(
    "db, table_id, expected",
    [
        (
            audformat.Database("test"),
            "non-existing",
            False,
        ),
        (
            audformat.testing.create_db(),
            "non-existing",
            False,
        ),
        (
            audformat.testing.create_db(),
            "misc",
            True,
        ),
        (
            audformat.testing.create_db(),
            "files",
            True,
        ),
    ],
)
def test_contains(db, table_id, expected):
    assert (table_id in db) == expected


@pytest.mark.parametrize(
    "files, num_workers",
    [
        (
            pytest.DB.files,
            1,
        ),
        (
            pytest.DB.files[:2],
            1,
        ),
        (
            pytest.DB.files[0],
            4,
        ),
        (
            lambda x: "1" in x,
            None,
        ),
    ],
)
def test_drop_files(files, num_workers):
    db = audformat.testing.create_db()
    db.drop_files(files, num_workers=num_workers)
    if callable(files):
        files = db.files.to_series().apply(files)
    else:
        if isinstance(files, str):
            files = [files]
    assert len(db.files.intersection(files)) == 0


def test_files():
    db = audformat.testing.create_db()
    # Shuffle order of files
    db["files"]._df = db["files"].df.sample(frac=1, random_state=0)
    # Check output is sorted
    assert list(db.files) == sorted(list(db.files))


@pytest.mark.parametrize(
    "files, expected",
    [
        (
            [],
            True,
        ),
        (
            ["file.txt"],
            True,
        ),
        (
            [".file.txt"],
            True,
        ),
        (
            ["file..txt"],
            True,
        ),
        (
            ["/a/b/c/.file.txt"],
            False,
        ),
        (
            ["a/b/c/.file.txt"],
            True,
        ),
        (
            ["./file.txt"],
            False,
        ),
        (
            ["../file.txt"],
            False,
        ),
        (
            ["a/b/c/./file.txt"],
            False,
        ),
        (
            ["a/b/c/../file.txt"],
            False,
        ),
        (
            ["D:\\absolute\\windows\\path"],
            False,
        ),
        (
            ["relative\\windows\\path"],
            False,
        ),
    ],
)
def test_is_portable(files, expected):
    db = audformat.testing.create_db(minimal=True)
    db["table"] = audformat.Table(index=audformat.filewise_index(files))
    assert db.is_portable == expected


@pytest.mark.parametrize(
    "files, num_workers",
    [
        (
            pytest.DB.files,
            1,
        ),
        (
            pytest.DB.files[:2],
            1,
        ),
        (
            pytest.DB.files[0],
            4,
        ),
        (
            lambda x: "1" in x,
            None,
        ),
    ],
)
def test_pick_files(files, num_workers):
    db = audformat.testing.create_db()
    db.pick_files(files, num_workers=num_workers)
    if callable(files):
        files = db.files[db.files.to_series().apply(files)]
    else:
        if isinstance(files, str):
            files = [files]
    pd.testing.assert_index_equal(
        db.files,
        audformat.filewise_index(files),
    )


@pytest.mark.parametrize(
    "db, tables, expected_tables",
    [
        (
            audformat.testing.create_db(),
            "segments",
            ["files", "misc"],
        ),
        (
            audformat.testing.create_db(),
            "misc",
            ["files", "segments"],
        ),
        (
            audformat.testing.create_db(),
            ["segments"],
            ["files", "misc"],
        ),
        (
            audformat.testing.create_db(),
            ["segments", "misc"],
            ["files"],
        ),
        pytest.param(  # non-existent table ID
            audformat.testing.create_db(),
            ["segments", "misc", "non-existing"],
            None,
            marks=pytest.mark.xfail(raises=audformat.errors.BadIdError),
        ),
    ],
)
def test_drop_tables(db, tables, expected_tables):
    if "misc" in audeer.to_list(tables):

        def error_msg(table_id):
            return re.escape(
                f"Misc table '{table_id}' is used as scheme(s): "
                "'label_map_misc', "
                "and cannot be removed."
            )

        with pytest.raises(RuntimeError, match=error_msg("misc")):
            db.drop_tables("misc")

        # Replace scheme with other misc table
        db["misc_copy"] = db["misc"].copy()
        db.schemes["label_map_misc"].replace_labels("misc_copy")
        db.drop_tables("misc")
        with pytest.raises(RuntimeError, match=error_msg("misc_copy")):
            db.drop_tables("misc_copy")

        # Delete scheme and remove copied table as well
        del db.schemes["label_map_misc"]
        db.drop_tables("misc_copy")

        tables = [t for t in audeer.to_list(tables) if t != "misc"]

    db.drop_tables(tables)
    assert list(db) == expected_tables


@pytest.mark.parametrize(
    "db, tables, expected_tables",
    [
        pytest.param(
            audformat.testing.create_db(),
            "segments",
            ["segments"],
        ),
        (
            audformat.testing.create_db(),
            "misc",
            ["misc"],
        ),
        (
            audformat.testing.create_db(),
            ["segments"],
            ["segments"],
        ),
        (
            audformat.testing.create_db(),
            ["segments", "misc"],
            ["misc", "segments"],
        ),
        pytest.param(  # non-existent table ID
            audformat.testing.create_db(),
            ["segments", "misc", "non-existing"],
            None,
            marks=pytest.mark.xfail(raises=audformat.errors.BadIdError),
        ),
    ],
)
def test_pick_tables(db, tables, expected_tables):
    if expected_tables is not None and "misc" not in expected_tables:
        error_msg = (
            "Misc table 'misc' is used as scheme(s): 'label_map_misc', "
            "and cannot be removed."
        )
        with pytest.raises(RuntimeError, match=re.escape(error_msg)):
            db.pick_tables(tables)
        del db.schemes["label_map_misc"]

    db.pick_tables(tables)
    assert list(db) == expected_tables


def test_files_duration():
    db = pytest.DB

    # prepare file names

    files_rel = db.files
    files_abs = [os.path.join(db.root, file) for file in files_rel]
    durs = [pd.to_timedelta(audiofile.duration(file), unit="s") for file in files_abs]

    # test with relative file names

    expected_rel = pd.Series(
        durs,
        index=files_rel.tolist(),
        name=audformat.define.IndexField.FILE,
    )
    for _ in range(2):
        y = db.files_duration(files_rel)
        pd.testing.assert_series_equal(y, expected_rel)

    # test with absolute file names

    expected_abs = pd.Series(
        durs,
        index=files_abs,
        name=audformat.define.IndexField.FILE,
    )
    for _ in range(2):
        y = db.files_duration(files_abs)
        pd.testing.assert_series_equal(y, expected_abs)

    # simulate that we have not loaded db from disk

    root = db._root
    db._root = None

    with pytest.raises(ValueError):
        db.files_duration(files_rel)

    for _ in range(2):
        y = db.files_duration(files_rel, root=root)
        pd.testing.assert_series_equal(y, expected_rel)

    # make sure we have only absolute file names in cache

    expected_cache = {os.path.normpath(file): dur for file, dur in zip(files_abs, durs)}
    assert db._files_duration == expected_cache

    # reset db

    db._files_duration = {}
    db._root = root


def test_iter():
    db = audformat.testing.create_db(minimal=True)
    assert list(db) == []
    db = audformat.testing.create_db()
    assert list(db) == ["files", "misc", "segments"]


@pytest.mark.parametrize(
    "license, license_url, expected_license, expected_url",
    [
        (
            audformat.define.License.CC0_1_0,
            None,
            "CC0-1.0",
            "https://creativecommons.org/publicdomain/zero/1.0/",
        ),
        (
            audformat.define.License.CC0_1_0,
            "https://custom.org",
            "CC0-1.0",
            "https://custom.org",
        ),
        (
            "custom",
            None,
            "custom",
            None,
        ),
        (
            "custom",
            "https://custom.org",
            "custom",
            "https://custom.org",
        ),
    ],
)
def test_license(license, license_url, expected_license, expected_url):
    db = audformat.Database(
        "test",
        license=license,
        license_url=license_url,
    )
    assert db.license == expected_license
    assert db.license_url == expected_url


def test_load(tmpdir):
    # Test loading a database containing a misc table as scheme,
    # see https://github.com/audeering/audformat/issues/294
    db = audformat.testing.create_db(minimal=True)
    db.schemes["scheme3"] = audformat.Scheme("str")
    db.schemes["scheme1"] = audformat.Scheme(labels=["some", "test", "labels"])
    audformat.testing.add_misc_table(
        db,
        "misc-in-scheme",
        pd.Index([0, 1, 2], dtype="Int64", name="idx"),
        columns={"emotion": ("scheme1", None)},
    )
    db.schemes["misc"] = audformat.Scheme(
        "int",
        labels="misc-in-scheme",
    )
    db.schemes["scheme2"] = audformat.Scheme("float")
    assert list(db.schemes) == ["misc", "scheme1", "scheme2", "scheme3"]
    db.save(tmpdir)
    db = audformat.Database.load(tmpdir)
    assert list(db.schemes) == ["misc", "scheme1", "scheme2", "scheme3"]


@pytest.mark.parametrize(
    "num_workers",
    [
        1,
        4,
        None,
    ],
)
def test_map_files(num_workers):
    db = audformat.testing.create_db()

    files = sorted(db.files)
    db.map_files(lambda x: x.upper(), num_workers=num_workers)
    assert [x.upper() for x in files] == sorted(db.files)


@pytest.mark.parametrize(
    "db, storage_format, load_data, num_workers",
    [
        (
            audformat.testing.create_db(minimal=True),
            audformat.define.TableStorageFormat.PARQUET,
            False,
            1,
        ),
        (
            audformat.testing.create_db(minimal=True),
            audformat.define.TableStorageFormat.CSV,
            False,
            1,
        ),
        (
            audformat.testing.create_db(minimal=True),
            audformat.define.TableStorageFormat.PICKLE,
            False,
            1,
        ),
        (
            audformat.testing.create_db(),
            audformat.define.TableStorageFormat.PARQUET,
            False,
            4,
        ),
        (
            audformat.testing.create_db(),
            audformat.define.TableStorageFormat.CSV,
            False,
            4,
        ),
        (
            audformat.testing.create_db(),
            audformat.define.TableStorageFormat.PICKLE,
            False,
            None,
        ),
        (
            audformat.testing.create_db(),
            audformat.define.TableStorageFormat.PICKLE,
            True,
            None,
        ),
    ],
)
def test_save_and_load(tmpdir, db, storage_format, load_data, num_workers):
    all_formats = audformat.define.TableStorageFormat._attribute_values()
    non_cache_formats = [
        ext for ext in all_formats if ext != audformat.define.TableStorageFormat.PICKLE
    ]

    assert db.root is None
    audformat.testing.create_attachment_files(db, tmpdir)
    db.save(
        tmpdir,
        storage_format=storage_format,
        num_workers=num_workers,
    )
    assert db.root == tmpdir

    expected_formats = [storage_format]
    for table_id in db.tables:
        for ext in all_formats:
            table_file = os.path.join(tmpdir, f"db.{table_id}.{ext}")
            if ext in expected_formats:
                assert os.path.exists(table_file)
            else:
                assert not os.path.exists(table_file)

    # Test update other formats
    if storage_format in non_cache_formats and db.tables:
        db2 = audformat.testing.create_db()
        assert db2.root is None
        db2.save(
            tmpdir,
            storage_format=audformat.define.TableStorageFormat.PICKLE,
            num_workers=num_workers,
        )
        assert db.root == tmpdir

        # Load prefers PKL files,
        # which means we are loading the second database here
        db_load = audformat.Database.load(
            tmpdir,
            load_data=load_data,
        )

        # Ensure no data is loaded
        if not load_data:
            for table in list(db_load.tables):
                assert db_load[table]._df is None

        assert db_load.root == db2.root
        assert db_load == db2
        assert db_load != db

        # After comparing the databases, tables should be loaded
        if not load_data:
            for table in list(db_load.tables):
                assert db_load[table]._df is not None

        # Save and not update PKL files,
        # now it should raise an error as CSV file is newer
        db.save(
            tmpdir,
            storage_format=audformat.define.TableStorageFormat.CSV,
            num_workers=num_workers,
            update_other_formats=False,
        )
        # The replace part handles Windows paths
        table_file = os.path.join(tmpdir, "db.files")
        table_path = table_file.replace("\\", "\\\\")
        error_msg = (
            f"The table CSV file '{table_path}.csv' is newer "
            f"than the table PKL file '{table_path}.pkl'. "
            "If you want to load from the CSV file, "
            "please delete the PKL file. "
            "If you want to load from the PKL file, "
            "please delete the CSV file."
        )
        with pytest.raises(RuntimeError, match=error_msg):
            db_load = audformat.Database.load(
                tmpdir,
                load_data=load_data,
            )
            db_load["files"].get()

        # Save and update PKL files
        db.save(
            tmpdir,
            storage_format=audformat.define.TableStorageFormat.CSV,
            num_workers=num_workers,
            update_other_formats=True,
        )
        db_load = audformat.Database.load(
            tmpdir,
            load_data=load_data,
        )
        # Ensure saving the database works
        # when loaded without tables before
        # https://github.com/audeering/audformat/issues/131
        if not load_data:
            db_load.save(
                tmpdir,
                storage_format=audformat.define.TableStorageFormat.CSV,
                num_workers=num_workers,
                update_other_formats=True,
            )

        assert db_load == db

    db_load = audformat.Database.load(
        tmpdir,
        load_data=load_data,
    )
    db_load.save(
        tmpdir,
        name="db-2",
        storage_format=storage_format,
        num_workers=num_workers,
    )

    assert filecmp.cmp(
        os.path.join(tmpdir, "db.yaml"),
        os.path.join(tmpdir, "db-2.yaml"),
    )

    for table_id, table in db.tables.items():
        assert db[table_id].df.equals(db_load[table_id].df)
        assert db[table_id].df.dtypes.equals(db_load[table_id].df.dtypes)
        assert db[table_id].files.equals(db_load[table_id].files)
        assert table._id == table_id
        assert table._db is db

    # Test load_data=False
    db_load = audformat.Database.load(
        tmpdir,
        load_data=load_data,
        num_workers=num_workers,
    )
    assert db_load.root == tmpdir
    for table_id, table in db_load.tables.items():
        pd.testing.assert_index_equal(db_load.files, db.files)
        assert table._id == table_id
        assert table._db == db_load
        assert str(db_load) == str(db)
        for column_id, column in table.columns.items():
            assert column._id == column_id
            assert column._table is table

    # Test missing table
    if db.tables:
        table_id = list(db.tables)[0]
        for ext in all_formats:
            table_file = os.path.join(tmpdir, f"db.{table_id}.{ext}")
            if os.path.exists(table_file):
                os.remove(table_file)

        # The replace part handles Windows paths
        table_path = table_file[:-4].replace("\\", "\\\\")
        error_msg = (
            r"No file found for table with path " rf"'{table_path}.{{csv|parquet|pkl}}'"
        )
        with pytest.raises(RuntimeError, match=error_msg):
            db = audformat.Database.load(
                tmpdir,
                load_data=load_data,
            )
            db[table_id].get()


def test_save_symlink(tmpdir):
    folder = audeer.mkdir(tmpdir, "folder")
    link = audeer.path(tmpdir, "link")
    os.symlink(folder, link)
    db = audformat.testing.create_db()
    db.save(link)
    assert db.root == folder


def test_segments():
    db = audformat.testing.create_db()
    df = pytest.DB["segments"].get()
    # Shuffle order of segments
    db["segments"]._df = df.sample(frac=1, random_state=0)
    # Check output is sorted
    assert db.segments.equals(pytest.DB.segments)


def test_string():
    db = audformat.testing.create_db(minimal=True)
    assert (
        str(db) == "name: unittest\n"
        "source: internal\n"
        "usage: unrestricted\n"
        "languages: [deu, eng]"
    )


def test_update(tmpdir):
    # original database

    db_root = audeer.mkdir(tmpdir, "db")
    db = audformat.testing.create_db(minimal=True)
    db.author = "author"
    db.organization = "organization"
    db.meta["key"] = "value"
    db.raters["rater"] = audformat.Rater()
    db.attachments["attachment"] = audformat.Attachment("file.txt")
    with open(audeer.path(db_root, "file.txt"), "w") as fp:
        fp.write("db")
    db.schemes["float"] = audformat.Scheme("float")
    db.schemes["labels"] = audformat.Scheme(labels=["a", "b"])
    audformat.testing.add_misc_table(
        db,
        "misc",
        pd.Index(["a", "b"], dtype="string", name="idx"),
        columns={
            "float": ("float", "rater"),
        },
    )
    db.schemes["misc"] = audformat.Scheme(dtype="str", labels="misc")
    audformat.testing.add_table(
        db,
        "table",
        audformat.define.IndexType.FILEWISE,
        num_files=[0, 1],
        columns={
            "float": ("float", "rater"),
            "labels": ("labels", None),
            "misc": ("misc", None),
        },
    )

    db.save(db_root)
    audformat.testing.create_audio_files(db, file_duration="0.1s")

    assert db.update(db) == db

    # databases with schemes that are not compatible

    other_bad = audformat.testing.create_db(minimal=True)
    other_bad.schemes["labels"] = audformat.Scheme(labels=[1, 2, 3])
    error_msg = "Elements or keys must have the same dtype"
    with pytest.raises(ValueError, match=error_msg):
        db.update(other_bad)

    other_bad = audformat.testing.create_db(minimal=True)
    other_bad.schemes["misc"] = audformat.Scheme("str")
    error_msg = (
        "Cannot join scheme 'misc' when one "
        "is using a misc table and the other is not."
    )
    with pytest.raises(ValueError, match=error_msg):
        db.update(other_bad)

    other_bad = audformat.testing.create_db(minimal=True)
    other_bad["misc-bla"] = audformat.MiscTable(
        pd.Index(["b", "c"], dtype="string", name="idx"),
    )
    other_bad.schemes["misc"] = audformat.Scheme(
        dtype="str",
        labels="misc-bla",
    )
    error_msg = (
        "Cannot join scheme 'misc' "
        "when using misc tables with "
        "different IDs: 'misc' != 'misc-bla'"
    )
    with pytest.raises(ValueError, match=error_msg):
        db.update(other_bad)

    # database with same table, but extra column

    other1_root = audeer.mkdir(tmpdir, "other1")
    other1 = audformat.testing.create_db(minimal=True)
    other1.raters["rater"] = audformat.Rater()
    other1.raters["rater2"] = audformat.Rater()
    other1.attachments["attachment"] = audformat.Attachment("file.txt")
    with open(audeer.path(other1_root, "file.txt"), "w") as fp:
        fp.write("other1")
    other1.schemes["int"] = audformat.Scheme(audformat.define.DataType.INTEGER)
    other1.schemes["float"] = audformat.Scheme(audformat.define.DataType.FLOAT)
    other1.schemes["labels"] = audformat.Scheme(labels=["b", "c"])
    audformat.testing.add_misc_table(
        other1,
        "misc",
        pd.Index(["b", "c"], dtype="string", name="idx"),
        columns={
            "float": ("float", "rater"),
            "labels": ("labels", None),
        },
    )
    other1.schemes["misc"] = audformat.Scheme(dtype="str", labels="misc")
    audformat.testing.add_table(
        other1,
        "table",
        audformat.define.IndexType.FILEWISE,
        num_files=[1, 2],
        columns={
            "int": ("int", "rater"),
            "float": ("float", "rater2"),
            "labels": ("labels", None),
            "misc": ("misc", None),
        },
    )
    other1.save(other1_root)
    audformat.testing.create_audio_files(other1, file_duration="0.1s")

    # database with new table

    other2_root = audeer.mkdir(tmpdir, "other2")
    other2 = audformat.testing.create_db(minimal=True)
    other2.raters["rater2"] = audformat.Rater()
    other2.attachments["attachment2"] = audformat.Attachment("file2.txt")
    with open(audeer.path(other2_root, "file2.txt"), "w") as fp:
        fp.write("other2")
    other2.schemes["str"] = audformat.Scheme("str")
    audformat.testing.add_table(
        other2,
        "table_new",
        audformat.define.IndexType.SEGMENTED,
        columns={"str": ("str", "rater2")},
    )
    audformat.testing.add_misc_table(
        other2,
        "misc_new",
        pd.Index([0, 1], dtype="string", name="idx"),
        columns={"str": ("str", "rater2")},
    )
    other2.save(other2_root)
    audformat.testing.create_audio_files(other2, file_duration="0.1s")

    # update db with [other1, other2]

    others = [other1, other2]

    # fails because overlapping values

    with pytest.raises(ValueError):
        db.update(others, overwrite=False)

    # fails because self.root is not given

    db_root = db.root
    db._root = None
    with pytest.raises(RuntimeError):
        db.update(others, overwrite=True, copy_attachments=True)
    with pytest.raises(RuntimeError):
        db.update(others, overwrite=True, copy_media=True)
    db._root = db_root

    # now should work...

    db.update(
        others,
        overwrite=True,
        copy_attachments=True,
        copy_media=True,
    )

    assert "int" in db["table"].columns
    assert "labels" in db["misc"].columns
    assert db["table_new"] == other2["table_new"]
    assert db["misc_new"] == other2["misc_new"]

    for other in others:
        for attachment_id, attachment in other.attachments.items():
            assert db.attachments[attachment_id] == attachment
        for rater_id, rater in other.raters.items():
            assert db.raters[rater_id] == rater
        for scheme_id, scheme in other.schemes.items():
            assert db.schemes[scheme_id] == scheme

    # fail if one of the others has no root folder

    db_root = other1._root
    other1._root = None
    with pytest.raises(RuntimeError):
        db.update(others, overwrite=True, copy_attachments=True)
    with pytest.raises(RuntimeError):
        db.update(others, overwrite=True, copy_media=True)
    other1._root = db_root

    # test attachment files

    expected_content = {
        "file.txt": "other1",
        "file2.txt": "other2",
    }
    for attachment_id in db.attachments:
        for file in db.attachments[attachment_id].files:
            assert os.path.exists(os.path.join(db.root, file))
            if file in expected_content:
                with open(audeer.path(db.root, file)) as fp:
                    assert fp.read() == expected_content[file]

    # test media files

    for file in db.files:
        assert os.path.exists(os.path.join(db.root, file))

    # fail if other has absolute path

    full_path(other2, other2_root)
    with pytest.raises(RuntimeError):
        db.update(other2, overwrite=True, copy_attachments=True)
    with pytest.raises(RuntimeError):
        db.update(other2, overwrite=True, copy_media=True)

    # test other fields

    db_author = db.author
    db_description = db.description
    db_languages = db.languages.copy()
    db_license_url = db.license_url
    db_meta = db.meta.copy()
    db_name = db.name
    db_organization = db.organization
    db_source = db.source

    other = audformat.Database(
        author="other",
        description="other",
        expires=datetime.date.today(),
        languages=audformat.utils.map_language("french"),
        license_url="other",
        meta={"other": "other"},
        name="other",
        organization="other",
        source="other",
    )
    db.update(other)

    assert db.author == f"{db_author}, {other.author}"
    assert db.name == db_name
    assert db.description == db_description
    assert db.expires == other.expires
    assert db.languages == db_languages + other.languages
    assert db.license_url == db_license_url
    db_meta.update(other.meta)
    assert db.meta == db_meta
    assert db.organization == f"{db_organization}, {other.organization}"
    assert db.source == f"{db_source}, {other.source}"

    # errors

    with pytest.raises(ValueError):
        other = audformat.testing.create_db(minimal=True)
        other.license = "other"
        db.update(other)

    with pytest.raises(ValueError):
        other = audformat.testing.create_db(minimal=True)
        other.usage = "other"
        db.update(other)

    with pytest.raises(ValueError):
        other = audformat.testing.create_db(minimal=True)
        other.raters["rater"] = audformat.Rater(type=audformat.define.RaterType.MACHINE)
        db.update(other)

    with pytest.raises(ValueError):
        other = audformat.testing.create_db(minimal=True)
        other.schemes["int"] = audformat.Scheme(str)
        db.update(other)

    with pytest.raises(ValueError):
        other = audformat.testing.create_db(minimal=True)
        other.meta["key"] = "other"
        db.update(other)

    # fail if self has absolute path
    full_path(db, db_root)
    with pytest.raises(RuntimeError):
        db.update(other1, overwrite=True, copy_attachments=True)
    with pytest.raises(RuntimeError):
        db.update(other1, overwrite=True, copy_media=True)


def test_update_misc_table():
    """Tests updating databases with misc tables.

    This addresses the particular case
    of adding a new column
    to a misc table
    in a database having only a single table,
    which was failing as described in
    https://github.com/audeering/audformat/issues/466

    """
    db1 = audformat.Database("mydb")
    db1.schemes["answer1"] = audformat.Scheme("str")
    db1["sessions"] = audformat.MiscTable(pd.Index(["a"], name="session"))
    db1["sessions"]["prompt_1"] = audformat.Column(scheme_id="answer1")
    db1["sessions"]["prompt_1"].set(["response1"])
    df1 = db1["sessions"].df.copy()

    db2 = audformat.Database("mydb")
    db2.schemes["answer1"] = audformat.Scheme("str")
    db2.schemes["answer2"] = audformat.Scheme("str")
    db2["sessions"] = audformat.MiscTable(pd.Index(["b"], name="session"))
    db2["sessions"]["prompt_1"] = audformat.Column(scheme_id="answer1")
    db2["sessions"]["prompt_2"] = audformat.Column(scheme_id="answer2")
    db2["sessions"]["prompt_1"].set(["response1"])
    db2["sessions"]["prompt_2"].set(["response2"])
    df2 = db2["sessions"].df.copy()

    df_expected = pd.concat([df1, df2])

    db1.update(db2)
    pd.testing.assert_frame_equal(db1["sessions"].df, df_expected)
    pd.testing.assert_frame_equal(db2["sessions"].df, df2)
