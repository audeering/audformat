import filecmp
import os

import pandas as pd
import pytest

import audformat
import audformat.testing


@pytest.mark.parametrize(
    'files, num_workers',
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
            lambda x: '1' in x,
            None,
        ),
    ]
)
def test_drop_files(files, num_workers):

    db = audformat.testing.create_db()
    db.drop_files(files, num_workers=num_workers)
    if callable(files):
        files = db.files.to_series().apply(files)
    else:
        if isinstance(files, str):
            files = [files]
    assert db.files.intersection(files).empty


@pytest.mark.parametrize(
    'files, num_workers',
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
            lambda x: '1' in x,
            None,
        ),
    ]
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


def test_drop_and_pick_tables():

    db = audformat.testing.create_db()

    assert 'segments' in db
    db.pick_tables('files')
    assert 'segments' not in db

    db = audformat.testing.create_db()

    assert 'segments' in db
    db.drop_tables('segments')
    assert 'segments' not in db


@pytest.mark.parametrize(
    'license, license_url, expected_license, expected_url',
    [
        (
            audformat.define.License.CC0_1_0,
            None,
            'CC0-1.0',
            'https://creativecommons.org/publicdomain/zero/1.0/',
        ),
        (
            audformat.define.License.CC0_1_0,
            'https://custom.org',
            'CC0-1.0',
            'https://custom.org',
        ),
        (
            'custom',
            None,
            'custom',
            None,
        ),
        (
            'custom',
            'https://custom.org',
            'custom',
            'https://custom.org',
        ),
    ]
)
def test_license(license, license_url, expected_license, expected_url):
    db = audformat.Database(
        'test',
        license=license,
        license_url=license_url,
    )
    assert db.license == expected_license
    assert db.license_url == expected_url


@pytest.mark.parametrize(
    'num_workers',
    [
        1,
        4,
        None,
    ]
)
def test_map_files(num_workers):

    db = audformat.testing.create_db()

    files = sorted(db.files)
    db.map_files(lambda x: x.upper(), num_workers=num_workers)
    assert [x.upper() for x in files] == sorted(db.files)


@pytest.mark.parametrize(
    'db, storage_format, num_workers',
    [
        (
            audformat.testing.create_db(minimal=True),
            audformat.define.TableStorageFormat.CSV,
            1,
        ),
        (
            audformat.testing.create_db(minimal=True),
            audformat.define.TableStorageFormat.PICKLE,
            1,
        ),
        (
            audformat.testing.create_db(),
            audformat.define.TableStorageFormat.CSV,
            4,
        ),
        (
            audformat.testing.create_db(),
            audformat.define.TableStorageFormat.PICKLE,
            None,
        ),
    ],
)
def test_save_and_load(tmpdir, db, storage_format, num_workers):

    db.save(
        tmpdir,
        storage_format=storage_format,
        num_workers=num_workers,
    )
    expected_formats = [storage_format]
    for table_id in db.tables:
        for ext in audformat.define.TableStorageFormat.attribute_values():
            table_file = os.path.join(tmpdir, f'db.{table_id}.{ext}')
            if ext in expected_formats:
                assert os.path.exists(table_file)
            else:
                assert not os.path.exists(table_file)

    # Test update other formats
    if (
            storage_format == audformat.define.TableStorageFormat.CSV
            and db.tables
    ):
        db2 = audformat.testing.create_db()
        db2.save(
            tmpdir,
            storage_format=audformat.define.TableStorageFormat.PICKLE,
            num_workers=num_workers,
        )

        # Load prefers PKL files over CSV files,
        # which means we are loading the second database here
        db_load = audformat.Database.load(tmpdir)
        assert db_load == db2
        assert db_load != db

        # Save and not update PKL files,
        # now it should raise an error as CSV file is newer
        db.save(
            tmpdir,
            storage_format=audformat.define.TableStorageFormat.CSV,
            num_workers=num_workers,
            update_other_formats=False,
        )
        # The replace part handles Windows paths
        table_file = os.path.join(tmpdir, 'db.files')
        table_path = table_file.replace('\\', '\\\\')
        error_msg = (
            f"The table CSV file '{table_path}.csv' is newer "
            f"than the table PKL file '{table_path}.pkl'. "
            "If you want to load from the CSV file, "
            "please delete the PKL file. "
            "If you want to load from the PKL file, "
            "please delete the CSV file."
        )
        with pytest.raises(RuntimeError, match=error_msg):
            db_load = audformat.Database.load(tmpdir)

        # Save and update PKL files
        db.save(
            tmpdir,
            storage_format=audformat.define.TableStorageFormat.CSV,
            num_workers=num_workers,
            update_other_formats=True,
        )
        db_load = audformat.Database.load(tmpdir)
        assert db_load == db

    db_load = audformat.Database.load(tmpdir)
    db_load.save(
        tmpdir,
        name='db-2',
        storage_format=storage_format,
        num_workers=num_workers,
    )

    assert filecmp.cmp(
        os.path.join(tmpdir, 'db.yaml'),
        os.path.join(tmpdir, 'db-2.yaml'),
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
        load_data=False,
        num_workers=num_workers,
    )
    for table_id, table in db_load.tables.items():
        assert list(db_load.files) == []
        assert table._id == table_id
        assert table._db == db_load
        assert str(db_load) == str(db)
        for column_id, column in table.columns.items():
            assert column._id == column_id
            assert column._table is table

    # Test missing table
    if db.tables:
        table_id = list(db.tables)[0]
        for ext in audformat.define.TableStorageFormat.attribute_values():
            table_file = os.path.join(tmpdir, f'db.{table_id}.{ext}')
            if os.path.exists(table_file):
                os.remove(table_file)

        # The replace part handles Windows paths
        table_path = table_file[:-4].replace('\\', '\\\\')
        error_msg = (
            r"No file found for table with path "
            rf"'{table_path}.{{pkl|csv}}'"
        )
        with pytest.raises(RuntimeError, match=error_msg):
            audformat.Database.load(tmpdir)


def test_string():

    db = audformat.testing.create_db(minimal=True)
    assert str(db) == 'name: unittest\n' \
                      'source: internal\n' \
                      'usage: unrestricted\n' \
                      'languages: [deu, eng]'
