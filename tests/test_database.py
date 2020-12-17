import os
import filecmp

import pandas as pd
import pytest

import audformat
import audformat.testing


@pytest.mark.parametrize(
    'files',
    [
        pytest.DB.files,
        pytest.DB.files[:2],
        pytest.DB.files[0],
        lambda x: '1' in x,
    ]
)
def test_drop_files(files):

    db = audformat.testing.create_db()
    db.drop_files(files)
    if callable(files):
        files = db.files.to_series().apply(files)
    else:
        if isinstance(files, str):
            files = [files]
    assert db.files.intersection(files).empty


@pytest.mark.parametrize(
    'files',
    [
        pytest.DB.files,
        pytest.DB.files[:2],
        pytest.DB.files[0],
        lambda x: '1' in x,
    ]
)
def test_pick_files(files):

    db = audformat.testing.create_db()
    db.pick_files(files)
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


def test_map_files():

    db = audformat.testing.create_db()

    files = sorted(db.files)
    db.map_files(lambda x: x.upper())
    assert [x.upper() for x in files] == sorted(db.files)


@pytest.mark.parametrize(
    'db, storage_format',
    [
        (
            audformat.testing.create_db(minimal=True),
            audformat.define.TableStorageFormat.CSV,
        ),
        (
            audformat.testing.create_db(minimal=True),
            audformat.define.TableStorageFormat.PICKLE,
        ),
        (
            audformat.testing.create_db(),
            audformat.define.TableStorageFormat.CSV,
        ),
        (
            audformat.testing.create_db(),
            audformat.define.TableStorageFormat.PICKLE,
        ),
    ],
)
def test_save_and_load(tmpdir, db, storage_format):

    db.save(tmpdir, storage_format=storage_format)

    db_load = audformat.Database.load(tmpdir)
    db_load.save(tmpdir, name='db-2', storage_format=storage_format)

    assert filecmp.cmp(os.path.join(tmpdir, 'db.yaml'),
                       os.path.join(tmpdir, 'db-2.yaml'))

    ext = f'.{storage_format}'
    for table_id, table in db.tables.items():
        if storage_format != audformat.define.TableStorageFormat.CSV:
            # TODO: why is this test failing if compression is turned off?
            assert db[table_id].df.equals(db_load[table_id].df)
        else:
            assert filecmp.cmp(
                os.path.join(tmpdir, 'db.{}{}'.format(table_id, ext)),
                os.path.join(tmpdir, 'db-2.{}{}'.format(table_id, ext)))
        assert db[table_id].df.dtypes.equals(db_load[table_id].df.dtypes)
        assert db[table_id].files.equals(db_load[table_id].files)
        assert table._id == table_id
        assert table._db is db

    # Test load_data=False
    db_load = audformat.Database.load(tmpdir, load_data=False)
    for table_id, table in db_load.tables.items():
        assert list(db_load.files) == []
        assert table._id == table_id
        assert table._db == db_load
        assert str(db_load) == str(db)
        for column_id, column in table.columns.items():
            assert column._id == column_id
            assert column._table is table


def test_string():

    db = audformat.testing.create_db(minimal=True)
    assert str(db) == 'name: unittest\n' \
                      'source: internal\n' \
                      'usage: unrestricted\n' \
                      'languages: [deu, eng]'
