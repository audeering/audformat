import os
import shutil
import filecmp

import pytest

import audformat
import audformat.testing


def test_map_files():

    db = audformat.testing.create_db()

    files = sorted(db.files)
    db.map_files(lambda x: x.upper())
    assert [x.upper() for x in files] == sorted(db.files)


def test_filter_files():

    db = audformat.testing.create_db()

    files = db.files
    n_removed = db.filter_files(lambda x: bool(
        int(os.path.splitext(os.path.basename(x))[0]) % 2))
    assert n_removed == len(files) - len(db.files)
    assert all(bool(int(os.path.splitext(
        os.path.basename(x))[0]) % 2) for x in db.files)


def test_drop_and_pick():

    db = audformat.testing.create_db()

    assert 'segments' in db
    db.pick_tables('files')
    assert 'segments' not in db

    db = audformat.testing.create_db()

    assert 'segments' in db
    db.drop_tables('segments')
    assert 'segments' not in db


@pytest.mark.parametrize(
    'db, compressed',
    [
        (audformat.testing.create_db(minimal=True), False),
        (audformat.testing.create_db(minimal=True), True),
        (audformat.testing.create_db(), False),
        (audformat.testing.create_db(), True)],
)
def test_save_and_load(tmpdir, db, compressed):

    db.save(tmpdir, compressed=compressed)

    db_load = audformat.Database.load(tmpdir)
    db_load.save(tmpdir, name='db-2', compressed=compressed)

    assert filecmp.cmp(os.path.join(tmpdir, 'db.yaml'),
                       os.path.join(tmpdir, 'db-2.yaml'))

    ext = '.pkl' if compressed else '.csv'
    for table_id, table in db.tables.items():
        if compressed:
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
