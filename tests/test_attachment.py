import os

import pytest

import audeer
import audformat


def test_attachment(tmpdir):

    # Path needs to be relative
    path = '/root/file.txt'
    error_msg = f"The provided path '{path}' needs to be relative."
    with pytest.raises(ValueError, match=error_msg):
        audformat.Attachment(path)

    # Create database (path does not need to exist)
    path = 'attachments/file.txt'
    db = audformat.Database('db')
    db.attachments['file'] = audformat.Attachment(
        path,
        description='Attached file',
        meta={'mime': 'text'},
    )

    assert list(db.attachments) == ['file']
    assert db.attachments['file'].path == path
    assert db.attachments['file'].description == 'Attached file'
    assert db.attachments['file'].meta == {'mime': 'text'}

    db_path = audeer.path(tmpdir, 'db')
    audeer.mkdir(db_path)

    # Save database, path needs to exist
    error_msg = (
        f"The provided path '{path}' "
        f"of attachment 'file' "
        "does not exist."
    )
    with pytest.raises(FileNotFoundError, match=error_msg):
        db.save(db_path)

    # Save database, path needs to be a file
    audeer.mkdir(audeer.path(db_path, path))
    error_msg = (
        f"The provided path '{path}' "
        f"of attachment 'file' "
        "is not a file."
    )
    with pytest.raises(FileNotFoundError, match=error_msg):
        db.save(db_path)

    audeer.rmdir(audeer.path(db_path, path))
    audeer.mkdir(audeer.path(db_path, os.path.dirname(path)))
    audeer.touch(audeer.path(db_path, path))
    db.save(db_path)

    # Load database
    db = audformat.Database.load(db_path)
    assert list(db.attachments) == ['file']
    assert db.attachments['file'].path == path
    assert db.attachments['file'].description == 'Attached file'
    assert db.attachments['file'].meta == {'mime': 'text'}

    # Load database, path needs to exist
    audeer.rmdir(audeer.path(db_path, os.path.dirname(path)))
    assert not os.path.exists(audeer.path(db_path, path))
    error_msg = (
        f"The provided path '{path}' "
        f"of attachment 'file' "
        "does not exist."
    )
    with pytest.raises(FileNotFoundError, match=error_msg):
        db = audformat.Database.load(db_path)
