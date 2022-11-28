import os
import re

import pytest

import audeer
import audformat


def test_attachment(tmpdir):

    # Path needs to be relative and not contain ., .., \
    for path in [
        '/root/file.txt',
        './root',
        './file.txt',
        '../file.txt',
        'doc/./file.txt',
        'doc/../file.txt',
        'doc/../../file.txt',
        r'C:\\doc\file.txt',
    ]:
        error_msg = (
            f"The provided path '{path}' needs to be relative "
            "and not contain '\\', '.', or '..'."
        )
        with pytest.raises(ValueError, match=re.escape(error_msg)):
            audformat.Attachment(path)

    # Create database (path does not need to exist)
    file_path = 'attachments/file.txt'
    folder_path = 'attachments/folder'
    db = audformat.Database('db')
    db.attachments['file'] = audformat.Attachment(
        file_path,
        description='Attached file',
        meta={'mime': 'text'},
    )
    db.attachments['folder'] = audformat.Attachment(
        folder_path,
        description='Attached folder',
        meta={'mime': 'inode/directory'},
    )

    assert list(db.attachments) == ['file', 'folder']
    assert db.attachments['file'].path == file_path
    assert db.attachments['file'].description == 'Attached file'
    assert db.attachments['file'].meta == {'mime': 'text'}
    assert db.attachments['folder'].path == folder_path
    assert db.attachments['folder'].description == 'Attached folder'
    assert db.attachments['folder'].meta == {'mime': 'inode/directory'}

    db_path = audeer.path(tmpdir, 'db')
    audeer.mkdir(db_path)

    # Save database, path needs to exist
    error_msg = (
        f"The provided path '{file_path}' "
        f"of attachment 'file' "
        "does not exist."
    )
    with pytest.raises(FileNotFoundError, match=error_msg):
        db.save(db_path)

    # Save database, path is not allowed to be a symlink
    audeer.mkdir(audeer.path(db_path, folder_path))
    os.symlink(
        audeer.path(db_path, folder_path),
        audeer.path(db_path, file_path),
    )
    error_msg = (
        f"The provided path '{file_path}' "
        f"of attachment 'file' "
        "is not allowed to be a symlink."
    )
    with pytest.raises(RuntimeError, match=error_msg):
        db.save(db_path)

    os.remove(os.path.join(db_path, file_path))
    audeer.touch(audeer.path(db_path, file_path))
    db.save(db_path)

    # Load database
    db = audformat.Database.load(db_path)
    assert list(db.attachments) == ['file', 'folder']
    assert db.attachments['file'].path == file_path
    assert db.attachments['file'].description == 'Attached file'
    assert db.attachments['file'].meta == {'mime': 'text'}
    assert db.attachments['folder'].path == folder_path
    assert db.attachments['folder'].description == 'Attached folder'
    assert db.attachments['folder'].meta == {'mime': 'inode/directory'}

    # Load database, path needs to exist
    audeer.rmdir(audeer.path(db_path, os.path.dirname(file_path)))
    assert not os.path.exists(audeer.path(db_path, file_path))
    error_msg = (
        f"The provided path '{file_path}' "
        f"of attachment 'file' "
        "does not exist."
    )
    with pytest.raises(FileNotFoundError, match=error_msg):
        db = audformat.Database.load(db_path)
