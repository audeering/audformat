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

    # Files do not exist yet on disc
    assert db.attachments['file'].files == []
    assert db.attachments['folder'].files == []

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

    # Remove file and check output of .files
    os.remove(os.path.join(db_path, file_path))
    assert db.attachments['file'].files == []
    assert db.attachments['folder'].files == []
    audeer.touch(audeer.path(db_path, file_path))

    # File exist now, folder is empty
    assert db.attachments['file'].files == [audeer.path(db_path, file_path)]
    assert db.attachments['folder'].files == []

    # Load database
    db = audformat.Database.load(db_path)
    assert list(db.attachments) == ['file', 'folder']
    assert db.attachments['file'].files == [audeer.path(db_path, file_path)]
    assert db.attachments['folder'].files == []
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


@pytest.mark.parametrize(
    # `expected` should be given as a relative path
    # starting from `db.root`
    # and will be converted to an absolute path
    # in the test
    'root, folders, files, expected',
    [
        (
            'extra',
            [],
            [],
            [],
        ),
        (
            'extra',
            [],
            ['file1.txt'],
            ['extra/file1.txt'],
        ),
        (
            'extra',
            [],
            ['sub/file1.txt'],
            ['extra/sub/file1.txt'],
        ),
        (
            'extra',
            ['sub1'],
            ['sub2/file1.txt'],
            ['extra/sub2/file1.txt'],
        ),
        (
            'extra',
            [],
            ['f1.txt', 'f2.txt'],
            ['extra/f1.txt', 'extra/f2.txt'],
        ),
    ]
)
def test_attachment_files(tmpdir, root, folders, files, expected):
    db_path = audeer.path(tmpdir, 'db')
    # Convert `expected` to absolute path
    expected = [audeer.path(db_path, p) for p in expected]
    root_path = audeer.path(db_path, root)
    audeer.mkdir(root_path)
    for folder in folders:
        path = audeer.path(root_path, folder)
        audeer.mkdir(path)
    for file in files:
        path = audeer.path(root_path, file)
        audeer.mkdir(os.path.dirname(path))
        audeer.touch(path)
    db = audformat.Database('db')
    db.attachments['extra'] = audformat.Attachment(root)
    db_path = audeer.path(tmpdir, 'db')
    db.save(db_path)
    assert db.attachments['extra'].files == expected


@pytest.mark.parametrize(
    'attachments',
    [
        [
            'extra/file1.txt',
            'extra/file2.txt',
        ],
        [
            'extra/file1.txt',
            'extra/sub/file1.txt',
        ],
        [
            'extra/file1.txt',
            'extra/sub/file1.txt',
            'extra/folder1',
        ],
        [
            'extra/file1.txt',
            'extra/sub/file1.txt',
            'extra/folder1',
            'extra/sub/folder2',
            'extra/sub/folder3',
        ],
        pytest.param(  # Same files
            [
                'extra/file1.txt',
                'extra/file1.txt',
            ],
            marks=pytest.mark.xfail(raises=RuntimeError),
        ),
        pytest.param(  # Same files
            [
                'extra/file1.txt',
                'extra/file1.txt',
                'extra/file2.txt',
                'extra/folder1',
            ],
            marks=pytest.mark.xfail(raises=RuntimeError),
        ),
        pytest.param(  # Overlapping file + folder
            [
                'extra/file1.txt',
                'extra',
            ],
            marks=pytest.mark.xfail(raises=RuntimeError),
        ),
        pytest.param(  # Overlapping file + folder
            [
                'extra/sub/file1.txt',
                'extra',
            ],
            marks=pytest.mark.xfail(raises=RuntimeError),
        ),
        pytest.param(  # Overlapping file + folder
            [
                'extra/sub/file1.txt',
                'extra/sub',
                'extra/file2.txt',
            ],
            marks=pytest.mark.xfail(raises=RuntimeError),
        ),
        pytest.param(  # Overlapping file + folder and same files
            [
                'extra/sub/file1.txt',
                'extra/sub',
                'extra/file2.txt',
                'extra/file2.txt',
            ],
            marks=pytest.mark.xfail(raises=RuntimeError),
        ),
        pytest.param(  # Nested folders
            [
                'extra/sub/folder1',
                'extra/sub',
            ],
            marks=pytest.mark.xfail(raises=RuntimeError),
        ),
        pytest.param(  # Nested folders
            [
                'extra/sub/folder1',
                'extra/sub/folder2',
                'extra/file1.txt',
                'extra/sub',
            ],
            marks=pytest.mark.xfail(raises=RuntimeError),
        ),
    ]
)
def test_attachment_overlapping(tmpdir, attachments):
    # Test for non-saved database
    db = audformat.Database('db')
    for n, attachment in enumerate(attachments):
        db.attachments[str(n)] = audformat.Attachment(attachment)
    db_path = audeer.path(tmpdir, 'db')
    audformat.testing.create_attachment_files(db, db_path)
    db.save(db_path)
    # Test for list of files
    # (an attachment is considered a file
    # if it contains at least one .)
    for n, attachment in enumerate(attachments):
        expected_files = []
        if '.' in attachment:
            expected_files = [audeer.path(db_path, attachment)]
        assert db.attachments[str(n)].files == expected_files

    # Test for saved database,
    # that contains attachment files
    db = audformat.Database('db')
    db.save(db_path)
    for n, attachment in enumerate(attachments):
        db.attachments[str(n)] = audformat.Attachment(attachment)
