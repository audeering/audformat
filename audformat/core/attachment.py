import os

import audeer

from audformat.core.common import HeaderBase
from audformat.core.common import is_relative_path


class Attachment(HeaderBase):
    r"""Database attachment.

    Adds a file or folder
    as attachment to a database.
    If a folder is provided,
    all of its sub-folders
    and files are included.

    Args:
        path: relative path to file or folder
        description: attachment description
        meta: additional meta fields

    Raises:
        ValueError: if ``path`` is absolute
            or contains ``\``, ``..`` or ``.``
        RuntimeError: when assigning an attachment
            to a database,
            but the database contains another attachment
            with an path
            that is identical
            or nested
            compared to the current attachment path

    Examples:
        >>> Attachment("file.txt", description="Attached file")
        {description: Attached file, path: file.txt}

    """

    def __init__(
        self,
        path: str,
        *,
        description: str = None,
        meta: dict = None,
    ):
        super().__init__(description=description, meta=meta)

        if not is_relative_path(path):
            raise ValueError(
                f"The provided path '{path}' needs to be relative "
                "and not contain '\\', '.', or '..'."
            )

        self._db = None
        self._id = None

        self.path = path
        r"""Attachment path"""

    @property
    def files(
        self,
    ) -> list[str]:
        r"""List all files that are part of the attachment.

        List recursively the relative path
        of all files that exist
        under :attr:`audformat.Attachment.path`
        on hard disk.

        Raises:
            FileNotFoundError: if a path
                associated with the attachment
                cannot be found
            RuntimeError: if a path
                associated with the attachment
                is a symlink
            RuntimeError: if attachment is not part
                of a database
            RuntimeError: if database is not saved to disk

        """
        if not self._db:
            raise RuntimeError(
                "The attachment needs to be assigned to a database "
                "before attached files can be listed."
            )
        if not self._db.root:
            raise RuntimeError(
                "The database needs to be saved to disk "
                "before attachment files can be listed."
            )
        self._check_path(self._db.root)

        files = []

        path = audeer.path(self._db.root, self.path)
        if os.path.isdir(path):
            files = audeer.list_file_names(
                path,
                recursive=True,
                hidden=True,
            )
            dirs = audeer.list_dir_names(
                path,
                recursive=True,
                hidden=True,
            )
            for path in files + dirs:
                if os.path.islink(path):
                    raise RuntimeError(
                        f"The path '{path}' "
                        f"included in attachment '{self._id}' "
                        "must not be a symlink."
                    )
        else:
            files = [path]

        # Remove absolute path
        files = [f.replace(f"{self._db.root}{os.path.sep}", "") for f in files]
        # Make sure we use `/` as sep
        files = [f.replace(os.path.sep, "/") for f in files]

        return files

    def _check_overlap(
        self,
        other: str,
    ):
        r"""Check if two attachment paths are nested."""
        if (
            self.path == other.path
            or self.path.startswith(other.path)
            or other.path.startswith(self.path)
        ):
            raise RuntimeError(
                f"Attachments '{self.path}' and '{other.path}' " "are nested."
            )

    def _check_path(
        self,
        root: str,
    ):
        r"""Check if path exists and is not a symlink."""
        if not os.path.exists(audeer.path(root, self.path, follow_symlink=True)):
            raise FileNotFoundError(
                f"The provided path '{self.path}' "
                f"of attachment '{self._id}' "
                "does not exist."
            )
        if os.path.islink(os.path.join(root, self.path)):
            raise RuntimeError(
                f"The provided path '{self.path}' "
                f"of attachment '{self._id}' "
                "must not be a symlink."
            )
