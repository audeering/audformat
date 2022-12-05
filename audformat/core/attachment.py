import os

import audeer

from audformat.core.common import (
    HeaderBase,
    is_relative_path,
)


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

    Example:
        >>> Attachment('file.txt', description='Attached file')
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
                f"Attachments '{self.path}' and '{other.path}' "
                "are nested."
            )

    def _check_path(
            self,
            root: str,
    ):
        r"""Check if path exists and is not a symlink."""
        if not os.path.exists(audeer.path(root, self.path)):
            raise FileNotFoundError(
                f"The provided path '{self.path}' "
                f"of attachment '{self._id}' "
                "does not exist."
            )
        if os.path.islink(os.path.join(root, self.path)):
            raise RuntimeError(
                f"The provided path '{self.path}' "
                f"of attachment '{self._id}' "
                "is not allowed to be a symlink."
            )
