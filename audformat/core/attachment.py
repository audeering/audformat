import os

import audeer

from audformat.core.common import HeaderBase


class Attachment(HeaderBase):
    r"""Database attachment.

    Adds a file as attachment to a database.

    Args:
        path: relative path to file
        description: attachment description
        meta: additional meta fields

    Raises:
        ValueError: if ``path`` is absolute

    Example:
        >>> import audeer
        >>> _ = audeer.touch('file.txt')
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

        if os.path.isabs(path):
            raise ValueError(
                f"The provided path '{path}' needs to be relative."
            )

        self._db = None
        self._id = None

        self.path = path
        r"""Attachment path"""

    def _check_path(
            self,
            root: str,
    ):
        if root is None:
            return
        if not os.path.exists(audeer.path(root, self.path)):
            raise FileNotFoundError(
                f"The provided path '{self.path}' "
                f"of attachment '{self._id}' "
                "does not exist."
            )
        if not os.path.isfile(audeer.path(root, self.path)):
            raise FileNotFoundError(
                f"The provided path '{self.path}' "
                f"of attachment '{self._id}' "
                "is not a file."
            )