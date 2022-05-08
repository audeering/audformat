import os
import typing

import pandas as pd

import audeer

from audformat.core.common import HeaderBase
import audformat.core.define as define


class Misc(HeaderBase):
    r"""Miscellaneous object.

    Args:
        obj: object
        description: database description
        meta: additional meta fields

    """
    def __init__(
            self,
            obj: typing.Union[pd.Series, pd.DataFrame],
            *,
            description: str = None,
            meta: dict = None,
    ):
        super().__init__(description=description, meta=meta)

        self.columns = None
        self.index = None
        self.name = None
        self.type = None

        self._db = None
        self._id = None
        self._obj = obj

        self._update_fields()

    @property
    def db(self):
        r"""Database object.

        Returns:
            database object or ``None`` if not assigned yet

        """
        return self._db

    @property
    def obj(self) -> typing.Union[pd.Series, pd.DataFrame]:
        r"""Object data.

        Returns:
            object

        """
        if self._obj is None:
            # if database was loaded with 'load_data=False'
            # we have to load the object data now
            path = audeer.path(
                self.db.root,
                f'{self.db._name}.{define.MISC_FILE_PREFIX}.{self._id}',
            )
            self.load(path)
        return self._obj

    def load(
            self,
            path: str,
    ):
        r"""Load object data from disk.

        Objects can be stored as PKL and/or CSV files to disk.
        If both files are present
        it will load the PKL file
        as long as its modification date is newer,
        otherwise it will raise an error
        and ask to delete one of the files.

        Args:
            path: file path without extension

        Raises:
            RuntimeError: if table file(s) are missing
            RuntimeError: if CSV file is newer than PKL file

        """
        path = audeer.path(path)
        pkl_file = f'{path}.{define.TableStorageFormat.PICKLE}'
        csv_file = f'{path}.{define.TableStorageFormat.CSV}'

        # Load from PKL if file exists and is newer then CSV file.
        # If both are written by Database.save() this is the case
        # as it stores first the PKL file
        pickled = False
        if os.path.exists(pkl_file):
            if (
                    os.path.exists(csv_file)
                    and os.path.getmtime(csv_file) > os.path.getmtime(pkl_file)
            ):
                raise RuntimeError(
                    f"The object CSV file '{csv_file}' is newer "
                    f"than the object PKL file '{pkl_file}'. "
                    "If you want to load from the CSV file, "
                    "please delete the PKL file. "
                    "If you want to load from the PKL file, "
                    "please delete the CSV file."
                )
            pickled = True

        if pickled:
            obj = pd.read_pickle(pkl_file)
        else:
            index_col = self.index
            obj = pd.read_csv(
                csv_file,
                index_col=index_col,
                float_precision='round_trip',
            )
            if self.type == 'series':
                obj = obj[self.name]

        self._obj = obj

    def save(
            self,
            path: str,
            *,
            storage_format: str = define.TableStorageFormat.CSV,
            update_other_formats: bool = True,
    ):
        r"""Save object data to disk.

        Existing files will be overwritten.

        Args:
            path: file path without extension
            storage_format: storage format of table.
                See :class:`audformat.define.TableStorageFormat`
                for available formats
            update_other_formats: if ``True`` it will not only save
                to the given ``storage_format``,
                but update all files stored in other storage formats as well

        """
        path = audeer.path(path)
        define.TableStorageFormat.assert_has_attribute_value(storage_format)

        pickle_file = path + f'.{define.TableStorageFormat.PICKLE}'
        csv_file = path + f'.{define.TableStorageFormat.CSV}'

        # Make sure the CSV file is always written first
        # as it is expected to be older by load()
        if storage_format == define.TableStorageFormat.PICKLE:
            if update_other_formats and os.path.exists(csv_file):
                self._save_csv(csv_file)
            self._save_pickled(pickle_file)

        if storage_format == define.TableStorageFormat.CSV:
            self._save_csv(csv_file)
            if update_other_formats and os.path.exists(pickle_file):
                self._save_pickled(pickle_file)

    def to_dict(self) -> dict:
        r"""Serialize object to dictionary.

        Returns:
            dictionary with attributes

        """
        self._update_fields()
        return super().to_dict()

    def _save_csv(self, path: str):
        # Load object before opening CSV file
        # to avoid creating a CSV file
        # that is newer than the PKL file
        obj = self.obj
        with open(path, 'w') as fp:
            obj.to_csv(fp, encoding='utf-8')

    def _save_pickled(self, path: str):
        self.obj.to_pickle(
            path,
            protocol=4,  # supported by Python >= 3.4
        )

    def _update_fields(self):
        if self._obj is not None:
            if isinstance(self.obj, pd.Series):
                if isinstance(self.obj.index, pd.MultiIndex):
                    index = list(self.obj.index.names)
                else:
                    index = self.obj.index.name
                self.type = 'series'
                self.index = index
                self.name = self.obj.name
            elif isinstance(self.obj, pd.DataFrame):
                if isinstance(self.obj.index, pd.MultiIndex):
                    index = list(self.obj.index.names)
                else:
                    index = [self.obj.index.name]
                self.type = 'frame'
                self.index = index
                self.columns = list(self.obj.columns)
