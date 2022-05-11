import os

import pandas as pd

import audeer

from audformat.core.common import HeaderBase
import audformat.core.define as define


class Misc(HeaderBase):
    r"""Miscellaneous table.

    Args:
        df: table
        description: table description
        meta: additional meta fields

    """
    def __init__(
            self,
            df: pd.DataFrame,
            *,
            description: str = None,
            meta: dict = None,
    ):
        super().__init__(description=description, meta=meta)

        if df is not None:
            if isinstance(df.index, pd.MultiIndex):
                index = list(df.index.names)
            else:
                index = [df.index.name]
            self.levels = index
            self.columns = list(df.columns)
        else:
            self.columns = None
            self.levels = None

        self._db = None
        self._df = df
        self._id = None

    def __eq__(
            self,
            other: 'Misc',
    ) -> bool:
        if self.dump() != other.dump():
            return False
        return self.df.equals(other.df)

    def __len__(self) -> int:
        return len(self.df)

    @property
    def db(self):
        r"""Database object.

        Returns:
            database object or ``None`` if not assigned yet

        """
        return self._db

    @property
    def df(self) -> pd.DataFrame:
        r"""Table data.

        Returns:
            data

        """
        if self._df is None:
            # if database was loaded with 'load_data=False'
            # we have to load the table data now
            path = audeer.path(
                self.db.root,
                f'{self.db._name}.{define.MISC_FILE_PREFIX}.{self._id}',
            )
            self.load(path)
        return self._df

    @property
    def index(self) -> pd.Index:
        r"""Table index.

        Returns:
            index

        """
        return self.df.index

    def copy(self) -> 'Misc':
        r"""Copy table.

        Return:
            new ``Misc`` object

        """
        misc = Misc(
            self.df.copy(),
            description=self.description,
            meta=self.meta.copy(),
        )
        misc._db = self.db
        return misc

    def get(
            self,
            *,
            copy: bool = True,
    ) -> pd.DataFrame:
        r"""Get table data.

        Args:
            copy: return a copy of the labels

        Returns:
            data

        """
        return self.df.copy() if copy else self.df

    def load(
            self,
            path: str,
    ):
        r"""Load table data from disk.

        Tables can be stored as PKL and/or CSV files to disk.
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
                    f"The table CSV file '{csv_file}' is newer "
                    f"than the table PKL file '{pkl_file}'. "
                    "If you want to load from the CSV file, "
                    "please delete the PKL file. "
                    "If you want to load from the PKL file, "
                    "please delete the CSV file."
                )
            pickled = True

        if pickled:
            df = pd.read_pickle(pkl_file)
        else:
            df = pd.read_csv(
                csv_file,
                index_col=self.levels,
                float_precision='round_trip',
            )

        self._df = df

    def save(
            self,
            path: str,
            *,
            storage_format: str = define.TableStorageFormat.CSV,
            update_other_formats: bool = True,
    ):
        r"""Save table data to disk.

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

    def _save_csv(self, path: str):
        # Load table before opening CSV file
        # to avoid creating a CSV file
        # that is newer than the PKL file
        obj = self.df
        with open(path, 'w') as fp:
            obj.to_csv(fp, encoding='utf-8')

    def _save_pickled(self, path: str):
        self.df.to_pickle(
            path,
            protocol=4,  # supported by Python >= 3.4
        )
