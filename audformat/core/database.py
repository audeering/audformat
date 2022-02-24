import datetime
import itertools
import os
import shutil
import typing

import audiofile
import oyaml as yaml
try:
    from yaml import CLoader as Loader
except ImportError:  # pragma: nocover
    from yaml import Loader
import pandas as pd

import audeer

from audformat.core import define
from audformat.core import utils
from audformat.core.column import Column
from audformat.core.common import HeaderBase, HeaderDict
from audformat.core.errors import BadIdError
from audformat.core.media import Media
from audformat.core.rater import Rater
from audformat.core.scheme import Scheme
from audformat.core.split import Split
from audformat.core.table import Table


class Database(HeaderBase):
    r"""Database object.

    A database consists of a header holding raters,
    schemes, splits, and other meta information.
    In addition it links to a number of tables
    listing files and labels.

    Args:
        name: name of database
        source: data source (e.g. link to website)
        usage: permission of usage, see :class:`audformat.define.Usage`.
            Set to ``'other'``
            if none of the other fields fit.
        expires: expiry date
        languages: list of languages.
            Will be mapped to ISO 639-3 strings
            with :func:`audformat.utils.map_language`
        description: database description
        author: database author(s)
        organization: organization(s) maintaining the database
        license: database license.
            You can use a custom license
            or pick one from :attr:`audformat.define.License`.
            In the later case,
            ``license_url`` will be automatically set
            if it is not given
        license_url: URL of database license
        meta: additional meta fields

    Raises:
        BadValueError: if an invalid ``usage`` value is passed
        ValueError: if language is unknown

    Example:
        >>> db = Database(
        ...     'mydb',
        ...     'https://www.audeering.com/',
        ...     define.Usage.COMMERCIAL,
        ...     languages=['English', 'de'],
        ... )
        >>> db
        name: mydb
        source: https://www.audeering.com/
        usage: commercial
        languages: [eng, deu]
        >>> labels = ['positive', 'neutral', 'negative']
        >>> db.schemes['emotion'] = Scheme(
        ...     labels=labels,
        ... )
        >>> db.raters['rater'] = Rater()
        >>> db.media['audio'] = Media(
        ...     define.MediaType.AUDIO,
        ...     format='wav',
        ...     sampling_rate=16000,
        ... )
        >>> db['table'] = Table(
        ...     media_id='audio',
        ... )
        >>> db['table']['column'] = Column(
        ...     scheme_id='emotion',
        ...     rater_id='rater',
        ... )
        >>> db
        name: mydb
        source: https://www.audeering.com/
        usage: commercial
        languages: [eng, deu]
        media:
          audio: {type: audio, format: wav, sampling_rate: 16000}
        raters:
          rater: {type: human}
        schemes:
          emotion:
            dtype: str
            labels: [positive, neutral, negative]
        tables:
          table:
            type: filewise
            media_id: audio
            columns:
              column: {scheme_id: emotion, rater_id: rater}

    """
    def __init__(
            self,
            name: str,
            source: str = '',
            usage: str = define.Usage.UNRESTRICTED,
            *,
            expires: datetime.date = None,
            languages: typing.Union[str, typing.Sequence[str]] = None,
            description: str = None,
            author: str = None,
            organization: str = None,
            license: typing.Union[str, define.License] = None,
            license_url: str = None,
            meta: dict = None,
    ):
        define.Usage.assert_has_attribute_value(usage)
        if (
                license_url is None
                and license in define.License.attribute_values()
        ):
            license_url = define.LICENSE_URLS[license]

        languages = [] if languages is None else audeer.to_list(languages)
        for idx in range(len(languages)):
            languages[idx] = utils.map_language(languages[idx])

        self.name = name
        r"""Name of database"""
        super().__init__(description=description, meta=meta)
        self.source = source
        r"""Database source"""
        self.usage = usage
        r"""Usage permission"""
        self.expires = expires
        r"""Expiry date"""
        self.languages = languages
        r"""List of included languages"""
        self.author = author
        r"""Author(s) of database"""
        self.organization = organization
        r"""Organization that created the database"""
        self.license = license
        r"""License of database"""
        self.license_url = license_url
        r"""URL of database license"""
        self.media = HeaderDict(value_type=Media)
        r"""Dictionary of media information"""
        self.raters = HeaderDict(value_type=Rater)
        r"""Dictionary of raters"""
        self.schemes = HeaderDict(
            value_type=Scheme,
            set_callback=self._set_scheme,
        )
        r"""Dictionary of schemes"""
        self.splits = HeaderDict(value_type=Split)
        r"""Dictionary of splits"""
        self.tables = HeaderDict(
            value_type=Table,
            set_callback=self._set_table,
        )
        r"""Dictionary of tables"""

        self._files_duration = {}
        self._name = None
        self._root = None

    @property
    def files(self) -> pd.Index:
        r"""Files referenced in the database.

        Includes files from filewise and segmented tables.

        Returns:
            files

        """
        index = utils.union(
            [table.files.drop_duplicates() for table in self.tables.values()]
        )
        # Sort alphabetical
        index, _ = index.sortlevel()
        return index

    @property
    def is_portable(
        self,
    ) -> bool:
        r"""Check if a database can be moved to another location.

        To be portable,
        media must not be referenced with an absolute path,
        and not contain ``\``,
        ``.``,
        or ``..``.
        If a database is portable
        it can be moved to another folder
        or updated by another database.

        Returns:
            ``True`` if the database is portable

        """
        if len(self.files) == 0:
            return True
        return not any(
            (
                os.path.isabs(f)
                or '\\' in f
                or f.startswith('./')
                or '/./' in f
                or f.startswith('../')
                or '/../' in f
            )
            for f in self.files
        )

    @property
    def root(self) -> typing.Optional[str]:
        r"""Database root directory.

        Returns ``None`` if database has not been stored yet.

        Returns:
            root directory

        """
        return self._root

    @property
    def segments(self) -> pd.MultiIndex:
        r"""Segments referenced in the database.

        Returns:
            segments

        """
        index = utils.union(
            [
                table.df.index
                for table in self.tables.values()
                if table.is_segmented
            ]
        )
        # Sort alphabetical
        index, _ = index.sortlevel()
        return index

    def drop_files(
            self,
            files: typing.Union[
                str,
                typing.Sequence[str],
                typing.Callable[[str], bool],
            ],
            num_workers: typing.Optional[int] = 1,
            verbose: bool = False,
    ):
        r"""Drop files from tables.

        Iterate through all tables and remove rows with a reference to
        listed or matching files.

        Args:
            files: list of files or condition function
            num_workers: number of parallel jobs.
                If ``None`` will be set to the number of processors
                on the machine multiplied by 5
            verbose: show progress bar

        """
        audeer.run_tasks(
            lambda x: x.drop_files(files, inplace=True),
            params=[([table], {}) for table in self.tables.values()],
            num_workers=num_workers,
            progress_bar=verbose,
            task_description='Drop files',
        )

    def drop_tables(
            self,
            table_ids: typing.Union[str, typing.Sequence[str]],
    ):
        r"""Drop tables by ID.

        Args:
            table_ids: table IDs to drop

        """
        if isinstance(table_ids, str):
            table_ids = [table_ids]
        for table_id in table_ids:
            self.tables.pop(table_id)

    def files_duration(
            self,
            files: typing.Union[str, typing.Sequence[str]],
            *,
            root: str = None,
    ) -> pd.Series:
        r"""Duration of files in the database.

        Use ``db.files_duration(db.files).sum()``
        to get the total duration of all files in a database.
        Or ``db.files_duration(db[table_id].files).sum()``
        to get the total duration of all files assigned to a table.

        .. note:: Durations are cached,
            i.e. changing the files on disk after calling
            this function can lead to wrong results.
            The cache is cleared when the
            database is reloaded from disk.

        Args:
            files: file names
            root: root directory under which the files are stored.
                Provide if file names are relative and
                database was not saved or loaded from disk.
                If ``None`` :attr:`audformat.Database.root` is used

        Returns:
            mapping from file to duration

        Raises:
            ValueError: if ``root`` is not set
                when using relative file names
                with a database that was not saved
                or loaded from disk

        """
        root = root or self.root

        def duration(file: str) -> pd.Timedelta:

            # expand file path
            if os.path.isabs(file):
                full_file = file
            else:
                if root is None:
                    raise ValueError(
                        f"Found relative file name "
                        f"{file}, "
                        f"but db.root is None. "
                        f"Please save database or "
                        f"provide a root folder."
                    )
                full_file = os.path.join(root, file)

            # check cache
            full_file = audeer.safe_path(full_file)
            if full_file in self._files_duration:
                return self._files_duration[full_file]

            # calculate duration and cache it
            dur = audiofile.duration(full_file)
            dur = pd.to_timedelta(dur, unit='s')
            self._files_duration[full_file] = dur

            return dur

        files = audeer.to_list(files)
        y = pd.Series(
            files,
            index=files,
            name=define.IndexField.FILE,
        ).map(duration)

        return y

    def map_files(
            self,
            func: typing.Callable[[str], str],
            num_workers: typing.Optional[int] = 1,
            verbose: bool = False,
    ):
        r"""Apply function to file names in all tables.

        Relies on :meth:`pandas.Index.map`,
        which can be slow.
        If speed is crucial,
        consider to change the index directly.
        In the following example we prefix every file with a folder:

        .. code-block:: python

            root = '/root/'
            for table in db.tables.values():
                if table.is_filewise:
                    table.df.index = root + table.df.index
                    table.df.index.name = audformat.define.IndexField.FILE
                elif len(table.df.index) > 0:
                    table.df.index.set_levels(
                        root + table.df.index.levels[0],
                        audformat.define.IndexField.FILE,
                        inplace=True,
                    )

        Args:
            func: map function
            num_workers: number of parallel jobs.
                If ``None`` will be set to the number of processors
                on the machine multiplied by 5
            verbose: show progress bar

        """
        def job(table):
            if table.is_segmented:
                table.df.index = table.df.index.map(
                    lambda x: (func(x[0]), x[1], x[2])
                )
            else:
                table.df.index = table.df.index.map(lambda x: func(x))

        audeer.run_tasks(
            job,
            params=[([table], {}) for table in self.tables.values()],
            num_workers=num_workers,
            progress_bar=verbose,
            task_description='Map files',
        )

    def pick_files(
            self,
            files: typing.Union[
                str,
                typing.Sequence[str],
                typing.Callable[[str], bool],
            ],
            num_workers: typing.Optional[int] = 1,
            verbose: bool = False,
    ):
        r"""Pick files from tables.

        Iterate through all tables and keep only rows with a reference
        to listed files or matching files.

        Args:
            files: list of files or condition function
            num_workers: number of parallel jobs.
                If ``None`` will be set to the number of processors
                on the machine multiplied by 5
            verbose: show progress bar

        """
        audeer.run_tasks(
            lambda x: x.pick_files(files, inplace=True),
            params=[([table], {}) for table in self.tables.values()],
            num_workers=num_workers,
            progress_bar=verbose,
            task_description='Pick files',
        )

    def pick_tables(
            self,
            table_ids: typing.Union[str, typing.Sequence[str]],
    ):
        r"""Pick tables by ID.

        Args:
            table_ids: table IDs to pick

        """
        if isinstance(table_ids, str):
            table_ids = [table_ids]
        drop_ids = []
        for table_id in list(self.tables):
            if table_id not in table_ids:
                drop_ids.append(table_id)
        self.drop_tables(drop_ids)

    def save(
            self,
            root: str,
            *,
            name: str = 'db',
            indent: int = 2,
            storage_format: str = define.TableStorageFormat.CSV,
            update_other_formats: bool = True,
            header_only: bool = False,
            num_workers: typing.Optional[int] = 1,
            verbose: bool = False,
    ):
        r"""Save database to disk.

        Creates a header ``<root>/<name>.yaml``
        and for every table a file ``<root>/<name>.<table-id>.[csv,pkl]``.

        Existing files will be overwritten.
        If ``update_other_formats`` is provided,
        it will overwrite all existing files in others formats as well.

        Args:
            root: root directory (possibly created)
            name: base name of files
            indent: indent size
            storage_format: storage format of tables.
                See :class:`audformat.define.TableStorageFormat`
                for available formats
            update_other_formats: if ``True`` it will not only save
                to the given ``storage_format``,
                but update all files stored in other storage formats as well
            header_only: store header only
            num_workers: number of parallel jobs.
                If ``None`` will be set to the number of processors
                on the machine multiplied by 5
            verbose: show progress bar

        """
        root = audeer.mkdir(root)

        ext = '.yaml'
        header_path = os.path.join(root, name + ext)
        with open(header_path, 'w') as fp:
            self.dump(fp, indent=indent)

        if not header_only:

            def job(table_id, table):
                table_path = os.path.join(root, name + '.' + table_id)
                table.save(
                    table_path,
                    storage_format=storage_format,
                    update_other_formats=update_other_formats,
                )

            audeer.run_tasks(
                job,
                params=[
                    ([table_id, table], {})
                    for table_id, table in self.tables.items()
                ],
                num_workers=num_workers,
                progress_bar=verbose,
                task_description='Save tables',
            )

        self._name = name
        self._root = root

    def update(
            self,
            others: typing.Union['Database', typing.Sequence['Database']],
            *,
            copy_media: bool = False,
            overwrite: bool = False,
    ) -> 'Database':
        r"""Update database with other database(s).

        In order to update a database, *license* and *usage* have to match.
        *Media*, *raters*, *schemes* and *splits* that are not part of
        the database yet are added. Other fields will be updated by
        applying the following rules:

        ============= =====================================
        **field**     **result**
        ------------- -------------------------------------
        author        'db.author, other.author'
        description   db.description
        expires       min(db.expires, other.expires)
        languages     db.languages + other.languages
        license_url   db.license_url
        meta          db.meta + other.meta
        name          db.name
        organization  'db.organization, other.organization'
        source        'db.source, other.source'
        ============= =====================================

        Args:
            others: database object(s)
            copy_media: if ``True`` it copies the media files
                associated with ``others`` to the current database root folder
            overwrite: overwrite table values where indices overlap

        Returns:
            the updated database

        Raises:
            ValueError: if database has different license or usage
            ValueError: if different media, rater, scheme or split with
                same ID is found
            ValueError: if table data cannot be combined (e.g. values in
                same position overlap)
            RuntimeError: if ``copy_media=True``,
                but one of the involved databases was not saved
                (contains files but no root folder)
            RuntimeError: if any involved database is not portable

        """

        if isinstance(others, Database):
            others = [others]

        def assert_equal(
                other: Database,
                field: str,
        ):
            r"""Assert fields are equal."""
            value1 = self.__dict__[field]
            value2 = other.__dict__[field]
            if value1 != value2:
                raise ValueError(
                    "Cannot update database, "
                    "found different value for "
                    f"'db.{field}':\n"
                    f"{value1}\n"
                    "!=\n"
                    f"{value2}"
                )

        def join_dict(
                field: str,
                ds: typing.Sequence[dict],
        ):
            r"""Join list of dictionaries.

            Raise error if dictionaries have same key with different values.

            """
            d = ds[0].copy()
            for d_other in ds[1:]:
                for key, value in d_other.items():
                    if key in d:
                        if d[key] != value:
                            raise ValueError(
                                "Cannot update database, "
                                "found different value for "
                                f"'db.{field}['{key}']':\n"
                                f"{d[key]}\n"
                                "!=\n"
                                f"{d_other[key]}"
                            )
                    else:
                        d[key] = value
            return d

        def join_field(
                other: Database,
                field: str,
                op: typing.Callable,
        ):
            r"""Join two fields of db header."""
            value1 = self.__dict__[field]
            value2 = other.__dict__[field]
            if value1 != value2:
                if value1 and value2:
                    self.__dict__[field] = op([value1, value2])
                elif value1:
                    self.__dict__[field] = value1
                elif value2:
                    self.__dict__[field] = value2

        # assert equal fields
        for other in others:
            assert_equal(other, 'license')
            assert_equal(other, 'usage')

        # can only join databases with relatvie paths
        for database in [self] + others:
            if not database.is_portable:
                raise RuntimeError(
                    f"You can only update with databases that are portable. "
                    f"The database '{database.name}' is not portable."
                )

        # join fields
        for other in others:
            join_field(other, 'author', ', '.join)
            join_field(other, 'expires', min)
            join_field(other, 'languages', itertools.chain.from_iterable)
            # remove duplicates whilst preserving order
            self.languages = list(dict.fromkeys(self.languages))
            join_field(other, 'media', lambda x: join_dict('media', x))
            join_field(other, 'meta', lambda x: join_dict('meta', x))
            join_field(other, 'organization', ', '.join)
            join_field(other, 'schemes', lambda x: join_dict('schemes', x))
            join_field(other, 'source', ', '.join)
            join_field(other, 'splits', lambda x: join_dict('splits', x))
            join_field(other, 'raters', lambda x: join_dict('raters', x))

        # join tables
        for other in others:
            for table_id, table in other.tables.items():
                if table_id in self.tables:
                    self[table_id].update(table, overwrite=overwrite)
                else:
                    self[table_id] = table.copy()

        # copy media files

        if copy_media:
            if self.root is None:
                raise RuntimeError(
                    f"You can only update a saved database. "
                    f"'{self.name}' was not saved yet."
                )
            for other in others:
                if len(other.files) > 0 and other.root is None:
                    raise RuntimeError(
                        f"You can only update with saved databases. "
                        f"The database '{other.name}' was not saved yet."
                    )
                for file in other.files:
                    src_file = os.path.join(other.root, file)
                    dst_file = os.path.join(self.root, file)
                    dst_dir = os.path.dirname(dst_file)
                    audeer.mkdir(dst_dir)
                    shutil.copy(src_file, dst_file)

        return self

    def __contains__(
            self,
            table_id: str,
    ) -> bool:
        r"""Check if table exists.

        Args:
            table_id: table identifier

        """
        return table_id in self.tables

    def __getitem__(
            self,
            table_id: str,
    ) -> Table:
        r"""Get table from database.

        Args:
            table_id: table identifier

        """
        return self.tables[table_id]

    def __eq__(
            self,
            other: 'Database',
    ) -> bool:
        if self.dump() != other.dump():
            return False
        for table_id in self.tables:
            if self[table_id] != other[table_id]:
                return False
        return True

    def __setitem__(
            self,
            table_id: str,
            table: Table,
    ) -> Table:
        r"""Add table to database.

        Args:
            table_id: table identifier
            table: the table

        Raises:
            BadIdError: if table has a ``split_id`` or ``media_id``,
                which is not specified in the underlying database

        """
        self.tables[table_id] = table
        return table

    @staticmethod
    def load(
            root: str,
            *,
            name: str = 'db',
            load_data: bool = False,
            num_workers: typing.Optional[int] = 1,
            verbose: bool = False,
    ) -> 'Database':
        r"""Load database from disk.

        Expects a header ``<root>/<name>.yaml``
        and for every table a file ``<root>/<name>.<table-id>.[csv|pkl]``
        Media files should be located under ``root``.

        Args:
            root: root directory
            name: base name of header and table files
            load_data: if ``False``,
                :class:`audformat.Table`
                data is only loaded on demand,
                e.g. when
                :meth:`audformat.Table.get`
                is called for the first time.
                Set to ``True`` to load all
                :class:`audformat.Table`
                data immediately
            num_workers: number of parallel jobs.
                If ``None`` will be set to the number of processors
                on the machine multiplied by 5
            verbose: show progress bar

        Returns:
            database object

        """
        ext = '.yaml'
        root = audeer.safe_path(root)
        path = os.path.join(root, name + ext)

        if not os.path.exists(path):
            raise FileNotFoundError(path)

        with open(path, 'r') as fp:

            header = yaml.load(fp, Loader=Loader)
            db = Database.load_header_from_yaml(header)

            if 'tables' in header and header['tables']:

                if load_data:

                    def job(table_id):
                        table = db[table_id]
                        path = os.path.join(root, name + '.' + table_id)
                        table.load(path)

                    # load all tables into memory
                    audeer.run_tasks(
                        job,
                        params=[
                            ([table_id], {}) for table_id in header['tables']
                        ],
                        num_workers=num_workers,
                        progress_bar=verbose,
                        task_description='Load tables',
                    )

                else:

                    # signal that table data is not loaded
                    # by setting the DataFrame to None
                    for table_id in header['tables']:
                        db[table_id]._df = None

        db._name = name
        db._root = root

        return db

    @staticmethod
    def load_header_from_yaml(header: dict) -> 'Database':
        r"""Load database header from YAML.

        Args:
            header: YAML header definition

        Returns:
            database object

        """
        # for backward compatibility
        if len(header) == 1:  # pragma: no cover
            id = next(iter(header))
            header = header[id]
            header['name'] = id

        db = Database(
            name=header['name'],
            source=header['source'],
            usage=header['usage'])
        db.from_dict(header, ignore_keys=['media', 'raters', 'schemes',
                                          'tables', 'splits'])

        if 'media' in header and header['media']:
            for media_id, media_d in header['media'].items():
                media = Media()
                media.from_dict(media_d)
                db.media[media_id] = media

        if 'raters' in header and header['raters']:
            for rater_id, rater_d in header['raters'].items():
                rater = Rater()
                rater.from_dict(rater_d)
                db.raters[rater_id] = rater

        if 'schemes' in header and header['schemes']:
            for scheme_id, scheme_d in header['schemes'].items():
                scheme = Scheme()
                scheme.from_dict(scheme_d)
                db.schemes[scheme_id] = scheme

        if 'splits' in header and header['splits']:
            for split_id, split_d in header['splits'].items():
                split = Split()
                split.from_dict(split_d)
                db.splits[split_id] = split

        if 'tables' in header and header['tables']:
            for table_id, table_d in header['tables'].items():
                table = Table()
                table.from_dict(table_d, ignore_keys=['is_segmented',
                                                      'columns'])
                if 'columns' in table_d and table_d['columns']:
                    tmp_callback = table.columns.set_callback
                    table.columns.set_callback = None
                    for column_id, column_d in \
                            table_d['columns'].items():
                        column = Column()
                        column.from_dict(
                            column_d, ignore_keys=['has_confidence']
                        )
                        column._id = column_id
                        column._table = table
                        table.columns[column_id] = column

                        # for backward compatibility we insert
                        # confidences as a regular column
                        if 'has_confidence' in column_d:  # pragma: no cover
                            column = Column()
                            column._id = '@' + column_id
                            column._table = table
                            table.columns['@' + column_id] = column

                    table.columns.set_callback = tmp_callback
                db[table_id] = table

        return db

    def _set_scheme(
            self,
            scheme_id: str,
            scheme: Scheme,
    ) -> Scheme:
        scheme._db = self
        scheme._id = scheme_id
        return scheme

    def _set_table(
            self,
            table_id: str,
            table: Table,
    ) -> Table:
        if table.split_id is not None and table.split_id not in self.splits:
            raise BadIdError('split', table.split_id, self.splits)
        if table.media_id is not None and table.media_id not in self.media:
            raise BadIdError('media', table.media_id, self.media)
        table._db = self
        table._id = table_id
        return table
