import datetime
import itertools
import os
import shutil
import typing

import oyaml as yaml


try:
    from yaml import CLoader as Loader
except ImportError:  # pragma: nocover
    from yaml import Loader
import pandas as pd

import audeer
import audiofile

from audformat.core import define
from audformat.core import utils
from audformat.core.attachment import Attachment
from audformat.core.column import Column
from audformat.core.common import HeaderBase
from audformat.core.common import HeaderDict
from audformat.core.common import is_relative_path
from audformat.core.errors import BadIdError
from audformat.core.errors import BadKeyError
from audformat.core.errors import TableExistsError
from audformat.core.index import filewise_index
from audformat.core.index import is_filewise_index
from audformat.core.index import is_segmented_index
from audformat.core.media import Media
from audformat.core.rater import Rater
from audformat.core.scheme import Scheme
from audformat.core.split import Split
from audformat.core.table import MiscTable
from audformat.core.table import Table


class Database(HeaderBase):
    r"""Database object.

    A database consists of a header holding raters,
    schemes, splits, and other meta information.
    In addition, it links to a number of tables
    listing files and labels.

    For a start
    see how to :ref:`create a database <create-a-database>`
    and inspect the :ref:`example of the emodb database <emodb-example>`.

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

    Examples:
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
        >>> db.schemes['emotion'] = Scheme(labels=labels)
        >>> db.schemes['match'] = Scheme(dtype='bool')
        >>> db.raters['rater'] = Rater()
        >>> db.media['audio'] = Media(
        ...     define.MediaType.AUDIO,
        ...     format='wav',
        ...     sampling_rate=16000,
        ... )
        >>> db['table'] = Table(media_id='audio')
        >>> db['table']['column'] = Column(
        ...     scheme_id='emotion',
        ...     rater_id='rater',
        ... )
        >>> index = pd.Index([], dtype='string', name='idx')
        >>> db['misc-table'] = MiscTable(index)
        >>> db['misc-table']['column'] = Column(scheme_id='match')
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
          match: {dtype: bool}
        tables:
          table:
            type: filewise
            media_id: audio
            columns:
              column: {scheme_id: emotion, rater_id: rater}
        misc_tables:
          misc-table:
            levels: {idx: str}
            columns:
              column: {scheme_id: match}
        >>> list(db)
        ['misc-table', 'table']

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
        define.Usage._assert_has_attribute_value(usage)
        if (
                license_url is None
                and license in define.License._attribute_values()
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
        r"""Usage permission.

        Possible return values are given by
        :class:`audformat.define.Usage`.

        """
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
        self.attachments = HeaderDict(
            value_type=Attachment,
            set_callback=self._set_attachment,
        )
        r"""Dictionary of attachments.

        Raises:
            RuntimeError: if the path of a newly assigned attachment
                overlaps with the path of an existing attachment

        """
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
        r"""Dictionary of audformat tables"""
        self.misc_tables = HeaderDict(
            value_type=MiscTable,
            set_callback=self._set_table,
        )
        r"""Dictionary of miscellaneous tables"""

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
        r"""Check if database can be moved to another location.

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
        return all(is_relative_path(f) for f in self.files)

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
        r"""Drop (miscellaneous) tables by ID.

        Args:
            table_ids: table IDs to drop

        Raises:
            audformat.errors.BadIdError: if a table with provided ID
                does not exist in the database
            RuntimeError: if a misc table
                that is used in a scheme
                would be removed

        """
        table_ids = audeer.to_list(table_ids)
        for table_id in table_ids:
            if table_id in self.tables:
                self.tables.pop(table_id)
            elif table_id in self.misc_tables:
                schemes = [
                    scheme._id for scheme in self.schemes.values()
                    if scheme.labels == table_id
                ]
                if len(schemes) > 0:
                    schemes = [f"'{scheme}'" for scheme in schemes]
                    raise RuntimeError(
                        f"Misc table '{table_id}' is used "
                        "as scheme(s): "
                        f"{', '.join(schemes)}, "
                        "and cannot be removed."
                    )
                self.misc_tables.pop(table_id)
            else:
                available_tables = {**self.tables, **self.misc_tables}
                raise BadIdError('table', table_id, available_tables)

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
            full_file = audeer.path(full_file)
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

    def get(
            self,
            schemes: typing.Union[str, typing.Sequence],
            *,
            tables: typing.Union[str, typing.Sequence] = None,
            splits: typing.Union[str, typing.Sequence] = None,
            aggregate_function: typing.Callable[pd.Series, typing.Any] = None,
            modify_function: typing.Callable[
                [pd.Series, 'Database', str, str],
                pd.Series
            ] = None,
    ) -> pd.DataFrame:
        r"""Get labels by scheme(s).

        Return all labels
        that match the requested schemes.
        A scheme is defined more broadly
        and does not only match
        schemes of the database,
        but also columns with the same name
        or labels of a scheme with the requested name.

        If at least one returned label belongs to a segmented table,
        the returned data frame has a segmented index.

        A custom ``modify_function`` can be provided
        that modifies how the labels are stored.
        The first argument of the function
        is a series ``y``
        with the labels
        for each table and column
        that matches the requested schemes.
        In addition,
        a database object ``db``,
        ``table_id``,
        and ``column_id``
        are its arguments.
        For example,
        to handle a database with stereo files,
        that has the labels for the different channels
        stored in tables with the IDs
        ``'channel0'``
        and ``'channel1'``
        one can provide an ``modify_function``
        that adds the table name to the scheme name::

            def add_channel_name(y, db, table_id, column_id):
                y.name = f'{y.name}-{table_id}'
                return y

        Alternatively,
        a user might select only labels for the first channel::

            def select_channel0(y, db, table_id, column_id):
                if table_id != 'channel0':
                    if audformat.is_filewise_index(y.index):
                        index = audformat.filewise_index()
                    else:
                        index = audformat.segmented_index()
                    y = pd.Series(index=index, name=y.name, dtype=y.dtype)
                return y

        Args:
            schemes: scheme or sequence of scheme
            tables: limit returned samples to selected tables
            splits: limit returned samples to selected splits
            aggregate_function: function to aggregate overlapping values.
                The function gets a :class:`pandas.Series`
                with overlapping values
                as input.
                E.g. set to
                ``lambda y: y.mean()``
                to average the values
                or to
                ``tuple``
                to return them as a tuple
            modify_function: callable to handle collection
                of scheme for certain tables and columns,
                see discussion above

        Returns:
            data frame with labels

        """

        def append_series(
                ys,
                y,
                table_id,
                column_id,
                modify_function,
        ):
            if y is not None:
                if modify_function is not None:
                    y = modify_function(y, self, table_id, column_id)
                ys.append(y)

        def scheme_in_column(scheme_id, column, column_id):
            # Check if scheme_id
            # is attached to a column,
            # or identical with the column name
            return (
                scheme_id == column_id
                or (
                    column.scheme_id is not None
                    and scheme_id == column.scheme_id
                )
            )

        requested_schemes = audeer.to_list(schemes)

        ys = []
        for requested_scheme in requested_schemes:

            # --- Check if requested scheme is stored as label in other schemes
            scheme_mappings = []
            for scheme_id, scheme in self.schemes.items():

                # Labels stored as misc table
                if scheme.uses_table and scheme_id in self.misc_tables:
                    for column_id, column in self[scheme_id].columns.items():
                        if scheme_in_column(
                                requested_scheme,
                                column,
                                column_id,
                        ):
                            scheme_mappings.append((scheme_id, column_id))

                # Labels stored in scheme
                elif isinstance(scheme.labels, dict):
                    labels = pd.DataFrame.from_dict(
                        scheme.labels,
                        orient='index',
                    )
                    if requested_scheme in labels:
                        scheme_mappings.append((scheme_id, requested_scheme))

            # --- Get data for requested schemes
            ys_requested_scheme = []
            for table_id, table in self.tables.items():

                for column_id, column in table.columns.items():
                    y = None
                    # Scheme directly stored in column
                    if scheme_in_column(requested_scheme, column, column_id):
                        y = self[table_id][column_id].get()
                        append_series(
                            ys_requested_scheme,
                            y,
                            table_id,
                            column_id,
                            modify_function,
                        )
                    # Get series based on label of scheme
                    else:
                        for (scheme_id, mapping) in scheme_mappings:
                            if scheme_in_column(scheme_id, column, column_id):
                                y = self[table_id][column_id].get(map=mapping)
                                append_series(
                                    ys_requested_scheme,
                                    y,
                                    table_id,
                                    column_id,
                                    modify_function,
                                )

            # Ensure we have a common dtype for requested scheme
            categorical_dtypes = [
                y.dtype for y in ys_requested_scheme
                if isinstance(y.dtype, pd.CategoricalDtype)
            ]
            dtypes_of_categories = [
                dtype.categories.dtype
                for dtype in categorical_dtypes
            ]
            if len(categorical_dtypes) > 0:
                if len(set(dtypes_of_categories)) > 1:
                    raise ValueError(
                        'All categorical data must have the same dtype.'
                    )
                dtype = dtypes_of_categories[0]
                # Convert everything to categorical data
                for n, y in enumerate(ys_requested_scheme):
                    if not isinstance(y.dtype, pd.CategoricalDtype):
                        ys_requested_scheme[n] = y.astype(
                            pd.CategoricalDtype(set(y.array.astype(dtype)))
                        )
                # Find union of categorical data
                data = [y.array for y in ys_requested_scheme]
                data = pd.api.types.union_categoricals(data)
                ys_requested_scheme = [
                    y.astype(data.dtype)
                    for y in ys_requested_scheme
                ]

            ys += ys_requested_scheme

        index = utils.union([y.index for y in ys])
        obj = utils.concat(
            ys,
            aggregate_function=aggregate_function,
        ).loc[index]

        if len(obj) == 0:
            obj = pd.DataFrame()

        if isinstance(obj, pd.Series):
            obj = obj.to_frame()

        # Start with column names matching requested schemes
        matching_columns = [
            column for column in requested_schemes
            if column in obj.columns
        ]
        additional_columns = [
            column for column in obj.columns
            if column not in requested_schemes
        ]
        obj = obj[matching_columns + additional_columns]

        # Limit returned samples by selected tables and splits
        indices = []
        if tables is not None or splits is not None:

            if tables is None:
                tables = list(self.tables)
            else:
                tables = audeer.to_list(tables)
            if splits is not None:
                splits = audeer.to_list(splits)

            for table_id, table in self.tables.items():
                if splits is None:
                    if table_id in tables:
                        indices.append(self[table_id].index)
                else:
                    if table_id in tables and table.split_id in splits:
                        indices.append(self[table_id].index)

            index = utils.union(indices)

            # Avoid matching segments in obj
            # by a filewise index
            if is_segmented_index(obj.index):
                return_filewise_index = is_filewise_index(index)
                index = utils.to_segmented_index(index)
                obj = obj.loc[index]
                if return_filewise_index:
                    index = filewise_index(index.get_level_values('file'))
                    obj = obj.set_index(index)
            else:
                obj = obj.loc[index]

        return obj

    def map_files(
            self,
            func: typing.Callable[[str], str],
            num_workers: typing.Optional[int] = 1,
            verbose: bool = False,
    ):
        r"""Apply function to file names in all tables.

        If speed is crucial,
        see :func:`audformat.utils.map_file_path`
        for further hints how to optimize your code.

        Args:
            func: map function
            num_workers: number of parallel jobs.
                If ``None`` will be set to the number of processors
                on the machine multiplied by 5
            verbose: show progress bar

        """
        def job(table):
            table.map_files(func)

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
        r"""Pick (miscellaneous) tables by ID.

        Args:
            table_ids: table IDs to pick

        Raises:
            audformat.errors.BadIdError: if a table with provided ID
                does not exist in the database
            RuntimeError: if a misc table
                that is used in a scheme
                would be removed

        """
        table_ids = audeer.to_list(table_ids)
        available_tables = {**self.tables, **self.misc_tables}
        for table_id in table_ids:
            if table_id not in available_tables:
                raise BadIdError('table', table_id, available_tables)
        drop_ids = [t for t in list(self) if t not in table_ids]
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

            # Store (misc) tables
            def job(obj_id, obj):
                path = audeer.path(root, f'{name}.{obj_id}')
                obj.save(
                    path,
                    storage_format=storage_format,
                    update_other_formats=update_other_formats,
                )

            objs = {**self.tables, **self.misc_tables}
            audeer.run_tasks(
                job,
                params=[
                    ([obj_id, obj], {})
                    for obj_id, obj in objs.items()
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
            copy_attachments: bool = False,
            copy_media: bool = False,
            overwrite: bool = False,
    ) -> 'Database':
        r"""Update database with other database(s).

        In order to :ref:`update a database <update-a-database>`,
        *license* and *usage* have to match.
        Labels and values of *schemes*
        with the same ID are combined.
        *Media*, *raters*, *schemes*, *splits*,
        and *attachments*
        that are not part of
        the database yet are added.
        Other fields will be updated by
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
            copy_attachments: if ``True`` it copies the attachment files
                associated with ``others`` to the current database root folder
            copy_media: if ``True`` it copies the media files
                associated with ``others`` to the current database root folder
            overwrite: overwrite table values where indices overlap

        Returns:
            the updated database

        Raises:
            ValueError: if database has different license or usage
            ValueError: if different media, rater, scheme, split, or attachment
                with same ID is found
            ValueError: if schemes cannot be combined,
                e.g. labels have different dtype
            ValueError: if tables cannot be combined
                (e.g. values in same position overlap or
                level and dtypes of table indices do not match)
            RuntimeError: if ``copy_media``
                or ``copy_attachments``
                is ``True``,
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

        # can only join databases with relative paths
        for database in [self] + others:
            if not database.is_portable:
                raise RuntimeError(
                    f"You can only update with databases that are portable. "
                    f"The database '{database.name}' is not portable."
                )

        # join schemes with labels
        for other in others:
            for scheme_id in other.schemes:
                if scheme_id in self.schemes:
                    other_scheme = other.schemes[scheme_id]
                    self_scheme = self.schemes[scheme_id]

                    # join labels from misc tables
                    # by combining the index of all tables,
                    # column values will be updated
                    # later when the tables are joined
                    if (
                        other_scheme.uses_table
                        or self_scheme.uses_table
                    ):

                        if (
                            other_scheme.uses_table
                            != self_scheme.uses_table
                        ):
                            raise ValueError(
                                f"Cannot join scheme "
                                f"'{scheme_id}' "
                                f"when one is using a misc table "
                                f"and the other is not."
                            )

                        if other_scheme.labels != self_scheme.labels:
                            raise ValueError(
                                f"Cannot join scheme "
                                f"'{scheme_id}' "
                                f"when using misc tables "
                                f"with different IDs: "
                                f"'{self_scheme.labels}' "
                                f"!= "
                                f"'{other_scheme.labels}'."
                            )

                        other_table = other[other_scheme.labels]
                        self_table = self[self_scheme.labels]
                        index = utils.union(
                            [
                                other_table.index,
                                self_table.index,
                            ],
                        )
                        other_table.extend_index(index, inplace=True)
                        self_table.extend_index(index, inplace=True)
                        # ensure same index order in both tables
                        other_table._df = other_table.df.reindex(index)
                        self_table._df = self_table.df.reindex(index)

                    # join other labels
                    elif (
                        other_scheme.labels is not None
                        and self_scheme.labels is not None
                    ):
                        utils.join_schemes([self, other], scheme_id)

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
            join_field(
                other,
                'attachments',
                lambda x: join_dict('attachments', x),
            )

        # join tables
        for other in others:
            for table_id in list(other.misc_tables) + list(other.tables):
                table = other[table_id]
                if table_id in self.tables:
                    self[table_id].update(table, overwrite=overwrite)
                else:
                    self[table_id] = table.copy()

        if copy_attachments or copy_media:
            if self.root is None:
                raise RuntimeError(
                    f"You can only update a saved database. "
                    f"'{self.name}' was not saved yet."
                )

        # copy attachment files

        if copy_attachments:
            for other in others:
                if other.root is None:
                    raise RuntimeError(
                        f"You can only update with saved databases. "
                        f"The database '{other.name}' was not saved yet."
                    )
                for attachment_id in other.attachments:
                    for file in other.attachments[attachment_id].files:
                        src_file = os.path.join(other.root, file)
                        dst_file = os.path.join(self.root, file)
                        dst_dir = os.path.dirname(dst_file)
                        audeer.mkdir(dst_dir)
                        shutil.copy(src_file, dst_file)

        # copy media files

        if copy_media:
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
        r"""Check if (miscellaneous) table exists.

        Args:
            table_id: table identifier

        """
        return table_id in self.tables or table_id in self.misc_tables

    def __getitem__(
            self,
            table_id: str,
    ) -> typing.Union[MiscTable, Table]:
        r"""Get (miscellaneous) table from database.

        Args:
            table_id: table identifier

        Raises:
            BadKeyError: if table does not exist

        """
        if table_id in self.tables:
            return self.tables[table_id]
        elif table_id in self.misc_tables:
            return self.misc_tables[table_id]

        raise BadKeyError(
            table_id,
            list(self.tables) + list(self.misc_tables),
        )

    def __eq__(
            self,
            other: 'Database',
    ) -> bool:
        r"""Comparison if database equals another database."""
        if self.dump() != other.dump():
            return False
        for table_id in list(self.tables) + list(self.misc_tables):
            if self[table_id] != other[table_id]:
                return False
        return True

    def __iter__(
            self,
    ) -> typing.Union[MiscTable, Table]:
        r"""Iterate over (miscellaneous) tables of database."""
        yield from sorted(list(self.tables) + list(self.misc_tables))

    def __setitem__(
            self,
            table_id: str,
            table: typing.Union[MiscTable, Table],
    ) -> typing.Union[MiscTable, Table]:
        r"""Add table to database.

        Args:
            table_id: table identifier
            table: the table

        Raises:
            BadIdError: if table has a ``split_id`` or ``media_id``,
                which is not specified in the underlying database
            TableExistsError: if setting a miscellaneous table
                when a filewise or segmented table with the same ID exists
                (or vice versa)

        """
        if isinstance(table, MiscTable):
            self.misc_tables[table_id] = table
        else:
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

        Raises:
            FileNotFoundError: if the database header file cannot be found
                under ``root``
            RuntimeError: if a CSV table file is newer
                than the corresponding PKL file

        """
        ext = '.yaml'
        root = audeer.path(root)
        path = os.path.join(root, name + ext)

        if not os.path.exists(path):
            raise FileNotFoundError(path)

        with open(path, 'r') as fp:

            header = yaml.load(fp, Loader=Loader)
            db = Database.load_header_from_yaml(header)

            params = []
            table_ids = []

            if 'tables' in header and header['tables']:
                for table_id in header['tables']:
                    table_ids.append(table_id)

            if 'misc_tables' in header and header['misc_tables']:
                for table_id in header['misc_tables']:
                    table_ids.append(table_id)

            for table_id in table_ids:
                table = db[table_id]
                if load_data:
                    table_path = audeer.path(root, name + '.' + table_id)
                    params.append(([table, table_path], {}))
                else:
                    table._df = None

            if params:
                def job(obj, obj_path):
                    obj.load(obj_path)

                # load all objects into memory
                audeer.run_tasks(
                    job,
                    params=params,
                    num_workers=num_workers,
                    progress_bar=verbose,
                    task_description='Load tables',
                )

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
            usage=header['usage'],
        )
        header_dicts = [
            'attachments',
            'media',
            'misc_tables',
            'raters',
            'schemes',
            'tables',
            'splits',
        ]
        db.from_dict(header, ignore_keys=header_dicts)

        if 'attachments' in header and header['attachments']:
            for attachment_id, attachment_d in header['attachments'].items():
                attachment = Attachment(attachment_d['path'])
                attachment.from_dict(attachment_d)
                db.attachments[attachment_id] = attachment

        if 'media' in header and header['media']:
            for media_id, media_d in header['media'].items():
                media = Media()
                media.from_dict(media_d)
                db.media[media_id] = media

        if 'misc_tables' in header and header['misc_tables']:
            for table_id, table_d in header['misc_tables'].items():
                table = MiscTable(None)
                table.from_dict(table_d, ignore_keys=['columns'])

                if 'columns' in table_d and table_d['columns']:
                    tmp_callback = table.columns.set_callback
                    table.columns.set_callback = None
                    for column_id, column_d in table_d['columns'].items():
                        column = Column()
                        column.from_dict(column_d)
                        column._id = column_id
                        column._table = table
                        table.columns[column_id] = column
                    table.columns.set_callback = tmp_callback

                db.misc_tables[table_id] = table

        if 'raters' in header and header['raters']:
            for rater_id, rater_d in header['raters'].items():
                rater = Rater()
                rater.from_dict(rater_d)
                db.raters[rater_id] = rater

        if 'schemes' in header and header['schemes']:
            misc_table_schemes = {}
            for scheme_id, scheme_d in header['schemes'].items():
                # ensure to load first all non misc table schemes
                # as they might be needed
                # when checking the column schemes
                # of the underlying misc table
                scheme = Scheme()
                scheme.from_dict(scheme_d)
                if scheme.uses_table:
                    misc_table_schemes[scheme_id] = scheme
                else:
                    db.schemes[scheme_id] = scheme
            for scheme_id, scheme in misc_table_schemes.items():
                db.schemes[scheme_id] = scheme
            # restore order of scheme IDs
            order = list(header['schemes'])
            db.schemes = HeaderDict(
                sorted(
                    db.schemes.items(),
                    key=lambda item: order.index(item[0]),
                ),
                value_type=Scheme,
                set_callback=db._set_scheme,
            )

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

    def _set_attachment(
            self,
            attachment_id: str,
            attachment: Attachment,
    ) -> Attachment:
        attachment._db = self
        attachment._id = attachment_id
        for other_id in list(self.attachments):
            attachment._check_overlap(self.attachments[other_id])
        return attachment

    def _set_scheme(
            self,
            scheme_id: str,
            scheme: Scheme,
    ) -> Scheme:
        scheme._db = self
        scheme._id = scheme_id
        if hasattr(scheme, 'labels') and scheme.labels is not None:
            scheme._check_labels(scheme.labels)
        return scheme

    def _set_table(
            self,
            table_id: str,
            table: typing.Union[MiscTable, Table],
    ) -> typing.Union[MiscTable, Table]:
        if isinstance(table, MiscTable) and table_id in self.tables:
            raise TableExistsError(self[table_id].type, table_id)
        elif isinstance(table, Table) and table_id in self.misc_tables:
            raise TableExistsError('miscellaneous', table_id)
        if table.split_id is not None and table.split_id not in self.splits:
            raise BadIdError('split', table.split_id, self.splits)
        if table.media_id is not None and table.media_id not in self.media:
            raise BadIdError('media', table.media_id, self.media)
        table._db = self
        table._id = table_id
        return table
