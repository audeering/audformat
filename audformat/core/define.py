from audformat.core.common import DefineBase


class DataType(DefineBase):
    r"""Data types of column content.

    Specifies string values
    representing data types
    of different columns
    within a table or misc table.
    Those string values
    have to be provided
    as ``dtype`` argument
    to :class:`audformat.Scheme`,
    and are returned
    by :attr:`audformat.Scheme.dtype`.
    The exact string values are part
    of the :ref:`scheme specifications <data-header:Scheme>`,
    and should never be changed by a user.

    Use ``DATE``
    to handle time and date information,
    e.g. as provided by :class:`datetime.datetime`.
    Use ``TIME``
    to handle duration values.

    """

    BOOL = "bool"
    """Bool data type."""

    DATE = "date"
    """Date data type.

    Data type to store date information,
    e.g. as provided by :class:`datetime.datetime`.

    """

    INTEGER = "int"
    """Integer data type."""

    FLOAT = "float"
    """Float data type."""

    OBJECT = "object"
    """Object data type.

    This should be used if no other data type fits.
    Inside :mod:`pandas` it is stored
    as the ``object`` data type,
    compare :attr:`pandas.DataFrame.dtypes`.

    """

    STRING = "str"
    """String data type."""

    TIME = "time"
    """Time data type.

    Data type to store durations,
    e.g. as provided by :class:`pandas.Timedelta`.

    """


class Gender(DefineBase):
    r"""Gender scheme definitions.

    Specifies string values
    representing gender labels
    that are recommended to use with a scheme
    that contains gender labels,
    e.g.
    ``audformat.Scheme('str', labels=['female', 'male'])``.
    The exact string values are not part
    of the :ref:`scheme specifications <data-header:Scheme>`,
    and can be changed if desired.

    """

    CHILD = "child"
    """Child gender label.

    Female and male children
    have a voice more common to each other
    than to a female or male grown up.
    Hence,
    we provide a common child label.

    """

    FEMALE = "female"
    """Female gender label."""

    MALE = "male"
    """Male gender label."""

    OTHER = "other"
    """Other gender label.

    Should include labels
    that are not covered by
    female, male, or child,
    e.g. diverse.

    """


class IndexField(DefineBase):
    r"""Index fields of a table.

    Specifies the string values
    representing column/field names
    for a filewise
    and segmented index.
    The exact string values are part
    of the :ref:`table specifications <data-tables:Tables>`,
    and should never be changed by a user.

    """

    FILE = "file"
    """File index field.

    Name of the index column
    listing files
    in a filewise or segmented table.

    """

    START = "start"
    """Start index field.

    Name of the index column
    listing start times
    in a segmented table.

    """

    END = "end"
    """End index field.

    Name of the index column
    listing end times
    in a segmented table.

    """


class IndexType(DefineBase):
    r"""Index types of a table.

    Specifies the string values
    representing a filewise or segmented index.
    Those string values are returned by
    :attr:`audformat.Table.type`
    and :func:`audformat.index_type`.
    The exact string values are part
    of the :ref:`table specifications <data-tables:Tables>`,
    and should never be changed by a user.

    """

    FILEWISE = "filewise"
    """Filewise index type."""

    SEGMENTED = "segmented"
    """Segmented index type."""


class License(DefineBase):
    r"""Common public licenses recommended to use with your data.

    Specifies string values
    representing public licences
    that are recommended to use with the database.
    If those string values
    are provided
    as ``license`` argument
    to :class:`audformat.Database`
    the corresponding
    ``license_url`` argument does not need
    to be provided
    but is set automatically.
    The exact string values are not part
    of the :ref:`database specifications <data-header:Database>`,
    and can be changed if desired.

    """

    CC0_1_0 = "CC0-1.0"
    """Creative Commons 1.0 Universal."""

    CC_BY_4_0 = "CC-BY-4.0"
    """Creative Commons Attribution 4.0."""

    CC_BY_NC_4_0 = "CC-BY-NC-4.0"
    """Creative Commons Attribution-NonCommercial 4.0."""

    CC_BY_NC_SA_4_0 = "CC-BY-NC-SA-4.0"
    """Creative Commons Attribution-NonCommercial-ShareAlike 4.0."""

    CC_BY_SA_4_0 = "CC-BY-SA-4.0"
    """Creative Commons Attribution-ShareAlike 4.0."""


LICENSE_URLS = {
    License.CC0_1_0: "https://creativecommons.org/publicdomain/zero/1.0/",
    License.CC_BY_4_0: "https://creativecommons.org/licenses/by/4.0/",
    License.CC_BY_NC_4_0: "https://creativecommons.org/licenses/by-nc/4.0/",
    License.CC_BY_NC_SA_4_0: "https://creativecommons.org/licenses/by-nc-sa/4.0/",
    License.CC_BY_SA_4_0: "https://creativecommons.org/licenses/by-sa/4.0/",
}


class MediaType(DefineBase):
    r"""Media type of table.

    Specifies string values
    representing media types
    of different tables
    or misc tables.
    Those string values
    have to be provided
    as ``type`` argument
    to :class:`audformat.Media`.
    The exact string values are part
    of the :ref:`media specifications <data-header:Media>`,
    and should never be changed by a user.

    """

    AUDIO = "audio"
    """Audio media type."""

    OTHER = "other"
    """Other media type."""

    VIDEO = "video"
    """Video media type."""


class RaterType(DefineBase):
    r"""Rater type of column.

    Specifies string values
    representing rater types
    of different columns
    in a tables or misc table.
    Those string values
    have to be provided
    as ``type`` argument
    to :class:`audformat.Rater`.
    The exact string values are part
    of the :ref:`rater specifications <data-header:Rater>`,
    and should never be changed by a user.

    """

    HUMAN = "human"
    """Human rater type."""

    MACHINE = "machine"
    """Machine rater type."""

    OTHER = "other"
    """Other rater type."""

    TRUTH = "ground truth"
    """Ground truth rater type."""

    VOTE = "vote"
    """Vote rater type."""


class SplitType(DefineBase):
    r"""Split type of table.

    Specifies string values
    representing split types
    of different tables
    or misc table.
    Those string values
    have to be provided
    as ``type`` argument
    to :class:`audformat.Split`.
    The exact string values are part
    of the :ref:`split specifications <data-header:Split>`,
    and should never be changed by a user.

    """

    TRAIN = "train"
    """Train split type."""

    DEVELOP = "dev"
    """Dev split type."""

    OTHER = "other"
    """Other split type."""

    TEST = "test"
    """Test split type."""


class TableStorageFormat(DefineBase):
    r"""Storage format of tables.

    Specifies string values
    used as file extensions
    of the CSV and PKL files
    that are used to store
    a table or misc table.
    Those string values
    have to be provided
    as ``storage_format`` argument
    to :meth:`audformat.Database.save`,
    :meth:`audformat.MiscTable.save`,
    :meth:`audformat.Table.save`,
    and when loading with
    to :meth:`audformat.Database.load`,
    :meth:`audformat.MiscTable.load`,
    :meth:`audformat.Table.load`,
    only files with an extension
    matching the string values
    are considered.

    The exact string values for CSV files
    are part
    of the :ref:`audformat specifications <data-format:Database>`,
    and should never be changed by a user.

    """

    CSV = "csv"
    """File extension for tables stored in CSV format."""

    PARQUET = "parquet"
    """File extension for tables stored in PARQUET format."""

    PICKLE = "pkl"
    """File extension for tables stored in PKL format."""


class Usage(DefineBase):
    r"""Usage permission of database.

    Specifies string values
    representing usage
    of a database.
    Those string values
    have to be provided
    as ``usage`` argument
    to :class:`audformat.Database`,
    and returned
    by :attr:`audformat.Database.usage`.
    The exact string values are part
    of the :ref:`database specifications <data-header:Database>`,
    and should never be changed by a user.

    """

    COMMERCIAL = "commercial"
    """Commercial usage."""

    OTHER = "other"
    """Other usage."""

    RESEARCH = "research"
    """Research only usage."""

    RESTRICTED = "restricted"
    """Restricted usage."""

    UNRESTRICTED = "unrestricted"
    """Unrestricted usage."""
