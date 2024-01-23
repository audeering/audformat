from audformat import define
from audformat import errors
from audformat import utils
from audformat.core.attachment import Attachment
from audformat.core.column import Column
from audformat.core.database import Database
from audformat.core.index import assert_index
from audformat.core.index import assert_no_duplicates
from audformat.core.index import filewise_index
from audformat.core.index import index_type
from audformat.core.index import is_filewise_index
from audformat.core.index import is_segmented_index
from audformat.core.index import segmented_index
from audformat.core.media import Media
from audformat.core.rater import Rater
from audformat.core.scheme import Scheme
from audformat.core.split import Split
from audformat.core.table import MiscTable
from audformat.core.table import Table


# Discourage from audformat import *
__all__ = []


# Dynamically get the version of the installed module
try:
    import importlib.metadata

    __version__ = importlib.metadata.version(__name__)
except Exception:  # pragma: no cover
    importlib = None  # pragma: no cover
finally:
    del importlib
