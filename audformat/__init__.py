from audformat import define
from audformat import errors
from audformat import utils
from audformat.core.database import Database
from audformat.core.index import (
    assert_index,
    filewise_index,
    segmented_index,
    index_type,
)
from audformat.core.media import Media
from audformat.core.rater import Rater
from audformat.core.scheme import Scheme
from audformat.core.split import Split
from audformat.core.table import (
    Column,
    Table,
)


# Discourage from audformat import *
__all__ = []


# Dynamically get the version of the installed module
try:
    import pkg_resources
    __version__ = pkg_resources.get_distribution(__name__).version
except Exception:  # pragma: no cover
    pkg_resources = None  # pragma: no cover
finally:
    del pkg_resources
