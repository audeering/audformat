from doctest import ELLIPSIS
from doctest import NORMALIZE_WHITESPACE

import sybil
from sybil.parsers.rest import DocTestParser
from sybil.parsers.rest import PythonCodeBlockParser
from sybil.parsers.rest import SkipParser


# Collect doctests
#
# We use several `sybil.Sybil` instances
# to pass different fixtures for different files
#
parsers = [
    DocTestParser(optionflags=ELLIPSIS + NORMALIZE_WHITESPACE),
    PythonCodeBlockParser(),
    SkipParser(),
]
pytest_collect_file = sybil.Sybil(
    parsers=parsers,
    pattern="*.rst",
    fixtures=["tmpdir"],
).pytest()
