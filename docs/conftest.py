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
pytest_collect_file = sybil.Sybil(parsers=parsers, pattern="*.rst").pytest()
# pytest_collect_file = sybil.sybil.SybilCollection(
#     (
#         sybil.Sybil(
#             parsers=parsers,
#             filenames=[
#                 "authentication.rst",
#                 "overview.rst",
#                 "quickstart.rst",
#                 "dependencies.rst",
#                 "load.rst",
#                 "audb.info.rst",
#             ],
#             fixtures=[
#                 "cache",
#                 "run_in_tmpdir",
#                 "public_repository",
#             ],
#             setup=imports,
#         ),
#         sybil.Sybil(
#             parsers=parsers,
#             filenames=["publish.rst"],
#             fixtures=["cache", "run_in_tmpdir"],
#             setup=imports,
#         ),
#         sybil.Sybil(
#             parsers=parsers,
#             filenames=["configuration.rst", "caching.rst"],
#             fixtures=["default_configuration"],
#             setup=imports,
#         ),
#     )
# ).pytest()
