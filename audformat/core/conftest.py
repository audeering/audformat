import os
import tempfile

import numpy as np
import pytest

import audiofile


@pytest.fixture(scope='package', autouse=True)
def prepare_docstring_tests(doctest_namespace):

    # Prepare files and tmp folder needed in doctests
    with tempfile.TemporaryDirectory() as tmp:
        doctest_namespace['tmp'] = tmp

        current_dir = os.getcwd()
        os.chdir(tmp)

        # audformat.utils.to_filewise_index()
        audiofile.write('f.wav', np.ones((1, 8000)), 8000)

        yield

        os.chdir(current_dir)
