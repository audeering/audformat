import os
import tempfile

import numpy as np
import pytest

import audeer
import audiofile


@pytest.fixture(scope='session', autouse=True)
def prepare_docstring_tests(doctest_namespace):

    # Prepare files and tmp folder needed in doctests
    with tempfile.TemporaryDirectory() as tmp:
        doctest_namespace['tmp'] = tmp

        # audformat.utils.to_filewise_index()
        audiofile.write(audeer.path(tmp, 'f.wav'), np.ones((1, 8000)), 8000)

        current_dir = os.getcwd()
        os.chdir(tmp)

        yield

        os.chdir(current_dir)
