import os

import numpy as np
import pytest

import audeer
import audiofile


@pytest.fixture(scope='session', autouse=True)
def prepare_docstring_tests():

    # Prepare WAV files needed in doctests
    #
    # audformat.utils.to_filewise_index()
    audiofile.write('f.wav', np.ones((1, 8000)), 8000)

    yield

    # Remove WAV files
    # and files generated during the doctests
    #
    # audformat.utils.to_filewise_index()
    audeer.rmdir('split')
    if os.path.exists('f.wav'):
        os.remove('f.wav')
