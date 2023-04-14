import os

import pandas as pd
import pytest

import audeer
import audformat.testing


pytest.ROOT = os.path.dirname(os.path.realpath(__file__))

pytest.DB = audformat.testing.create_db()
pytest.DB_ROOT = os.path.join(pytest.ROOT, 'db')
audformat.testing.create_attachment_files(pytest.DB, pytest.DB_ROOT)
pytest.DB.save(pytest.DB_ROOT)

pytest.FILE_DUR = pd.to_timedelta('1s')


@pytest.fixture(scope='session', autouse=True)
def create_audio_files():
    audformat.testing.create_audio_files(
        pytest.DB,
        file_duration=pytest.FILE_DUR,
    )
    yield
    audeer.rmdir(pytest.DB_ROOT)
    # Clean up docstring generated files
    # from audformat.utils.to_filewise_index()
    audeer.rmdir('split')
    os.remove('f.wav')
