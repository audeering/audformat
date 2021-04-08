import os
import shutil

import pandas as pd
import pytest

import audformat.testing


pytest.ROOT = os.path.dirname(os.path.realpath(__file__))

pytest.DB = audformat.testing.create_db()
pytest.DB_ROOT = os.path.join(pytest.ROOT, 'db')
pytest.DB.save(pytest.DB_ROOT)

pytest.FILE_DUR = pd.to_timedelta('1s')


@pytest.fixture(scope='session', autouse=True)
def create_audio_files():
    audformat.testing.create_audio_files(
        pytest.DB,
        root=pytest.DB_ROOT,
        file_duration=pytest.FILE_DUR,
    )
    yield
    shutil.rmtree(pytest.DB_ROOT)
