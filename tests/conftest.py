import os
import tempfile

import pandas as pd
import pytest

import audformat.testing


pytest.DB = audformat.testing.create_db()
# Adjust scheme dictionary to contain one missing label
labels = pytest.DB.schemes["label_map_str"].labels
labels["label3"].pop("prop2")
pytest.DB.schemes["label_map_str"].replace_labels(labels)

pytest.DB_ROOT = "db"
pytest.FILE_DUR = pd.to_timedelta("1s")


@pytest.fixture(scope="package", autouse=True)
def prepare_tests():
    # Prepare files used in tests
    # and set a temporary working directory
    with tempfile.TemporaryDirectory() as tmp:
        current_dir = os.getcwd()
        os.chdir(tmp)

        pytest.DB.save(pytest.DB_ROOT)

        audformat.testing.create_audio_files(
            pytest.DB,
            file_duration=pytest.FILE_DUR,
        )
        audformat.testing.create_attachment_files(
            pytest.DB,
            pytest.DB.root,
        )

        yield

        os.chdir(current_dir)
