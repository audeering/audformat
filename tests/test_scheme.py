import pandas as pd
import pytest

import audformat
import audformat.testing


def test_scheme():

    db = audformat.testing.create_db()

    assert 'tests' in db.schemes['string']
    assert 0.0 in db.schemes['float']
    assert 1.0 in db.schemes['float']
    assert -1.0 in db.schemes['float']
    assert 2.0 not in db.schemes['float']
    assert -2.0 not in db.schemes['float']
    assert pd.Timedelta(0.5, unit='s') in db.schemes['time']
    assert 'label1' in db.schemes['label']
    assert 'label1' in db.schemes['label_map_str']
    assert 1 in db.schemes['label_map_int']

    # dtype mismatch
    with pytest.raises(ValueError):
        audformat.Scheme(
            audformat.define.DataType.STRING,
            labels=[1, 2, 3],
        )

    # unknown type
    with pytest.raises(ValueError):
        audformat.Scheme('bad')

    # labels not list or dictionary
    with pytest.raises(ValueError):
        audformat.Scheme(labels=set([1, 2, 3]))

    # labels do not have the same type
    with pytest.raises(ValueError):
        audformat.Scheme(labels=[1, '2', 3])
