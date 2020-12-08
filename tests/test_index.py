import pandas as pd
import pytest

import audformat


@pytest.mark.parametrize(
    'files,starts,ends',
    [
        (
            '1.wav',
            None,
            None,
        ),
        (
            ['1.wav', '2.wav'],
            None,
            None,
        ),
        (
            '1.wav',
            pd.Timedelta('0s'),
            None,
        ),
        (
            '1.wav',
            None,
            pd.Timedelta('1s'),
        ),
        (
            '1.wav',
            pd.Timedelta('0s'),
            pd.Timedelta('1s'),
        ),
        (
            ['1.wav', '2.wav'],
            [pd.Timedelta('0s'), pd.Timedelta('1s')],
            None,
        ),
        (
            ['1.wav', '2.wav'],
            None,
            [pd.Timedelta('1s'), pd.Timedelta('2s')],
        ),
        (
            ['1.wav', '2.wav'],
            [pd.Timedelta('0s'), pd.Timedelta('1s')],
            [pd.Timedelta('1s'), pd.Timedelta('2s')],
        ),
        (
            pd.Series(['1.wav', '2.wav']),
            pd.timedelta_range('0s', freq='1s', periods=2),
            pd.timedelta_range('1s', freq='1s', periods=2),
        ),
        (
            pytest.DB['segments'].files,
            pytest.DB['segments'].starts,
            pytest.DB['segments'].ends,
        ),
        pytest.param(
            ['1.wav'],
            [pd.Timedelta('0s'), pd.Timedelta('1s')],
            [pd.Timedelta('1s'), pd.Timedelta('2s')],
            marks=pytest.mark.xfail(raises=ValueError),
        ),
    ]
)
def test_to_index(files, starts, ends):

    def to_array(value):
        if value is not None:
            if isinstance(value, (pd.Series, pd.DataFrame, pd.Index)):
                value = value.tolist()
            elif not isinstance(value, list):
                value = [value]
        return value

    files = to_array(files)
    starts = to_array(starts)
    ends = to_array(ends)

    index = audformat.index(files, starts=starts, ends=ends)
    index_type = audformat.index_type(index)

    assert index.get_level_values(
        audformat.define.IndexField.FILE
    ).tolist() == files
    if index_type == audformat.define.IndexType.SEGMENTED:
        if starts is not None:
            assert index.get_level_values(
                audformat.define.IndexField.START
            ).tolist() == starts
        else:
            assert index.get_level_values(
                audformat.define.IndexField.START
            ).tolist() == [pd.Timedelta(0)] * len(ends)
        if ends is not None:
            assert index.get_level_values(
                audformat.define.IndexField.END
            ).tolist() == ends
        else:
            assert index.get_level_values(
                audformat.define.IndexField.END
            ).tolist() == [pd.NaT] * len(starts)


@pytest.mark.parametrize(
    'index, index_type',
    [
        (pytest.DB['files'].index, audformat.define.IndexType.FILEWISE),
        (pytest.DB['segments'].index, audformat.define.IndexType.SEGMENTED),
    ]
)
def test_index_type(index, index_type):
    assert audformat.index_type(index) == index_type
