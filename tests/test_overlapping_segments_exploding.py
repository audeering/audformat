import pytest
import pandas as pd

from audformat.core.utils import explode_overlapping_segments


@pytest.mark.parametrize(
    'obj, expected',
    [
        (
            pd.Series(index=pd.MultiIndex.from_arrays([
                ['audio_file.wav',
                 'audio_file.wav'],
                [pd.to_timedelta('0 days 00:00:00.861768592'),
                 pd.to_timedelta('0 days 00:00:03.428162853')],
                [pd.to_timedelta('0 days 00:00:09.581768592'),
                 pd.to_timedelta('0 days 00:00:04.488162853')]
            ], names=('file', 'start', 'end')),
                data=[(1, 0, 0, 0, 0), (0, 0, 1, 0, 0)]),

            pd.Series(index=pd.MultiIndex.from_arrays([
                ['audio_file.wav',
                 'audio_file.wav',
                 'audio_file.wav'],
                [pd.to_timedelta('0 days 00:00:00.861768592'),
                 pd.to_timedelta('0 days 00:00:03.428162853'),
                 pd.to_timedelta('0 days 00:00:04.488162853')],
                [pd.to_timedelta('0 days 00:00:03.428162853'),
                 pd.to_timedelta('0 days 00:00:04.488162853'),
                 pd.to_timedelta('0 days 00:00:09.581768592')]
            ], names=('file', 'start', 'end')),
                data=[(1, 0, 0, 0, 0), (1, 0, 1, 0, 0), (1, 0, 0, 0, 0)]),
        )
    ]
)
def test_explode_overlapping_segments_case1(obj, expected):
    exploded_obj = explode_overlapping_segments(obj)
    assert expected.equals(exploded_obj)


@pytest.mark.parametrize(
    'obj, expected',
    [
        (
            pd.DataFrame(index=pd.MultiIndex.from_arrays([
                ['audio_file.wav',
                 'audio_file.wav'],
                [pd.to_timedelta('0 days 00:00:00.861768592'),
                 pd.to_timedelta('0 days 00:00:03.428162853')],
                [pd.to_timedelta('0 days 00:00:09.581768592'),
                 pd.to_timedelta('0 days 00:00:04.488162853')]
            ], names=('file', 'start', 'end')),
                data=['[1, 0, 0, 0, 0]', '[0, 0, 1, 0, 0]'],
            ),

            pd.Series(index=pd.MultiIndex.from_arrays([
                ['audio_file.wav',
                 'audio_file.wav',
                 'audio_file.wav'],
                [pd.to_timedelta('0 days 00:00:00.861768592'),
                 pd.to_timedelta('0 days 00:00:03.428162853'),
                 pd.to_timedelta('0 days 00:00:04.488162853')],
                [pd.to_timedelta('0 days 00:00:03.428162853'),
                 pd.to_timedelta('0 days 00:00:04.488162853'),
                 pd.to_timedelta('0 days 00:00:09.581768592')]
            ], names=('file', 'start', 'end')),
                data=[(1, 0, 0, 0, 0), (1, 0, 1, 0, 0), (1, 0, 0, 0, 0)]),
        )
    ]
)
def test_explode_overlapping_segments_case2(obj, expected):
    exploded_obj = explode_overlapping_segments(obj)
    assert expected.equals(exploded_obj)


@pytest.mark.parametrize(
    'obj, expected',
    [
        (
            pd.Series(index=pd.MultiIndex.from_arrays([
                ['audio_file.wav',
                 'audio_file.wav'],
                [pd.to_timedelta('0 days 00:00:00.861768592'),
                 pd.to_timedelta('0 days 00:00:03.428162853')],
                [pd.to_timedelta('0 days 00:00:09.581768592'),
                 pd.to_timedelta('0 days 00:00:04.488162853')]
            ], names=('file', 'start', 'end')),
                data=['1', '2']),

            ValueError,
        )
    ]
)
def test_explode_overlapping_segments_case3(obj, expected):
    try:
        exploded_obj = explode_overlapping_segments(obj)
    except Exception as e:
        assert isinstance(e, ValueError)
