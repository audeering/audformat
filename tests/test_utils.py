import os
import shutil

import pytest
import pandas as pd
import numpy as np

import audformat
from audformat import testing
from audformat import utils
from audformat import define


@pytest.mark.parametrize(
    'frames',
    [
        # empty
        [],
        # filewise
        [
            table.get() for table in pytest.DB.tables.values()
            if table.is_filewise
        ],
        # segmented
        [
            utils.to_segmented(table.get()) for table in
            pytest.DB.tables.values()
        ],
        # mixed
        [
            table.get() for table in pytest.DB.tables.values()
        ],
        # not Unified Format
        pytest.param(
            [pd.DataFrame([1, 2, 3])],
            marks=pytest.mark.xfail(
                raises=audformat.errors.NotConformToUnifiedFormat
            ),
        )
    ],
)
def test_concat(frames):
    df = utils.concat(frames)
    if not frames:
        assert df.empty
    elif audformat.index_type(df) == define.IndexType.SEGMENTED:
        frames = [utils.to_segmented(frame) for frame in frames]
        pd.testing.assert_frame_equal(
            df, pd.concat(frames, axis='index').sort_index(),
        )
    else:
        pd.testing.assert_frame_equal(
            df, pd.concat(frames, axis='index').sort_index(),
        )


@pytest.mark.parametrize(
    'table_id,table_type',
    [
        ('files', define.IndexType.FILEWISE),
        ('segments', define.IndexType.SEGMENTED),
    ]
)
def test_index_to_dict(table_id, table_type):
    d = utils.index_to_dict(pytest.DB[table_id].index)
    np.testing.assert_equal(
        d[define.IndexField.FILE + 's'],
        pytest.DB[table_id].files,
    )
    if table_type == define.IndexType.SEGMENTED:
        np.testing.assert_equal(
            d[define.IndexField.START + 's'],
            pytest.DB[table_id].starts,
        )
        np.testing.assert_equal(
            d[define.IndexField.END + 's'],
            pytest.DB[table_id].ends,
        )
    elif table_type == define.IndexType.FILEWISE:
        assert d[define.IndexField.START + 's'] is None
        assert d[define.IndexField.END + 's'] is None


@pytest.mark.parametrize(
    'table_id,table_type',
    [
        ('files', define.IndexType.FILEWISE),
        ('segments', define.IndexType.SEGMENTED),
    ]
)
def test_series_to_dict(table_id, table_type):
    for _, column in pytest.DB[table_id].get().items():
        d = utils.series_to_dict(column)
        np.testing.assert_equal(
            d[define.IndexField.FILE + 's'],
            pytest.DB[table_id].files,
        )
        if table_type == define.IndexType.SEGMENTED:
            np.testing.assert_equal(
                d[define.IndexField.START + 's'],
                pytest.DB[table_id].starts,
            )
            np.testing.assert_equal(
                d[define.IndexField.END + 's'],
                pytest.DB[table_id].ends,
            )
        elif table_type == define.IndexType.FILEWISE:
            assert d[define.IndexField.START + 's'] is None
            assert d[define.IndexField.END + 's'] is None
        pd.testing.assert_series_equal(
            pd.Series(d['values']),
            pd.Series(utils.series_to_array(column)),
        )


@pytest.mark.parametrize(
    'table_id,table_type',
    [
        ('files', define.IndexType.FILEWISE),
        ('segments', define.IndexType.SEGMENTED),
    ]
)
def test_series_to_frame(table_id, table_type):

    d = utils.frame_to_dict(pytest.DB[table_id].get())
    np.testing.assert_equal(
        d[define.IndexField.FILE + 's'],
        pytest.DB[table_id].files,
    )
    if table_type == define.IndexType.SEGMENTED:
        np.testing.assert_equal(
            d[define.IndexField.START + 's'],
            pytest.DB[table_id].starts,
        )
        np.testing.assert_equal(
            d[define.IndexField.END + 's'],
            pytest.DB[table_id].ends,
        )
    elif table_type == define.IndexType.FILEWISE:
        assert d[define.IndexField.START + 's'] is None
        assert d[define.IndexField.END + 's'] is None

    for column_id, column in pytest.DB[table_id].get().items():
        pd.testing.assert_series_equal(
            pd.Series(d['values'][column_id]),
            pd.Series(utils.series_to_array(column)),
        )


@pytest.mark.parametrize(
    'table_id',
    ['files', 'segments']
)
def test_to_segmented_frame(table_id):
    for column_id, column in pytest.DB[table_id].get().items():
        series = utils.to_segmented(column)
        pd.testing.assert_series_equal(series.reset_index(drop=True),
                                       column.reset_index(drop=True))
    df = utils.to_segmented(pytest.DB[table_id].get())
    pd.testing.assert_frame_equal(
        df.reset_index(drop=True),
        pytest.DB[table_id].get().reset_index(drop=True),
    )
    if pytest.DB[table_id].is_filewise:
        pd.testing.assert_index_equal(
            df.index.get_level_values(define.IndexField.FILE),
            pytest.DB[table_id].index.get_level_values(
                define.IndexField.FILE
            )
        )
        assert df.index.get_level_values(
            define.IndexField.START).drop_duplicates()[0] == pd.Timedelta(0)
        assert df.index.get_level_values(
            define.IndexField.END).dropna().empty
    else:
        pd.testing.assert_index_equal(df.index, pytest.DB[table_id].index)


@pytest.mark.parametrize(
    'output_folder,table_id,expected_file_names',
    [
        pytest.param(
            '.',
            'segments',
            None,
            marks=pytest.mark.xfail(raises=ValueError)
        ),
        pytest.param(
            os.path.abspath(''),
            'segments',
            None,
            marks=pytest.mark.xfail(raises=ValueError)
        ),
        (
            'tmp',
            'segments',
            [
                str(i).zfill(3) + f'_{j}'
                for i in range(1, 11)
                for j in range(10)
            ]
        ),
        (
            'tmp',
            'files',
            [str(i).zfill(3) for i in range(1, 101)]
        )
    ]
)
def test_to_filewise_frame(tmpdir, output_folder, table_id,
                           expected_file_names):

    testing.create_audio_files(pytest.DB, root=tmpdir, file_duration='1s')

    has_existed = os.path.exists(output_folder)

    frame = utils.to_filewise(
        obj=pytest.DB[table_id].get(),
        root=tmpdir,
        output_folder=output_folder,
        num_workers=3,
    )

    assert audformat.index_type(frame) == define.IndexType.FILEWISE
    pd.testing.assert_frame_equal(
        pytest.DB[table_id].get().reset_index(drop=True),
        frame.reset_index(drop=True),
    )
    files = frame.index.get_level_values(define.IndexField.FILE).values

    if table_id == 'segmented':  # already `framewise` frame is unprocessed
        assert os.path.isabs(output_folder) == os.path.isabs(files[0])

    if table_id == 'files':
        # files of unprocessed frame are relative to `root`
        files = [os.path.join(tmpdir, f) for f in files]
    assert all(os.path.exists(f) for f in files)

    file_names = [f.split('/')[-1].rsplit('.', 1)[0] for f in files]
    assert file_names == expected_file_names

    # clean-up
    if not has_existed:  # output folder was created and can be removed
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
    else:
        if table_id == 'segments':
            for f in frame.index.get_level_values(
                    define.IndexField.FILE):
                if os.path.exists(f):
                    os.remove(f)


@pytest.mark.parametrize(
    'language, expected',
    [
        ('en', 'eng'),
        ('en', 'eng'),
        ('english', 'eng'),
        ('English', 'eng'),
        pytest.param(
            'xx', None,
            marks=pytest.mark.xfail(raises=ValueError)
        ),
        pytest.param(
            'xxx', None,
            marks=pytest.mark.xfail(raises=ValueError)
        ),
        pytest.param(
            'Bad language', None,
            marks=pytest.mark.xfail(raises=ValueError)
        )
    ]
)
def test_map_language(language, expected):
    assert utils.map_language(language) == expected
