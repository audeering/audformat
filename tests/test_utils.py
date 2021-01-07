from io import StringIO
import os
import shutil

import pytest
import pandas as pd

import audformat
from audformat import testing
from audformat import utils
from audformat import define


@pytest.mark.parametrize(
    'objects, axis',
    [
        # empty
        ([], 'index'),
        # filewise
        (
            [
                table.get() for table in pytest.DB.tables.values()
                if table.is_filewise
            ], 'index',
        ),
        # segmented
        (
            [
                utils.to_segmented_index(table.get()) for table in
                pytest.DB.tables.values()
            ], 'index',
        ),
        # mixed
        (
            [
                table.get() for table in pytest.DB.tables.values()
            ], 'index',
        ),
        # series filewise
        (
            [
                pytest.DB['files']['string'].get(),
                pytest.DB['files']['int'].get(),
            ], 'columns',
        ),
        # series segmented
        (
            [
                pytest.DB['segments']['string'].get(),
                pytest.DB['segments']['int'].get(),
            ], 'columns',
        ),
        # series filewise + segmented
        (
            [
                pytest.DB['files']['string'].get(),
                pytest.DB['segments']['int'].get(),
            ], 'columns',
        ),
        # invalid index
        pytest.param(
            [pd.DataFrame([1, 2, 3])], None,
            marks=pytest.mark.xfail(raises=ValueError),
        )
    ],
)
def test_concat(objects, axis):
    df = utils.concat(objects)
    if not objects:
        assert df.empty
    elif audformat.index_type(df) == define.IndexType.SEGMENTED:
        objects = [utils.to_segmented_index(obj) for obj in objects]
        pd.testing.assert_frame_equal(
            df, pd.concat(objects, axis=axis).sort_index(),
        )
    else:
        pd.testing.assert_frame_equal(
            df, pd.concat(objects, axis=axis).sort_index(),
        )


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


@pytest.mark.parametrize('csv,result', [
    (
        StringIO('''file
f1
f2
f3'''),
        pd.Index(
            ['f1', 'f2', 'f3'],
            name='file',
        ),
    ),
    (
        StringIO('''file,value
f1,0.0
f2,1.0
f3,2.0'''),
        pd.Series(
            [0.0, 1.0, 2.0],
            index=audformat.filewise_index(['f1', 'f2', 'f3']),
            name='value',
        ),
    ),
    (
        StringIO('''file,value1,value2
f1,0.0,a
f2,1.0,b
f3,2.0,c'''),
        pd.DataFrame(
            {
                'value1': [0.0, 1.0, 2.0],
                'value2': ['a', 'b', 'c'],
            },
            index=audformat.filewise_index(['f1', 'f2', 'f3']),
            columns=['value1', 'value2'],
        ),
    ),
    (
        StringIO('''file,start,value
f1,00:00:00,0.0
f1,00:00:01,1.0
f2,00:00:02,2.0'''),
        pd.Series(
            [0.0, 1.0, 2.0],
            index=audformat.segmented_index(
                ['f1', 'f1', 'f2'],
                starts=['0s', '1s', '2s'],
                ends=pd.to_timedelta([pd.NaT, pd.NaT, pd.NaT]),
            ),
            name='value',
        ),
    ),
    (
        StringIO('''file,end,value
f1,00:00:01,0.0
f1,00:00:02,1.0
f2,00:00:03,2.0'''),
        pd.Series(
            [0.0, 1.0, 2.0],
            index=audformat.segmented_index(
                ['f1', 'f1', 'f2'],
                starts=['0s', '0s', '0s'],
                ends=['1s', '2s', '3s'],
            ),
            name='value',
        ),
    ),
    (
        StringIO('''file,start,end
f1,00:00:00,00:00:01
f1,00:00:01,00:00:02
f2,00:00:02,00:00:03'''),
        pd.MultiIndex.from_arrays(
            [
                ['f1', 'f1', 'f2'],
                pd.to_timedelta(['0s', '1s', '2s']),
                pd.to_timedelta(['1s', '2s', '3s']),
            ],
            names=['file', 'start', 'end'],
        ),
    ),
    (
        StringIO('''file,start,end,value
f1,00:00:00,00:00:01,0.0
f1,00:00:01,00:00:02,1.0
f2,00:00:02,00:00:03,2.0'''),
        pd.Series(
            [0.0, 1.0, 2.0],
            index=audformat.segmented_index(
                ['f1', 'f1', 'f2'],
                starts=['0s', '1s', '2s'],
                ends=['1s', '2s', '3s'],
            ),
            name='value',
        ),
    ),
    (
        StringIO('''file,start,end,value1,value2
f1,00:00:00,00:00:01,0.0,a
f1,00:00:01,00:00:02,1.0,b
f2,00:00:02,00:00:03,2.0,c'''),
        pd.DataFrame(
            {
                'value1': [0.0, 1.0, 2.0],
                'value2': ['a', 'b', 'c'],
            },
            index=audformat.segmented_index(
                ['f1', 'f1', 'f2'],
                starts=['0s', '1s', '2s'],
                ends=['1s', '2s', '3s'],
            ),
            columns=['value1', 'value2'],
        ),

    ),
    pytest.param(
        StringIO('''value
0.0
1.0
2.0'''),
        None,
        marks=pytest.mark.xfail(raises=ValueError)
    )
])
def test_read_csv(csv, result):
    obj = audformat.utils.read_csv(csv)
    if isinstance(result, pd.Index):
        pd.testing.assert_index_equal(obj, result)
    elif isinstance(result, pd.Series):
        pd.testing.assert_series_equal(obj, result)
    else:
        pd.testing.assert_frame_equal(obj, result)


@pytest.mark.parametrize(
    'table_id',
    ['files', 'segments']
)
def test_to_segmented(table_id):
    for column_id, column in pytest.DB[table_id].get().items():
        series = utils.to_segmented_index(column)
        pd.testing.assert_series_equal(series.reset_index(drop=True),
                                       column.reset_index(drop=True))
    df = utils.to_segmented_index(pytest.DB[table_id].get())
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

        start = df.index.get_level_values(define.IndexField.START)
        assert start.drop_duplicates()[0] == pd.Timedelta(0)
        assert type(start) == pd.core.indexes.timedeltas.TimedeltaIndex

        end = df.index.get_level_values(define.IndexField.END)
        assert end.dropna().empty
        assert type(end) == pd.core.indexes.timedeltas.TimedeltaIndex
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
def test_to_filewise(tmpdir, output_folder, table_id, expected_file_names):

    testing.create_audio_files(pytest.DB, root=tmpdir, file_duration='1s')

    has_existed = os.path.exists(output_folder)

    frame = utils.to_filewise_index(
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

    file_names = [f.split(os.path.sep)[-1].rsplit('.', 1)[0] for f in files]
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
