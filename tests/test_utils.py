# import pytest
# import pandas as pd
# import numpy as np
# import os
# import shutil
#
# from audformat import testing
# from audformat import utils
# from audformat import define
#
# xfail = pytest.mark.xfail
#
#
# db = testing.create_db()
# root = os.path.dirname(__file__)
# testing.create_audio_files(db, root=root)
#
#
# @pytest.mark.parametrize('table_id,table_type',
#                          [('files', define.TableType.FILEWISE),
#                           ('segments', define.TableType.SEGMENTED),
#                           ])
# def test_index_to_table_type(table_id, table_type):
#     assert utils.index_to_table_type(db[table_id].df) == table_type
#
#
# def test_concat_frames():
#     dfs = [utils.to_segmented_frame(table.df) for table in db.tables.values()]
#     pd.testing.assert_frame_equal(utils.concat_frames(dfs),
#                                   pd.concat(dfs, axis='index').sort_index())
#
#
# @pytest.mark.parametrize('table_id,table_type',
#                          [('files', define.TableType.FILEWISE),
#                           ('segments', define.TableType.SEGMENTED),
#                           ])
# def test_index_to_dict(table_id, table_type):
#     d = utils.index_to_dict(db[table_id].df.index)
#     np.testing.assert_equal(d[define.IndexField.FILE + 's'],
#                             db[table_id].files)
#     if table_type == define.TableType.SEGMENTED:
#         np.testing.assert_equal(d[define.IndexField.SEGMENT_START + 's'],
#                                 db[table_id].starts)
#         np.testing.assert_equal(d[define.IndexField.SEGMENT_END + 's'],
#                                 db[table_id].ends)
#     elif table_type == define.TableType.FILEWISE:
#         assert d[define.IndexField.SEGMENT_START + 's'] is None
#         assert d[define.IndexField.SEGMENT_END + 's'] is None
#
#
# @pytest.mark.parametrize('table_id,table_type',
#                          [('files', define.TableType.FILEWISE),
#                           ('segments', define.TableType.SEGMENTED),
#                           ])
# def test_series_to_dict(table_id, table_type):
#     for _, column in db[table_id].df.items():
#         d = utils.series_to_dict(column)
#         np.testing.assert_equal(d[define.IndexField.FILE + 's'],
#                                 db[table_id].files)
#         if table_type == define.TableType.SEGMENTED:
#             np.testing.assert_equal(d[define.IndexField.SEGMENT_START + 's'],
#                                     db[table_id].starts)
#             np.testing.assert_equal(d[define.IndexField.SEGMENT_END + 's'],
#                                     db[table_id].ends)
#         elif table_type == define.TableType.FILEWISE:
#             assert d[define.IndexField.SEGMENT_START + 's'] is None
#             assert d[define.IndexField.SEGMENT_END + 's'] is None
#         pd.testing.assert_series_equal(
#             pd.Series(d['values']),
#             pd.Series(utils.series_to_array(column)),
#         )
#
#
# @pytest.mark.parametrize('table_id,table_type',
#                          [('files', define.TableType.FILEWISE),
#                           ('segments', define.TableType.SEGMENTED),
#                           ])
# def test_series_to_frame(table_id, table_type):
#
#     d = utils.frame_to_dict(db[table_id].df)
#     np.testing.assert_equal(d[define.IndexField.FILE + 's'],
#                             db[table_id].files)
#     if table_type == define.TableType.SEGMENTED:
#         np.testing.assert_equal(d[define.IndexField.SEGMENT_START + 's'],
#                                 db[table_id].starts)
#         np.testing.assert_equal(d[define.IndexField.SEGMENT_END + 's'],
#                                 db[table_id].ends)
#     elif table_type == define.TableType.FILEWISE:
#         assert d[define.IndexField.SEGMENT_START + 's'] is None
#         assert d[define.IndexField.SEGMENT_END + 's'] is None
#
#     for column_id, column in db[table_id].df.items():
#         pd.testing.assert_series_equal(
#             pd.Series(d['values'][column_id]),
#             pd.Series(utils.series_to_array(column)),
#         )
#
#
# @pytest.mark.parametrize('table_id',
#                          ['files', 'segments'])
# def test_to_segmented_frame(table_id):
#     for column_id, column in db[table_id].df.items():
#         series = utils.to_segmented_frame(column)
#         pd.testing.assert_series_equal(series.reset_index(drop=True),
#                                        column.reset_index(drop=True))
#     df = utils.to_segmented_frame(db[table_id].df)
#     pd.testing.assert_frame_equal(df.reset_index(drop=True),
#                                   db[table_id].df.reset_index(drop=True))
#     if db[table_id].is_filewise:
#         pd.testing.assert_index_equal(
#             df.index.get_level_values(define.IndexField.FILE),
#             db[table_id].df.index.get_level_values(define.IndexField.FILE))
#         assert df.index.get_level_values(
#             define.IndexField.SEGMENT_START).drop_duplicates()[0] == \
#             pd.Timedelta(0)
#         assert df.index.get_level_values(
#             define.IndexField.SEGMENT_END).dropna().empty
#     else:
#         pd.testing.assert_index_equal(df.index, db[table_id].df.index)
#
#
# @pytest.mark.parametrize('table_type,files,starts,ends',
#                          [
#                              (
#                                  define.TableType.FILEWISE,
#                                  ['1.wav', '2.wav'], None, None,
#                              ),
#                              (
#                                  define.TableType.SEGMENTED,
#                                  ['1.wav', '2.wav'],
#                                  [pd.Timedelta('0s'), pd.Timedelta('1s')],
#                                  [pd.Timedelta('1s'), pd.Timedelta('2s')],
#                              ),
#                          ])
# def test_to_index(table_type, files, starts, ends):
#     index = utils.to_index(files, starts, ends)
#     assert utils.index_to_table_type(index) == table_type
#     d = utils.index_to_dict(index)
#     assert all(d['files'] == files)
#     if starts is not None:
#         pd.testing.assert_index_equal(pd.to_timedelta(d['starts']),
#                                       pd.to_timedelta(starts))
#     else:
#         assert d['starts'] is None
#     if ends is not None:
#         pd.testing.assert_index_equal(pd.to_timedelta(d['ends']),
#                                       pd.to_timedelta(ends))
#     else:
#         assert d['ends'] is None
#
#
# @pytest.mark.parametrize(
#     'output_folder,table_id,expected_file_names',
#     [
#         pytest.param(
#             '.',
#             'segments',
#             None,
#             marks=xfail(raises=ValueError)
#         ),
#         pytest.param(
#             os.path.abspath(''),
#             'segments',
#             None,
#             marks=xfail(raises=ValueError)
#         ),
#         (
#             'tmp',
#             'segments',
#             [
#                 str(i).zfill(3) + f'_{j}'
#                 for i in range(1, 11)
#                 for j in range(10)
#             ]
#         ),
#         (
#             'tmp',
#             'files',
#             [str(i).zfill(3) for i in range(1, 101)]
#         )
#     ]
# )
# def test_to_filewise_frame(output_folder, table_id, expected_file_names):
#     has_existed = os.path.exists(output_folder)
#
#     frame = utils.to_filewise_frame(frame=db[table_id].df,
#                                     root=root,
#                                     output_folder=output_folder,
#                                     num_workers=3)
#
#     assert utils.index_to_table_type(frame) == define.TableType.FILEWISE
#     pd.testing.assert_frame_equal(db[table_id].df.reset_index(drop=True),
#                                   frame.reset_index(drop=True))
#     files = frame.index.get_level_values(define.IndexField.FILE).values
#
#     if table_id == 'segmented':  # already `framewise` frame is unprocessed
#         assert os.path.isabs(output_folder) == os.path.isabs(files[0])
#
#     if table_id == 'files':
#         # files of unprocessed frame are relative to `root`
#         files = [os.path.join(root, f) for f in files]
#     assert all(os.path.exists(f) for f in files)
#
#     file_names = [f.split('/')[-1].rsplit('.', 1)[0] for f in files]
#     assert file_names == expected_file_names
#
#     # clean-up
#     if not has_existed:  # output folder was created and can be removed
#         if os.path.exists(output_folder):
#             shutil.rmtree(output_folder)
#     else:
#         if table_id == 'segments':
#             for f in frame.index.get_level_values(
#                     define.IndexField.FILE):
#                 if os.path.exists(f):
#                     os.remove(f)
