import os

import pandas as pd

import audeer

from audformat.core.common import HeaderDict
import audformat.core.define as define


class Misc(HeaderDict):

    def __init__(self):
        super().__init__(
            value_type=[pd.Series, pd.DataFrame],
        )

    def load(
            self,
            root: str,
            d: dict,
    ):
        for obj_id, obj_d in d.items():

            path = audeer.path(root, 'misc', obj_id)
            pkl_file = f'{path}.{define.TableStorageFormat.PICKLE}'
            csv_file = f'{path}.{define.TableStorageFormat.CSV}'

            if os.path.exists(pkl_file):
                obj = pd.read_pickle(pkl_file)
            else:
                index_col = obj_d['index']
                obj = pd.read_csv(
                    csv_file,
                    index_col=index_col,
                    float_precision='round_trip',
                )
                if obj_d['type'] == 'series':
                    obj = obj[obj_d['name']]

            self[obj_id] = obj

    def save(
            self,
            root: str,
            *,
            storage_format: str = define.TableStorageFormat.CSV,
    ):
        for obj_id, obj in self.items():
            misc_root = audeer.mkdir(audeer.path(root, 'misc'))
            if storage_format == define.TableStorageFormat.PICKLE:
                path = audeer.path(
                    misc_root,
                    f'{obj_id}.{define.TableStorageFormat.PICKLE}',
                )
                obj.to_pickle(
                    path,
                    protocol=4,  # supported by Python >= 3.4
                )
            else:
                path = audeer.path(
                    misc_root,
                    f'{obj_id}.{define.TableStorageFormat.CSV}',
                )
                with open(path, 'w') as fp:
                    obj.to_csv(fp, encoding='utf-8')
