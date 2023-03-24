import os
import time
import typing

import pandas as pd

import audeer
import audformat


# Benchmark for the utility function
# audformat.utils.union()
# that concatenates a list of index objects.
# This can be achieved with pd.concat().
# However,
# for indices with less than 500 entries,
# it is faster to convert the level values to lists
# and create the index from those.
#
# See also https://github.com/audeering/audformat/pull/354


cache_root = audeer.mkdir('cache')


def benchmark(
        segmented: bool,
        num_segs: typing.Tuple[int],
        num_objs: typing.Tuple[int],
        num_repeat: int,
        threshold: int,
) -> pd.Series:

    cache_name = (
            hash(segmented) +
            hash(num_segs) +
            hash(num_objs) +
            hash(num_repeat) +
            hash(threshold)
    )
    cache_path = audeer.path(cache_root, f'{cache_name}.pkl')

    if os.path.exists(cache_path):
        return pd.read_pickle(cache_path)

    audformat.core.utils.UNION_MAX_INDEX_LEN_THRES = threshold

    ds = []

    for num_seg in num_segs:
        for num_obj in num_objs:

            objs = []
            for idx in range(num_obj):
                files = [f'file-{idx}-{seg}' for seg in range(num_seg)]
                if segmented:
                    starts = [0] * len(files)
                    ends = [1] * len(files)
                    index = audformat.segmented_index(files, starts, ends)
                else:
                    index = audformat.filewise_index(files)
                objs.append(index)

            t = time.time()
            for _ in range(num_repeat):
                _ = audformat.utils.union(objs)
            dt = (time.time() - t) / num_repeat

            d = {
                'num_obj': num_obj,
                'num_seg': num_seg,
                'elapsed': dt,
            }
            ds.append(d)

    y = pd.DataFrame(ds).set_index(['num_obj', 'num_seg'])['elapsed']
    y.to_pickle(cache_path)

    return y


def main():

    default_threshold = audformat.core.utils.UNION_MAX_INDEX_LEN_THRES

    num_segs = tuple(range(100, 1000, 100))
    num_objs = (10, 100, 1000)
    num_repeat = 5

    for segmented in [False, True]:

        print(f'{"segmented" if segmented else "filewise"} index')

        # always use pd.concat()
        y_pandas = benchmark(
            segmented,
            num_segs,
            num_objs,
            num_repeat,
            0,
        )

        # use pd.concat() only when at least one index
        # has more than UNION_MAX_INDEX_LEN_THRES entries
        # otherwise create index from lists
        y_audformat = benchmark(
            segmented,
            num_segs,
            num_objs,
            num_repeat,
            default_threshold,
        )

        df = pd.DataFrame(
            {
                'pandas': y_pandas.values,
                'audformat': y_audformat.values,
                'factor': (y_audformat / y_pandas).values,
            },
            index=y_pandas.index,
        )
        print(df.round(2))
        print(df.mean())


if __name__ == '__main__':
    main()
