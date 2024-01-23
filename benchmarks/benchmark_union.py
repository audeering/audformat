import time
import typing

import numpy as np
import pandas as pd

import audformat


# Benchmark for the utility function
# audformat.utils.union()
# that concatenates a list of index objects.
# This can be achieved with pd.concat().
#
# See also https://github.com/audeering/audformat/pull/354


np.random.seed(1)


def benchmark(
    segmented: bool,
    num_segs: typing.Tuple[int],
    num_objs: typing.Tuple[int],
    num_repeat: int,
) -> pd.Series:
    ds = []

    for num_seg, num_obj in zip(num_segs, num_objs):
        objs = []
        for idx in range(num_obj):
            files = [f"file-{idx}-{seg}" for seg in range(num_seg)]
            if segmented:
                starts = np.random.randn(len(files))
                ends = np.random.randn(len(files)) + 2
                index = audformat.segmented_index(files, starts, ends)
            else:
                index = audformat.filewise_index(files)
            objs.append(index)

        t = time.time()
        for _ in range(num_repeat):
            _ = audformat.utils.union(objs)
        dt = (time.time() - t) / num_repeat

        d = {
            "num_obj": num_obj,
            "num_seg": num_seg,
            "elapsed": dt,
        }
        ds.append(d)

    y = pd.DataFrame(ds).set_index(["num_obj", "num_seg"])["elapsed"]

    return y


def main():
    num_segs = [1000000, 1000, 10000, 100000, 100, 1000, 10000, 100, 1000]
    num_objs = [2, 10, 10, 10, 100, 100, 100, 1000, 1000]
    num_repeat = 10

    print(f"Execution time averaged over {num_repeat} runs.")

    for segmented in [False, True]:
        print()
        print(f'{"segmented" if segmented else "filewise"} index')

        y = benchmark(
            segmented,
            num_segs,
            num_objs,
            num_repeat,
        )
        y.name = "time (s)"
        print(y.round(2))
        print(f"Average: {y.mean()}")


if __name__ == "__main__":
    main()
