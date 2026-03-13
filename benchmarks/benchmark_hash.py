"""Benchmark for hash(strict=True) and _save_parquet().

Run on different branches to compare performance.

"""

import tempfile
import time

import numpy as np
import pandas as pd

import audformat


def create_numeric_df(n_rows):
    """Create a DataFrame with numeric columns and filewise index."""
    files = [f"audio/file-{i}.wav" for i in range(n_rows)]
    return pd.DataFrame(
        {
            "value1": np.random.randn(n_rows),
            "value2": np.random.randint(0, 100, n_rows),
        },
        index=audformat.filewise_index(files),
    )


def create_mixed_df(n_rows):
    """Create a DataFrame with numeric and string columns."""
    files = [f"audio/file-{i}.wav" for i in range(n_rows)]
    return pd.DataFrame(
        {
            "value": np.random.randn(n_rows),
            "label": np.random.choice(
                ["category_a", "category_b", "category_c"], n_rows
            ),
        },
        index=audformat.filewise_index(files),
    )


def create_segmented_df(n_rows):
    """Create a DataFrame with a segmented index."""
    files = [f"audio/file-{i}.wav" for i in range(n_rows)]
    starts = pd.to_timedelta(np.random.rand(n_rows), unit="s")
    ends = starts + pd.to_timedelta(np.random.rand(n_rows) * 0.5, unit="s")
    return pd.DataFrame(
        {"value": np.random.randn(n_rows)},
        index=audformat.segmented_index(files, starts, ends),
    )


def bench(func, *args, n_runs=5, **kwargs):
    """Run benchmark and return average time in seconds."""
    func(*args, **kwargs)  # warmup
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        func(*args, **kwargs)
        times.append(time.perf_counter() - start)
    return sum(times) / len(times)


def save_parquet(df, path):
    """Simulate Table._save_parquet() without needing a full Database."""
    import pyarrow as pa
    import pyarrow.parquet as parquet

    table_hash = audformat.utils.hash(df, strict=True)
    table = pa.Table.from_pandas(df.reset_index(), preserve_index=False)
    metadata = {"hash": table_hash}
    table = table.replace_schema_metadata({**metadata, **table.schema.metadata})
    parquet.write_table(table, path, compression="snappy")


def main():
    np.random.seed(42)

    print("=" * 70)
    print("hash(strict=True)")
    print("=" * 70)
    print()
    print(f"{'Rows':<15} {'Type':<25} {'Time':<10}")
    print("-" * 50)

    for n_rows in [10_000, 100_000, 1_000_000]:
        for label, create_fn in [
            ("numeric (filewise)", create_numeric_df),
            ("mixed (filewise)", create_mixed_df),
            ("numeric (segmented)", create_segmented_df),
        ]:
            df = create_fn(n_rows)
            t = bench(audformat.utils.hash, df, strict=True)
            print(f"{n_rows:<15,} {label:<25} {t:.4f}s")
        print()

    print("=" * 70)
    print("_save_parquet() (hash + file write)")
    print("=" * 70)
    print()
    print(f"{'Rows':<15} {'Type':<25} {'Time':<10}")
    print("-" * 50)

    for n_rows in [10_000, 100_000, 1_000_000]:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/table.parquet"
            for label, create_fn in [
                ("numeric (filewise)", create_numeric_df),
                ("mixed (filewise)", create_mixed_df),
            ]:
                df = create_fn(n_rows)
                t = bench(save_parquet, df, path)
                print(f"{n_rows:<15,} {label:<25} {t:.4f}s")
        print()


if __name__ == "__main__":
    main()
