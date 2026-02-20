"""Benchmark for audformat.utils.hash() with strict=True.

This benchmark measures hash performance on DataFrames of different sizes
and with different column types:

1. Numeric-only DataFrame: Contains only numeric columns (float, int).

2. Mixed DataFrame: Contains object dtype columns (strings).
"""

import time

import numpy as np
import pandas as pd

import audformat


def create_numeric_df(n_rows):
    """Create a DataFrame with only numeric columns."""
    files = [f"audio/file-{i}.wav" for i in range(n_rows)]
    return pd.DataFrame(
        {
            "value1": np.random.randn(n_rows),
            "value2": np.random.randint(0, 100, n_rows),
        },
        index=audformat.filewise_index(files),
    )


def create_mixed_df(n_rows):
    """Create a DataFrame with mixed column types including strings."""
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


def benchmark(df, n_runs=10):
    """Run benchmark and return average time."""
    # Warmup run
    audformat.utils.hash(df, strict=True)

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        audformat.utils.hash(df, strict=True)
        times.append(time.perf_counter() - start)
    return sum(times) / len(times)


def main():
    print("=" * 70)
    print("Benchmark: audformat.utils.hash(df, strict=True)")
    print("=" * 70)
    print()

    row_counts = [10_000, 100_000, 1_000_000]

    # Benchmark numeric-only DataFrames
    print("Numeric-only DataFrame")
    print("-" * 50)
    print(f"{'Rows':<15} {'Time':<15} {'Rows/sec':<15}")
    print("-" * 50)

    for n_rows in row_counts:
        df = create_numeric_df(n_rows)
        elapsed = benchmark(df)
        rows_per_sec = n_rows / elapsed
        print(f"{n_rows:<15,} {elapsed:<15.4f}s {rows_per_sec:<15,.0f}")

    print()

    # Benchmark mixed DataFrames (with object columns)
    print("Mixed DataFrame (includes object dtype)")
    print("-" * 50)
    print(f"{'Rows':<15} {'Time':<15} {'Rows/sec':<15}")
    print("-" * 50)

    for n_rows in row_counts:
        df = create_mixed_df(n_rows)
        elapsed = benchmark(df)
        rows_per_sec = n_rows / elapsed
        print(f"{n_rows:<15,} {elapsed:<15.4f}s {rows_per_sec:<15,.0f}")

    print()


if __name__ == "__main__":
    main()
