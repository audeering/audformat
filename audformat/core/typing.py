import typing


import numpy as np
import pandas as pd


Files = typing.Union[
    str, typing.Sequence[str], pd.Index, pd.Series,
]
Timestamps = typing.Union[
    float,
    int,
    str,
    pd.Timedelta,
    typing.Sequence[typing.Union[float, int, str, pd.Timedelta]],
    pd.Index,
    pd.Series,
]
Values = typing.Union[
    int, float, str, pd.Timedelta,
    typing.Sequence[
        typing.Union[int, float, str, pd.Timedelta],
    ],
    np.ndarray,
    pd.Series,
]
