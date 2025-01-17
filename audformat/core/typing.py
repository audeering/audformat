from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


Files = str | Sequence[str] | pd.Index | pd.Series
Timestamps = (
    float
    | int
    | str
    | pd.Timedelta
    | Sequence[float | int | str | pd.Timedelta]
    | pd.Index
    | pd.Series
)
Values = (
    int
    | float
    | str
    | pd.Timedelta
    | Sequence[int | float | str | pd.Timedelta]
    | np.ndarray
    | pd.Series
)
