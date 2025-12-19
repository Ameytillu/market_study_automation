import pandas as pd
import numpy as np
from typing import Tuple


def _infer_lag_from_dates(dates: pd.Series) -> int:
    """Infer sensible lag (in periods) for YoY given a date series.

    Returns 12 for monthly-like, 4 for quarterly-like, 1 for yearly/daily fallback.
    """
    dates = pd.to_datetime(dates.dropna()).sort_values()
    if len(dates) < 2:
        return 12
    diffs = dates.diff().dt.days.dropna()
    med = diffs.median()
    if 25 <= med <= 35:
        return 12
    if 80 <= med <= 100:
        return 4
    return 12


def calculate_yoy_growth(df: pd.DataFrame, date_col: str = 'date', value_col: str = 'value') -> pd.DataFrame:
    """Calculate YoY growth for a time series in `df`.

    Args:
        df: DataFrame containing at least a date column and a value column.
        date_col: name of the date column.
        value_col: name of the value column.

    Returns:
        DataFrame with an added `yoy_growth` column (pct change over inferred lag).
    """
    df = df.copy()
    if date_col not in df.columns:
        df['yoy_growth'] = np.nan
        return df
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.sort_values(date_col)
    lag = _infer_lag_from_dates(df[date_col])
    df['yoy_growth'] = df[value_col].pct_change(lag)
    return df


def trend_direction(df: pd.DataFrame, date_col: str = 'date', value_col: str = 'value', lookback: int = 12) -> Tuple[str, float]:
    """Determine trend direction using linear regression on the most recent `lookback` points.

    Returns a tuple (direction, slope). Direction is one of 'up', 'down', 'flat'.
    """
    if date_col not in df.columns or value_col not in df.columns:
        return 'unknown', 0.0
    tmp = df[[date_col, value_col]].dropna()
    if tmp.empty:
        return 'unknown', 0.0
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors='coerce')
    tmp = tmp.sort_values(date_col).tail(lookback)
    if len(tmp) < 2:
        return 'flat', 0.0
    # convert dates to ordinal for regression
    x = tmp[date_col].map(pd.Timestamp.toordinal).astype(float).values
    y = tmp[value_col].astype(float).values
    # simple linear fit
    try:
        slope, intercept = np.polyfit(x, y, 1)
    except Exception:
        return 'unknown', 0.0
    # normalize slope relative to median value magnitude
    med = np.median(np.abs(y)) if np.median(np.abs(y)) != 0 else 1.0
    rel = slope / med
    thresh = 1e-6
    if rel > thresh:
        return 'up', float(slope)
    if rel < -thresh:
        return 'down', float(slope)
    return 'flat', float(slope)


def segment_share(df: pd.DataFrame, date_col: str = 'date', segment_col: str = 'segment', value_col: str = 'value') -> pd.DataFrame:
    """Compute segment share per date. Returns a DataFrame with columns [date, segment, value, share].

    Args:
        df: canonical long-format DataFrame with date, segment, value.
    """
    tmp = df[[date_col, segment_col, value_col]].copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors='coerce')
    grouped = tmp.groupby([date_col, segment_col], dropna=False)[value_col].sum().reset_index()
    totals = grouped.groupby(date_col)[value_col].transform('sum')
    grouped['share'] = grouped[value_col] / totals.replace({0: np.nan})
    return grouped.sort_values([date_col, segment_col])
