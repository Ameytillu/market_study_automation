import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from typing import Optional


def plot_line_trend(df: pd.DataFrame, date_col: str = 'date', value_col: str = 'value', title: Optional[str] = None):
    """Create a simple line chart for a trend series and return the matplotlib Figure.

    Args:
        df: DataFrame with date and value columns.
        date_col: name of date column.
        value_col: name of value column.
        title: optional chart title.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.sort_values(date_col)
    ax.plot(df[date_col], df[value_col], marker='o', linewidth=1)
    ax.set_title(title or f'{value_col} over time')
    ax.set_xlabel('Date')
    ax.set_ylabel(value_col)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    fig.tight_layout()
    return fig


def plot_stacked_segment(df: pd.DataFrame, date_col: str = 'date', segment_col: str = 'segment', value_col: str = 'value', title: Optional[str] = None):
    """Create a stacked bar chart for segment mix over time and return the Figure.

    Args:
        df: canonical long-format DataFrame.
    """
    tmp = df.copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors='coerce')
    pivot = tmp.pivot_table(index=date_col, columns=segment_col, values=value_col, aggfunc='sum', fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.sort_index().plot(kind='bar', stacked=True, ax=ax, width=0.8)
    ax.set_title(title or 'Segment Mix')
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.legend(title=segment_col, bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    fig.tight_layout()
    return fig
