import pandas as pd
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class SheetInfo:
    """Structured information about a single sheet.

    Attributes:
        sheet_name: original sheet name
        analysis_type: one of 'trend', 'segment', 'growth', 'seasonality', or 'unknown'
        aggregation: one of 'Monthly', 'Quarterly', '12MMA', or 'Unknown'
        rows: number of rows in original sheet
        cols: number of cols in original sheet
        normalized: list of dict rows in canonical format (not raw DataFrame)
        source_file: original file name or identifier
    """
    sheet_name: str
    analysis_type: str
    aggregation: str
    rows: int
    cols: int
    normalized: List[Dict[str, Any]]
    source_file: Optional[str] = None


def _parse_dates_if_possible(series: pd.Series) -> Optional[pd.Series]:
    """Try to parse a Series as datetimes. Return Series of datetimes or None."""
    try:
        parsed = pd.to_datetime(series, errors='coerce')
        if parsed.notna().sum() >= max(1, int(len(parsed) * 0.5)):
            return parsed
    except Exception:
        pass
    return None


def _detect_date_column(df: pd.DataFrame) -> Optional[str]:
    """Heuristic to find a date-like column name in a DataFrame."""
    # 1) obvious name
    for col in df.columns:
        low = str(col).lower()
        if 'date' in low or low == 'period' or low.startswith('month'):
            return col

    # 2) dtype
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.datetime64):
            return col

    # 3) try parsing each column
    for col in df.columns:
        parsed = _parse_dates_if_possible(df[col])
        if parsed is not None:
            return col

    # 4) no date column found
    return None


def _detect_aggregation_from_dates(dates: pd.Series) -> str:
    """Infer aggregation by median difference between dates."""
    dates = pd.to_datetime(dates.dropna()).sort_values()
    if len(dates) < 2:
        return 'Unknown'
    diffs = dates.diff().dt.days.dropna()
    if diffs.empty:
        return 'Unknown'
    med = diffs.median()
    if med <= 3:
        return 'Daily'
    if 25 <= med <= 35:
        return 'Monthly'
    if 80 <= med <= 100:
        return 'Quarterly'
    return 'Unknown'


def _detect_aggregation_from_columns(cols: List[str]) -> Optional[str]:
    """Look for signals in column names such as '12MMA'."""
    for c in cols:
        low = str(c).lower()
        if '12mma' in low or '12-mma' in low or '12 ma' in low or '12m' in low:
            return '12MMA'
    return None


def _classify_analysis_type(df: pd.DataFrame, date_col: Optional[str]) -> str:
    """Classify sheet into analysis types using heuristics."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if date_col is None:
        # if many numeric columns and columns look like dates, maybe trend matrix
        parsed_cols = [pd.to_datetime(c, errors='coerce') for c in df.columns]
        if sum([1 for p in parsed_cols if not pd.isna(p)]) >= 2:
            return 'trend'
        return 'unknown'

    if len(numeric_cols) == 1:
        return 'trend'
    if len(numeric_cols) > 1:
        return 'segment'
    # fallback: look for growth keywords
    colnames = ' '.join(map(str, df.columns)).lower()
    if 'growth' in colnames or 'yoy' in colnames or 'year over year' in colnames:
        return 'growth'
    return 'unknown'


def _normalize_to_canonical(df: pd.DataFrame, sheet_name: str, source_file: Optional[str]) -> pd.DataFrame:
    """Normalize a DataFrame into canonical long format with columns:
    date, metric, value, segment, aggregation, analysis_type, source_file
    """
    df = df.copy()
    # drop fully-empty rows/cols
    df = df.dropna(how='all')
    df = df.dropna(axis=1, how='all')

    # detect date column or date-y columns
    date_col = _detect_date_column(df)

    # if dates are in columns (wide format), transpose
    parsed_col_dates = [pd.to_datetime(c, errors='coerce') for c in df.columns]
    if date_col is None and sum([1 for p in parsed_col_dates if not pd.isna(p)]) >= 2:
        # treat columns as dates, rows as metrics/segments
        wide = df.copy()
        wide.index = wide.index.astype(str)
        melted = wide.reset_index().melt(id_vars=wide.reset_index().columns[0], var_name='date', value_name='value')
        melted.rename(columns={melted.columns[0]: 'metric'}, inplace=True)
        melted['date'] = pd.to_datetime(melted['date'], errors='coerce')
        melted['metric'] = melted['metric'].astype(str)
        melted['segment'] = melted['metric']
        aggregation = _detect_aggregation_from_columns(df.columns.tolist()) or _detect_aggregation_from_dates(melted['date'])
        melted['aggregation'] = aggregation
        melted['analysis_type'] = _classify_analysis_type(df, None)
        melted['source_file'] = source_file
        melted = melted[['date', 'metric', 'value', 'segment', 'aggregation', 'analysis_type', 'source_file']]
        return melted

    # if we have a date column, melt the other numeric columns
    if date_col is not None:
        date_series = _parse_dates_if_possible(df[date_col]) or pd.to_datetime(df[date_col], errors='coerce')
        df[date_col] = date_series
        value_cols = [c for c in df.columns if c != date_col]
        if not value_cols:
            # nothing to melt
            return pd.DataFrame(columns=['date', 'metric', 'value', 'segment', 'aggregation', 'analysis_type', 'source_file'])

        melted = df.melt(id_vars=[date_col], value_vars=value_cols, var_name='metric', value_name='value')
        melted.rename(columns={date_col: 'date'}, inplace=True)
        melted['date'] = pd.to_datetime(melted['date'], errors='coerce')
        melted['segment'] = melted['metric'].astype(str)
        aggregation = _detect_aggregation_from_columns(value_cols) or _detect_aggregation_from_dates(melted['date'])
        analysis_type = _classify_analysis_type(df, date_col)
        melted['aggregation'] = aggregation
        melted['analysis_type'] = analysis_type
        melted['source_file'] = source_file
        return melted[['date', 'metric', 'value', 'segment', 'aggregation', 'analysis_type', 'source_file']]

    # fallback: return empty canonical frame
    return pd.DataFrame(columns=['date', 'metric', 'value', 'segment', 'aggregation', 'analysis_type', 'source_file'])


def _detect_report_style_sheet(df: pd.DataFrame) -> bool:
    """Detect if a sheet uses a report-style layout (matrix format).

    Args:
        df: DataFrame to analyze.

    Returns:
        True if the sheet is report-style, False otherwise.
    """
    # Check if the first column is mostly text
    first_col_text_ratio = df.iloc[:, 0].apply(lambda x: isinstance(x, str)).mean()
    if first_col_text_ratio < 0.5:
        return False

    # Check if column headers contain years or months
    parsed_col_dates = [pd.to_datetime(c, errors='coerce') for c in df.columns]
    if sum([1 for p in parsed_col_dates if not pd.isna(p)]) >= 2:
        return True

    return False


def _reshape_report_style_sheet(df: pd.DataFrame, sheet_name: str, source_file: Optional[str]) -> pd.DataFrame:
    """Reshape a report-style sheet from wide matrix format into long format.

    Args:
        df: DataFrame to reshape.
        sheet_name: Name of the sheet.
        source_file: Name of the source file.

    Returns:
        Reshaped DataFrame with columns: date, metric, value.
    """
    df = df.copy()
    # Drop fully-empty rows/columns
    df = df.dropna(how='all')
    df = df.dropna(axis=1, how='all')

    # Assume the first column contains metrics and the rest are dates
    metrics = df.iloc[:, 0].astype(str)
    date_columns = df.columns[1:]

    melted = df.melt(id_vars=[df.columns[0]], value_vars=date_columns, var_name='date', value_name='value')
    melted.rename(columns={df.columns[0]: 'metric'}, inplace=True)
    melted['date'] = pd.to_datetime(melted['date'], errors='coerce')
    melted['metric'] = melted['metric'].astype(str)
    melted['source_file'] = source_file
    melted['sheet_name'] = sheet_name

    return melted[['date', 'metric', 'value', 'source_file', 'sheet_name']]


def _detect_matrix_style_sheet(df: pd.DataFrame) -> bool:
    """Detect if a sheet uses a matrix-style layout typical of CoStar reports.

    Args:
        df: DataFrame to analyze.

    Returns:
        True if the sheet is matrix-style, False otherwise.
    """
    # Check if the first column is mostly text
    first_col_text_ratio = df.iloc[:, 0].apply(lambda x: isinstance(x, str)).mean()
    if first_col_text_ratio < 0.5:
        return False

    # Check if remaining columns are mostly numeric
    numeric_cols_ratio = df.iloc[:, 1:].applymap(lambda x: isinstance(x, (int, float))).mean().mean()
    if numeric_cols_ratio < 0.5:
        return False

    # Check if column headers contain years, months, or time indicators
    parsed_col_dates = [pd.to_datetime(c, errors='coerce') for c in df.columns[1:]]
    if sum([1 for p in parsed_col_dates if not pd.isna(p)]) >= 2:
        return True

    return False


def _reshape_matrix_style_sheet(df: pd.DataFrame, sheet_name: str, source_file: Optional[str]) -> pd.DataFrame:
    """Reshape a matrix-style sheet into long format typical of CoStar reports.

    Args:
        df: DataFrame to reshape.
        sheet_name: Name of the sheet.
        source_file: Name of the source file.

    Returns:
        Reshaped DataFrame with columns: date, metric, value.
    """
    df = df.copy()
    # Drop fully-empty rows/columns
    df = df.dropna(how='all')
    df = df.dropna(axis=1, how='all')

    # Assume the first column contains metrics and the rest are dates
    metrics = df.iloc[:, 0].astype(str)
    date_columns = df.columns[1:]

    melted = df.melt(id_vars=[df.columns[0]], value_vars=date_columns, var_name='date', value_name='value')
    melted.rename(columns={df.columns[0]: 'metric'}, inplace=True)
    melted['date'] = pd.to_datetime(melted['date'], errors='coerce')
    melted['metric'] = melted['metric'].astype(str)
    melted['source_file'] = source_file
    melted['sheet_name'] = sheet_name

    return melted[['date', 'metric', 'value', 'source_file', 'sheet_name']]


def analyze_excel(file, source_file: Optional[str] = None) -> Dict[str, Any]:
    """Analyze an uploaded Excel file and return structured sheet information.

    Args:
        file: path or file-like object accepted by pandas.ExcelFile
        source_file: optional display name for the source file

    Returns:
        Dict with keys: 'file' and 'sheets'. 'sheets' is a list of SheetInfo as dicts.
    """
    xls = pd.ExcelFile(file)
    sheets_out: List[Dict[str, Any]] = []

    for sheet in xls.sheet_names:
        try:
            df = pd.read_excel(xls, sheet_name=sheet)
            if _detect_matrix_style_sheet(df):
                norm = _reshape_matrix_style_sheet(df, sheet, source_file)
                sheet_info = SheetInfo(
                    sheet_name=sheet,
                    analysis_type='matrix-style',
                    aggregation='Unknown',
                    rows=df.shape[0],
                    cols=df.shape[1],
                    normalized=norm.fillna('').to_dict(orient='records'),
                    source_file=source_file,
                )
                sheets_out.append(asdict(sheet_info))
            elif _detect_report_style_sheet(df):
                norm = _reshape_report_style_sheet(df, sheet, source_file)
                sheet_info = SheetInfo(
                    sheet_name=sheet,
                    analysis_type='report-style',
                    aggregation='Unknown',
                    rows=df.shape[0],
                    cols=df.shape[1],
                    normalized=norm.fillna('').to_dict(orient='records'),
                    source_file=source_file,
                )
                sheets_out.append(asdict(sheet_info))
            else:
                # Skip if no numeric values or time dimension
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if numeric_cols.empty:
                    sheets_out.append(
                        SheetInfo(
                            sheet_name=sheet,
                            analysis_type='skipped',
                            aggregation='Unknown',
                            rows=df.shape[0],
                            cols=df.shape[1],
                            normalized=[],
                            source_file=source_file,
                        ).__dict__
                    )
                    continue

                sheet_info = SheetInfo(
                    sheet_name=sheet,
                    analysis_type='unknown',
                    aggregation='Unknown',
                    rows=df.shape[0],
                    cols=df.shape[1],
                    normalized=[],
                    source_file=source_file,
                )
                sheets_out.append(asdict(sheet_info))
        except Exception as e:
            # Include diagnostic message for skipped sheets
            sheets_out.append(
                SheetInfo(
                    sheet_name=sheet,
                    analysis_type='error',
                    aggregation='Unknown',
                    rows=0,
                    cols=0,
                    normalized=[],
                    source_file=source_file,
                ).__dict__
            )
            continue

    return {'file': source_file, 'sheets': sheets_out}
