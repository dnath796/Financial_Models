from pathlib import Path

import numpy as np
import pandas as pd


DATE_CANDIDATES = {
    "date",
    "dates",
    "datetime",
    "time",
    "timestamp",
    "day",
}

VALUE_CANDIDATES = {
    "value",
    "values",
    "close",
    "price",
    "index",
    "index value",
    "adj close",
    "adj_close",
    "last",
    "level",
}


def normalize_name(name):
    return str(name).strip().lower().replace("_", " ")


def make_series_label(name):
    cleaned = str(name).strip().replace("_", " ").replace("-", " ")
    return " ".join(cleaned.split()) or "Series"


def safe_sheet_title(name, fallback="Series"):
    invalid = set(r'[]:*?/\\')
    cleaned = "".join(ch for ch in make_series_label(name) if ch not in invalid).strip()
    return (cleaned or fallback)[:31]


def read_tabular_data(data_path):
    path = Path(data_path)
    suffix = path.suffix.lower()

    if suffix in {".csv", ".txt", ".tsv"}:
        df = pd.read_csv(path, sep=None, engine="python")
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    df.columns = [str(col).strip() for col in df.columns]
    return df


def resolve_column(columns, requested_name=None, candidates=None):
    if requested_name:
        requested_norm = normalize_name(requested_name)
        for column in columns:
            if normalize_name(column) == requested_norm:
                return column
        raise ValueError(f"Column '{requested_name}' not found in {list(columns)}")

    if candidates:
        for column in columns:
            if normalize_name(column) in candidates:
                return column

    return None


def infer_date_column(df):
    candidate = resolve_column(df.columns, candidates=DATE_CANDIDATES)
    if candidate is not None:
        return candidate

    best_column = None
    best_score = 0.0
    sample_size = max(1, min(len(df), 200))

    for column in df.columns:
        parsed = pd.to_datetime(df[column], errors="coerce")
        score = parsed.notna().head(sample_size).mean()
        if score > best_score:
            best_score = score
            best_column = column

    if best_score >= 0.8:
        return best_column
    return None


def infer_value_column(df, date_column=None, requested_name=None):
    if requested_name:
        column = resolve_column(df.columns, requested_name=requested_name)
        series = pd.to_numeric(df[column], errors="coerce")
        if series.notna().sum() == 0:
            raise ValueError(f"Column '{requested_name}' does not contain numeric data.")
        return column

    candidate_cols = [col for col in df.columns if col != date_column]

    for column in candidate_cols:
        if normalize_name(column) in VALUE_CANDIDATES:
            series = pd.to_numeric(df[column], errors="coerce")
            if series.notna().sum() > 0:
                return column

    numeric_scores = []
    for column in candidate_cols:
        series = pd.to_numeric(df[column], errors="coerce")
        score = series.notna().mean()
        if score > 0:
            numeric_scores.append((score, column))

    if not numeric_scores:
        raise ValueError("No numeric value column found in the input data.")

    numeric_scores.sort(reverse=True)
    return numeric_scores[0][1]


def load_time_series(data_path, date_col=None, value_col=None):
    df = read_tabular_data(data_path)
    date_column = resolve_column(df.columns, requested_name=date_col) if date_col else infer_date_column(df)
    value_column = infer_value_column(df, date_column=date_column, requested_name=value_col)

    working_df = df.copy()

    if date_column is not None:
        working_df[date_column] = pd.to_datetime(working_df[date_column], errors="coerce")
        working_df = working_df.dropna(subset=[date_column]).sort_values(date_column)
        index = working_df[date_column]
    else:
        index = pd.RangeIndex(start=1, stop=len(working_df) + 1)

    values = pd.to_numeric(working_df[value_column], errors="coerce")
    series = pd.Series(values.to_numpy(), index=index, name=value_column).dropna()

    if date_column is not None:
        series.index.name = date_column

    if series.empty:
        raise ValueError("Selected series is empty after removing invalid rows.")

    series_label = make_series_label(value_column or Path(data_path).stem)
    metadata = {
        "date_column": date_column,
        "value_column": value_column,
        "series_label": series_label,
        "source_name": Path(data_path).stem,
    }
    return series, metadata
