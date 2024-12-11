import logging
from datetime import datetime
from typing import Any, Optional

import pandas as pd


def is_date_column(series: pd.Series) -> bool:
    """Check if a pandas Series likely contains date values

    Args:
        series: pandas Series to check

    Returns:
        bool: True if series likely contains dates, False otherwise
    """
    # Skip if series is empty
    if series.empty:
        return False

    # Get non-null values
    sample = series.dropna().head(100)
    if sample.empty:
        return False

    # Common date patterns
    date_patterns = [
        r"\d{4}-\d{2}-\d{2}",  # YYYY-MM-DD
        r"\d{2}-\d{2}-\d{4}",  # DD-MM-YYYY or MM-DD-YYYY
        r"\d{2}/\d{2}/\d{4}",  # DD/MM/YYYY or MM/DD/YYYY
        r"\d{4}/\d{2}/\d{2}",  # YYYY/MM/DD
        r"\d{2}\.\d{2}\.\d{4}",  # DD.MM.YYYY or MM.DD.YYYY
        r"\d{4}\.\d{2}\.\d{2}",  # YYYY.MM.DD
    ]

    # Check if any values match date patterns
    pattern = "|".join(date_patterns)
    matches = sample.astype(str).str.match(pattern)
    match_ratio = matches.mean() if not matches.empty else 0

    return match_ratio > 0.8  # Return True if >80% of values match date patterns


def convert_datetime_series(series: pd.Series) -> pd.Series:
    """Convert a series of values to datetime using vectorized operations

    Args:
        series: pandas Series to convert

    Returns:
        pandas Series with ISO format datetime strings
    """
    try:
        # Convert to datetime
        result = pd.to_datetime(series, infer_datetime_format=True, errors="coerce")
        # Convert to ISO format strings
        return result.dt.strftime("%Y-%m-%dT%H:%M:%S")
    except Exception as e:
        logging.warning(f"Initial datetime conversion failed: {str(e)}")
        return series
