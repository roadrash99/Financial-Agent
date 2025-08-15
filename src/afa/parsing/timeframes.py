# src/afa/parsing/timeframes.py
from __future__ import annotations

"""
Deterministic timeframe parsing (no LLM).

Converts free-form text like:
  - "last 3 months", "past week", "YTD"
  - "3m", "1y", "5y"
  - "since 2024-01-15", "from 2024-01-15 to 2024-07-01"
into a normalized (start, end, interval) tuple:

    start: 'YYYY-MM-DD' | None
    end:   'YYYY-MM-DD' | None
    interval: one of {'1d', '1wk', '1mo'}

Rules:
- Defaults to last 6 months at daily ('1d') if nothing is recognized.
- For relative phrases, end = today (normalized date).
- If start > end is parsed, the dates are swapped.
- Interval heuristic based on window length:
    > 5 years  → '1mo'
    > 2 years  → '1wk'
    else       → '1d'

Examples:
    >>> resolve_timeframe("last 3 months", today="2025-08-15")
    ('2025-05-15', '2025-08-15', '1d')

    >>> resolve_timeframe("YTD on NVDA", today="2025-08-15")
    ('2025-01-01', '2025-08-15', '1d')

    >>> resolve_timeframe("past 2 years SPY", today="2025-08-15")
    ('2023-08-15', '2025-08-15', '1wk')

    >>> resolve_timeframe("from 2024-01-15 to 2024-07-01")
    ('2024-01-15', '2024-07-01', '1d')
"""

from typing import Optional, Tuple
import re

import pandas as pd

__all__ = ["resolve_timeframe", "coerce_dates"]

# --- Regexes (compiled once) ---

# last/past N units
_RELATIVE_RE = re.compile(
    r"\b(?:last|past)\s+(\d{1,3})\s*(day|days|week|weeks|month|months|year|years)\b",
    re.IGNORECASE,
)

# "past week/month/year" (no number)
_SINGLE_UNIT_RE = re.compile(
    r"\b(?:last|past)\s*(day|week|month|year)\b", re.IGNORECASE
)

# compact tokens like 1d,5d,1w,1m,3m,6m,9m,1y,2y,5y
_COMPACT_RE = re.compile(r"\b(1d|5d|1w|1m|3m|6m|9m|1y|2y|5y)\b", re.IGNORECASE)

# ytd / year to date
_YTD_RE = re.compile(r"\b(ytd|year\s*to\s*date)\b", re.IGNORECASE)

# absolute ranges
_FROM_TO_RE = re.compile(
    r"\b(?:from\s+)?(\d{4}-\d{2}-\d{2})\s*(?:to|-|–)\s*(\d{4}-\d{2}-\d{2})\b",
    re.IGNORECASE,
)
_SINCE_RE = re.compile(
    r"\b(?:since|from)\s+(\d{4}-\d{2}-\d{2})\b", re.IGNORECASE
)


def _norm_ts(x: pd.Timestamp | None) -> Optional[pd.Timestamp]:
    """Normalize to date (00:00) or return None."""
    if x is None:
        return None
    return pd.Timestamp(x).normalize()


def coerce_dates(
    start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]
) -> Tuple[Optional[str], Optional[str]]:
    """
    Normalize and format to 'YYYY-MM-DD'. Returns (start_str|None, end_str|None).
    """
    s = _norm_ts(start)
    e = _norm_ts(end)
    if s is not None and e is not None and s > e:
        s, e = e, s  # swap if reversed
    s_str = s.strftime("%Y-%m-%d") if s is not None else None
    e_str = e.strftime("%Y-%m-%d") if e is not None else None
    return s_str, e_str


def _infer_interval(start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> str:
    """Choose '1d' | '1wk' | '1mo' based on window length."""
    if start is None or end is None:
        return "1d"
    delta_days = int((end - start).days)
    if delta_days > 365 * 5:
        return "1mo"
    if delta_days > 365 * 2:
        return "1wk"
    return "1d"


def _apply_offset(today: pd.Timestamp, **kwargs) -> pd.Timestamp:
    """today - DateOffset(**kwargs), normalized."""
    return _norm_ts(today - pd.DateOffset(**kwargs))  # type: ignore[arg-type]


def _parse_relative(text: str, today: pd.Timestamp) -> tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """Parse 'last/past N units' and 'past {unit}' (no number)."""
    m = _RELATIVE_RE.search(text)
    if m:
        n = int(m.group(1))
        unit = m.group(2).lower()
        if "day" in unit:
            start = _apply_offset(today, days=n)
        elif "week" in unit:
            start = _apply_offset(today, weeks=n)
        elif "month" in unit:
            start = _apply_offset(today, months=n)
        else:  # years
            start = _apply_offset(today, years=n)
        return start, today

    m2 = _SINGLE_UNIT_RE.search(text)
    if m2:
        unit = m2.group(1).lower()
        if unit == "day":
            start = _apply_offset(today, days=1)
        elif unit == "week":
            start = _apply_offset(today, weeks=1)
        elif unit == "month":
            start = _apply_offset(today, months=1)
        else:
            start = _apply_offset(today, years=1)
        return start, today

    return None, None


def _parse_compact(text: str, today: pd.Timestamp) -> tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """Parse compact tokens like 3m, 1y, 5y, 1w, 1m, 1d, 5d."""
    m = _COMPACT_RE.search(text)
    if not m:
        return None, None

    token = m.group(1).lower()
    start: Optional[pd.Timestamp] = None

    if token.endswith("d"):
        days = int(token[:-1])
        start = _apply_offset(today, days=days)
    elif token.endswith("w"):
        weeks = int(token[:-1])
        start = _apply_offset(today, weeks=weeks)
    elif token.endswith("m"):
        months = int(token[:-1])
        start = _apply_offset(today, months=months)
    elif token.endswith("y"):
        years = int(token[:-1])
        start = _apply_offset(today, years=years)

    return start, today if start is not None else (None, None)


def _parse_ytd(text: str, today: pd.Timestamp) -> tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """Parse YTD / Year to date."""
    if _YTD_RE.search(text):
        start = pd.Timestamp(year=today.year, month=1, day=1)
        return _norm_ts(start), today
    return None, None


def _parse_absolute(text: str, today: pd.Timestamp) -> tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """
    Parse absolute date ranges:
      - 'from YYYY-MM-DD to YYYY-MM-DD' (or 'YYYY-MM-DD to YYYY-MM-DD')
      - 'since YYYY-MM-DD' / 'from YYYY-MM-DD'
    """
    m = _FROM_TO_RE.search(text)
    if m:
        s = pd.to_datetime(m.group(1), errors="coerce")
        e = pd.to_datetime(m.group(2), errors="coerce")
        return _norm_ts(s), _norm_ts(e)

    m2 = _SINCE_RE.search(text)
    if m2:
        s = pd.to_datetime(m2.group(1), errors="coerce")
        return _norm_ts(s), today

    return None, None


def resolve_timeframe(
    text: str, today: str | pd.Timestamp | None = None
) -> tuple[Optional[str], Optional[str], str]:
    """
    Resolve (start, end, interval) from free-form text.

    Args:
        text: user question or instruction.
        today: optional anchor date ('YYYY-MM-DD' or Timestamp). Defaults to today().

    Returns:
        (start_str|None, end_str|None, interval_str)

    Resolution order (first match wins):
      1) Absolute ranges ("from 2024-01-15 to 2024-07-01"; "since 2024-01-15")
      2) YTD / Year to date
      3) Relative phrases ("last 3 months", "past week")
      4) Compact tokens ("3m", "1y", "5y")
      5) Default to last 6 months

    Interval heuristic runs after dates are chosen.
    """
    # Normalize 'today'
    if today is None:
        t0 = pd.Timestamp.today().normalize()
    else:
        t0 = _norm_ts(pd.to_datetime(today, errors="coerce"))  # type: ignore[arg-type]
        if t0 is None:
            t0 = pd.Timestamp.today().normalize()

    # Try absolute first
    s, e = _parse_absolute(text, t0)
    if s is None and e is None:
        # YTD
        s, e = _parse_ytd(text, t0)
    if s is None and e is None:
        # Relative (last/past N units or single-unit)
        s, e = _parse_relative(text, t0)
    if s is None and e is None:
        # Compact tokens
        s, e = _parse_compact(text, t0)

    # Default to last 6 months if still nothing
    if s is None and e is None:
        s = _apply_offset(t0, months=6)
        e = t0

    # If one side missing, fill with today
    if s is not None and e is None:
        e = t0
    if e is not None and s is None:
        # if only end is present (rare), back off 6 months
        s = _apply_offset(e, months=6)

    # Normalize/format and choose interval
    start_str, end_str = coerce_dates(s, e)
    # Convert back to Timestamp for interval heuristic
    s_ts = pd.to_datetime(start_str) if start_str else None
    e_ts = pd.to_datetime(end_str) if end_str else None
    interval = _infer_interval(s_ts, e_ts)

    return start_str, end_str, interval
