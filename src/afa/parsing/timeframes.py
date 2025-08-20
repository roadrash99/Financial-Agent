from __future__ import annotations

from typing import Optional, Tuple
import re

import pandas as pd

__all__ = ["resolve_timeframe", "coerce_dates"]


_RELATIVE_RE = re.compile(
    r"\b(?:last|past)\s+(\d{1,3})\s*(day|days|week|weeks|month|months|year|years)\b",
    re.IGNORECASE,
)

_SINGLE_UNIT_RE = re.compile(
    r"\b(?:last|past)\s*(day|week|month|year)\b", re.IGNORECASE
)

_COMPACT_RE = re.compile(r"\b(1d|5d|1w|1m|3m|6m|9m|1y|2y|5y)\b", re.IGNORECASE)

_YTD_RE = re.compile(r"\b(ytd|year\s*to\s*date)\b", re.IGNORECASE)

_FROM_TO_RE = re.compile(
    r"\b(?:from\s+)?(\d{4}-\d{2}-\d{2})\s*(?:to|-|–)\s*(\d{4}-\d{2}-\d{2})\b",
    re.IGNORECASE,
)
_SINCE_RE = re.compile(
    r"\b(?:since|from)\s+(\d{4}-\d{2}-\d{2})\b", re.IGNORECASE
)


def _norm_ts(x: pd.Timestamp | None) -> Optional[pd.Timestamp]:
    if x is None:
        return None
    return pd.Timestamp(x).normalize()


def coerce_dates(
    start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]
) -> Tuple[Optional[str], Optional[str]]:
    s = _norm_ts(start)
    e = _norm_ts(end)
    if s is not None and e is not None and s > e:
        s, e = e, s
    s_str = s.strftime("%Y-%m-%d") if s is not None else None
    e_str = e.strftime("%Y-%m-%d") if e is not None else None
    return s_str, e_str


def _infer_interval(start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> str:
    if start is None or end is None:
        return "1d"
    delta_days = int((end - start).days)
    if delta_days > 365 * 5:
        return "1mo"
    if delta_days > 365 * 2:
        return "1wk"
    return "1d"


def _apply_offset(today: pd.Timestamp, **kwargs) -> pd.Timestamp:
    return _norm_ts(today - pd.DateOffset(**kwargs))


def _parse_relative(text: str, today: pd.Timestamp) -> tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
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
        else:
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
    if _YTD_RE.search(text):
        start = pd.Timestamp(year=today.year, month=1, day=1)
        return _norm_ts(start), today
    return None, None


def _parse_absolute(text: str, today: pd.Timestamp) -> tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
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

    if today is None:
        t0 = pd.Timestamp.today().normalize()
    else:
        t0 = _norm_ts(pd.to_datetime(today, errors="coerce"))
        if t0 is None:
            t0 = pd.Timestamp.today().normalize()


    s, e = _parse_absolute(text, t0)
    if s is None and e is None:

        s, e = _parse_ytd(text, t0)
    if s is None and e is None:

        s, e = _parse_relative(text, t0)
    if s is None and e is None:

        s, e = _parse_compact(text, t0)


    if s is None and e is None:
        s = _apply_offset(t0, months=6)
        e = t0


    if s is not None and e is None:
        e = t0
    if e is not None and s is None:

        s = _apply_offset(e, months=6)


    start_str, end_str = coerce_dates(s, e)

    s_ts = pd.to_datetime(start_str) if start_str else None
    e_ts = pd.to_datetime(end_str) if end_str else None
    interval = _infer_interval(s_ts, e_ts)

    return start_str, end_str, interval
