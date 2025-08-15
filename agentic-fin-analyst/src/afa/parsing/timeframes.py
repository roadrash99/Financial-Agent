"""
Timeframe parsing module for converting natural language and compact tokens to date ranges.

Maps phrases like "last 3 months", "YTD", "past week", or tokens like "1y/3m" 
into (start, end, interval) tuples using pandas date utilities.
"""

import re
from typing import Optional

import pandas as pd

__all__ = ["resolve_timeframe", "coerce_dates"]


def coerce_dates(start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> tuple[Optional[str], Optional[str]]:
    """
    Helper function to normalize timestamps to date strings.
    
    Args:
        start: Start timestamp or None
        end: End timestamp or None
        
    Returns:
        Tuple of (start_str, end_str) where dates are formatted as 'YYYY-MM-DD' or None
        
    Examples:
        >>> import pandas as pd
        >>> ts1 = pd.Timestamp('2024-01-15')
        >>> ts2 = pd.Timestamp('2024-07-01')
        >>> coerce_dates(ts1, ts2)
        ('2024-01-15', '2024-07-01')
        >>> coerce_dates(None, ts2)
        (None, '2024-07-01')
    """
    start_str = start.strftime('%Y-%m-%d') if start is not None else None
    end_str = end.strftime('%Y-%m-%d') if end is not None else None
    return start_str, end_str


def resolve_timeframe(text: str, today: Optional[str] = None) -> tuple[Optional[str], Optional[str], str]:
    """
    Parse natural language timeframe expressions into standardized date ranges.
    
    Supports various patterns:
    - Relative: "last 3 months", "past 2 weeks"
    - Compact tokens: "1d", "5d", "1w", "1m", "3m", "6m", "9m", "1y", "2y", "3y", "5y"
    - YTD: "ytd", "year to date"
    - Simple past: "past week", "past month", "past year"
    - Absolute dates: "since 2024-01-01", "from 2024-01-15 to 2024-07-01"
    
    Args:
        text: Input text containing timeframe expressions
        today: Optional date string 'YYYY-MM-DD' for "today" reference. 
               If None, uses current date.
    
    Returns:
        Tuple of (start_date, end_date, interval) where:
        - start_date: 'YYYY-MM-DD' string or None
        - end_date: 'YYYY-MM-DD' string or None  
        - interval: one of "1d", "1wk", "1mo"
    
    Examples:
        >>> resolve_timeframe("last 3 months")
        ('2024-01-15', '2024-04-15', '1d')
        
        >>> resolve_timeframe("ytd on nvda")
        ('2024-01-01', '2024-04-15', '1d')
        
        >>> resolve_timeframe("2y on AAPL")
        ('2022-04-15', '2024-04-15', '1wk')
        
        >>> resolve_timeframe("from 2024-01-15 to 2024-07-01")
        ('2024-01-15', '2024-07-01', '1d')
        
        >>> resolve_timeframe("")  # defaults
        ('2023-10-15', '2024-04-15', '1d')
    """
    # Normalize today reference
    if today is None:
        today_ts = pd.Timestamp.today().normalize()
    else:
        today_ts = pd.Timestamp(today).normalize()
    
    # Initialize result variables
    start_ts = None
    end_ts = None
    
    # Convert to lowercase for case-insensitive matching
    text_lower = text.lower().strip()
    
    # Pattern 1: Relative "last/past N units"
    # Matches: "last 3 months", "past 2 weeks", "LAST 1 YEAR"
    relative_pattern = r"\b(last|past)\s+(\d{1,3})\s*(day|days|week|weeks|month|months|year|years)\b"
    relative_match = re.search(relative_pattern, text_lower)
    
    if relative_match:
        _, num_str, unit = relative_match.groups()
        num = int(num_str)
        
        # Map units to DateOffset parameters
        if unit.startswith('day'):
            offset = pd.DateOffset(days=num)
        elif unit.startswith('week'):
            offset = pd.DateOffset(weeks=num)
        elif unit.startswith('month'):
            offset = pd.DateOffset(months=num)
        elif unit.startswith('year'):
            offset = pd.DateOffset(years=num)
        else:
            offset = pd.DateOffset(days=num)  # fallback
        
        start_ts = today_ts - offset
        end_ts = today_ts
    
    # Pattern 2: Compact tokens
    # Matches: "1d", "5d", "1w", "1m", "3m", "6m", "9m", "1y", "2y", "3y", "5y"
    if start_ts is None:  # Only if not already matched
        compact_pattern = r"\b(1d|5d|1w|1m|3m|6m|9m|1y|2y|3y|5y)\b"
        compact_match = re.search(compact_pattern, text_lower)
        
        if compact_match:
            token = compact_match.group(1)
            
            # Map compact tokens to date offsets
            token_map = {
                '1d': pd.DateOffset(days=1),
                '5d': pd.DateOffset(days=5),
                '1w': pd.DateOffset(weeks=1),
                '1m': pd.DateOffset(months=1),
                '3m': pd.DateOffset(months=3),
                '6m': pd.DateOffset(months=6),
                '9m': pd.DateOffset(months=9),
                '1y': pd.DateOffset(years=1),
                '2y': pd.DateOffset(years=2),
                '3y': pd.DateOffset(years=3),
                '5y': pd.DateOffset(years=5)
            }
            
            if token in token_map:
                start_ts = today_ts - token_map[token]
                end_ts = today_ts
    
    # Pattern 3: YTD / Year to date
    # Matches: "ytd", "YTD", "year to date"
    if start_ts is None:
        ytd_pattern = r"\b(ytd|year\s+to\s+date)\b"
        if re.search(ytd_pattern, text_lower):
            start_ts = pd.Timestamp(today_ts.year, 1, 1)
            end_ts = today_ts
    
    # Pattern 4: Simple past patterns (no number)
    # Matches: "past week", "past month", "past year"
    if start_ts is None:
        simple_past_pattern = r"\bpast\s+(week|month|year)\b"
        simple_past_match = re.search(simple_past_pattern, text_lower)
        
        if simple_past_match:
            unit = simple_past_match.group(1)
            
            if unit == 'week':
                offset = pd.DateOffset(weeks=1)
            elif unit == 'month':
                offset = pd.DateOffset(months=1)
            elif unit == 'year':
                offset = pd.DateOffset(years=1)
            else:
                offset = pd.DateOffset(weeks=1)  # fallback
            
            start_ts = today_ts - offset
            end_ts = today_ts
    
    # Pattern 5: Absolute ISO dates
    if start_ts is None:
        # "since YYYY-MM-DD"
        since_pattern = r"\bsince\s+(\d{4}-\d{2}-\d{2})\b"
        since_match = re.search(since_pattern, text_lower)
        
        if since_match:
            start_ts = pd.Timestamp(since_match.group(1))
            end_ts = today_ts
        else:
            # "from YYYY-MM-DD to YYYY-MM-DD"
            from_to_pattern = r"\bfrom\s+(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})\b"
            from_to_match = re.search(from_to_pattern, text_lower)
            
            if from_to_match:
                start_ts = pd.Timestamp(from_to_match.group(1))
                end_ts = pd.Timestamp(from_to_match.group(2))
    
    # Apply defaults if nothing matched
    if start_ts is None and end_ts is None:
        start_ts = today_ts - pd.DateOffset(months=6)
        end_ts = today_ts
    
    # Ensure we have both start and end dates
    if start_ts is None:
        start_ts = today_ts - pd.DateOffset(months=6)
    if end_ts is None:
        end_ts = today_ts
    
    # Swap if start > end (handle reversed dates)
    if start_ts > end_ts:
        start_ts, end_ts = end_ts, start_ts
    
    # Determine interval based on window length heuristic
    delta_days = (end_ts - start_ts).days
    
    if delta_days > 365 * 5:  # > 5 years
        interval = "1mo"
    elif delta_days > 365 * 2:  # > 2 years
        interval = "1wk"
    else:  # <= 2 years
        interval = "1d"
    
    # Convert to normalized date strings
    start_str, end_str = coerce_dates(start_ts, end_ts)
    
    return start_str, end_str, interval