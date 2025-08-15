"""Price data fetching utilities using yfinance."""

from typing import Dict, List, Optional, Union
import pandas as pd
import yfinance as yf


__all__ = ["fetch_prices"]


def fetch_prices(
    tickers: List[str],
    start: Optional[str],
    end: Optional[str],
    interval: str = "1d",
    max_tickers: int = 5,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch OHLCV price data for given tickers using yfinance.
    
    Args:
        tickers: List of ticker symbols (e.g., ['AAPL', 'MSFT'])
        start: Start date in YYYY-MM-DD format or None
        end: End date in YYYY-MM-DD format or None
        interval: Data interval (default "1d"). Options: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        max_tickers: Maximum number of tickers to process (default 5)
    
    Returns:
        Dictionary mapping ticker symbols to DataFrames with OHLCV data.
        Each DataFrame has DatetimeIndex and columns: ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        Invalid tickers return empty DataFrames with correct column structure.
    
    Examples:
        >>> # Fetch 6 months of daily data for single ticker
        >>> data = fetch_prices(["AAPL"], None, None)
        >>> print(data["AAPL"].head())
                      Open    High     Low   Close  Adj Close    Volume
        Date                                                          
        2024-01-01  100.0   105.0    99.0   104.0      104.0  1000000
        
        >>> # Fetch specific date range for multiple tickers
        >>> data = fetch_prices(["AAPL", "MSFT"], "2024-01-01", "2024-02-01")
        >>> print(list(data.keys()))
        ['AAPL', 'MSFT']
        
        >>> # Invalid ticker returns empty DataFrame with correct structure
        >>> data = fetch_prices(["INVALID_TICKER"], None, None)
        >>> print(data["INVALID_TICKER"].columns.tolist())
        ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    """
    # Normalize tickers: uppercase, deduplicate, and cap at max_tickers
    normalized_tickers = list(dict.fromkeys([ticker.upper() for ticker in tickers]))[:max_tickers]
    
    if not normalized_tickers:
        return {}
    
    # Define expected columns for consistent output
    expected_columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    
    # Determine download parameters
    download_kwargs = {
        "auto_adjust": True,
        "progress": False,
        "threads": True,
        "interval": interval,
    }
    
    if start is None and end is None:
        download_kwargs["period"] = "6mo"
    else:
        if start is not None:
            download_kwargs["start"] = start
        if end is not None:
            download_kwargs["end"] = end
    
    try:
        # Download data for all tickers
        raw_data = yf.download(normalized_tickers, **download_kwargs)
    except Exception:
        # If download fails completely, return empty DataFrames for all tickers
        result = {}
        for ticker in normalized_tickers:
            empty_df = pd.DataFrame(columns=expected_columns)
            empty_df.index = pd.DatetimeIndex([], name="Date")
            result[ticker] = empty_df
        return result
    
    result = {}
    
    # Check if we have MultiIndex columns (happens with multiple tickers or single ticker with auto_adjust=True)
    if isinstance(raw_data.columns, pd.MultiIndex):
        # MultiIndex columns case - split into per-ticker DataFrames
        for ticker in normalized_tickers:
            try:
                if raw_data.empty or ticker not in [col[1] for col in raw_data.columns if isinstance(col, tuple)]:
                    # Ticker not found in response or empty data
                    empty_df = pd.DataFrame(columns=expected_columns)
                    empty_df.index = pd.DatetimeIndex([], name="Date")
                    result[ticker] = empty_df
                else:
                    # Extract data for this ticker
                    ticker_data = raw_data.xs(ticker, level=1, axis=1)
                    df = _clean_dataframe(ticker_data, expected_columns)
                    result[ticker] = df
            except (KeyError, IndexError):
                # Handle case where ticker data is not available
                empty_df = pd.DataFrame(columns=expected_columns)
                empty_df.index = pd.DatetimeIndex([], name="Date")
                result[ticker] = empty_df
    else:
        # Simple columns case (single ticker without auto_adjust)
        if len(normalized_tickers) == 1:
            ticker = normalized_tickers[0]
            df = _clean_dataframe(raw_data, expected_columns)
            result[ticker] = df
        else:
            # This shouldn't happen but handle gracefully
            for ticker in normalized_tickers:
                empty_df = pd.DataFrame(columns=expected_columns)
                empty_df.index = pd.DatetimeIndex([], name="Date")
                result[ticker] = empty_df
    
    # Ensure all requested tickers are in result (add empty DataFrames for missing ones)
    for ticker in normalized_tickers:
        if ticker not in result:
            empty_df = pd.DataFrame(columns=expected_columns)
            empty_df.index = pd.DatetimeIndex([], name="Date")
            result[ticker] = empty_df
    
    return result


def _clean_dataframe(df: pd.DataFrame, expected_columns: List[str]) -> pd.DataFrame:
    """
    Clean and normalize a DataFrame with OHLCV data.
    
    Args:
        df: Raw DataFrame from yfinance
        expected_columns: List of expected column names
    
    Returns:
        Cleaned DataFrame with consistent structure
    """
    if df.empty:
        # Return empty DataFrame with correct structure
        clean_df = pd.DataFrame(columns=expected_columns)
        clean_df.index = pd.DatetimeIndex([], name="Date")
        return clean_df
    
    # Make a copy to avoid modifying original
    clean_df = df.copy()
    
    # Ensure index is named "Date"
    if clean_df.index.name != "Date":
        clean_df.index.name = "Date"
    
    # Convert timezone-aware index to timezone-naive
    if hasattr(clean_df.index, 'tz') and clean_df.index.tz is not None:
        clean_df.index = clean_df.index.tz_localize(None)
    
    # Ensure all expected columns exist (add missing ones with NaN)
    for col in expected_columns:
        if col not in clean_df.columns:
            clean_df[col] = pd.NA
    
    # Reorder columns to match expected order
    clean_df = clean_df.reindex(columns=expected_columns)
    
    # Sort by date index
    clean_df = clean_df.sort_index()
    
    # Drop rows where all OHLCV values are NaN
    # Note: We check all columns since Volume might be missing for some instruments
    ohlcv_columns = ["Open", "High", "Low", "Close", "Volume"]
    clean_df = clean_df.dropna(subset=ohlcv_columns, how="all")
    
    return clean_df
