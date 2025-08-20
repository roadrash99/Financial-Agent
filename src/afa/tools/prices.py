

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

    normalized_tickers = list(dict.fromkeys([ticker.upper() for ticker in tickers]))[:max_tickers]
    
    if not normalized_tickers:
        return {}
    

    expected_columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    

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

        raw_data = yf.download(normalized_tickers, **download_kwargs)
    except Exception:

        result = {}
        for ticker in normalized_tickers:
            empty_df = pd.DataFrame(columns=expected_columns)
            empty_df.index = pd.DatetimeIndex([], name="Date")
            result[ticker] = empty_df
        return result
    
    result = {}
    

    if isinstance(raw_data.columns, pd.MultiIndex):

        for ticker in normalized_tickers:
            try:
                if raw_data.empty or ticker not in [col[1] for col in raw_data.columns if isinstance(col, tuple)]:

                    empty_df = pd.DataFrame(columns=expected_columns)
                    empty_df.index = pd.DatetimeIndex([], name="Date")
                    result[ticker] = empty_df
                else:

                    ticker_data = raw_data.xs(ticker, level=1, axis=1)
                    df = _clean_dataframe(ticker_data, expected_columns)
                    result[ticker] = df
            except (KeyError, IndexError):

                empty_df = pd.DataFrame(columns=expected_columns)
                empty_df.index = pd.DatetimeIndex([], name="Date")
                result[ticker] = empty_df
    else:

        if len(normalized_tickers) == 1:
            ticker = normalized_tickers[0]
            df = _clean_dataframe(raw_data, expected_columns)
            result[ticker] = df
        else:

            for ticker in normalized_tickers:
                empty_df = pd.DataFrame(columns=expected_columns)
                empty_df.index = pd.DatetimeIndex([], name="Date")
                result[ticker] = empty_df
    

    for ticker in normalized_tickers:
        if ticker not in result:
            empty_df = pd.DataFrame(columns=expected_columns)
            empty_df.index = pd.DatetimeIndex([], name="Date")
            result[ticker] = empty_df
    
    return result


def _clean_dataframe(df: pd.DataFrame, expected_columns: List[str]) -> pd.DataFrame:
    if df.empty:

        clean_df = pd.DataFrame(columns=expected_columns)
        clean_df.index = pd.DatetimeIndex([], name="Date")
        return clean_df
    

    clean_df = df.copy()
    

    if clean_df.index.name != "Date":
        clean_df.index.name = "Date"
   
    if hasattr(clean_df.index, 'tz') and clean_df.index.tz is not None:
        clean_df.index = clean_df.index.tz_localize(None)
    
    
    for col in expected_columns:
        if col not in clean_df.columns:
            clean_df[col] = pd.NA
    
    clean_df = clean_df.reindex(columns=expected_columns)
   
    clean_df = clean_df.sort_index()
    
    
    ohlcv_columns = ["Open", "High", "Low", "Close", "Volume"]
    clean_df = clean_df.dropna(subset=ohlcv_columns, how="all")
    
    return clean_df
