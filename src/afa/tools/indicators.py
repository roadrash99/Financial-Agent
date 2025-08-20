

from typing import List, Optional
import pandas as pd
import numpy as np


__all__ = ["compute_indicators_pandas"]



SUPPORTED_INDICATORS = ["sma20", "sma50", "ema20", "rsi14", "macd", "bbands"]


def compute_indicators_pandas(
    df: pd.DataFrame,
    indicators: Optional[List[str]] = None
) -> pd.DataFrame:

    if 'Close' not in df.columns:
        raise ValueError(
            "DataFrame must contain a 'Close' column for technical indicator computation. "
            f"Available columns: {list(df.columns)}"
        )
    

    if indicators is None:
        indicators = SUPPORTED_INDICATORS.copy()
    

    invalid_indicators = set(indicators) - set(SUPPORTED_INDICATORS)
    if invalid_indicators:
        raise ValueError(
            f"Unsupported indicator(s): {sorted(invalid_indicators)}. "
            f"Supported indicators: {SUPPORTED_INDICATORS}"
        )
    

    result = df.copy()
    

    for indicator in indicators:
        if indicator == "sma20":
            result["sma20"] = _sma(result["Close"], period=20)
        elif indicator == "sma50":
            result["sma50"] = _sma(result["Close"], period=50)
        elif indicator == "ema20":
            result["ema20"] = _ema(result["Close"], span=20)
        elif indicator == "rsi14":
            result["rsi14"] = _rsi(result["Close"], period=14)
        elif indicator == "macd":
            macd_data = _macd(result["Close"])
            result["macd"] = macd_data["macd"]
            result["macd_signal"] = macd_data["macd_signal"]
            result["macd_hist"] = macd_data["macd_hist"]
        elif indicator == "bbands":
            bbands_data = _bbands(result["Close"], period=20, std_dev=2)
            result["bb_mid"] = bbands_data["bb_mid"]
            result["bb_high"] = bbands_data["bb_high"]
            result["bb_low"] = bbands_data["bb_low"]
    
    return result


def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:

    delta = series.diff()
    

    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    


    alpha = 1.0 / period
    avg_gains = gains.ewm(alpha=alpha, adjust=False).mean()
    avg_losses = losses.ewm(alpha=alpha, adjust=False).mean()
    

    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))
    

    rsi.iloc[:period] = np.nan
    
    return rsi


def _macd(series: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> dict:

    ema_fast = _ema(series, span=fast_period)
    ema_slow = _ema(series, span=slow_period)
    

    macd = ema_fast - ema_slow
    

    macd_signal = _ema(macd, span=signal_period)
    

    macd_hist = macd - macd_signal
    
    return {
        "macd": macd,
        "macd_signal": macd_signal,
        "macd_hist": macd_hist
    }


def _bbands(series: pd.Series, period: int = 20, std_dev: float = 2) -> dict:

    bb_mid = _sma(series, period=period)
    

    bb_std = series.rolling(window=period, min_periods=period).std(ddof=0)
    

    bb_high = bb_mid + (bb_std * std_dev)
    bb_low = bb_mid - (bb_std * std_dev)
    
    return {
        "bb_mid": bb_mid,
        "bb_high": bb_high,
        "bb_low": bb_low
    }
