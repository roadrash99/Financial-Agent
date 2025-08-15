"""Technical indicators computation using pandas and numpy."""

from typing import List, Optional
import pandas as pd
import numpy as np


__all__ = ["compute_indicators_pandas"]


# Supported indicator IDs
SUPPORTED_INDICATORS = ["sma20", "sma50", "ema20", "rsi14", "macd", "bbands"]


def compute_indicators_pandas(
    df: pd.DataFrame,
    indicators: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compute technical indicators and append them as new columns to the DataFrame.
    
    Args:
        df: DataFrame with OHLCV data. Must contain 'Close' column.
        indicators: List of indicator IDs to compute. If None, computes all supported indicators.
                   Supported: ["sma20", "sma50", "ema20", "rsi14", "macd", "bbands"]
    
    Returns:
        DataFrame with original OHLCV data plus new indicator columns.
        Warm-up NaNs are preserved (no forward-filling).
    
    Raises:
        ValueError: If 'Close' column is missing from the DataFrame.
    
    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> 
        >>> # Create sample OHLCV data
        >>> dates = pd.date_range('2024-01-01', periods=100, freq='D')
        >>> df = pd.DataFrame({
        ...     'Open': np.random.randn(100).cumsum() + 100,
        ...     'High': np.random.randn(100).cumsum() + 102,
        ...     'Low': np.random.randn(100).cumsum() + 98,
        ...     'Close': np.random.randn(100).cumsum() + 100,
        ...     'Volume': np.random.randint(1000000, 10000000, 100)
        ... }, index=dates)
        >>> 
        >>> # Compute all indicators
        >>> result = compute_indicators_pandas(df)
        >>> print(result.columns.tolist())
        ['Open', 'High', 'Low', 'Close', 'Volume', 'sma20', 'sma50', 'ema20', 'rsi14', 'macd', 'macd_signal', 'macd_hist', 'bb_mid', 'bb_high', 'bb_low']
        >>> 
        >>> # Compute specific indicators
        >>> result = compute_indicators_pandas(df, indicators=['sma20', 'rsi14'])
        >>> print(result.columns.tolist())
        ['Open', 'High', 'Low', 'Close', 'Volume', 'sma20', 'rsi14']
        >>> 
        >>> # Verify MACD histogram calculation
        >>> result = compute_indicators_pandas(df, indicators=['macd'])
        >>> assert (result['macd_hist'] == result['macd'] - result['macd_signal']).all()
    """
    # Validate input
    if 'Close' not in df.columns:
        raise ValueError(
            "DataFrame must contain a 'Close' column for technical indicator computation. "
            f"Available columns: {list(df.columns)}"
        )
    
    # Use all indicators if none specified
    if indicators is None:
        indicators = SUPPORTED_INDICATORS.copy()
    
    # Validate indicator IDs
    invalid_indicators = set(indicators) - set(SUPPORTED_INDICATORS)
    if invalid_indicators:
        raise ValueError(
            f"Unsupported indicator(s): {sorted(invalid_indicators)}. "
            f"Supported indicators: {SUPPORTED_INDICATORS}"
        )
    
    # Create a copy to avoid modifying the original DataFrame
    result = df.copy()
    
    # Compute indicators based on requested list
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
    """
    Calculate Simple Moving Average.
    
    Args:
        series: Price series (typically Close prices)
        period: Number of periods for the moving average
    
    Returns:
        Series with SMA values. First (period-1) values will be NaN.
    """
    return series.rolling(window=period, min_periods=period).mean()


def _ema(series: pd.Series, span: int) -> pd.Series:
    """
    Calculate Exponential Moving Average.
    
    Args:
        series: Price series (typically Close prices)
        span: Span for the EMA calculation
    
    Returns:
        Series with EMA values.
    """
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    
    Uses exponential weighted moving average for smoothing gains and losses
    with alpha = 1/period.
    
    Args:
        series: Price series (typically Close prices)
        period: Number of periods for RSI calculation (default 14)
    
    Returns:
        Series with RSI values (0-100). First period values will be NaN.
    """
    # Calculate price changes
    delta = series.diff()
    
    # Separate gains and losses
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    
    # Calculate exponential weighted moving averages
    # alpha = 1/period for traditional RSI calculation
    alpha = 1.0 / period
    avg_gains = gains.ewm(alpha=alpha, adjust=False).mean()
    avg_losses = losses.ewm(alpha=alpha, adjust=False).mean()
    
    # Calculate RS and RSI
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))
    
    # Set first period values to NaN to match warm-up behavior
    rsi.iloc[:period] = np.nan
    
    return rsi


def _macd(series: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> dict:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Args:
        series: Price series (typically Close prices)
        fast_period: Fast EMA period (default 12)
        slow_period: Slow EMA period (default 26)
        signal_period: Signal line EMA period (default 9)
    
    Returns:
        Dictionary with 'macd', 'macd_signal', and 'macd_hist' Series.
    """
    # Calculate fast and slow EMAs
    ema_fast = _ema(series, span=fast_period)
    ema_slow = _ema(series, span=slow_period)
    
    # Calculate MACD line
    macd = ema_fast - ema_slow
    
    # Calculate signal line (EMA of MACD)
    macd_signal = _ema(macd, span=signal_period)
    
    # Calculate histogram
    macd_hist = macd - macd_signal
    
    return {
        "macd": macd,
        "macd_signal": macd_signal,
        "macd_hist": macd_hist
    }


def _bbands(series: pd.Series, period: int = 20, std_dev: float = 2) -> dict:
    """
    Calculate Bollinger Bands.
    
    Args:
        series: Price series (typically Close prices)
        period: Number of periods for moving average and standard deviation (default 20)
        std_dev: Number of standard deviations for the bands (default 2)
    
    Returns:
        Dictionary with 'bb_mid', 'bb_high', and 'bb_low' Series.
    """
    # Calculate middle band (SMA)
    bb_mid = _sma(series, period=period)
    
    # Calculate standard deviation (using ddof=0 as specified)
    bb_std = series.rolling(window=period, min_periods=period).std(ddof=0)
    
    # Calculate upper and lower bands
    bb_high = bb_mid + (bb_std * std_dev)
    bb_low = bb_mid - (bb_std * std_dev)
    
    return {
        "bb_mid": bb_mid,
        "bb_high": bb_high,
        "bb_low": bb_low
    }
