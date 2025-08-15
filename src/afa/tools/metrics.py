"""Financial metrics computation for summarizing price and indicator data."""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import math


__all__ = ["summarize_metrics"]


def summarize_metrics(
    df: pd.DataFrame,
    interval: str = "1d"
) -> Dict[str, Any]:
    """
    Produce a compact numeric digest of financial metrics from price data and indicators.
    
    Args:
        df: DataFrame with OHLCV data and optional technical indicators.
            Must contain 'Close' column if non-empty.
        interval: Data interval (default "1d"). Used for annualized volatility calculation.
    
    Returns:
        Dictionary containing:
        - period_start: First date as 'YYYY-MM-DD' string
        - period_end: Last date as 'YYYY-MM-DD' string
        - period_return: Total return (last_close/first_close - 1)
        - annualized_vol: Annualized volatility (only for 1d interval with ≥2 returns)
        - max_drawdown: Maximum drawdown (≤ 0)
        - trend_slope: OLS slope of Close vs index
        - rsi_last: Last RSI value if available
        - macd_state: MACD state classification if available
        - bb_position: Bollinger Band position if available
        - note: Additional information for edge cases
    
    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> 
        >>> # Create sample data
        >>> dates = pd.date_range('2024-01-01', periods=30, freq='D')
        >>> df = pd.DataFrame({
        ...     'Close': np.random.randn(30).cumsum() + 100,
        ...     'rsi14': np.random.uniform(20, 80, 30)
        ... }, index=dates)
        >>> 
        >>> # Get metrics summary
        >>> metrics = summarize_metrics(df)
        >>> print(metrics.keys())
        dict_keys(['period_start', 'period_end', 'period_return', 'annualized_vol', 'max_drawdown', 'trend_slope', 'rsi_last', 'macd_state', 'bb_position'])
        >>> 
        >>> # Check for insufficient data
        >>> empty_df = pd.DataFrame({'Close': [100]}, index=[pd.Timestamp('2024-01-01')])
        >>> metrics = summarize_metrics(empty_df)
        >>> print(metrics['note'])
        'insufficient data'
    """
    # Initialize result with basic structure
    result = {}
    
    # Handle empty DataFrame
    if df.empty:
        return {
            "period_start": None,
            "period_end": None,
            "note": "no data"
        }
    
    # Check for Close column
    if 'Close' not in df.columns:
        return {
            "period_start": df.index[0].strftime('%Y-%m-%d') if len(df) > 0 else None,
            "period_end": df.index[-1].strftime('%Y-%m-%d') if len(df) > 0 else None,
            "note": "missing Close column"
        }
    
    # Get period dates
    result["period_start"] = df.index[0].strftime('%Y-%m-%d')
    result["period_end"] = df.index[-1].strftime('%Y-%m-%d')
    
    # Check for insufficient data (less than 2 rows)
    if len(df) < 2:
        result["note"] = "insufficient data"
        return result
    
    # Get close prices
    close_prices = df['Close'].dropna()
    
    if len(close_prices) < 2:
        result["note"] = "insufficient data"
        return result
    
    # Calculate basic metrics
    first_close = close_prices.iloc[0]
    last_close = close_prices.iloc[-1]
    
    # Period return
    result["period_return"] = (last_close / first_close) - 1
    
    # Annualized volatility (only for 1d interval)
    if interval == "1d" and len(close_prices) >= 2:
        returns = close_prices.pct_change().dropna()
        if len(returns) >= 1:
            result["annualized_vol"] = returns.std() * math.sqrt(252)
        else:
            result["annualized_vol"] = None
    else:
        result["annualized_vol"] = None
    
    # Maximum drawdown
    cummax = close_prices.cummax()
    drawdowns = (close_prices / cummax) - 1
    result["max_drawdown"] = drawdowns.min()
    
    # Trend slope using OLS
    if len(close_prices) >= 2:
        x = np.arange(len(close_prices))
        y = close_prices.values
        slope = np.polyfit(x, y, 1)[0]
        result["trend_slope"] = slope
    else:
        result["trend_slope"] = None
    
    # RSI last value
    if 'rsi14' in df.columns:
        rsi_values = df['rsi14'].dropna()
        if len(rsi_values) > 0:
            result["rsi_last"] = rsi_values.iloc[-1]
        else:
            result["rsi_last"] = None
    else:
        result["rsi_last"] = None
    
    # MACD state
    result["macd_state"] = _classify_macd_state(df)
    
    # Bollinger Band position
    result["bb_position"] = _classify_bb_position(df)
    
    return result


def _classify_macd_state(df: pd.DataFrame) -> str:
    """
    Classify MACD state based on last two rows.
    
    Args:
        df: DataFrame with potential 'macd' and 'macd_signal' columns
    
    Returns:
        String classification: "cross_up", "cross_down", "above", "below", "flat", or "unknown"
    """
    if 'macd' not in df.columns or 'macd_signal' not in df.columns:
        return "unknown"
    
    macd = df['macd'].dropna()
    macd_signal = df['macd_signal'].dropna()
    
    # Need at least 2 valid data points for both series
    if len(macd) < 2 or len(macd_signal) < 2:
        return "unknown"
    
    # Get last two values for each series
    # Align the indices to ensure we're comparing the same time periods
    common_index = macd.index.intersection(macd_signal.index)
    if len(common_index) < 2:
        return "unknown"
    
    # Get the last two common points
    last_two_index = common_index[-2:]
    macd_last_two = macd.loc[last_two_index]
    signal_last_two = macd_signal.loc[last_two_index]
    
    prev_macd, curr_macd = macd_last_two.iloc[0], macd_last_two.iloc[1]
    prev_signal, curr_signal = signal_last_two.iloc[0], signal_last_two.iloc[1]
    
    # Check for crossing
    prev_above = prev_macd > prev_signal
    curr_above = curr_macd > curr_signal
    
    if not prev_above and curr_above:
        return "cross_up"
    elif prev_above and not curr_above:
        return "cross_down"
    elif curr_above:
        return "above"
    elif not curr_above:
        return "below"
    else:
        return "flat"


def _classify_bb_position(df: pd.DataFrame) -> str:
    """
    Classify Bollinger Band position based on current Close price.
    
    Args:
        df: DataFrame with potential 'bb_low', 'bb_high', 'bb_mid' columns and 'Close'
    
    Returns:
        String classification: "above_upper", "below_lower", "near_upper", "near_lower", "inside", or None
    """
    required_cols = ['bb_low', 'bb_high', 'bb_mid', 'Close']
    if not all(col in df.columns for col in required_cols):
        return None
    
    # Get the last valid values
    last_row = df[required_cols].dropna().iloc[-1:] if len(df[required_cols].dropna()) > 0 else None
    
    if last_row is None or len(last_row) == 0:
        return None
    
    close = last_row['Close'].iloc[0]
    bb_low = last_row['bb_low'].iloc[0]
    bb_high = last_row['bb_high'].iloc[0]
    bb_mid = last_row['bb_mid'].iloc[0]
    
    # Check for NaN values
    if pd.isna(close) or pd.isna(bb_low) or pd.isna(bb_high):
        return None
    
    # Position classification
    if close > bb_high:
        return "above_upper"
    elif close < bb_low:
        return "below_lower"
    else:
        # Calculate band width
        band_width = bb_high - bb_low
        
        # Handle zero or NaN band width
        if pd.isna(band_width) or band_width == 0:
            return "inside"
        
        # Calculate distances as fractions of band width
        dist_from_upper = (bb_high - close) / band_width
        dist_from_lower = (close - bb_low) / band_width
        
        # Check if close to upper band (distance from upper < 10% of band width)
        if dist_from_upper < 0.1:
            return "near_upper"
        # Check if close to lower band (distance from lower < 10% of band width)
        elif dist_from_lower < 0.1:
            return "near_lower"
        else:
            return "inside"
