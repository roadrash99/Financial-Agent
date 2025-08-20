

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import math


__all__ = ["summarize_metrics"]


def summarize_metrics(
    df: pd.DataFrame,
    interval: str = "1d"
) -> Dict[str, Any]:

    result = {}
    

    if df.empty:
        return {
            "period_start": None,
            "period_end": None,
            "note": "no data"
        }
    

    if 'Close' not in df.columns:
        return {
            "period_start": df.index[0].strftime('%Y-%m-%d') if len(df) > 0 else None,
            "period_end": df.index[-1].strftime('%Y-%m-%d') if len(df) > 0 else None,
            "note": "missing Close column"
        }
    

    result["period_start"] = df.index[0].strftime('%Y-%m-%d')
    result["period_end"] = df.index[-1].strftime('%Y-%m-%d')
    

    if len(df) < 2:
        result["note"] = "insufficient data"
        return result
    

    close_prices = df['Close'].dropna()
    
    if len(close_prices) < 2:
        result["note"] = "insufficient data"
        return result
    

    first_close = close_prices.iloc[0]
    last_close = close_prices.iloc[-1]
    

    result["period_return"] = (last_close / first_close) - 1
    

    if interval == "1d" and len(close_prices) >= 2:
        returns = close_prices.pct_change().dropna()
        if len(returns) >= 1:
            result["annualized_vol"] = returns.std() * math.sqrt(252)
        else:
            result["annualized_vol"] = None
    else:
        result["annualized_vol"] = None
    

    cummax = close_prices.cummax()
    drawdowns = (close_prices / cummax) - 1
    result["max_drawdown"] = drawdowns.min()
    

    if len(close_prices) >= 2:
        x = np.arange(len(close_prices))
        y = close_prices.values
        slope = np.polyfit(x, y, 1)[0]
        result["trend_slope"] = slope
    else:
        result["trend_slope"] = None
    

    if 'rsi14' in df.columns:
        rsi_values = df['rsi14'].dropna()
        if len(rsi_values) > 0:
            result["rsi_last"] = rsi_values.iloc[-1]
        else:
            result["rsi_last"] = None
    else:
        result["rsi_last"] = None
    

    result["macd_state"] = _classify_macd_state(df)
    

    result["bb_position"] = _classify_bb_position(df)
    
    return result


def _classify_macd_state(df: pd.DataFrame) -> str:
    if 'macd' not in df.columns or 'macd_signal' not in df.columns:
        return "unknown"
    
    macd = df['macd'].dropna()
    macd_signal = df['macd_signal'].dropna()
    

    if len(macd) < 2 or len(macd_signal) < 2:
        return "unknown"
    


    common_index = macd.index.intersection(macd_signal.index)
    if len(common_index) < 2:
        return "unknown"
    

    last_two_index = common_index[-2:]
    macd_last_two = macd.loc[last_two_index]
    signal_last_two = macd_signal.loc[last_two_index]
    
    prev_macd, curr_macd = macd_last_two.iloc[0], macd_last_two.iloc[1]
    prev_signal, curr_signal = signal_last_two.iloc[0], signal_last_two.iloc[1]
    

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
    required_cols = ['bb_low', 'bb_high', 'bb_mid', 'Close']
    if not all(col in df.columns for col in required_cols):
        return None
    
    
    last_row = df[required_cols].dropna().iloc[-1:] if len(df[required_cols].dropna()) > 0 else None
    
    if last_row is None or len(last_row) == 0:
        return None
    
    close = last_row['Close'].iloc[0]
    bb_low = last_row['bb_low'].iloc[0]
    bb_high = last_row['bb_high'].iloc[0]
    bb_mid = last_row['bb_mid'].iloc[0]
    
    
    if pd.isna(close) or pd.isna(bb_low) or pd.isna(bb_high):
        return None
    
    
    if close > bb_high:
        return "above_upper"
    elif close < bb_low:
        return "below_lower"
    else:
        
        band_width = bb_high - bb_low
        
        
        if pd.isna(band_width) or band_width == 0:
            return "inside"
        
        
        dist_from_upper = (bb_high - close) / band_width
        dist_from_lower = (close - bb_low) / band_width
        
        
        if dist_from_upper < 0.1:
            return "near_upper"
        
        elif dist_from_lower < 0.1:
            return "near_lower"
        else:
            return "inside"
