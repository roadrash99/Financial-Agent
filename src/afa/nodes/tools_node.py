

from typing import Dict, List, Any
import pandas as pd

from afa.schemas import RouterPlan
from afa.tools.prices import fetch_prices
from afa.tools.indicators import compute_indicators_pandas
from afa.tools.metrics import summarize_metrics
from afa.state import ConversationState

__all__ = ["tools_node"]


def tools_node(state: ConversationState) -> Dict[str, Any]:

    plan = state.get("__plan__", {})
    tool_calls = plan.get("tool_calls", [])
    

    dfs: Dict[str, pd.DataFrame] = dict(state.get("dataframes", {}))
    metrics: Dict[str, Dict[str, Any]] = dict(state.get("metrics", {}))
    

    for tool_call in tool_calls:
        tool_name = tool_call.get("name", "")
        args = tool_call.get("args", {})
        
        if tool_name == "fetch_prices":
            _handle_fetch_prices(dfs, args)
        elif tool_name == "compute_indicators":
            _handle_compute_indicators(dfs, args)
        elif tool_name == "summarize_metrics":
            _handle_summarize_metrics(dfs, metrics, args, state)
    
    return {
        "dataframes": dfs,
        "metrics": metrics
    }


def _handle_fetch_prices(dfs: Dict[str, pd.DataFrame], args: Dict[str, Any]) -> None:

    tickers = args.get("tickers", [])
    start = args.get("start")
    end = args.get("end")
    interval = args.get("interval", "1d")
    

    if not tickers:
        return
    
    try:

        fetched_data = fetch_prices(
            tickers=tickers,
            start=start,
            end=end,
            interval=interval
        )
        

        for ticker, df in fetched_data.items():
            if not df.empty:
                dfs[ticker] = df

            elif ticker not in dfs:
                dfs[ticker] = df
                
    except Exception:


        pass


def _handle_compute_indicators(dfs: Dict[str, pd.DataFrame], args: Dict[str, Any]) -> None:
    # Determine target tickers
    target_tickers = args.get("tickers")
    if target_tickers is None:
        # Use all tickers that have data
        target_tickers = list(dfs.keys())
    
    # Get indicators to compute
    indicators = args.get("indicators")  # None means compute all supported indicators
    
    # Process each target ticker
    for ticker in target_tickers:
        if ticker not in dfs:
            continue
            
        df = dfs[ticker]
        
        # Skip empty DataFrames
        if df.empty:
            continue
            
        try:
            # Compute indicators and update the DataFrame
            updated_df = compute_indicators_pandas(df, indicators=indicators)
            dfs[ticker] = updated_df
        except Exception:
            # Handle computation errors gracefully (e.g., insufficient data)
            # Keep the original DataFrame if indicator computation fails
            pass


def _handle_summarize_metrics(
    dfs: Dict[str, pd.DataFrame], 
    metrics: Dict[str, Dict[str, Any]], 
    args: Dict[str, Any],
    state: ConversationState
) -> None:
    # Determine target tickers
    target_tickers = args.get("tickers")
    if target_tickers is None:
        # Use all tickers that have data
        target_tickers = list(dfs.keys())
    
    # Determine interval for metrics calculation
    interval = args.get("interval")
    if interval is None:
        # Fallback to parsed interval from state
        parsed = state.get("parsed", {})
        interval = parsed.get("interval", "1d")
    
    # Process each target ticker
    for ticker in target_tickers:
        if ticker not in dfs:
            continue
            
        df = dfs[ticker]
        
        # Skip empty DataFrames
        if df.empty:
            continue
            
        try:
            # Compute metrics summary
            ticker_metrics = summarize_metrics(df, interval=interval)
            metrics[ticker] = ticker_metrics
        except Exception:
            # Handle metrics computation errors gracefully
            # Skip this ticker if metrics computation fails
            pass
