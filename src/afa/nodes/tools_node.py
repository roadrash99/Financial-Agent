"""Tools execution node for the Agentic Financial Analyst LangGraph project.

This module implements the Tools node that executes tool calls planned by the Router.
It processes fetch_prices, compute_indicators, and summarize_metrics operations,
updating state dataframes and metrics without making any LLM calls.

The Tools node operates as a pure function transformer:
- Reads the router's plan from state['__plan__']['tool_calls']
- Executes each tool call in sequence
- Updates state['dataframes'] and state['metrics'] per ticker
- Returns only the updates without mutating other state keys

Example usage:
    >>> state = ConversationState(...)
    >>> state['__plan__'] = RouterPlan(
    ...     next_action="CALL_TOOLS",
    ...     tool_calls=[
    ...         {"name": "fetch_prices", "args": {"tickers": ["AAPL"], "interval": "1d"}},
    ...         {"name": "compute_indicators", "args": {}},
    ...         {"name": "summarize_metrics", "args": {}}
    ...     ]
    ... )
    >>> updates = tools_node(state)
    >>> # updates contains {"dataframes": {...}, "metrics": {...}}
"""

from typing import Dict, List, Any
import pandas as pd

from afa.schemas import RouterPlan
from afa.tools.prices import fetch_prices
from afa.tools.indicators import compute_indicators_pandas
from afa.tools.metrics import summarize_metrics
from afa.state import ConversationState

__all__ = ["tools_node"]


def tools_node(state: ConversationState) -> Dict[str, Any]:
    """
    Executes tool calls from state['__plan__']['tool_calls'].
    Updates state['dataframes'] and state['metrics'] per ticker.
    Returns a dict of updates; does not mutate other keys.
    
    Args:
        state: ConversationState containing the router's plan
        
    Returns:
        Dictionary with 'dataframes' and 'metrics' updates
        
    Behavior:
        - Reads plan from state.get("__plan__", {})
        - Gets tool_calls from plan.get("tool_calls", [])
        - Executes each tool call in order:
          * fetch_prices: Fetches price data and merges into dataframes
          * compute_indicators: Computes technical indicators for specified/all tickers
          * summarize_metrics: Generates metric summaries for specified/all tickers
        - Returns updates without modifying other state keys
        
    Edge cases:
        - Invalid tickers are skipped silently
        - Empty DataFrames are handled gracefully
        - Missing tool arguments use sensible defaults
        - Never raises exceptions on data issues
    """
    # Read plan from state
    plan = state.get("__plan__", {})
    tool_calls = plan.get("tool_calls", [])
    
    # Prepare local copies of dataframes and metrics
    dfs: Dict[str, pd.DataFrame] = dict(state.get("dataframes", {}))
    metrics: Dict[str, Dict[str, Any]] = dict(state.get("metrics", {}))
    
    # Execute each tool call in sequence
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
    """
    Handle fetch_prices tool call.
    
    Args:
        dfs: Dictionary of ticker -> DataFrame to update
        args: Tool arguments containing tickers, start, end, interval
    """
    # Extract arguments with defaults
    tickers = args.get("tickers", [])
    start = args.get("start")
    end = args.get("end")
    interval = args.get("interval", "1d")
    
    # Skip if no tickers specified
    if not tickers:
        return
    
    try:
        # Fetch prices for all tickers
        fetched_data = fetch_prices(
            tickers=tickers,
            start=start,
            end=end,
            interval=interval
        )
        
        # Merge fetched data into existing dataframes
        for ticker, df in fetched_data.items():
            if not df.empty:
                dfs[ticker] = df
            # If df is empty but ticker was requested, still add it for consistency
            elif ticker not in dfs:
                dfs[ticker] = df
                
    except Exception:
        # Handle any fetch errors gracefully - don't let them crash the node
        # Invalid tickers or network issues should not break the workflow
        pass


def _handle_compute_indicators(dfs: Dict[str, pd.DataFrame], args: Dict[str, Any]) -> None:
    """
    Handle compute_indicators tool call.
    
    Args:
        dfs: Dictionary of ticker -> DataFrame to update
        args: Tool arguments containing optional tickers and indicators
    """
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
    """
    Handle summarize_metrics tool call.
    
    Args:
        dfs: Dictionary of ticker -> DataFrame
        metrics: Dictionary of ticker -> metrics to update
        args: Tool arguments containing optional tickers and interval
        state: Full state for accessing parsed interval as fallback
    """
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
