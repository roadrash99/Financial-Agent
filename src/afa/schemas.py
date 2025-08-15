"""JSON contracts between Router LLM and Tools node.

This module defines the structured data formats that the Router LLM must emit
and the Tools node must consume. Provides lightweight validation to fail fast
on malformed router outputs.

The contract supports two main flows:
1. CALL_TOOLS: Router requests data fetching, indicator computation, and metric summarization
2. FINALIZE: Router signals completion and hands off to finalizer

Example CALL_TOOLS plan:
```json
{
  "next_action": "CALL_TOOLS",
  "tool_calls": [
    {
      "name": "fetch_prices", 
      "args": {
        "tickers": ["AAPL", "MSFT"], 
        "start": "2025-05-15", 
        "end": "2025-08-14", 
        "interval": "1d"
      }
    },
    {
      "name": "compute_indicators", 
      "args": {
        "indicators": ["sma20", "rsi14", "macd"]
      }
    },
    {
      "name": "summarize_metrics", 
      "args": {}
    }
  ]
}
```

Example FINALIZE plan:
```json
{
  "next_action": "FINALIZE"
}
```
"""

from __future__ import annotations

import json
from typing import Literal, TypedDict

__all__ = [
    "NextAction", "ToolName", "Interval", "IndicatorId",
    "FetchPricesArgs", "ComputeIndicatorsArgs", "SummarizeMetricsArgs",
    "ToolCall", "RouterPlan",
    "validate_tool_call", "validate_plan", "coerce_plan", "plan_to_json",
    "ALLOWED_INDICATORS", "MAX_TICKERS_PER_CALL"
]

# Type aliases for controlled vocabularies
NextAction = Literal["CALL_TOOLS", "FINALIZE"]
ToolName = Literal["fetch_prices", "compute_indicators", "summarize_metrics"]
Interval = Literal["1d", "1wk", "1mo"]
IndicatorId = Literal["sma20", "sma50", "ema20", "rsi14", "macd", "bbands"]

# Module constants
ALLOWED_INDICATORS: set[str] = {"sma20", "sma50", "ema20", "rsi14", "macd", "bbands"}
MAX_TICKERS_PER_CALL: int = 5


class FetchPricesArgs(TypedDict, total=False):
    """Arguments for fetch_prices tool call.
    
    Fields:
        tickers: List of 1-5 stock symbols to fetch
        start: Start date in 'YYYY-MM-DD' format (optional)
        end: End date in 'YYYY-MM-DD' format (optional)  
        interval: Data interval frequency
    """
    tickers: list[str]
    start: str | None
    end: str | None
    interval: Interval


class ComputeIndicatorsArgs(TypedDict, total=False):
    """Arguments for compute_indicators tool call.
    
    Fields:
        tickers: Specific tickers to compute indicators for (optional, defaults to all fetched)
        indicators: Subset of indicators to compute (optional, defaults to all supported)
    """
    tickers: list[str]
    indicators: list[IndicatorId]


class SummarizeMetricsArgs(TypedDict, total=False):
    """Arguments for summarize_metrics tool call.
    
    Fields:
        tickers: Specific tickers to summarize (optional, defaults to all available)
        interval: Interval for annualized volatility calculations (optional)
    """
    tickers: list[str]
    interval: Interval


class ToolCall(TypedDict):
    """Individual tool call specification.
    
    Fields:
        name: Tool function to invoke
        args: Tool-specific arguments (validated based on name)
    """
    name: ToolName
    args: dict


class RouterPlan(TypedDict, total=False):
    """Complete router decision with optional tool calls.
    
    Fields:
        next_action: Whether to call tools or finalize
        tool_calls: List of tools to execute (required if CALL_TOOLS, optional if FINALIZE)
    """
    next_action: NextAction
    tool_calls: list[ToolCall]


def validate_tool_call(obj: dict) -> tuple[bool, str | None]:
    """Validate a single tool call structure and arguments.
    
    Args:
        obj: Dictionary to validate as ToolCall
        
    Returns:
        Tuple of (is_valid, error_message)
        
    Example:
        >>> valid, error = validate_tool_call({"name": "fetch_prices", "args": {"tickers": ["AAPL"]}})
        >>> valid
        True
    """
    # Check required fields
    if "name" not in obj:
        return False, "Missing required field 'name'"
    
    if "args" not in obj:
        return False, "Missing required field 'args'"
    
    # Validate tool name
    if obj["name"] not in {"fetch_prices", "compute_indicators", "summarize_metrics"}:
        return False, f"Unknown tool name: {obj['name']}"
    
    # Validate args is dict
    if not isinstance(obj["args"], dict):
        return False, f"Tool args must be dict, got {type(obj['args']).__name__}"
    
    # Tool-specific validation
    args = obj["args"]
    
    if obj["name"] == "fetch_prices":
        # Check tickers
        if "tickers" in args:
            if not isinstance(args["tickers"], list):
                return False, "fetch_prices.tickers must be list"
            if len(args["tickers"]) > MAX_TICKERS_PER_CALL:
                return False, f"fetch_prices.tickers exceeds {MAX_TICKERS_PER_CALL} (got {len(args['tickers'])})"
            if not all(isinstance(t, str) for t in args["tickers"]):
                return False, "fetch_prices.tickers must contain only strings"
        
        # Check interval
        if "interval" in args:
            if args["interval"] not in {"1d", "1wk", "1mo"}:
                return False, f"fetch_prices.interval must be 1d/1wk/1mo, got '{args['interval']}'"
        
        # Check date formats (basic validation)
        for date_field in ["start", "end"]:
            if date_field in args and args[date_field] is not None:
                if not isinstance(args[date_field], str):
                    return False, f"fetch_prices.{date_field} must be string or null"
    
    elif obj["name"] == "compute_indicators":
        # Check indicators subset
        if "indicators" in args:
            if not isinstance(args["indicators"], list):
                return False, "compute_indicators.indicators must be list"
            invalid_indicators = set(args["indicators"]) - ALLOWED_INDICATORS
            if invalid_indicators:
                return False, f"Unknown indicators: {sorted(invalid_indicators)}"
        
        # Check tickers
        if "tickers" in args:
            if not isinstance(args["tickers"], list):
                return False, "compute_indicators.tickers must be list"
    
    elif obj["name"] == "summarize_metrics":
        # Check tickers
        if "tickers" in args:
            if not isinstance(args["tickers"], list):
                return False, "summarize_metrics.tickers must be list"
        
        # Check interval
        if "interval" in args:
            if args["interval"] not in {"1d", "1wk", "1mo"}:
                return False, f"summarize_metrics.interval must be 1d/1wk/1mo, got '{args['interval']}'"
    
    return True, None


def validate_plan(obj: dict) -> tuple[bool, str | None]:
    """Validate a complete router plan structure.
    
    Args:
        obj: Dictionary to validate as RouterPlan
        
    Returns:
        Tuple of (is_valid, error_message)
        
    Example:
        >>> plan = {"next_action": "CALL_TOOLS", "tool_calls": [{"name": "fetch_prices", "args": {}}]}
        >>> valid, error = validate_plan(plan)
        >>> valid
        True
    """
    # Check next_action
    if "next_action" not in obj:
        return False, "Missing required field 'next_action'"
    
    if obj["next_action"] not in {"CALL_TOOLS", "FINALIZE"}:
        return False, f"next_action must be CALL_TOOLS or FINALIZE, got '{obj['next_action']}'"
    
    # Validate tool_calls based on action
    if obj["next_action"] == "CALL_TOOLS":
        if "tool_calls" not in obj:
            return False, "tool_calls required when next_action is CALL_TOOLS"
        
        if not isinstance(obj["tool_calls"], list):
            return False, "tool_calls must be list"
        
        if len(obj["tool_calls"]) == 0:
            return False, "tool_calls cannot be empty when next_action is CALL_TOOLS"
        
        # Validate each tool call
        for i, tool_call in enumerate(obj["tool_calls"]):
            if not isinstance(tool_call, dict):
                return False, f"tool_calls[{i}] must be dict"
            
            valid, error = validate_tool_call(tool_call)
            if not valid:
                return False, f"tool_calls[{i}]: {error}"
    
    elif obj["next_action"] == "FINALIZE":
        # tool_calls optional for FINALIZE, but if present should be valid
        if "tool_calls" in obj and obj["tool_calls"]:
            return False, "tool_calls should be empty or absent when next_action is FINALIZE"
    
    return True, None


def coerce_plan(obj: dict) -> RouterPlan:
    """Convert and validate dictionary to typed RouterPlan.
    
    Args:
        obj: Dictionary to convert
        
    Returns:
        RouterPlan: Validated and typed plan
        
    Raises:
        ValueError: If validation fails
        
    Example:
        >>> obj = {"next_action": "FINALIZE"}
        >>> plan = coerce_plan(obj)
        >>> plan["next_action"]
        'FINALIZE'
    """
    valid, error = validate_plan(obj)
    if not valid:
        raise ValueError(f"Invalid router plan: {error}")
    
    # Safe to cast after validation
    result = RouterPlan(next_action=obj["next_action"])
    
    if "tool_calls" in obj:
        result["tool_calls"] = obj["tool_calls"]
    
    return result


def plan_to_json(plan: RouterPlan) -> str:
    """Convert RouterPlan to pretty JSON string.
    
    Args:
        plan: RouterPlan to serialize
        
    Returns:
        Pretty-formatted JSON string with stable key ordering
        
    Example:
        >>> plan = RouterPlan(next_action="FINALIZE")
        >>> json_str = plan_to_json(plan)
        >>> "FINALIZE" in json_str
        True
    """
    return json.dumps(plan, indent=2, sort_keys=True)
