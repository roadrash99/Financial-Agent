

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

NextAction = Literal["CALL_TOOLS", "FINALIZE"]
ToolName = Literal["fetch_prices", "compute_indicators", "summarize_metrics"]
Interval = Literal["1d", "1wk", "1mo"]
IndicatorId = Literal["sma20", "sma50", "ema20", "rsi14", "macd", "bbands"]

ALLOWED_INDICATORS: set[str] = {"sma20", "sma50", "ema20", "rsi14", "macd", "bbands"}
MAX_TICKERS_PER_CALL: int = 5


class FetchPricesArgs(TypedDict, total=False):
    tickers: list[str]
    start: str | None
    end: str | None
    interval: Interval


class ComputeIndicatorsArgs(TypedDict, total=False):
    tickers: list[str]
    indicators: list[IndicatorId]


class SummarizeMetricsArgs(TypedDict, total=False):
    tickers: list[str]
    interval: Interval


class ToolCall(TypedDict):
    name: ToolName
    args: dict


class RouterPlan(TypedDict, total=False):
    next_action: NextAction
    tool_calls: list[ToolCall]


def validate_tool_call(obj: dict) -> tuple[bool, str | None]:

    if "name" not in obj:
        return False, "Missing required field 'name'"
    
    if "args" not in obj:
        return False, "Missing required field 'args'"
    

    if obj["name"] not in {"fetch_prices", "compute_indicators", "summarize_metrics"}:
        return False, f"Unknown tool name: {obj['name']}"
    

    if not isinstance(obj["args"], dict):
        return False, f"Tool args must be dict, got {type(obj['args']).__name__}"
    

    args = obj["args"]
    
    if obj["name"] == "fetch_prices":

        if "tickers" in args:
            if not isinstance(args["tickers"], list):
                return False, "fetch_prices.tickers must be list"
            if len(args["tickers"]) > MAX_TICKERS_PER_CALL:
                return False, f"fetch_prices.tickers exceeds {MAX_TICKERS_PER_CALL} (got {len(args['tickers'])})"
            if not all(isinstance(t, str) for t in args["tickers"]):
                return False, "fetch_prices.tickers must contain only strings"
        

        if "interval" in args:
            if args["interval"] not in {"1d", "1wk", "1mo"}:
                return False, f"fetch_prices.interval must be 1d/1wk/1mo, got '{args['interval']}'"
        

        for date_field in ["start", "end"]:
            if date_field in args and args[date_field] is not None:
                if not isinstance(args[date_field], str):
                    return False, f"fetch_prices.{date_field} must be string or null"
    
    elif obj["name"] == "compute_indicators":

        if "indicators" in args:
            if not isinstance(args["indicators"], list):
                return False, "compute_indicators.indicators must be list"
            invalid_indicators = set(args["indicators"]) - ALLOWED_INDICATORS
            if invalid_indicators:
                return False, f"Unknown indicators: {sorted(invalid_indicators)}"
        

        if "tickers" in args:
            if not isinstance(args["tickers"], list):
                return False, "compute_indicators.tickers must be list"
    
    elif obj["name"] == "summarize_metrics":

        if "tickers" in args:
            if not isinstance(args["tickers"], list):
                return False, "summarize_metrics.tickers must be list"
        

        if "interval" in args:
            if args["interval"] not in {"1d", "1wk", "1mo"}:
                return False, f"summarize_metrics.interval must be 1d/1wk/1mo, got '{args['interval']}'"
    
    return True, None


def validate_plan(obj: dict) -> tuple[bool, str | None]:

    if "next_action" not in obj:
        return False, "Missing required field 'next_action'"
    
    if obj["next_action"] not in {"CALL_TOOLS", "FINALIZE"}:
        return False, f"next_action must be CALL_TOOLS or FINALIZE, got '{obj['next_action']}'"
    

    if obj["next_action"] == "CALL_TOOLS":
        if "tool_calls" not in obj:
            return False, "tool_calls required when next_action is CALL_TOOLS"
        
        if not isinstance(obj["tool_calls"], list):
            return False, "tool_calls must be list"
        
        if len(obj["tool_calls"]) == 0:
            return False, "tool_calls cannot be empty when next_action is CALL_TOOLS"
        

        for i, tool_call in enumerate(obj["tool_calls"]):
            if not isinstance(tool_call, dict):
                return False, f"tool_calls[{i}] must be dict"
            
            valid, error = validate_tool_call(tool_call)
            if not valid:
                return False, f"tool_calls[{i}]: {error}"
    
    elif obj["next_action"] == "FINALIZE":

        if "tool_calls" in obj and obj["tool_calls"]:
            return False, "tool_calls should be empty or absent when next_action is FINALIZE"
    
    return True, None


def coerce_plan(obj: dict) -> RouterPlan:
    valid, error = validate_plan(obj)
    if not valid:
        raise ValueError(f"Invalid router plan: {error}")
    

    result = RouterPlan(next_action=obj["next_action"])
    
    if "tool_calls" in obj:
        result["tool_calls"] = obj["tool_calls"]
    
    return result


def plan_to_json(plan: RouterPlan) -> str:
    return json.dumps(plan, indent=2, sort_keys=True)
