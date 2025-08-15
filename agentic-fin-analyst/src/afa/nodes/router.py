"""Router node for the Agentic Financial Analyst.

The Router node reads state.question, state.parsed, and any existing state.metrics;
calls the Groq LLM (ChatGroq via config) with the router prompt; parses JSON;
validates it against schemas.py; and writes the next action + plan into state.

This module implements the "decide" phase of the decide → do → tell workflow.
"""

import json
import re
from typing import Dict

from langchain_core.messages import SystemMessage, HumanMessage

from afa.config import get_router_llm
from afa.prompts import get_router_prompt, get_system_prompt
from afa.schemas import coerce_plan, RouterPlan
from afa.state import ConversationState

__all__ = ["router_node"]


def router_node(state: ConversationState) -> Dict:
    """
    Reads question/parsed/metrics; emits {'route': 'CALL_TOOLS'|'FINALIZE', '__plan__': RouterPlan, 'iterations': <int>}.
    Never includes DataFrames in prompts. Keeps tokens tiny.
    
    Args:
        state: Current conversation state with question, parsed data, and optionally metrics
        
    Returns:
        Dict with updated state containing:
        - iterations: Incremented iteration counter
        - route: Next action ('CALL_TOOLS' or 'FINALIZE') 
        - __plan__: Full RouterPlan dict with tool calls
        
    Behavior:
        - Increments state['iterations'] (init to 0 if missing)
        - Builds minimal LLM messages excluding DataFrames
        - Calls router LLM to get JSON plan
        - Validates plan with fallback to minimal default
        - Enforces iteration limit guardrail (≥3 forces FINALIZE)
    """
    # Increment iteration counter
    current_iterations = state.get("iterations", 0)
    new_iterations = current_iterations + 1
    
    # Guardrail: Force finalize if too many iterations
    if new_iterations >= 3:
        fallback_plan = RouterPlan(next_action="FINALIZE")
        return {
            "iterations": new_iterations,
            "route": "FINALIZE", 
            "__plan__": fallback_plan
        }
    
    # Build minimal prompt data (exclude DataFrames to keep tokens low)
    have_metrics_for = list(state.get("metrics", {}).keys())
    
    prompt_data = {
        "question": state.get("question", ""),
        "parsed": state.get("parsed", {}),
        "have_metrics_for": have_metrics_for
    }
    
    # Build LLM messages
    messages = [
        SystemMessage(content=get_system_prompt()),
        HumanMessage(content=json.dumps(prompt_data))
    ]
    
    # Call router LLM
    llm = get_router_llm()
    response = llm.invoke(messages)
    content = response.content
    
    # Parse JSON with error handling and fallback
    try:
        plan_obj = _parse_json_with_fallback(content)
        plan = coerce_plan(plan_obj)
    except (json.JSONDecodeError, ValueError, KeyError):
        # Fallback plan based on current state
        plan = _create_fallback_plan(state)
    
    return {
        "iterations": new_iterations,
        "route": plan["next_action"],
        "__plan__": plan
    }


def _parse_json_with_fallback(content: str) -> Dict:
    """Parse JSON content with regex fallback for markdown fences.
    
    Args:
        content: Raw LLM response content
        
    Returns:
        Parsed JSON object
        
    Raises:
        json.JSONDecodeError: If JSON parsing fails after all attempts
    """
    # Try direct parsing first
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    
    # Strip markdown code fences and try again
    # Pattern matches ```json...``` or ```...``` blocks
    fence_pattern = r'```(?:json\s*)?\s*(.*?)\s*```'
    match = re.search(fence_pattern, content, re.DOTALL)
    
    if match:
        cleaned_content = match.group(1).strip()
        return json.loads(cleaned_content)
    
    # Strip any leading/trailing non-JSON text and try once more
    # Look for content between first { and last }
    brace_pattern = r'.*?(\{.*\}).*'
    match = re.search(brace_pattern, content, re.DOTALL)
    
    if match:
        json_content = match.group(1).strip()
        return json.loads(json_content)
    
    # If all else fails, raise the original error
    raise json.JSONDecodeError("Could not parse JSON from LLM response", content, 0)


def _create_fallback_plan(state: ConversationState) -> RouterPlan:
    """Create a minimal fallback plan based on current state.
    
    Args:
        state: Current conversation state
        
    Returns:
        RouterPlan: Minimal plan to either fetch data or finalize
    """
    parsed = state.get("parsed", {})
    metrics = state.get("metrics", {})
    requested_tickers = parsed.get("tickers", [])
    
    # Check if metrics missing for requested tickers
    missing_metrics = [ticker for ticker in requested_tickers if ticker not in metrics]
    
    if missing_metrics or not metrics:
        # Need to fetch data and compute metrics
        tool_calls = []
        
        # Add fetch_prices call
        fetch_args = {
            "tickers": requested_tickers or ["AAPL"],  # Default ticker if none specified
            "interval": parsed.get("interval", "1d")
        }
        
        # Add optional date range if available
        if parsed.get("start"):
            fetch_args["start"] = parsed["start"]
        if parsed.get("end"):
            fetch_args["end"] = parsed["end"]
            
        tool_calls.append({
            "name": "fetch_prices",
            "args": fetch_args
        })
        
        # Add compute_indicators call with conservative indicator set
        tool_calls.append({
            "name": "compute_indicators", 
            "args": {"indicators": ["sma20", "rsi14", "macd"]}
        })
        
        # Add summarize_metrics call
        tool_calls.append({
            "name": "summarize_metrics",
            "args": {}
        })
        
        return RouterPlan(
            next_action="CALL_TOOLS",
            tool_calls=tool_calls
        )
    else:
        # Have sufficient metrics, can finalize
        return RouterPlan(next_action="FINALIZE")
