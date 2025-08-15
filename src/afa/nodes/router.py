"""Router node for the Agentic Financial Analyst LangGraph project.

This module implements the Router node that decides the next action based on current state.
It acts as the control flow hub, determining whether to call tools for data processing
or finalize the analysis with a natural language summary.

The Router uses LLM-based planning to generate structured RouterPlan objects that
specify the next action and any required tool calls. It also enforces iteration
limits to prevent infinite loops.

Note: This is a basic implementation. The router should be enhanced with proper
LLM integration and planning logic according to the project specifications.
"""

from typing import Dict, Any

from afa.state import ConversationState

__all__ = ["router_node"]


def router_node(state: ConversationState) -> Dict[str, Any]:
    """
    Determines next action based on current state and increments iteration counter.
    
    Args:
        state: Current conversation state
        
    Returns:
        Dictionary with plan and updated iteration count
        
    Note:
        This is a basic implementation that needs to be enhanced with:
        - LLM-based planning using get_router_llm()
        - Proper RouterPlan generation based on prompts
        - Analysis of current state to determine needed actions
    """
    # Increment iteration counter
    current_iterations = state.get("iterations", 0)
    new_iterations = current_iterations + 1
    
    # Basic safety check - finalize if too many iterations
    if new_iterations >= 3:
        plan = {
            "next_action": "FINALIZE",
            "tool_calls": []
        }
    else:
        # Basic logic - if no metrics, fetch data and compute
        metrics = state.get("metrics", {})
        if not metrics:
            plan = {
                "next_action": "CALL_TOOLS",
                "tool_calls": [
                    {
                        "name": "fetch_prices",
                        "args": {
                            "tickers": state.get("parsed", {}).get("tickers", []),
                            "interval": state.get("parsed", {}).get("interval", "1d")
                        }
                    },
                    {
                        "name": "compute_indicators", 
                        "args": {}
                    },
                    {
                        "name": "summarize_metrics",
                        "args": {}
                    }
                ]
            }
        else:
            # Already have metrics, finalize
            plan = {
                "next_action": "FINALIZE",
                "tool_calls": []
            }
    
    return {
        "__plan__": plan,
        "iterations": new_iterations
    }
