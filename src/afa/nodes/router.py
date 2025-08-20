

from typing import Dict, Any

from afa.state import ConversationState

__all__ = ["router_node"]


def router_node(state: ConversationState) -> Dict[str, Any]:

    current_iterations = state.get("iterations", 0)
    new_iterations = current_iterations + 1

    if new_iterations >= 3:
        plan = {
            "next_action": "FINALIZE",
            "tool_calls": []
        }
    else:

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

            plan = {
                "next_action": "FINALIZE",
                "tool_calls": []
            }
    
    return {
        "__plan__": plan,
        "iterations": new_iterations
    }
