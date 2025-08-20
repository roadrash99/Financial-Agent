from typing import Literal

from langgraph.graph import StateGraph, END

from afa.state import ConversationState
from afa.nodes.router import router_node
from afa.nodes.tools_node import tools_node
from afa.nodes.finalizer import finalizer_node

__all__ = ["build_graph", "get_app"]

_compiled_graph = None


def _route_decider(state: ConversationState) -> Literal["CALL_TOOLS", "FINALIZE"]:
    plan = state.get("__plan__", {})
    next_action = plan.get("next_action", "FINALIZE")
    
    if next_action == "CALL_TOOLS":
        return "CALL_TOOLS"
    else:
        return "FINALIZE"


def build_graph():
    graph = StateGraph(ConversationState)
    
    graph.add_node("router", router_node)
    graph.add_node("tools_node", tools_node)
    graph.add_node("finalizer", finalizer_node)
    
    graph.add_conditional_edges(
        "router",
        _route_decider,
        {
            "CALL_TOOLS": "tools_node",
            "FINALIZE": "finalizer",
        }
    )
    
    graph.add_edge("tools_node", "router")
    graph.set_entry_point("router")
    
    return graph.compile()


def get_app():
    global _compiled_graph
    
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    
    return _compiled_graph
