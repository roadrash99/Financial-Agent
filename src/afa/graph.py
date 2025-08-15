"""LangGraph workflow definition for the Agentic Financial Analyst.

This module defines the complete graph workflow that orchestrates the three main nodes:
- Router: Decides what actions to take based on current state
- Tools: Executes data fetching, indicator computation, and metric summarization
- Finalizer: Generates natural language summaries from computed metrics

The graph implements a controlled loop with iteration limits and proper termination
conditions to prevent infinite execution while allowing for iterative refinement.

Workflow:
    router → tools_node → router (loop, max 3 iterations)
    router → finalizer (when next_action == "FINALIZE")
    Stop when final_answer is set or iteration limit reached

Example usage:
    >>> from afa.graph import build_graph
    >>> from afa.state import initial_state, Parsed
    >>> 
    >>> graph = build_graph()
    >>> parsed = Parsed(tickers=["AAPL"], interval="1d", compare=False)
    >>> state = initial_state("How is AAPL performing?", parsed)
    >>> result = graph.invoke(state)
    >>> print(result["final_answer"])
"""

from typing import Literal

from langgraph.graph import StateGraph, END

from afa.state import ConversationState
from afa.nodes.router import router_node
from afa.nodes.tools_node import tools_node
from afa.nodes.finalizer import finalizer_node

__all__ = ["build_graph", "get_app"]

# Global variable to cache the compiled graph
_compiled_graph = None


def _route_decider(state: ConversationState) -> Literal["CALL_TOOLS", "FINALIZE"]:
    """
    Determine the next node based on router's decision in the plan.
    
    Args:
        state: Current conversation state containing the router's plan
        
    Returns:
        Next action to take: either "CALL_TOOLS" or "FINALIZE"
        
    Logic:
        - Reads the router's plan from state["__plan__"]["next_action"]
        - If no plan exists or plan is malformed, defaults to "FINALIZE"
        - This ensures the graph always has a valid route and terminates safely
        
    Safety:
        - Always returns a valid Literal value
        - Defaults to termination if routing information is unclear
        - Prevents infinite loops by falling back to finalization
    """
    # Get the plan from state
    plan = state.get("__plan__", {})
    
    # Extract next_action with safe default
    next_action = plan.get("next_action", "FINALIZE")
    
    # Ensure we return a valid literal value
    if next_action == "CALL_TOOLS":
        return "CALL_TOOLS"
    else:
        # Default to FINALIZE for any other value (including None, malformed, etc.)
        return "FINALIZE"


def build_graph():
    """
    Build and compile the LangGraph workflow for financial analysis.
    
    Returns:
        Compiled graph ready for execution
        
    Graph Structure:
        - Entry point: router
        - Router decides between tools_node and finalizer
        - Tools_node always returns to router (for iterative refinement)
        - Finalizer terminates the workflow
        - No explicit END edges needed (finalizer has no outgoing edges)
        
    Termination:
        - Automatic when finalizer completes (no outgoing edges)
        - Router should enforce iteration limits internally
        - Graph will stop when final_answer is set by finalizer
        
    Configuration:
        - Uses ConversationState as the state schema
        - No checkpointer (stateless execution)
        - Conditional routing based on router decisions
    """
    # Initialize the state graph
    graph = StateGraph(ConversationState)
    
    # Add the three main nodes
    graph.add_node("router", router_node)
    graph.add_node("tools_node", tools_node)
    graph.add_node("finalizer", finalizer_node)
    
    # Add conditional edge from router
    graph.add_conditional_edges(
        "router",
        _route_decider,
        {
            "CALL_TOOLS": "tools_node",
            "FINALIZE": "finalizer",
        }
    )
    
    # Add direct edge from tools_node back to router (creates the loop)
    graph.add_edge("tools_node", "router")
    
    # Set entry point
    graph.set_entry_point("router")
    
    # Compile and return the graph
    return graph.compile()


def get_app():
    """
    Get a memoized compiled graph instance.
    
    Returns:
        Compiled graph (cached for efficiency)
        
    Benefits:
        - Avoids recompilation overhead on repeated calls
        - Ensures consistent graph structure across invocations
        - Suitable for production deployment patterns
        
    Note:
        - First call compiles the graph and caches it
        - Subsequent calls return the cached instance
        - Thread-safe for read-only usage patterns
    """
    global _compiled_graph
    
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    
    return _compiled_graph
