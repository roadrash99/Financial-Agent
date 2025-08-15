"""Finalizer node for the Agentic Financial Analyst LangGraph project.

This module implements the Finalizer node that generates natural language summaries
from the computed metrics. It uses only the compact metrics data (no DataFrames)
to ask the Groq LLM for a concise 4-7 sentence explanation of the financial analysis.

The Finalizer operates as the final step in the analysis pipeline:
- Reads question, parsed data, and metrics from state
- Creates a compact JSON context (excluding DataFrames)
- Sends system and human messages to the LLM
- Generates a final natural language answer
- Sets state['final_answer'] to terminate the graph

Example usage:
    >>> state = ConversationState(
    ...     question="How is AAPL performing?",
    ...     parsed={"tickers": ["AAPL"], "interval": "1d"},
    ...     metrics={"AAPL": {"period_return": 0.15, "rsi_last": 65.5}}
    ... )
    >>> result = finalizer_node(state)
    >>> print(result["final_answer"])
    # Returns a 4-7 sentence analysis summary
"""

import json
from typing import Dict, Any

from langchain_core.messages import SystemMessage, HumanMessage

from afa.config import get_finalizer_llm
from afa.prompts import get_finalizer_prompt, get_system_prompt
from afa.state import ConversationState

__all__ = ["finalizer_node"]


def finalizer_node(state: ConversationState) -> Dict[str, Any]:
    """
    Uses only 'parsed' and 'metrics' to ask Groq for a concise natural-language summary.
    Writes 'final_answer' and returns {'final_answer': text}.
    
    Args:
        state: ConversationState containing question, parsed data, and metrics
        
    Returns:
        Dictionary with 'final_answer' containing the LLM-generated summary
        
    Behavior:
        - Builds compact JSON context from question, parsed, and metrics
        - Creates system message with role definition
        - Creates human message with finalizer prompt and JSON context
        - Invokes LLM to generate natural language explanation
        - Returns final answer as cleaned string
        
    Context Structure:
        {
            "question": str,
            "parsed": Parsed,
            "metrics": Dict[str, Metrics]
        }
        
    LLM Requirements:
        - Uses get_finalizer_llm() for model configuration
        - Generates 4-7 sentence explanations
        - Focuses on historical data and technical indicators
        - Avoids investment advice or future predictions
        - Handles empty metrics gracefully per prompt instructions
    """
    # Build compact JSON payload (no DataFrames)
    context = {
        "question": state["question"],
        "parsed": state.get("parsed", {}),
        "metrics": state.get("metrics", {})
    }
    
    # Create LLM messages
    messages = [
        SystemMessage(content=get_system_prompt()),
        HumanMessage(content=_build_human_message(context))
    ]
    
    # Get LLM instance and invoke
    llm = get_finalizer_llm()
    response = llm.invoke(messages)
    
    # Extract content and clean it
    final_answer = response.content.strip()
    
    return {"final_answer": final_answer}


def _build_human_message(context: Dict[str, Any]) -> str:
    """
    Build the human message combining finalizer prompt with JSON context.
    
    Args:
        context: Dictionary containing question, parsed, and metrics data
        
    Returns:
        String combining the finalizer prompt with JSON-serialized context
    """
    prompt = get_finalizer_prompt()
    context_json = json.dumps(context, indent=2, sort_keys=True)
    
    return f"{prompt}\n\nJSON: {context_json}"
