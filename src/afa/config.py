"""Configuration module for the Agentic Financial Analyst.

This module provides LLM configuration and initialization functions for different
components of the analysis pipeline. It centralizes model setup and API key
management for consistent behavior across the application.

Environment Variables:
    GROQ_API_KEY: Required for Groq LLM access
    
Example usage:
    >>> from afa.config import get_finalizer_llm
    >>> llm = get_finalizer_llm()
    >>> response = llm.invoke(messages)
"""

import os
from typing import Any

from langchain_groq import ChatGroq

__all__ = ["get_finalizer_llm", "get_router_llm"]


def get_finalizer_llm() -> ChatGroq:
    """
    Get configured Groq LLM instance for finalizer tasks.
    
    Returns:
        ChatGroq: Configured LLM for generating natural language summaries
        
    Configuration:
        - Model: llama3-8b-8192 (fast and efficient for text generation)
        - Temperature: 0.1 (slightly creative but mostly deterministic)
        - Max tokens: 512 (sufficient for 4-7 sentence responses)
        
    Raises:
        ValueError: If GROQ_API_KEY environment variable is not set
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY environment variable is required for LLM access. "
            "Please set it to your Groq API key."
        )
    
    return ChatGroq(
        model="llama3-8b-8192",
        temperature=0.1,
        max_tokens=512,
        api_key=api_key
    )


def get_router_llm() -> ChatGroq:
    """
    Get configured Groq LLM instance for router planning tasks.
    
    Returns:
        ChatGroq: Configured LLM for generating JSON RouterPlan objects
        
    Configuration:
        - Model: llama3-8b-8192 (good balance of speed and JSON capability)
        - Temperature: 0.0 (deterministic for consistent JSON output)
        - Max tokens: 256 (sufficient for RouterPlan JSON structures)
        
    Raises:
        ValueError: If GROQ_API_KEY environment variable is not set
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY environment variable is required for LLM access. "
            "Please set it to your Groq API key."
        )
    
    return ChatGroq(
        model="llama3-8b-8192",
        temperature=0.0,
        max_tokens=256,
        api_key=api_key
    )
