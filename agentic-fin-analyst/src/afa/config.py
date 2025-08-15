"""Configuration module for AFA (Agentic Financial Analyst).

Provides LangChain ChatGroq clients optimized for different roles:

- **Router**: Fast, lightweight model with low temperature for consistent routing decisions.
  Default: llama-3.1-8b-instant, temp=0.1, max_tokens=500

- **Finalizer**: More capable model with higher temperature for rich responses.
  Default: llama-3.1-70b-versatile, temp=0.3, max_tokens=900

Environment Variables:
    GROQ_API_KEY (required): Groq API key
    GROQ_ROUTER_MODEL: Model for router (default: "llama-3.1-8b-instant")
    GROQ_FINALIZER_MODEL: Model for finalizer (default: "llama-3.1-70b-versatile")
    AFA_ROUTER_TEMP: Router temperature (default: 0.1)
    AFA_FINALIZER_TEMP: Finalizer temperature (default: 0.3)
    AFA_LLM_TIMEOUT_S: LLM timeout in seconds (default: 20)
    AFA_LLM_MAX_RETRIES: Max retry attempts (default: 1)
    AFA_ROUTER_MAX_TOKENS: Router max tokens (default: 500)
    AFA_FINALIZER_MAX_TOKENS: Finalizer max tokens (default: 900)

Usage:
    >>> from afa.config import get_router_llm
    >>> llm = get_router_llm()
    >>> llm  # ChatGroq(model="llama-3.1-8b-instant", temperature=0.1)
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

# Optional dotenv loading - no hard dependency
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from langchain_groq import ChatGroq

__all__ = ["get_router_llm", "get_finalizer_llm", "get_models"]


def _get_env_str(name: str, default: Optional[str] = None) -> Optional[str]:
    """Get string environment variable with optional default."""
    return os.getenv(name, default)


def _get_env_float(name: str, default: float) -> float:
    """Get float environment variable with default fallback."""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _get_env_int(name: str, default: int) -> int:
    """Get integer environment variable with default fallback."""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _require(key: str) -> str:
    """Get required environment variable or raise ValueError with friendly message."""
    value = os.getenv(key)
    if not value:
        raise ValueError(f"{key} not set. Define it in your environment or .env")
    return value


def _make_chatgroq(
    *,
    model: str,
    temperature: float,
    timeout_s: int,
    max_tokens: Optional[int],
    max_retries: int
) -> ChatGroq:
    """Factory function to create ChatGroq instance with common configuration.
    
    Args:
        model: Groq model name
        temperature: Sampling temperature
        timeout_s: Request timeout in seconds
        max_tokens: Maximum tokens to generate (None for model default)
        max_retries: Maximum retry attempts
    
    Returns:
        Configured ChatGroq instance
    
    Raises:
        ValueError: If GROQ_API_KEY is not set
    """
    api_key = _require("GROQ_API_KEY")
    
    return ChatGroq(
        model_name=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout_s,
        max_retries=max_retries,
        api_key=api_key
    )


def get_router_llm() -> ChatGroq:
    """Get ChatGroq client optimized for routing decisions.
    
    Uses fast, lightweight model with low temperature for consistent
    and predictable routing behavior.
    
    Returns:
        ChatGroq instance configured for routing
    """
    model = _get_env_str("GROQ_ROUTER_MODEL", "llama-3.1-8b-instant")
    temperature = _get_env_float("AFA_ROUTER_TEMP", 0.1)
    timeout_s = _get_env_int("AFA_LLM_TIMEOUT_S", 20)
    max_tokens = _get_env_int("AFA_ROUTER_MAX_TOKENS", 500)
    max_retries = _get_env_int("AFA_LLM_MAX_RETRIES", 1)
    
    return _make_chatgroq(
        model=model,
        temperature=temperature,
        timeout_s=timeout_s,
        max_tokens=max_tokens,
        max_retries=max_retries
    )


def get_finalizer_llm() -> ChatGroq:
    """Get ChatGroq client optimized for final response generation.
    
    Uses more capable model with higher temperature for rich,
    detailed responses.
    
    Returns:
        ChatGroq instance configured for final response generation
    """
    model = _get_env_str("GROQ_FINALIZER_MODEL", "llama-3.1-70b-versatile")
    temperature = _get_env_float("AFA_FINALIZER_TEMP", 0.3)
    timeout_s = _get_env_int("AFA_LLM_TIMEOUT_S", 20)
    max_tokens = _get_env_int("AFA_FINALIZER_MAX_TOKENS", 900)
    max_retries = _get_env_int("AFA_LLM_MAX_RETRIES", 1)
    
    return _make_chatgroq(
        model=model,
        temperature=temperature,
        timeout_s=timeout_s,
        max_tokens=max_tokens,
        max_retries=max_retries
    )


def get_models() -> Tuple[ChatGroq, ChatGroq]:
    """Convenience function to get both router and finalizer LLMs.
    
    Returns:
        Tuple of (router_llm, finalizer_llm)
    """
    return (get_router_llm(), get_finalizer_llm())
