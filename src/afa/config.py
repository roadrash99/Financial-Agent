import os
from typing import Any

from langchain_groq import ChatGroq

__all__ = ["get_finalizer_llm", "get_router_llm"]


def get_finalizer_llm() -> ChatGroq:
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
