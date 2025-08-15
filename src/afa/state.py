"""Shared state definitions for the Agentic Financial Analyst LangGraph project.

This module defines the core state types used throughout the application following
a philosophy of:
- **decide** (router): determines next action based on state
- **do** (tools): fetches data and computes metrics
- **tell** (finalizer): produces human-readable response

The state is designed for token economy - metrics summaries rather than raw DataFrames
are passed to LLMs. Termination occurs via `final_answer` field or `iterations` limit.

State invariants:
- All monetary values as decimals (not percentages)
- Dates in 'YYYY-MM-DD' format
- Tickers as uppercase strings
- No validation side effects in state construction

Example usage:
    parsed = Parsed(tickers=["AAPL"], interval="1d", compare=False)
    state = initial_state("How is AAPL doing?", parsed)
    
    # Tools node might write:
    state["metrics"]["AAPL"] = Metrics(
        period_return=0.15,
        annualized_vol=0.25,
        rsi_last=65.5
    )
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, TypedDict
import pandas as pd

__all__ = ["Parsed", "Metrics", "ToolMessage", "ConversationState", "initial_state"]


class Parsed(TypedDict, total=False):
    """Deterministic interpretation of the user question.
    
    Set by parse step before router to provide structured inputs for tools.
    All fields are optional to handle partial parsing scenarios.
    
    Fields:
        tickers: List of uppercase stock symbols to analyze
        start: Start date in 'YYYY-MM-DD' format
        end: End date in 'YYYY-MM-DD' format  
        interval: Data interval (e.g., '1d', '1h', '5m')
        compare: Whether to perform comparative analysis between tickers
    """
    tickers: List[str]
    start: Optional[str]
    end: Optional[str] 
    interval: str
    compare: bool


class Metrics(TypedDict, total=False):
    """Tiny numeric digest per ticker for LLM verbalization.
    
    Computed by tools nodes from raw DataFrames. Never send full DataFrames 
    to models - only these compact summaries. Some fields may be unavailable
    depending on data quality and timeframe.
    
    Fields:
        period_start: Start date of analysis period
        period_end: End date of analysis period
        period_return: Total return as decimal (0.15 = 15%)
        annualized_vol: Annualized volatility as decimal
        max_drawdown: Maximum drawdown as decimal (negative value)
        trend_slope: Linear trend slope coefficient
        rsi_last: Most recent RSI value (0-100)
        macd_state: MACD signal interpretation
        bb_position: Position relative to Bollinger Bands
    """
    period_start: str
    period_end: str
    period_return: float
    annualized_vol: Optional[float]
    max_drawdown: Optional[float]
    trend_slope: Optional[float]
    rsi_last: Optional[float]
    macd_state: Optional[Literal["above", "below", "cross_up", "cross_down", "flat", "unknown"]]
    bb_position: Optional[Literal["below_lower", "near_lower", "inside", "near_upper", "above_upper", "unknown"]]


class ToolMessage(TypedDict, total=False):
    """Optional lightweight trace of interactions for debugging.
    
    Safe to omit in downstream logic. Useful for tracking tool calls
    and responses during development and debugging.
    
    Fields:
        role: Message role in conversation
        content: Message content/text
        name: Optional tool or function name
    """
    role: Literal["user", "assistant", "system", "tool"]
    content: str
    name: Optional[str]


class ConversationState(TypedDict, total=False):
    """The shared blackboard across all nodes.
    
    Keep minimal and predictable. All nodes read/write to this state.
    Router uses this to determine next actions. Tools populate data and metrics.
    Finalizer reads metrics to generate human-readable responses.
    
    Termination conditions:
    - final_answer is set (successful completion)
    - iterations exceeds limit (loop prevention)
    
    Fields:
        question: Raw user input question
        parsed: Structured interpretation of question
        dataframes: Per-ticker OHLCV data with indicators
        metrics: Per-ticker numeric summaries for LLM consumption
        messages: Optional trace of tool interactions
        final_answer: Set by finalizer to terminate graph
        iterations: Loop guard counter for router (recommend ≤3)
    """
    question: str
    parsed: Parsed
    dataframes: Dict[str, pd.DataFrame]
    metrics: Dict[str, Metrics]
    messages: List[ToolMessage]
    final_answer: Optional[str]
    iterations: int


def initial_state(question: str, parsed: Parsed) -> ConversationState:
    """Create a well-formed ConversationState for graph initialization.
    
    Returns a state with all required fields properly initialized:
    - Empty containers for data collection
    - Zero iteration counter
    - No final answer (allows graph to run)
    
    No validation or type coercion performed - inputs used as-is.
    
    Args:
        question: Raw user question/input
        parsed: Structured interpretation from parsing step
        
    Returns:
        ConversationState: Initialized state ready for graph execution
        
    Example:
        >>> parsed = Parsed(tickers=["AAPL", "MSFT"], interval="1d", compare=True)
        >>> state = initial_state("Compare AAPL vs MSFT", parsed)
        >>> state["iterations"]
        0
        >>> len(state["dataframes"])
        0
    """
    return ConversationState(
        question=question,
        parsed=parsed,
        dataframes={},
        metrics={},
        messages=[],
        final_answer=None,
        iterations=0
    )