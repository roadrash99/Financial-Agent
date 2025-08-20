from __future__ import annotations

from typing import Dict, List, Literal, Optional, TypedDict
import pandas as pd

__all__ = ["Parsed", "Metrics", "ToolMessage", "ConversationState", "initial_state"]


class Parsed(TypedDict, total=False):
    tickers: List[str]
    start: Optional[str]
    end: Optional[str] 
    interval: str
    compare: bool


class Metrics(TypedDict, total=False):
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
    role: Literal["user", "assistant", "system", "tool"]
    content: str
    name: Optional[str]


class ConversationState(TypedDict, total=False):
    question: str
    parsed: Parsed
    dataframes: Dict[str, pd.DataFrame]
    metrics: Dict[str, Metrics]
    messages: List[ToolMessage]
    final_answer: Optional[str]
    iterations: int


def initial_state(question: str, parsed: Parsed) -> ConversationState:
    return ConversationState(
        question=question,
        parsed=parsed,
        dataframes={},
        metrics={},
        messages=[],
        final_answer=None,
        iterations=0
    )