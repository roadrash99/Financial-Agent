"""
Intent parsing module for extracting tickers and comparison intent from free-form questions.

This module provides deterministic extraction of stock tickers and comparison intent
without making any model calls, using only regex patterns and heuristics.
"""

import re
from typing import List, Dict

__all__ = ["extract_tickers", "detect_compare", "parse_intent"]

# Common technical analysis terms and frequent uppercased terms to exclude
STOPWORDS = {
    "RSI", "MACD", "EMA", "SMA", "BB", "BBANDS", "VWAP", "YTD",
    "USD", "EPS", "PE", "ETF", "IPO", "EV", "EBITDA", "AI",
    "AND", "OR", "VS", "V", "VS.", "THE", "A", "AN"
}

# Regex pattern to match tickers with optional $ prefix
TICKER_PATTERN = re.compile(r"\b\$?([A-Z]{1,5})\b")

# Comparison keywords pattern (case-insensitive, with spaces to avoid partials)
COMPARE_PATTERN = re.compile(r" vs | versus | compare ", re.IGNORECASE)


def extract_tickers(text: str) -> List[str]:
    """
    Extract stock tickers from text using regex pattern with conservative filtering.
    
    Recognizes tickers via regex with optional $ prefix, normalizes to uppercase,
    deduplicates while preserving order, and filters out common false positives.
    
    Args:
        text: Input text to extract tickers from
        
    Returns:
        List of unique ticker symbols in order of first appearance
        
    Examples:
        >>> extract_tickers("Compare AAPL vs MSFT last 3 months")
        ['AAPL', 'MSFT']
        >>> extract_tickers("rsi on nvda ytd")
        ['NVDA']
        >>> extract_tickers("is macd bullish on spy?")
        ['SPY']
        >>> extract_tickers("Check $T and $F performance")
        ['T', 'F']
        >>> extract_tickers("Looking at AI trends")
        []
    """
    if not text:
        return []
    
    # Find all potential ticker matches
    matches = TICKER_PATTERN.findall(text.upper())
    
    # Filter and deduplicate while preserving order
    seen = set()
    tickers = []
    
    for match in matches:
        ticker = match.upper()
        
        # Skip if already seen
        if ticker in seen:
            continue
            
        # Skip stopwords
        if ticker in STOPWORDS:
            continue
            
        # Heuristic for short tokens
        if len(ticker) <= 2:
            # Check if the original match in text was $-prefixed
            # We need to look for the actual occurrence in the original text
            dollar_prefixed = f"${ticker}" in text.upper()
            if len(ticker) == 1:
                # Single letter only allowed if $-prefixed
                if not dollar_prefixed:
                    continue
            elif len(ticker) == 2:
                # Two letter tokens: only if $-prefixed (reduces "IN", "AT" false hits)
                if not dollar_prefixed:
                    continue
        
        seen.add(ticker)
        tickers.append(ticker)
    
    return tickers


def detect_compare(text: str, tickers: List[str]) -> bool:
    """
    Detect comparison intent based on number of tickers and comparison keywords.
    
    Returns True if there are 2 or more tickers, or if the text contains
    comparison keywords like "vs", "versus", or "compare" (case-insensitive,
    with spaces to avoid partial matches).
    
    Args:
        text: Input text to analyze
        tickers: List of extracted tickers
        
    Returns:
        Boolean indicating if comparison intent is detected
        
    Examples:
        >>> detect_compare("Compare AAPL vs MSFT", ["AAPL", "MSFT"])
        True
        >>> detect_compare("AAPL versus GOOGL performance", ["AAPL", "GOOGL"])
        True
        >>> detect_compare("How is AAPL doing?", ["AAPL"])
        False
        >>> detect_compare("Compare these stocks", [])
        True
    """
    if not text:
        return False
        
    # Check if we have 2 or more tickers
    if len(tickers) >= 2:
        return True
        
    # Check for comparison keywords with spaces to avoid partials
    if COMPARE_PATTERN.search(text):
        return True
        
    return False


def parse_intent(question: str) -> Dict[str, any]:
    """
    Parse a free-form question to extract tickers and comparison intent.
    
    This is the main entry point that combines ticker extraction and
    comparison detection into a single result.
    
    Args:
        question: Free-form question text
        
    Returns:
        Dictionary with keys:
        - "tickers": List of extracted ticker symbols
        - "compare": Boolean indicating comparison intent
        
    Examples:
        >>> parse_intent("Compare AAPL vs MSFT last 3 months")
        {'tickers': ['AAPL', 'MSFT'], 'compare': True}
        >>> parse_intent("rsi on nvda ytd")
        {'tickers': ['NVDA'], 'compare': False}
        >>> parse_intent("is macd bullish on spy?")
        {'tickers': ['SPY'], 'compare': False}
        >>> parse_intent("Show me $T performance versus $VZ")
        {'tickers': ['T', 'VZ'], 'compare': True}
        >>> parse_intent("What's the AI market outlook?")
        {'tickers': [], 'compare': False}
    """
    if not question:
        return {"tickers": [], "compare": False}
        
    # Extract tickers from the question
    tickers = extract_tickers(question)
    
    # Detect comparison intent
    compare = detect_compare(question, tickers)
    
    return {
        "tickers": tickers,
        "compare": compare
    }
