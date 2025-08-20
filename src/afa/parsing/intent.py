

import re
from typing import List, Dict

__all__ = ["extract_tickers", "detect_compare", "parse_intent"]


STOPWORDS = {
    "RSI", "MACD", "EMA", "SMA", "BB", "BBANDS", "VWAP", "YTD",
    "USD", "EPS", "PE", "ETF", "IPO", "EV", "EBITDA", "AI",
    "AND", "OR", "VS", "V", "VS.", "THE", "A", "AN",
    "HOW", "WHAT", "WHEN", "WHERE", "WHO", "WHY", "WHICH", "THAT",
    "THIS", "THESE", "THOSE", "ARE", "IS", "WAS", "WERE", "BE", "BEEN",
    "HAVE", "HAS", "HAD", "DO", "DOES", "DID", "WILL", "WOULD", "COULD",
    "SHOULD", "MAY", "MIGHT", "CAN", "MUST", "SHALL",
    "IN", "ON", "AT", "BY", "FOR", "WITH", "FROM", "TO", "OF", "AS",
    "BUT", "IF", "SO", "UP", "OUT", "ALL", "ANY", "SOME", "NO", "NOT",
    "LAST", "PAST", "RECENT", "NOW", "TODAY", "LATE", "EARLY", "NEW", "OLD",
    "GOOD", "BAD", "BIG", "SMALL", "HIGH", "LOW", "LONG", "SHORT",
    "STOCK", "STOCKS", "SHARE", "PRICE", "TREND", "RALLY", "CRASH",
    "LIKE", "ABOUT", "OVER", "UNDER", "BEST", "WORST", "TOP", "SHOW",
    "CHECK", "LOOK", "SEE", "GET", "TAKE", "MAKE", "GIVE", "TELL",
    "MUCH", "MANY", "MORE", "MOST", "LESS", "LEAST", "VERY", "QUITE",
    "JUST", "ONLY", "ALSO", "EVEN", "STILL", "BACK", "DOWN", "AWAY"
}

TICKER_PATTERN = re.compile(r"\b\$?([A-Z]{1,5})\b")

COMPARE_PATTERN = re.compile(r" vs | versus | compare ", re.IGNORECASE)


def extract_tickers(text: str) -> List[str]:
    if not text:
        return []
    

    matches = TICKER_PATTERN.findall(text.upper())

    seen = set()
    tickers = []
    
    for match in matches:
        ticker = match.upper()
        

        if ticker in seen:
            continue

        if ticker in STOPWORDS:
            continue

        if len(ticker) <= 2:


            dollar_prefixed = f"${ticker}" in text.upper()
            if len(ticker) == 1:

                if not dollar_prefixed:
                    continue
            elif len(ticker) == 2:

                if not dollar_prefixed:
                    continue
        
        seen.add(ticker)
        tickers.append(ticker)
    
    return tickers


def detect_compare(text: str, tickers: List[str]) -> bool:
    if not text:
        return False

    if len(tickers) >= 2:
        return True

    if COMPARE_PATTERN.search(text):
        return True
        
    return False


def parse_intent(question: str) -> Dict[str, any]:
    if not question:
        return {"tickers": [], "compare": False}

    tickers = extract_tickers(question)

    compare = detect_compare(question, tickers)
    
    return {
        "tickers": tickers,
        "compare": compare
    }
