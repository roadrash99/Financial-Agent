

import re
from typing import List, Dict

__all__ = ["extract_tickers", "detect_compare", "parse_intent"]


TICKER_ALIASES = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "nvidia": "NVDA",
    "amazon": "AMZN",
    "meta": "META",
    "tesla": "TSLA",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "netflix": "NFLX",
    "intel": "INTC",
    "ibm": "IBM",
    "paypal": "PYPL",
    "adobe": "ADBE",
    "salesforce": "CRM"
    
}


COMPARE_PATTERN = re.compile(r" vs | versus | compare ", re.IGNORECASE)


def extract_tickers(text: str) -> List[str]:
    if not text:
        return []
    
    seen = set()
    tickers = []

    for name, ticker in TICKER_ALIASES.items():
        pattern = re.compile(r'\b' + re.escape(name) + r'\b', re.IGNORECASE)
        if pattern.search(text):
            if ticker not in seen:
                seen.add(ticker)
                tickers.append(ticker)
    

    alias_values = set(TICKER_ALIASES.values())
   
    direct_candidates = re.findall(r'\b\$?([A-Z]{1,5})\b', text)
    for candidate in direct_candidates:
        if candidate in alias_values and candidate not in seen:
            seen.add(candidate)
            tickers.append(candidate)
    
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
