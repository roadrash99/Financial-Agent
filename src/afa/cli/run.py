"""Minimal command-line entry point for the Agentic Financial Analyst.

Example usage:
$ python -m afa.cli.run "Compare AAPL vs MSFT last 3 months. Are we overbought?"
$ python -m afa.cli.run "AAPL last 3 months" --show-parsed
$ python -m afa.cli.run "SPY YTD performance" --today 2025-08-15 --timeout 120
$ echo "What's the RSI on NVDA past week?" | python -m afa.cli.run
Note that GROQ_API_KEY must be set (or in .env).
"""

import argparse
import json
import sys
from typing import Optional

# Optional dotenv support - graceful fallback if not installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from afa.parsing.intent import parse_intent
from afa.parsing.timeframes import resolve_timeframe
from afa.state import initial_state
from afa.graph import build_graph


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Agentic Financial Analyst - Ask questions about stocks",
        prog="python -m afa.cli.run"
    )
    
    parser.add_argument(
        "question",
        nargs="?",
        help="Free-form question about stocks (if omitted, reads from stdin)"
    )
    
    parser.add_argument(
        "--today",
        metavar="YYYY-MM-DD",
        help="Anchor date for relative time parsing (useful for deterministic tests)"
    )
    
    parser.add_argument(
        "--show-parsed",
        action="store_true",
        help="Print the parsed intent and timeframe data before running"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        metavar="SECONDS",
        help="Overall run timeout in seconds (default: 60)"
    )
    
    return parser


def get_question(args) -> str:
    """Get the question from args or stdin, with proper error handling."""
    if args.question:
        return args.question.strip()
    
    # Read from stdin
    try:
        question = sys.stdin.read().strip()
        if not question:
            print("Error: No question provided via argument or stdin.", file=sys.stderr)
            print("Use --help for usage information.", file=sys.stderr)
            sys.exit(2)
        return question
    except (EOFError, KeyboardInterrupt):
        print("Error: Failed to read question from stdin.", file=sys.stderr)
        sys.exit(2)


def parse_question(question: str, today: Optional[str]) -> dict:
    """Parse the question into structured data."""
    # Parse intent (tickers and comparison)
    intent_data = parse_intent(question)
    
    # Parse timeframe
    start, end, interval = resolve_timeframe(question, today=today)
    
    # Merge into single parsed dict
    parsed = {
        "tickers": intent_data["tickers"],
        "compare": intent_data["compare"],
        "start": start,
        "end": end,
        "interval": interval
    }
    
    return parsed


def run_analysis(question: str, parsed: dict, timeout: int) -> str:
    """Run the analysis and return the final answer."""
    try:
        # Initialize state
        state = initial_state(question, parsed)
        
        # Build and invoke graph
        app = build_graph()
        result = app.invoke(state, {"recursion_limit": 10, "timeout": timeout})
        
        # Extract final answer
        final_answer = result.get("final_answer")
        if not final_answer:
            return "No answer produced; check logs."
        
        return final_answer
        
    except Exception as e:
        # Handle any runtime errors
        print(f"Error during analysis: {str(e)}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Get the question
    question = get_question(args)
    
    # Parse the question
    try:
        parsed = parse_question(question, args.today)
    except Exception as e:
        print(f"Error parsing question: {str(e)}", file=sys.stderr)
        sys.exit(1)
    
    # Show parsed data if requested
    if args.show_parsed:
        print(json.dumps(parsed, indent=2))
    
    # Run analysis
    answer = run_analysis(question, parsed, args.timeout)
    
    # Print final answer
    print(answer)


if __name__ == "__main__":
    main()
