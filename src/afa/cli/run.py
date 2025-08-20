

import argparse
import json
import sys
from typing import Optional


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
    if args.question:
        return args.question.strip()
    

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

    intent_data = parse_intent(question)

    start, end, interval = resolve_timeframe(question, today=today)

    parsed = {
        "tickers": intent_data["tickers"],
        "compare": intent_data["compare"],
        "start": start,
        "end": end,
        "interval": interval
    }
    
    return parsed


def run_analysis(question: str, parsed: dict, timeout: int) -> str:
    try:

        state = initial_state(question, parsed)

        app = build_graph()
        result = app.invoke(state, {"recursion_limit": 10, "timeout": timeout})

        final_answer = result.get("final_answer")
        if not final_answer:
            return "No answer produced; check logs."
        
        return final_answer
        
    except Exception as e:

        print(f"Error during analysis: {str(e)}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    

    question = get_question(args)

    try:
        parsed = parse_question(question, args.today)
    except Exception as e:
        print(f"Error parsing question: {str(e)}", file=sys.stderr)
        sys.exit(1)

    if args.show_parsed:
        print(json.dumps(parsed, indent=2))

    answer = run_analysis(question, parsed, args.timeout)

    print(answer)


if __name__ == "__main__":
    main()
