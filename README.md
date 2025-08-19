# Agentic Financial Analyst (LangGraph + Groq + pandas + yfinance)

LLM-orchestrated agent that **plans → fetches/analyzes → explains** stock behavior on demand.
A **Router** LLM emits a strict-JSON tool plan; Python tools fetch OHLCV (yfinance), compute pandas-only indicators (SMA/EMA/RSI/MACD/Bollinger) and risk metrics; a **Finalizer** LLM writes a concise, date-anchored summary. No investment advice—just numbers.

---

## ✨ What it does

* **Ask in plain English:** "Compare AAPL vs MSFT last 3 months. Are either overbought?"
* **Agent executes:** fetches prices → computes indicators → summarizes returns, volatility, drawdown, trend, RSI/MACD/BB context.
* **Deterministic parsing:** tickers/timeframes resolved without LLM; strict JSON contracts for tool plans; bounded loop (≤3).

---

## 🧠 Architecture (decide → do → tell)

```mermaid
flowchart LR
  U[User Question] --> P[Parse intent + timeframes<br/>(deterministic)]
  P --> R{Router (LLM)<br/>plan as JSON}
  R -- CALL_TOOLS --> T[Tools node (Python)]
  T --> R
  R -- FINALIZE --> F[Finalizer (LLM)]
  F --> A[Final Answer]
```

* **Parse (deterministic):** extract `tickers`, `start/end`, `interval`, `compare`.
* **Router (LLM via Groq):** outputs **strict JSON** plan per `schemas.py`.
* **Tools (Python):** `fetch_prices` (yfinance) → `compute_indicators_pandas` → `summarize_metrics`.
* **Finalizer (LLM):** crafts a 4–7 sentence summary using **metrics only** (no DataFrames to the LLM).

---

## 📁 Repo structure

```
agentic-fin-analyst/
├─ src/afa/
│  ├─ config.py                # Groq (ChatGroq) clients: router/finalizer
│  ├─ state.py                 # ConversationState, Parsed, Metrics, initial_state()
│  ├─ prompts.py               # system / router / finalizer prompts
│  ├─ schemas.py               # RouterPlan + validators (JSON contracts)
│  ├─ graph.py                 # LangGraph wiring (router → tools ↺ → finalizer)
│  ├─ parsing/
│  │  ├─ intent.py             # tickers + compare (regex, stopwords)
│  │  └─ timeframes.py         # "last 3 months", "YTD" → start/end/interval
│  ├─ nodes/
│  │  ├─ router.py             # calls Groq, validates plan
│  │  ├─ tools_node.py         # runs tool_calls; updates dataframes/metrics
│  │  └─ finalizer.py          # calls Groq to write the final answer
│  ├─ tools/
│  │  ├─ prices.py             # yfinance download → per-ticker OHLCV
│  │  ├─ indicators.py         # pandas-only SMA/EMA/RSI/MACD/BB
│  │  └─ metrics.py            # returns, vol, drawdown, slope, RSI/MACD/BB
│  └─ cli/
│     └─ run.py                # CLI entry: one-command analysis
├─ requirements.txt
└─ README.md
```

---

## ⚙️ Setup

```bash
# 1) Clone & enter
git clone <your-repo-url> && cd agentic-fin-analyst

# 2) Create venv (recommended)
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3) Install deps
pip install -r requirements.txt

# 4) Configure Groq
# Set your GROQ_API_KEY environment variable
export GROQ_API_KEY=xxxxxxxxxxxxxxxx
```

**requirements.txt**

```
langgraph
groq
pandas
numpy
yfinance
python-dotenv
pytest
```

---

## 🚀 Quickstart (CLI)

```bash
# Show parsed window/tickers then run the agent
python -m afa.cli.run --show-parsed "Compare AAPL vs MSFT last 3 months. Are either overbought?"
```

**What you'll see**

* Parsed dict (tickers/start/end/interval/compare)
* Final paragraph with dates, returns, RSI/MACD/BB context, volatility/drawdown, and a relative performance sentence.

> Tip: No API key needed for yfinance. It fetches public data; we cache intelligently in code (optional).

---

## 🔑 Configuration

Set via environment variables:

* `GROQ_API_KEY` (required)

The current implementation uses:
* **Router model:** `llama3-8b-8192` (temperature: 0.0, max_tokens: 256)
* **Finalizer model:** `llama3-8b-8192` (temperature: 0.1, max_tokens: 512)

---

## 🧪 Testing

```bash
pytest -q
```

Covers:

* `prices.py` (columns/index, empty ticker handling)
* `indicators.py` (NaN warm-ups; `macd_hist == macd - macd_signal`)
* `metrics.py` (return/vol/drawdown/slope; MACD/BB states)
* Graph smoke test (end-to-end final answer with recent window)

---

## 📏 Design principles

* **Decide / Do / Tell:** separate planning (router), execution (tools), and narration (finalizer).
* **Deterministic parsing:** no LLM for dates/tickers; same input → same parsed output.
* **Schema-validated plans:** router must output JSON that matches `RouterPlan`.
* **Token economy:** LLMs see **metrics only**, never full DataFrames.
* **Termination guarantees:** bounded loop (≤3); `final_answer` ends runs gracefully.
* **No advice:** prompts forbid recommendations or forward-looking statements.

---

## 🔧 Extending

* **More indicators:** add in `tools/indicators.py`; surface optional fields in `Metrics`.
* **New data sources:** create a new tool; reference it in `schemas.py` and the router prompt.
* **API server:** wrap CLI logic in FastAPI (e.g., `POST /analyze`) and containerize.

---

## 🗂 Example router plan (strict JSON)

```json
{
  "next_action": "CALL_TOOLS",
  "tool_calls": [
    {"name":"fetch_prices","args":{"tickers":["AAPL","MSFT"],"start":"2025-05-20","end":"2025-08-19","interval":"1d"}},
    {"name":"compute_indicators","args":{"indicators":["sma20","rsi14","macd"]}},
    {"name":"summarize_metrics","args":{}}
  ]
}
```

---

## ❓FAQ

**Does yfinance need an API key?**
No—just works. Be mindful of rate limits; keep tickers ≤5 per call.

**Why two LLMs?**
Split responsibilities: the **Router** plans minimal tools; the **Finalizer** writes. Smaller prompts, fewer tokens, clearer tests.

**What about fundamentals/news?**
Out of scope by design. This project focuses on technical/price analysis only.

---

## ⚠️ Disclaimer

This project is for educational/analytical purposes. It **does not provide investment advice**. Always do your own research.

---

## 📄 License

MIT (suggested). Add your preferred license file to the repo.