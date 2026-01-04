# Autogen Stock Analysis (AG2)

Streamlit app for stock analysis built on **AG2 (AutoGen) v0.10.3**.

It uses a small toolchain (yfinance-backed) for fundamentals, technical indicators, strategy signals, and risk metrics, then produces a concise markdown report.

## What you get
- **Streamlit UI**: enter a ticker or a question, run analysis, download a markdown report.
- **Two run modes**:
  - **Multi-agent** (default): finance + technical + strategy agents with a lightweight deterministic router.
  - **Single-agent** (optional): one assistant with tools (enabled by a sidebar toggle) that returns a bounded, tool-backed recommendation.
- **Tool-backed metrics** (no invented data):
  - Fundamentals: price, market cap, P/E, P/B, dividend rate, business summary (truncated)
  - Technicals: SMA/EMA/RSI, last close
  - Signals: MACD + MACD signal
  - Risk: beta, volatility, dividend yield, risk rating
- **Usage/cost visibility**: per-agent usage plus totals (shown in the UI).

## Repo structure
- `app.py`: Streamlit UI (model selector, single-agent toggle, Run button, report rendering)
- `agents.py`: agent setup + single-agent runner
- `agent_orchestrator.py`: multi-agent orchestration + deterministic stage routing + report consolidation
- `tools.py`: FinanceTools tool implementations
- `agent_config.py`: model + runtime config (temperature enforced for supported models)

## Requirements
- Python 3.12
- Dependencies: see `requirements.txt` (includes `ag2==0.10.3`, `streamlit`, `openai`, `yfinance`, `python-dotenv`)

## Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configure
The app reads env vars and also loads a local `.env` file (if present).

Required:
- `OPENAI_API_KEY`

Optional:
- `OPENAI_BASE_URL` (custom OpenAI-compatible endpoint)
- `OPENAI_MODEL` (overridden by the UI model selector when you use the app)

## Run
```bash
streamlit run app.py
```

In the sidebar you can:
- select model (`gpt-5-mini` or `gpt-5-nano`)
- enable/disable **Single-agent mode**
- provide the API key

## Deploy to Streamlit Community Cloud
This repo is ready to deploy on Streamlit Community Cloud.

1) Push this repo to GitHub.
2) In Streamlit Community Cloud, create a new app from the repo.
3) Set the entrypoint to:
  - `app.py`
4) Set app secrets (Settings → Secrets):
  - `OPENAI_API_KEY` (required)
  - Optional: `OPENAI_BASE_URL`, `AG2_ROUTING_MODE`, `AG2_MANAGER_LLM`, `AG2_INCLUDE_MANAGER_USAGE`

Notes:
- `.env` is for local development; it will not be present in Streamlit Cloud.
- Python is pinned via `runtime.txt`.

## Single-agent mode behavior (cost-bounded)
Single-agent mode is designed to be predictable and cheap.

- If the input is **ticker-only** (e.g., `AAPL`), it runs a bounded strategy workflow and returns **Buy/Sell/Hold**:
  - Calls (once each): `finance_data_fetch` → `strategy_signal_tool` → `risk_assessment_tool`
  - Does **not** call `technical_analysis_tool` unless you explicitly ask for technicals.
- If the input explicitly asks for technicals (RSI/SMA/EMA), it may call `technical_analysis_tool`.
- Tool calls are capped to avoid loops and runaway token usage.

## Multi-agent routing and manager controls
The multi-agent orchestrator supports two routing styles:

- `AG2_ROUTING_MODE=deterministic` (default)
  - Stage router: fundamentals → technicals → strategy → terminate
  - Manager LLM is disabled by default to reduce overhead
- `AG2_ROUTING_MODE=nondeterministic`
  - Uses the GroupChat manager’s natural speaker selection (typically higher variance)
  - Typically results in higher token usage and cost than `deterministic`

Manager toggles:
- `AG2_MANAGER_LLM=1|0` : enable/disable manager LLM configuration
- `AG2_INCLUDE_MANAGER_USAGE=1|0` : include manager usage object in the usage output

## Output format
The final report is markdown and includes a unified:
- **Key Metrics** section assembled from tool payloads (valuation + indicators + MACD + risk fields)

## Troubleshooting
- If you see **insufficient_quota**: the API key/org has no remaining credits. Add billing/credits or switch providers via `OPENAI_BASE_URL`.
- If you hit rate limits (429): try smaller prompts, switch to deterministic routing, or re-run after a short wait.

## Contributing
Follow the repo Copilot guidance:
- `.copilot/README.copilot.md`
- `.copilot/agents/senior_ag2_pair_programmer.md`
- `.copilot/plans/stock_analyzer_plan.md`