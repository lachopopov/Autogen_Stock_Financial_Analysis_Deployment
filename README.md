# Autogen_Stock_Analysis
# Autogen_Stock_Analysis

This project builds an AI/Agent application for stock analysis using Autogen (AG2). It leverages multiple specialized agents to fetch, analyze, and report on stock data, enabling automated financial insights through conversational AI. The app integrates yfinance for data retrieval and OpenAI for LLM-powered analysis.

## Features
- **Data Fetching**: Retrieve real-time stock information, including price, market cap, P/E ratio, and historical data.
- **Technical Analysis**: Compute indicators like SMA, EMA, RSI for trend analysis.
- **Risk Assessment**: Evaluate beta, volatility, dividend yield, and assign risk ratings.
- **Strategy Signals**: Generate trading signals using MACD, RSI, and price data.
- **Agent Collaboration**: Multi-agent group chat for comprehensive analysis (e.g., finance reporting, technical, and strategy agents).
- **Interactive UI**: Built with Streamlit for user-friendly input and output.

## Architecture
- **Agents**: Defined in `agents.py` (e.g., `finance_reporting_analyst`, `technical_analyst`, `strategy_agent`, `user` proxy).
- **Tools**: Implemented in `tools.py` (`FinanceTools` class) for data processing.
- **Configuration**: Centralized in `agent_config.py` (LLM, tools, code executor).
- **Orchestration**: Managed in `agent_orchestrator.py` using AG2's GroupChat.
- **App Entry**: `app.py` for Streamlit interface.

## Requirements
- Python 3.8+
- Dependencies: See `requirements.txt` (autogen, streamlit, openai, yfinance, python-dotenv).

## Installation
1. Clone the repository.
2. Create a virtual environment: `python -m venv ag2_env`.
3. Activate it: `source ag2_env/bin/activate` (Linux/Mac) or `ag2_env\Scripts\activate` (Windows).
4. Install dependencies: `pip install -r requirements.txt`.
5. Set environment variables in a `.env` file or shell:
   - `OPENAI_API_KEY`: Your OpenAI API key.
   - `OPENAI_BASE_URL`: Optional, for custom OpenAI endpoints.
6. Run the app: `streamlit run app.py`.

## Usage
1. Launch the app via Streamlit.
2. Input a stock ticker (e.g., "AAPL").
3. The agents will collaborate in a group chat to provide analysis.
4. View results, including data summaries, technical indicators, risk assessments, and signals.

Example Output:
- Fetched data: Current price, market cap, etc.
- Analysis: SMA/EMA/RSI values.
- Risk: Beta-based rating.
- Signals: MACD/RSI-based recommendations.

## Development Best Practices
- Follow `aitk-get_agent_code_gen_best_practices` for agent design.
- Use `aitk-get_tracing_code_gen_best_practices` for logging and debugging.
- Adhere to `aitk-get_ai_model_guidance` for LLM usage.
- For evaluation, use `aitk-evaluation_planner` and related tools.

## Changelog
### v0.10.3 to Latest AG2 Refactoring (Applied [Current Date])
- **File: `agent_orchestrator.py`**
  - Changed `register_function` to `register_for_llm` for tool registration (latest AG2 auto-generates schemas from function signatures).
  - Removed `tools_list` from `AgentConfig.get_tools_list()`; now relies on function docstrings for descriptions.
  - Simplified registration loop; directly passes callable functions.
  - **Why**: Improves compatibility with latest AG2, reduces manual schema maintenance.
  - **Rollback**: Revert import to `from autogen.agentchat import register_function`, restore `tools_list` usage, and use schemas from `get_tools_list()`. Ensure AG2 v0.10.3 is installed.

- **File: `tools.py`**
  - Added type hints (e.g., `ticker: str, period: str = "1mo"`) and docstrings to all methods for better AG2 schema auto-generation.
  - **Why**: Enhances tool discoverability and follows `aitk-get_agent_code_gen_best_practices`.
  - **Rollback**: Remove type hints and docstrings; code remains functional in v0.10.3.

- **General Notes**
  - Tested with AG2 latest; if issues, rollback to v0.10.3 by changing `requirements.txt` to `autogen==0.10.3` and reverting code.
  - Follows `stock_analyzer_plan.md` for phased refactoring.

## Contributing
- Follow the project's Copilot instructions: Reference `.copilot/README.copilot.md` for scope, use `.copilot/agents/senior_ag2.md` as pair programmer, and align with `.copilot/plans/stock_analyzer_plan.md` for workflows.
- Ensure code adheres to AI Toolkit best practices.

## License
[Add license if applicable, e.g., MIT]