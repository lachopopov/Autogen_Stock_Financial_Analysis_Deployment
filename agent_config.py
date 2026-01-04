import os
import tempfile
from autogen.coding import LocalCommandLineCodeExecutor
from autogen import AssistantAgent, UserProxyAgent
from autogen.agentchat import register_function  # Added for correct tool registration.
from tools import FinanceTools

class AgentConfig:
    """Configuration settings for the AutoGen application."""

    @staticmethod
    def get_selected_model() -> str:
        # Selected via Streamlit (OPENAI_MODEL) with a safe default.
        return os.environ.get("OPENAI_MODEL", "gpt-5-mini")

    @staticmethod
    def get_default_temperature_for_model(model: str | None = None) -> int:
        # NOTE: Some models restrict allowed temperature values.
        # - gpt-5-mini: only supports the default temperature (1).
        # - gpt-5-nano: only supports the default temperature (1).
        m = (model or AgentConfig.get_selected_model()).strip()
        # Both gpt-5-mini and gpt-5-nano require the default temperature.
        return 1
    
    @staticmethod
    def get_llm_config():
        # Returns the LLM configuration list for AG2 agents, including model, API key, and base URL.
        # This is used by all agents for language model interactions.
        return [
            {
                "model": AgentConfig.get_selected_model(),
                "api_key": os.environ.get("OPENAI_API_KEY"),  # Retrieves API key from environment variables for security.
                "base_url": os.environ.get("OPENAI_BASE_URL"),  # Optional base URL for custom OpenAI endpoints.
            }
        ]

    @staticmethod
    def get_llm_runtime_config(*, timeout: int, temperature: int | None = None) -> dict:
        # Convenience wrapper for passing into AssistantAgent/GroupChatManager llm_config.
        temp = AgentConfig.get_default_temperature_for_model() if temperature is None else temperature
        return {"config_list": AgentConfig.get_llm_config(), "timeout": timeout, "temperature": temp}
    
    @staticmethod
    def get_code_executor_config():
        # Sets up a temporary directory for code execution and returns the LocalCommandLineCodeExecutor config.
        # This allows agents to run code safely in an isolated environment.
        temp_dir = tempfile.TemporaryDirectory()
        return LocalCommandLineCodeExecutor(
            timeout=30,  # Timeout in seconds for code execution to prevent hangs.
            work_dir=temp_dir.name,  # Working directory set to temporary folder for security.
        ), temp_dir
    
    @staticmethod
    def get_assistant_agent(executor: UserProxyAgent | None = None) -> AssistantAgent:
        # Creates and returns an AssistantAgent with all FinanceTools registered for single-agent mode.
        # This serves as a graceful fallback for direct, comprehensive stock analysis when multi-agent orchestration is not needed.
        # Conditions for use: Triggered by user selection of "single-agent mode" in the app UI (e.g., via a toggle in streamlit).
        # Triggers: Activated when the app detects single-agent mode preference, bypassing GroupChat for simpler queries.
        # Integration: Used in agents.py for initialization, paired with user_proxy for direct chat interactions.
        # Purpose: Provides flexibility for users who want quick, all-in-one analysis without specialized agent roles.
        # Note: All tools are registered here for broad capability, unlike selective registration in multi-agent mode.
        if executor is None:
            raise ValueError("Single-agent mode requires a UserProxyAgent executor for tool execution.")

        assistant = AssistantAgent(
            name="Stock_Analyst",  # Unique name for the agent in conversations.
            llm_config=AgentConfig.get_llm_runtime_config(timeout=280),
            system_message=(
                "You are a financial analyst. Use ONLY the provided tools to analyze stocks. "
                "Return concise markdown. Do not invent data. Never paste raw tool JSON; summarize it. "
                "\n\nCost/behavior limits (MUST follow):"
                "\n- Call finance_data_fetch at most once."
                "\n- Do not call the same tool more than once."
                "\n- Default behavior: if the user provides only a ticker (e.g., 'AAPL') or asks whether to invest, you MUST call (once each) finance_data_fetch, strategy_signal_tool, and risk_assessment_tool, then give a Buy/Sell/Hold recommendation with a brief rationale."
                "\n- Only call technical_analysis_tool if the user explicitly asks for technicals/RSI/SMA/EMA."
                "\n- If you lack data, say so and stop."
                "\n- Do NOT ask follow-up questions. End after your answer."
                "\n- Do not include any question marks (?) in your response."
                "\n- After you finish your final answer, output a final line containing exactly: TERMINATE"
                "\n\nIf the user wants more, they can explicitly ask: 'AAPL technicals'."
            ),
        )
        # Register all tools for comprehensive analysis capability in single-agent mode.
        register_function(
            f=FinanceTools.finance_data_fetch,  # Registers tool for fetching stock data.
            caller=assistant,  # The agent that can call the tool.
            executor=executor,
            name="finance_data_fetch",  # Unique name for the tool.
            description="Fetch recent stock information for a ticker symbol"  # Description for the LLM.
        )
        register_function(
            f=FinanceTools.technical_analysis_tool,
            caller=assistant,
            executor=executor,
            name="technical_analysis_tool",
            description="Perform technical analysis using moving averages and RSI"
        )
        register_function(
            f=FinanceTools.risk_assessment_tool,
            caller=assistant,
            executor=executor,
            name="risk_assessment_tool",
            description="Perform risk evaluation using beta, volatility, and dividend yield"
        )
        register_function(
            f=FinanceTools.strategy_signal_tool,
            caller=assistant,
            executor=executor,
            name="strategy_signal_tool",
            description="Evaluate trading signals using MACD, RSI, and closing price"
        )
        return assistant  # Returns the fully configured agent for use in the app.