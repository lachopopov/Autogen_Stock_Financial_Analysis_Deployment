import os
import tempfile
from autogen.coding import LocalCommandLineCodeExecutor
from autogen import AssistantAgent
from tools import FinanceTools

class AgentConfig:
    """Configuration settings for the AutoGen application."""
    
    @staticmethod
    def get_llm_config():
        # Returns the LLM configuration list for AG2 agents, including model, API key, and base URL.
        # This is used by all agents for language model interactions.
        return [
            {
                "model": "gpt-4",  # Updated from invalid "gpt-4.1-nano" to valid OpenAI model for compatibility.
                "api_key": os.environ.get("OPENAI_API_KEY"),  # Retrieves API key from environment variables for security.
                "base_url": os.environ.get("OPENAI_BASE_URL"),  # Optional base URL for custom OpenAI endpoints.
            }
        ]
    
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
    def get_assistant_agent():
        # Creates and returns an AssistantAgent with all FinanceTools registered for single-agent mode.
        # This serves as a graceful fallback for direct, comprehensive stock analysis when multi-agent orchestration is not needed.
        # Conditions for use: Triggered by user selection of "single-agent mode" in the app UI (e.g., via a toggle in streamlit).
        # Triggers: Activated when the app detects single-agent mode preference, bypassing GroupChat for simpler queries.
        # Integration: Used in agents.py for initialization, paired with user_proxy for direct chat interactions.
        # Purpose: Provides flexibility for users who want quick, all-in-one analysis without specialized agent roles.
        # Note: All tools are registered here for broad capability, unlike selective registration in multi-agent mode.
        assistant = AssistantAgent(
            name="Stock_Analyst",  # Unique name for the agent in conversations.
            llm_config=AgentConfig.get_llm_config()[0],  # Uses the standard LLM config for consistency.
            system_message="You are a financial analyst. Use the provided tools to analyze stocks."  # Defines the agent's role and behavior.
        )
        # Register all tools for comprehensive analysis capability in single-agent mode.
        assistant.register_for_llm(FinanceTools.finance_data_fetch)  # Registers tool for fetching stock data.
        assistant.register_for_llm(FinanceTools.technical_analysis_tool)  # Registers tool for technical indicators.
        assistant.register_for_llm(FinanceTools.risk_assessment_tool)  # Registers tool for risk evaluation.
        assistant.register_for_llm(FinanceTools.strategy_signal_tool)  # Registers tool for trading signals.
        return assistant  # Returns the fully configured agent for use in the app.