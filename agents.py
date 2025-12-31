import streamlit as st  # For error display in the streamlit app.
import tempfile  # For temporary directories in code execution.
from autogen import AssistantAgent, UserProxyAgent, register_for_llm  # Imports for AG2 agents and tool registration.
import os  # For environment variable access.
from agent_config import AgentConfig  # Imports config utilities.

class Agents:
    """Manages AutoGen agents and their interactions for the stock analysis app."""
    
    def __init__(self):
        # Initializes the agent configuration with LLM settings and code executor.
        # Sets up the config list for all agents, using GPT-4 model and environment variables for API access.
        self.config_list = [
            {
                "model": "gpt-4",  # Valid OpenAI model for LLM interactions.
                "api_key": os.environ.get("OPENAI_API_KEY"),  # Secure API key retrieval.
                "base_url": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),  # Default or custom OpenAI endpoint.
            }
        ]
        # Initializes code executor config for safe code execution by agents.
        self.code_executor_config, self.temp_dir = AgentConfig.get_code_executor_config()

    def initialize_agents(self):
        # Main method to create and configure all agents for the stock analysis workflow.
        # Returns a tuple of agents: finance_reporting_analyst, technical_analyst, strategy_agent, user (supervisor), assistant_agent, user_proxy.
        # Handles exceptions by displaying errors in streamlit and returning None values.
        try:
            # Creates the finance reporting analyst agent for fundamental stock analysis and reporting.
            # This agent focuses on data fetching and comprehensive reports, using only the finance_data_fetch tool.
            finance_reporting_analyst = AssistantAgent(
                name="finance_reporting_analyst",  # Unique identifier for the agent.
                system_message="""
                    You are a Finance Reporting Analyst. You have to analyze the stock data for the given ticker 
                    Perform the Stock as per the user request and create a comprehensive report in markdown format.
                    To extract the data, use the tools provided.

                    Constraints:
                    - Think step by step.
                    - Use only the available tools.
                    - Do not invent data. Reflect if unsure.
                    - Provide actionable insights and recommendations.
                    - Include key financial metrics and ratios.
                    """,  # Detailed prompt for the agent's role and constraints.
                human_input_mode="NEVER",  # Agent responds automatically without human input.
                llm_config={
                    "config_list": self.config_list,  # Uses shared LLM config.
                    "timeout": 280,  # Timeout for LLM responses.
                    "temperature": 0.5,  # Balanced creativity for analysis.
                },
            )

            # Registers the finance_data_fetch tool specifically for this agent.
            # This allows the agent to call the tool during conversations for data retrieval.
            register_for_llm(
                FinanceTools.finance_data_fetch,  # The tool function to register.
                caller=finance_reporting_analyst,  # The agent that can call the tool.
                executor=user  # The user proxy that executes the tool (defined later).
            )

            # Creates the technical analyst agent for indicator-based trend analysis.
            # This agent analyzes SMA, EMA, RSI, and close prices, providing insights without summaries.
            technical_analyst = AssistantAgent(
                name="technical_analyst",  # Unique identifier.
                system_message="""
                    You are a Technical Analyst specializing in identifying stock trends using technical indicators.
                    Use only the tools provided to analyze data using the following indicators:
                        - Simple Moving Average (SMA)
                        - Exponential Moving Average (EMA)
                        - Relative Strength Index (RSI)
                        - Last Close Price

                    Focus on identifying trends, patterns, and signals (e.g., crossovers, momentum shifts, overbought/oversold conditions).
                    Provide concise insights and interpretations for each indicator.
                    Do not include any summaries, financial reports, or performance overviews—these are handled by a separate reporting agent.
                    Avoid responding to any human input directly.
                    """,  # Prompt defining technical analysis focus.
                human_input_mode="NEVER",  # Automatic responses.
                llm_config={
                    "config_list": self.config_list,  # Shared config.
                    "timeout": 200,  # Shorter timeout for technical data.
                    "temperature": 0.5,  # Consistent analysis.
                },
            )

            # Registers the technical_analysis_tool for this agent.
            register_for_llm(
                FinanceTools.technical_analysis_tool,
                caller=technical_analyst,
                executor=user
            )

            # Creates the strategy agent for buy/sell recommendations and risk assessment.
            # This agent evaluates signals (MACD, RSI) and risks (beta, volatility), providing recommendations.
            strategy_agent = AssistantAgent(
                name="strategy_agent",  # Unique identifier.
                system_message="""
                    You are a Strategy Analyst responsible for recommending Buy/Sell actions while also evaluating the risk profile of a stock.

                    Your responsibilities include:
                    - Analyzing trading signals:
                        • MACD and its signal line (for momentum)
                        • RSI (for overbought/oversold signals)
                        • Last Close Price
                    - Assessing risk indicators:
                        • Volatility (e.g., 52-week change)
                        • Beta (systematic market risk)
                        • Dividend Yield and Stability (consistency)

                    Recommendation Logic:
                    - If MACD > Signal and RSI < 70, consider "Buy" (bullish signal).
                    - If MACD < Signal or RSI > 70, consider "Sell" (bearish/overbought).
                    - If RSI is near 50 or indicators conflict, consider "Hold".
                    - Cross-check recommendation against risk metrics:
                        • Flag "High Risk" if Beta > 1.2 or high Volatility
                        • Mention "Stable" if Beta < 0.9 and steady dividends

                    Your response must:
                    - Include a Buy/Sell/Hold recommendation.
                    - Clearly state 2-3 sentence rationale using both signals and risk metrics.
                    - Optionally suggest caution if risk is high despite positive signals.

                    Do NOT perform raw calculations. Use only the tools provided.
                    Do NOT summarize financial performance or market news.
                    Do NOT respond to user inputs.
                    """,  # Prompt for strategy and risk logic.
                human_input_mode="NEVER",  # Automatic.
                llm_config={
                    "config_list": self.config_list,  # Shared config.
                    "timeout": 300,  # Longer timeout for complex recommendations.
                    "temperature": 0.5,  # Balanced for decision-making.
                },
            )

            # Registers risk_assessment_tool and strategy_signal_tool for this agent.
            register_for_llm(
                FinanceTools.risk_assessment_tool,
                caller=strategy_agent,
                executor=user
            )
            register_for_llm(
                FinanceTools.strategy_signal_tool,
                caller=strategy_agent,
                executor=user
            )

            # Creates the supervisor user proxy agent to orchestrate the workflow.
            # This agent manages the conversation flow between user and other agents, ensuring the analysis sequence.
            user = UserProxyAgent(
                name="supervisor",  # Identifier as coordinator.
                system_message="""
                    You are the coordinator agent responsible for managing and orchestrating the financial analysis workflow between the user and the following agents:

                    
                    1. finance_reporting_analyst - Creates high-level reports.
                    2. technical_analyst - Performs technical indicator analysis.
                    3. strategy_agent - Recommends Buy/Sell/Hold based on trading signals and risk assessment.

                    ### Workflow:
                    1. Start with `finance_reporting_analyst` to get stock data and initial analysis report.
                    2. Pass the output to:
                        - `finance_reporting_analyst` for adding fundamental points to above report
                        - `technical_analyst` for adding indicator-based insights to above report
                    3. After both are done, call `strategy_agent` using the combined data.
                    4. Present a final, consolidated summary to the user.

                    ### Rules:
                    - Do not invent information.
                    - Use only available tools and agent responses.
                    - Complete the full flow before presenting output.
                    - If any agent fails, return a helpful explanation and gracefully handle the failure.
                    """,  # Workflow instructions for orchestration.
                human_input_mode="NEVER",  # Handles orchestration automatically.
                max_consecutive_auto_reply=3,  # Limits auto-replies to prevent loops.
                code_execution_config={"executor": self.code_executor_config},  # Enables code execution.
            )

            # Creates the assistant agent as a graceful fallback for single-agent mode.
            # This agent has all tools registered for comprehensive, direct analysis when multi-agent is not needed.
            assistant_agent = AgentConfig.get_assistant_agent()

            # Creates the user proxy for direct interaction in single-agent mode.
            # This allows human input for chat with the assistant_agent.
            user_proxy = UserProxyAgent(
                name="User",  # Identifier for user interactions.
                human_input_mode="ALWAYS",  # Requires human input for responses.
                code_execution_config=AgentConfig.get_code_executor_config()[0]  # Code execution setup.
            )

            # Returns all initialized agents for use in the app.
            return finance_reporting_analyst, technical_analyst, strategy_agent, user, assistant_agent, user_proxy
        
        except Exception as e:
            # Displays any initialization errors in the streamlit app.
            st.error(f"Error initializing agents: {e}")
            # Returns None for all agents on failure.
            return None, None, None, None, None, None