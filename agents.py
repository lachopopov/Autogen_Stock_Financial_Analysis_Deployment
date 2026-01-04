import streamlit as st  # For error display in the streamlit app.
from autogen.agentchat import AssistantAgent, UserProxyAgent, register_function
import os
import json
import re
from agent_config import AgentConfig
from tools import FinanceTools  # Imports the financial tools.

class Agents:
    """Manages AutoGen agents and their interactions for the stock analysis app."""
    
    def __init__(self):
        # Initializes the agent configuration with code executor.
        # LLM model/temperature selection is centralized in AgentConfig.
        # Initializes code executor config for safe code execution by agents.
        self.code_executor_config, self.temp_dir = AgentConfig.get_code_executor_config()

    def initialize_agents(self):
        """
        Multi-agent (GroupChat) initialization.

        IMPORTANT CONTRACT:
        - Returns exactly 4 agents in this order:
          (finance_reporting_analyst, technical_analyst, strategy_agent, supervisor_user)
        - This matches agent_orchestrator.orchestrate_agents(...).
        """
        try:
            # Create the supervisor FIRST so it can be used as the executor in register_function(...).
            user = UserProxyAgent(
                name="supervisor",
                system_message="""
You coordinate a 3-agent stock analysis workflow:

1) finance_reporting_analyst: fundamentals + short markdown report
2) technical_analyst: SMA/EMA/RSI interpretation (no fundamentals)
3) strategy_agent: Buy/Sell/Hold + risk note using tools

Rules:
- Use only tool outputs; do not invent data.
- NEVER paste raw tool JSON into responses; summarize it.
- Keep responses short (<= 12 bullet lines unless user asks for more).
""",  # Workflow instructions for orchestration.
                human_input_mode="NEVER",  # Handles orchestration automatically.
                max_consecutive_auto_reply=10,  # Limits auto-replies to prevent loops.
                code_execution_config={"executor": self.code_executor_config},  # Enables code execution.
            )

            finance_reporting_analyst = AssistantAgent(
                name="finance_reporting_analyst",  # Unique identifier for the agent.
                system_message="""
Role: Finance Reporting Analyst.
Use ONLY available tools. Write concise markdown: key metrics + brief insights.
NEVER paste raw tool JSON; summarize.
Do not invent data. If unsure, say so.

Scope:
- Fundamentals/valuation only (P/E, P/B, market cap, dividend, business summary).
- Do NOT include technical indicators (SMA/EMA/RSI/MACD) and do NOT give trading triggers.

IMPORTANT:
- Call finance_data_fetch at most once (default period) unless the user explicitly requests a different period.
""",  # Detailed prompt for the agent's role and constraints.
                human_input_mode="NEVER",  # Agent responds automatically without human input.
                llm_config=AgentConfig.get_llm_runtime_config(timeout=280),
            )

            technical_analyst = AssistantAgent(
                name="technical_analyst",  # Unique identifier.
                system_message="""
Role: Technical Analyst.
Use ONLY available tools. Explain SMA_20/EMA_20/RSI/Last_Close briefly.
NEVER paste raw tool JSON; summarize.
No fundamentals, no long report, no user interaction.

Do not mention trading volume unless you have a tool-provided volume metric.
""",  # Prompt defining technical analysis focus.
                human_input_mode="NEVER",  # Automatic responses.
                llm_config=AgentConfig.get_llm_runtime_config(timeout=200),
            )

            strategy_agent = AssistantAgent(
                name="strategy_agent",  # Unique identifier.
                system_message="""
Role: Strategy Analyst.
Use ONLY available tools. Output: Buy/Sell/Hold + 2-3 sentence rationale using signals + risk metrics.
NEVER paste raw tool JSON; summarize.
If risk is high, explicitly flag it.

Do not mention trading volume unless you have a tool-provided volume metric.
""",  # Prompt for strategy and risk logic.
                human_input_mode="NEVER",  # Automatic.
                llm_config=AgentConfig.get_llm_runtime_config(timeout=300),
            )

            # Register tools AFTER caller + executor exist.
            register_function(
                f=FinanceTools.finance_data_fetch,
                caller=finance_reporting_analyst,
                executor=user,
                name="finance_data_fetch",
                description="Fetch recent stock information for a ticker symbol",
            )
            register_function(
                f=FinanceTools.technical_analysis_tool,
                caller=technical_analyst,
                executor=user,
                name="technical_analysis_tool",
                description="Perform technical analysis using moving averages and RSI",
            )
            register_function(
                f=FinanceTools.risk_assessment_tool,
                caller=strategy_agent,
                executor=user,
                name="risk_assessment_tool",
                description="Perform risk evaluation using beta, volatility, and dividend yield",
            )
            register_function(
                f=FinanceTools.strategy_signal_tool,
                caller=strategy_agent,
                executor=user,
                name="strategy_signal_tool",
                description="Evaluate trading signals using MACD, RSI, and closing price",
            )

            # IMPORTANT: return ONLY what app.py/orchestrator expects (4 agents).
            return finance_reporting_analyst, technical_analyst, strategy_agent, user

        except Exception as e:
            # Displays any initialization errors in the streamlit app.
            st.error(f"Error initializing agents: {e}")
            return None, None, None, None

    def initialize_single_agent(self):
        """
        Single-agent initialization.

        Returns:
            (assistant, executor_user)
        """
        try:
            def _is_termination_msg(msg):
                # Robust: some adapters add whitespace or extra tokens around TERMINATE.
                try:
                    content = (msg or {}).get("content")
                    if content is None:
                        return False
                    return "TERMINATE" in str(content).upper()
                except Exception:
                    return False

            executor_user = UserProxyAgent(
                name="single_agent_executor",
                system_message="You execute tools for the assistant.",
                # Never ask a human; use auto-replies. We'll auto-terminate via default_auto_reply + is_termination_msg.
                human_input_mode="NEVER",
                # Hard cap: keep single-agent runs predictable and low-cost.
                # Needs to be >= number of tool executions in the bounded flow.
                max_consecutive_auto_reply=10,
                is_termination_msg=_is_termination_msg,
                code_execution_config={"executor": self.code_executor_config},
            )

            # Critical: prevent empty auto-replies that trigger redundant assistant turns.
            # When the agent is done, the executor will send TERMINATE.
            executor_user.default_auto_reply = "TERMINATE"

            assistant = AgentConfig.get_assistant_agent(executor=executor_user)
            return assistant, executor_user
        except Exception as e:
            st.error(f"Error initializing single-agent mode: {e}")
            return None, None

    def run_single_agent(self, user_request: str) -> dict:
        """Runs single-agent mode and returns {result, usage} for the Streamlit UI."""
        assistant, executor_user = self.initialize_single_agent()
        if not assistant or not executor_user:
            return {
                "result": "Single-agent initialization failed. Check Streamlit error output above.",
                "usage": None,
            }

        def _normalize_usage(obj):
            if obj is None:
                return None
            try:
                return json.loads(json.dumps(obj, default=str))
            except Exception:
                return None

        def _round_usage_summary(summary: dict | None) -> dict | None:
            if not isinstance(summary, dict):
                return summary
            try:
                if "total_cost" in summary:
                    summary["total_cost"] = round(_to_float(summary.get("total_cost")), 10)
                for k, v in list(summary.items()):
                    if k == "total_cost":
                        continue
                    if isinstance(v, dict) and "cost" in v:
                        v["cost"] = round(_to_float(v.get("cost")), 10)
                return summary
            except Exception:
                return summary

        def _agent_client_usage(agent_obj):
            """Best-effort per-agent usage from the underlying OpenAIWrapper client."""
            try:
                client = (
                    getattr(agent_obj, "client", None)
                    or getattr(agent_obj, "_client", None)
                    or getattr(agent_obj, "oai_client", None)
                    or getattr(agent_obj, "_oai_client", None)
                )
                if client is None:
                    return None
                summary = getattr(client, "total_usage_summary", None)
                if summary is None:
                    return None
                normalized = _normalize_usage(summary)
                return _round_usage_summary(normalized)
            except Exception:
                return None

        def _to_float(v):
            try:
                return float(v)
            except Exception:
                return 0.0

        def _to_int(v):
            try:
                return int(v)
            except Exception:
                return 0

        def _aggregate_usage(by_agent: dict) -> dict:
            totals = {
                "total_cost": 0.0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
            by_model: dict[str, dict] = {}

            if not isinstance(by_agent, dict):
                return {"totals": totals, "by_model": by_model}

            for _agent_name, agent_usage in by_agent.items():
                if not isinstance(agent_usage, dict):
                    continue

                totals["total_cost"] += _to_float(agent_usage.get("total_cost"))

                for k, v in agent_usage.items():
                    if k == "total_cost":
                        continue
                    if not isinstance(v, dict):
                        continue

                    model_bucket = by_model.setdefault(
                        k,
                        {"cost": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    )
                    model_bucket["cost"] += _to_float(v.get("cost"))
                    model_bucket["prompt_tokens"] += _to_int(v.get("prompt_tokens"))
                    model_bucket["completion_tokens"] += _to_int(v.get("completion_tokens"))
                    model_bucket["total_tokens"] += _to_int(v.get("total_tokens"))

                    totals["prompt_tokens"] += _to_int(v.get("prompt_tokens"))
                    totals["completion_tokens"] += _to_int(v.get("completion_tokens"))
                    totals["total_tokens"] += _to_int(v.get("total_tokens"))

            # Clean up float artifacts for display/storage.
            totals["total_cost"] = round(_to_float(totals.get("total_cost")), 10)
            for _model_name, bucket in by_model.items():
                if isinstance(bucket, dict):
                    bucket["cost"] = round(_to_float(bucket.get("cost")), 10)

            return {"totals": totals, "by_model": by_model}

        def _build_single_agent_message(raw: str) -> str:
            s = (raw or "").strip()
            # If the user enters only a ticker, force a bounded strategy workflow.
            # This avoids relying solely on system prompt compliance.
            if re.fullmatch(r"[A-Za-z]{1,10}", s or ""):
                ticker = s.upper()
                return (
                    f"{ticker}\n\n"
                    "TASK: Provide a Buy/Sell/Hold recommendation. "
                    "Use these tools exactly once each. First call finance_data_fetch. "
                    "Then, in a SINGLE subsequent message, request BOTH strategy_signal_tool and risk_assessment_tool (back-to-back) before writing the final answer. "
                    "Do NOT call technical_analysis_tool. "
                    "Summarize in concise markdown (<= 10 bullet lines). "
                    "After the final answer, output a final line containing exactly: TERMINATE"
                )
            return raw

        def _strip_termination_sentinel(text: str) -> str:
            s = (text or "").strip()
            lines = [ln.rstrip() for ln in s.splitlines()]
            while lines and not lines[-1].strip():
                lines.pop()
            if lines and lines[-1].strip() == "TERMINATE":
                lines.pop()
            return "\n".join(lines).strip()

        try:
            result = executor_user.initiate_chat(
                assistant,
                message=_build_single_agent_message(user_request),
                # Hard cap: prevent long/expensive single-agent runs.
                # Allow a bounded tool chain for ticker-only queries:
                # finance_data_fetch -> strategy_signal_tool -> risk_assessment_tool -> final answer
                # Tool calls/responses each count as turns; allow enough for 3 tools + final.
                # With the "request both tools in one message" rule above, this fits in 7 turns:
                # user -> finance call -> finance result -> (signals+risk calls) -> signals result -> risk result -> final answer
                max_turns=7,
                summary_method="last_msg",
            )

            # A1 tracking for single-agent mode: assistant only.
            by_agent = {assistant.name: _agent_client_usage(assistant)}
            agg = _aggregate_usage(by_agent)
            usage = {
                "totals": agg.get("totals"),
                "by_model": agg.get("by_model"),
                "by_agent": by_agent,
            }

            # Deterministic capture: prefer the last assistant message.
            if hasattr(result, "chat_history") and result.chat_history:
                for msg in reversed(result.chat_history):
                    # Some AG2 adapters set `name` without a stable `role`.
                    is_assistant_msg = (msg.get("role") == "assistant") or (msg.get("name") == assistant.name)
                    if is_assistant_msg:
                        content = msg.get("content")
                        if content is not None and str(content).strip():
                            cleaned = _strip_termination_sentinel(str(content))
                            if cleaned:
                                return {"result": cleaned, "usage": usage}

            # Fallback to summary (depends on AG2 result object).
            if hasattr(result, "summary") and result.summary is not None and str(result.summary).strip():
                return {"result": _strip_termination_sentinel(str(result.summary)), "usage": usage}

            return {"result": "Single-agent run completed, but no output was captured.", "usage": usage}
        except Exception as e:
            return {"result": f"Error during single-agent analysis: {str(e)}", "usage": None}

    