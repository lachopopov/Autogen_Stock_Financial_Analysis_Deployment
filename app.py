import streamlit as st
import os
from datetime import datetime
import uuid
import json  # Added for JSON parsing with error handling.
from agents import Agents
from agent_orchestrator import orchestrate_agents
from app_config import AppConfig
from tools import FinanceTools

# Load .env early so OPENAI_API_KEY / OPENAI_BASE_URL are available during Streamlit runs.
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # ...existing code... (do not hard-fail if dotenv isn't available for some reason)
    pass

class StockAnalysisApp:
    """Main application class for the AI Stock Analysis Platform."""
    def __init__(self):
        self.config = AppConfig()
        self.agents = Agents()

    @staticmethod
    def _apply_streamlit_secrets_to_env() -> None:
        """Copies Streamlit Secrets into env vars (when missing).

        Streamlit Community Cloud stores secrets in a TOML blob. The rest of this app
        expects config via `os.environ`, so we bridge secrets -> env here.
        """
        try:
            secrets = getattr(st, "secrets", None)
            if not secrets:
                return
        except Exception:
            return

        for key in (
            "OPENAI_API_KEY",
            "OPENAI_BASE_URL",
            "OPENAI_MODEL",
            "AG2_ROUTING_MODE",
            "AG2_MANAGER_LLM",
            "AG2_INCLUDE_MANAGER_USAGE",
        ):
            try:
                if os.environ.get(key):
                    continue
                if key in secrets and secrets.get(key) is not None:
                    os.environ[key] = str(secrets.get(key))
            except Exception:
                continue

    def render_sidebar(self):
        with st.sidebar:
            st.header("🔧 Configuration")

            # Ensure Streamlit Cloud secrets (TOML) are available through env vars.
            self._apply_streamlit_secrets_to_env()

            selected_model = st.selectbox(
                "Model",
                options=["gpt-5-mini", "gpt-5-nano"],
                index=0 if os.environ.get("OPENAI_MODEL", "gpt-5-mini") != "gpt-5-nano" else 1,
                help="Choose which OpenAI model to use for agent reasoning.",
            )
            os.environ["OPENAI_MODEL"] = selected_model

            single_agent_mode = st.checkbox(
                "Single-agent mode",
                key="single_agent_mode",
                help="Run a single assistant instead of the multi-agent GroupChat.",
            )

            if single_agent_mode:
                st.info("Single-agent mode enabled")

            has_configured_key = bool(os.environ.get("OPENAI_API_KEY"))
            use_own_key = st.checkbox(
                "Use my own OpenAI API key",
                value=False,
                help="Enable this only if you want to override the configured key.",
            )

            if not use_own_key and has_configured_key:
                st.caption("OpenAI API key is configured (hidden).")
            else:
                api_key = st.text_input(
                    "OpenAI API Key",
                    type="password",
                    value="",
                    help="Enter your OpenAI API key",
                )
                if api_key:
                    os.environ["OPENAI_API_KEY"] = api_key

            st.divider()
            st.header("📊 Quick Stock Info")
            quick_ticker = st.text_input("Enter ticker for quick view:", placeholder="e.g., AAPL")
            if quick_ticker:
                quick_ticker = quick_ticker.upper()
                try:
                    metrics = json.loads(FinanceTools.get_stock_metrics(quick_ticker))  # Parse JSON with error handling.
                except json.JSONDecodeError:
                    st.error("Failed to parse stock metrics. Please check the ticker or try again.")
                    metrics = {}
                for metric, value in metrics.items():
                    st.metric(metric, value)

    def render_main_content(self):
        st.markdown('<h1 class="main-header">📈 AI Stock Analysis Platform</h1>', unsafe_allow_html=True)
        st.markdown('<div style="text-align:center;"><strong>Powered by AutoGen AI Agents</strong></div>', unsafe_allow_html=True)

        # Persist last request so results can be shown after reruns even if input changes.
        if "last_request" not in st.session_state:
            st.session_state.last_request = None

        user_request = st.text_input(
            "Enter your query:",
            placeholder="Should I invest in MSFT based on recent trends?",
            help="Mention stock symbol you want to analyze"
        )

        _, center_col, _ = st.columns([3, 1, 3])
        with center_col:
            run_clicked = st.button("Run Analysis")

        if run_clicked:
            user_request = (user_request or "").strip()
            if not user_request:
                st.error("⚠️ Please enter a query (e.g., AAPL)")
                return
            if not os.environ.get("OPENAI_API_KEY"):
                st.error("⚠️ Please provide your OpenAI API key in the sidebar")
            else:
                st.session_state.last_request = user_request

                # Always create/overwrite the entry so the UI never shows stale None.
                st.session_state.analysis_results[user_request] = {
                    "timestamp": datetime.now(),
                    "request": user_request,
                    "result": "Running analysis...",
                }

                with st.spinner("🤖 AI agents are analyzing the stock... This may take a few moments."):
                    if st.session_state.get("single_agent_mode"):
                        analysis_data = self.agents.run_single_agent(user_request)
                    else:
                        agents = self.agents.initialize_agents()
                        if not all(agents):
                            analysis_data = "Agent initialization failed. Check Streamlit error output above."
                        else:
                            analysis_data = orchestrate_agents(user_request, *agents)

                    # Support structured payloads: {"result": <markdown str>, "usage": <dict|None>}
                    usage = None
                    if isinstance(analysis_data, dict):
                        usage = analysis_data.get("usage")
                        analysis_text = analysis_data.get("result")
                    else:
                        analysis_text = analysis_data

                    # Normalize and guard against None/"None"/non-string outputs.
                    if analysis_text is None:
                        analysis_text = "No analysis output was returned. Check logs for errors."
                    elif not isinstance(analysis_text, str):
                        analysis_text = str(analysis_text)
                    elif analysis_text.strip().lower() == "none":
                        analysis_text = "No analysis output was returned (got 'None'). Check logs for errors."

                    # Persist final output
                    st.session_state.analysis_results[user_request] = {
                        "timestamp": datetime.now(),
                        "request": user_request,
                        "result": analysis_text,
                        "usage": usage,
                    }
                    st.session_state.chat_history.append({
                        "timestamp": datetime.now(),
                        "ticker": user_request,
                        "request": user_request,
                        "result": analysis_text,
                        "usage": usage,
                    })

        # Show the last run result (not strictly tied to current input value).
        last = st.session_state.get("last_request")
        if last and last in st.session_state.analysis_results:
            st.header("📋 Analysis Results")
            result_text = st.session_state.analysis_results[last].get("result")
            if not result_text or str(result_text).strip().lower() == "none":
                result_text = "No analysis output was stored."

            result_text_str = str(result_text)
            lowered = result_text_str.lower()
            if (
                "insufficient_quota" in lowered
                or "exceeded your current quota" in lowered
                or "no remaining credits/quota" in lowered
            ):
                st.error(
                    "OpenAI reports insufficient quota/credits for this API key/org. "
                    "Add billing/credits, use a different API key, or switch providers (OPENAI_BASE_URL)."
                )
            st.markdown(result_text_str, unsafe_allow_html=True)

            usage = st.session_state.analysis_results[last].get("usage")
            if usage:
                with st.expander("Usage (tokens/cost)"):
                    st.json(usage)

            file_id = uuid.uuid1()
            st.download_button(
                label="📥 Download Analysis Report",
                data=st.session_state.analysis_results[last]["result"],
                file_name=f"{file_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown"
            )

        st.markdown("---")
        st.markdown(
            "**Disclaimer:** This analysis is for educational purposes only. "
            "Always consult with a financial advisor before making investment decisions."
        )

    def run(self):
        self.config.setup_page()
        self.config.initialize_session_state()
        self.render_sidebar()
        self.render_main_content()

if __name__ == "__main__":
    app = StockAnalysisApp()
    app.run()