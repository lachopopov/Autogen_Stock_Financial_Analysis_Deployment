import autogen
from agent_config import AgentConfig
import time  # added
import random
import re
import json
import os
import ast

def orchestrate_agents(user_request, finance_reporting_analyst, technical_analyst, strategy_agent, user):
    # Helper for consolidation from either result.chat_history or groupchat.messages
    def _is_empty(content) -> bool:
        if content is None:
            return True
        s = str(content).strip()
        return (not s) or (s.lower() == "none")

    def _last_message_by(messages, agent_name: str):
        if not messages:
            return None
        for msg in reversed(messages):
            if msg.get("name") != agent_name:
                continue
            content = msg.get("content")
            if _is_empty(content):
                continue
            return str(content)
        return None

    def _extract_last_tool_payloads(messages) -> dict:
        """
        Returns: { tool_name: data_dict } from the latest tool-result JSON found in messages.
        Expected tool JSON shape: {"name": "<tool_name>", "data": {...}}
        """
        payloads: dict[str, dict] = {}
        if not messages:
            return payloads

        def _maybe_parse_json_blob(content: object) -> dict | None:
            if content is None:
                return None
            if isinstance(content, dict):
                return content
            s = str(content).strip()
            if not s:
                return None

            # Some adapters may emit python-literal dicts; try that as a fallback.
            def _try_parse(text: str):
                try:
                    obj = json.loads(text)
                    return obj if isinstance(obj, dict) else None
                except Exception:
                    pass
                try:
                    obj = ast.literal_eval(text)
                    return obj if isinstance(obj, dict) else None
                except Exception:
                    return None

            # If content contains extra text, try to extract the largest {...} span.
            if not (s.startswith("{") and s.endswith("}")):
                i = s.find("{")
                j = s.rfind("}")
                if i >= 0 and j > i:
                    s = s[i : j + 1]

            return _try_parse(s)

        def _record_from_obj(obj: dict | None):
            if not isinstance(obj, dict):
                return
            tool_name = obj.get("name")
            data = obj.get("data")
            if isinstance(tool_name, str) and isinstance(data, dict):
                payloads[tool_name] = data

        for msg in messages:
            if not isinstance(msg, dict):
                continue

            # Common: tool output is embedded as JSON in content
            _record_from_obj(_maybe_parse_json_blob(msg.get("content")))

            # Some AutoGen adapters attach tool outputs as a structured list
            tool_responses = msg.get("tool_responses") or msg.get("toolResponses")
            if isinstance(tool_responses, list):
                for tr in tool_responses:
                    if isinstance(tr, dict):
                        _record_from_obj(_maybe_parse_json_blob(tr.get("content")))
                        _record_from_obj(_maybe_parse_json_blob(tr.get("result")))

            # role='tool' messages sometimes store tool name in `name` and raw JSON in `content`
            if msg.get("role") == "tool" and isinstance(msg.get("name"), str):
                tool_name = msg.get("name")
                content_obj = _maybe_parse_json_blob(msg.get("content"))
                if isinstance(content_obj, dict):
                    # If the tool already wrapped as {name,data}, record it.
                    if "name" in content_obj and "data" in content_obj:
                        _record_from_obj(content_obj)
                    # Otherwise treat the parsed dict as the tool data.
                    else:
                        if isinstance(tool_name, str):
                            payloads[tool_name] = content_obj

        return payloads

    def _fmt_money(v) -> str:
        try:
            if v is None:
                return "N/A"
            return f"{float(v):,.2f}"
        except Exception:
            return "N/A"

    def _fmt_pct(v) -> str:
        try:
            if v is None:
                return "N/A"
            return f"{float(v):.2f}%"
        except Exception:
            return "N/A"

    def _fmt_price(v) -> str:
        try:
            if v is None:
                return "N/A"
            return f"{float(v):.2f}"
        except Exception:
            return str(v)

    def _fmt_ratio(v) -> str:
        try:
            if v is None:
                return "N/A"
            return f"{float(v):.2f}"
        except Exception:
            return "N/A"

    def _fmt_volatility(v) -> str:
        """Normalize volatility display.

        Some tools return volatility as a fraction (e.g., 0.1096), others as percent.
        """
        try:
            if v is None:
                return "N/A"
            x = float(v)
            if 0 <= x <= 1.5:
                return f"{x * 100:.2f}%"
            return f"{x:.2f}%"
        except Exception:
            return "N/A"

    def _fmt_market_cap(v) -> str:
        try:
            if v is None:
                return "N/A"
            x = float(v)
            if x >= 1e12:
                return f"${x/1e12:.2f}T"
            if x >= 1e9:
                return f"${x/1e9:.2f}B"
            if x >= 1e6:
                return f"${x/1e6:.2f}M"
            return f"${x:,.0f}"
        except Exception:
            return "N/A"

    def _fundamentals_from_tool(data: dict) -> str:
        # Summarize only; do not paste raw JSON.
        period_stats = data.get("periodStats") if isinstance(data.get("periodStats"), dict) else {}
        lines = [
            f"**Company:** {data.get('name', 'N/A')} ({data.get('symbol', 'N/A')})",
            f"**Price:** {data.get('currentPrice', 'N/A')} {data.get('currency', '')}".strip(),
            f"**Market cap:** {_fmt_market_cap(data.get('marketCap'))}",
            f"**P/E (TTM):** {data.get('peRatio', 'N/A')}",
            f"**P/B:** {data.get('priceToBook', 'N/A')}",
            f"**Dividend (rate):** {data.get('dividend', 'N/A')}",
        ]
        if period_stats:
            lines.append(
                f"**{period_stats.get('period','period')} return:** {_fmt_pct(period_stats.get('returnPct'))} "
                f"(min {period_stats.get('minClose','N/A')}, max {period_stats.get('maxClose','N/A')})"
            )

        summary = str(data.get("summary", "") or "").strip()
        if summary:
            lines.append(f"**Business summary:** {summary}")

        return "\n".join(f"- {ln}" for ln in lines if ln and str(ln).strip())

    def _period_stats_from_finance_tool(data: dict | None) -> dict:
        """Return a consistent period stats dict from finance_data_fetch payload."""
        if not isinstance(data, dict):
            return {}

        period_stats = data.get("periodStats")
        if isinstance(period_stats, dict) and period_stats:
            return period_stats

        # Fallback: compute from recentClosePrices if present.
        rcp = data.get("recentClosePrices")
        if not isinstance(rcp, dict) or not rcp:
            return {}

        # Keys are timestamps; values are prices.
        items = []
        for k, v in rcp.items():
            try:
                items.append((str(k), float(v)))
            except Exception:
                continue
        if len(items) < 2:
            return {}

        items.sort(key=lambda t: t[0])
        first_date, first_close = items[0]
        last_date, last_close = items[-1]

        closes = [p for _, p in items]
        mn = min(closes)
        mx = max(closes)
        ret_pct = ((last_close - first_close) / first_close) * 100.0 if first_close else None

        return {
            "period": "Recent period",
            "startDate": first_date,
            "endDate": last_date,
            "startClose": first_close,
            "endClose": last_close,
            "minClose": mn,
            "maxClose": mx,
            "returnPct": ret_pct,
        }

    def _key_metrics_from_tools(tool_payloads: dict) -> str | None:
        """Produce a single unified Key Metrics block from tool payloads (no raw JSON)."""
        if not isinstance(tool_payloads, dict) or not tool_payloads:
            return None

        f_raw = tool_payloads.get("finance_data_fetch")
        t_raw = tool_payloads.get("technical_analysis_tool")
        s_raw = tool_payloads.get("strategy_signal_tool")
        r_raw = tool_payloads.get("risk_assessment_tool")

        f: dict = f_raw if isinstance(f_raw, dict) else {}
        t: dict = t_raw if isinstance(t_raw, dict) else {}
        s: dict = s_raw if isinstance(s_raw, dict) else {}
        r: dict = r_raw if isinstance(r_raw, dict) else {}

        currency = str(f.get("currency") or "").strip()
        current_price = f.get("currentPrice")
        last_close = t.get("Last_Close")
        if last_close is None:
            # Fallback to computed period end close.
            ps = _period_stats_from_finance_tool(f)
            last_close = ps.get("endClose")

        period_stats = _period_stats_from_finance_tool(f)

        bullets = []

        # Price / period
        if current_price is not None:
            bullets.append(f"**Current price:** {_fmt_price(current_price)}{(' ' + currency) if currency else ''}")
        if last_close is not None:
            bullets.append(f"**Last close:** {_fmt_price(last_close)}{(' ' + currency) if currency else ''}")

        if period_stats:
            label = period_stats.get("period") or "Period"
            ret = period_stats.get("returnPct")
            mn = period_stats.get("minClose")
            mx = period_stats.get("maxClose")
            if ret is not None:
                bullets.append(f"**{label} return:** {_fmt_pct(ret)}")
            if mn is not None and mx is not None:
                bullets.append(f"**{label} range:** {_fmt_price(mn)} – {_fmt_price(mx)}")

        # Fundamentals
        if f:
            if f.get("marketCap") is not None:
                bullets.append(f"**Market cap:** {_fmt_market_cap(f.get('marketCap'))}")
            if f.get("peRatio") is not None:
                bullets.append(f"**P/E (TTM):** {_fmt_ratio(f.get('peRatio'))}")
            if f.get("priceToBook") is not None:
                bullets.append(f"**P/B:** {_fmt_ratio(f.get('priceToBook'))}")
            if f.get("dividend") is not None:
                bullets.append(f"**Dividend (rate):** {_fmt_price(f.get('dividend'))}")

        # Technicals
        if t:
            if t.get("SMA_20") is not None:
                bullets.append(f"**20-day SMA:** {_fmt_price(t.get('SMA_20'))}")
            if t.get("EMA_20") is not None:
                bullets.append(f"**20-day EMA:** {_fmt_price(t.get('EMA_20'))}")
            rsi_val = t.get("RSI")
            if rsi_val is not None:
                try:
                    bullets.append(f"**RSI (14):** {float(rsi_val):.2f}")
                except Exception:
                    bullets.append(f"**RSI (14):** {rsi_val}")

        # Signals
        if s:
            if s.get("MACD") is not None:
                bullets.append(f"**MACD:** {_fmt_ratio(s.get('MACD'))}")
            macd_signal = s.get("MACD_Signal") if s.get("MACD_Signal") is not None else s.get("MACDSignal")
            if macd_signal is not None:
                bullets.append(f"**MACD signal:** {_fmt_ratio(macd_signal)}")

        # Risk
        if r:
            if r.get("Beta") is not None:
                bullets.append(f"**Beta:** {r.get('Beta')}")
            if r.get("Volatility") is not None:
                bullets.append(f"**Volatility:** {_fmt_volatility(r.get('Volatility'))}")
            if r.get("DividendYield") is not None:
                bullets.append(f"**Dividend yield:** {_fmt_pct(r.get('DividendYield'))}")
            if r.get("RiskRating") is not None:
                bullets.append(f"**Risk rating:** {r.get('RiskRating')}")

        if not bullets:
            return None

        return "\n".join(f"- {b}" for b in bullets if b and str(b).strip())

    def _technicals_from_tool(data: dict) -> str:
        last_close = data.get("Last_Close")
        sma20 = data.get("SMA_20")
        ema20 = data.get("EMA_20")
        rsi = data.get("RSI")

        bullets = [
            f"**Last close:** {last_close}",
            f"**SMA(20):** {sma20}",
            f"**EMA(20):** {ema20}",
            f"**RSI(14):** {rsi}",
        ]

        # Light interpretation derived strictly from tool numbers.
        try:
            lc = float(last_close) if last_close is not None else None
            s20 = float(sma20) if sma20 is not None else None
            e20 = float(ema20) if ema20 is not None else None
            rv = float(rsi) if rsi is not None else None

            if lc is not None and s20 is not None and e20 is not None:
                if lc < s20 and lc < e20:
                    bullets.append("**Trend (20d):** below SMA/EMA (bearish bias)")
                elif lc > s20 and lc > e20:
                    bullets.append("**Trend (20d):** above SMA/EMA (bullish bias)")
                else:
                    bullets.append("**Trend (20d):** mixed vs SMA/EMA")

            if rv is not None:
                if rv <= 30:
                    bullets.append("**Momentum:** RSI near/under 30 (oversold zone)")
                elif rv >= 70:
                    bullets.append("**Momentum:** RSI near/over 70 (overbought zone)")
                else:
                    bullets.append("**Momentum:** RSI mid-range")
        except Exception:
            pass

        return "\n".join(f"- {b}" for b in bullets if b and str(b).strip())

    def _risk_from_tool(data: dict) -> str:
        bullets = [
            f"**Risk rating:** {data.get('RiskRating', 'N/A')}",
            f"**Beta:** {data.get('Beta', 'N/A')}",
            f"**Dividend yield:** {data.get('DividendYield', 'N/A')}",
            f"**Volatility proxy:** {data.get('Volatility', 'N/A')}",
            f"**Market cap:** {_fmt_market_cap(data.get('MarketCap'))}",
        ]
        return "\n".join(f"- {b}" for b in bullets if b and str(b).strip())

    def _consolidate(messages, note: str | None = None) -> str:
        finance_md = _last_message_by(messages, "finance_reporting_analyst")
        technical_md = _last_message_by(messages, "technical_analyst")
        strategy_md = _last_message_by(messages, "strategy_agent")

        tool_payloads = _extract_last_tool_payloads(messages)

        key_metrics_md = _key_metrics_from_tools(tool_payloads)

        # Fallback: if agents never wrote narrative, synthesize from tool outputs (summarized).
        if _is_empty(finance_md) and isinstance(tool_payloads.get("finance_data_fetch"), dict):
            finance_md = _fundamentals_from_tool(tool_payloads["finance_data_fetch"])
        if _is_empty(technical_md) and isinstance(tool_payloads.get("technical_analysis_tool"), dict):
            technical_md = _technicals_from_tool(tool_payloads["technical_analysis_tool"])

        risk_md = None
        if isinstance(tool_payloads.get("risk_assessment_tool"), dict):
            risk_md = _risk_from_tool(tool_payloads["risk_assessment_tool"])

        parts: list[str] = []
        parts.append(f"# Stock Analysis\n\n**Query:** {user_request}\n")
        if note:
            parts.append(f"> {note}\n")

        if key_metrics_md and not _is_empty(key_metrics_md):
            parts.append("## Key Metrics\n")
            parts.append(key_metrics_md.strip() + "\n")

        if finance_md and not _is_empty(finance_md):
            parts.append("## Fundamentals\n")
            parts.append(finance_md.strip() + "\n")

        if technical_md and not _is_empty(technical_md):
            parts.append("## Technicals\n")
            parts.append(technical_md.strip() + "\n")

        parts.append("## Strategy / Risk\n")
        if strategy_md and not _is_empty(strategy_md):
            parts.append(strategy_md.strip() + "\n")
        else:
            parts.append("_No strategy recommendation was produced before the run ended._\n")

        if risk_md and not _is_empty(risk_md):
            parts.append("### Risk metrics\n")
            parts.append(risk_md.strip() + "\n")

        final_report = "\n".join(parts).strip()
        return final_report if not _is_empty(final_report) else "No usable message content was captured."

    def _looks_like_rate_limit(msg: str) -> bool:
        m = (msg or "").lower()
        return (
            "error code: 429" in m
            or "rate_limit_exceeded" in m
            or "rate limit reached" in m
            or "tokens per min" in m
        )

    def _looks_like_insufficient_quota(msg: str) -> bool:
        m = (msg or "").lower()
        # OpenAI commonly returns 429 with code 'insufficient_quota' when the account/org
        # has no remaining credit or quota. This is not transient and should not be retried.
        return "insufficient_quota" in m or "exceeded your current quota" in m

    def _parse_retry_after_seconds(msg: str) -> float | None:
        # Common OpenAI-style hint: "Please try again in 384ms"
        m = re.search(r"try again in\s+(\d+)\s*ms", msg or "", flags=re.IGNORECASE)
        if m:
            return max(0.0, int(m.group(1)) / 1000.0)

        # Sometimes appears as "retry after 2s"
        m = re.search(r"retry after\s+(\d+(?:\.\d+)?)\s*s", msg or "", flags=re.IGNORECASE)
        if m:
            return max(0.0, float(m.group(1)))

        # Generic header-like text (best-effort)
        m = re.search(r"retry-after:\s*(\d+)", msg or "", flags=re.IGNORECASE)
        if m:
            return max(0.0, float(m.group(1)))

        return None

    def _backoff_seconds(attempt: int, msg: str) -> float:
        hinted = _parse_retry_after_seconds(msg)
        if hinted is not None:
            # Add small jitter to avoid herd effects when multiple runs retry together.
            return min(30.0, hinted + random.uniform(0.05, 0.25))
        # Exponential backoff w/ cap + jitter
        base = 0.6
        cap = 20.0
        return min(cap, base * (2 ** max(0, attempt - 1)) + random.uniform(0.05, 0.35))

    def _normalize_usage(obj):
        if obj is None:
            return None
        try:
            # Ensure Streamlit can render it and that session_state stays serializable.
            return json.loads(json.dumps(obj, default=str))
        except Exception:
            return None

    def _round_usage_summary(summary: dict | None) -> dict | None:
        """Round cost fields in an OpenAIWrapper-style usage summary."""
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

    def _env_flag(name: str, default: str = "0") -> bool:
        v = str(os.environ.get(name, default)).strip().lower()
        return v in {"1", "true", "yes", "y", "on"}

    def _routing_mode() -> str:
        v = str(os.environ.get("AG2_ROUTING_MODE", "deterministic")).strip().lower()
        if v in {"nondeterministic", "non_deterministic", "non-deterministic", "auto", "llm"}:
            return "nondeterministic"
        return "deterministic"

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
        """Aggregate totals across agents.

        Expected per-agent shape (best effort):
        {
          "total_cost": 0.001,
          "<model>": {"cost": 0.001, "prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
        }
        """
        totals = {
            "total_cost": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        by_model: dict[str, dict] = {}

        if not isinstance(by_agent, dict):
            return {"totals": totals, "by_model": by_model}

        for agent_name, agent_usage in by_agent.items():
            if not isinstance(agent_usage, dict):
                continue

            totals["total_cost"] += _to_float(agent_usage.get("total_cost"))

            for k, v in agent_usage.items():
                # Skip non-model keys
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
        for model_name, bucket in by_model.items():
            if isinstance(bucket, dict):
                bucket["cost"] = round(_to_float(bucket.get("cost")), 10)

        return {"totals": totals, "by_model": by_model}

    def _get_usage_a1():
        """Usage tracking.

        Default = A1 (assistants only).
        If AG2_INCLUDE_MANAGER_USAGE=1, include GroupChatManager usage as well.
        """
        mode = _routing_mode()
        include_manager = _env_flag("AG2_INCLUDE_MANAGER_USAGE", "1" if mode == "nondeterministic" else "0")

        by_agent = {
            finance_reporting_analyst.name: _agent_client_usage(finance_reporting_analyst),
            technical_analyst.name: _agent_client_usage(technical_analyst),
            strategy_agent.name: _agent_client_usage(strategy_agent),
        }

        if include_manager and manager is not None:
            mgr_usage = _agent_client_usage(manager)
            # With deterministic routing, the manager often makes no LLM calls even if llm_config is enabled.
            # Return a non-null stub to make that explicit in the UI.
            by_agent["groupchat_manager"] = mgr_usage if mgr_usage is not None else {
                "total_cost": 0.0,
                "note": "No LLM usage recorded for GroupChatManager (likely not invoked).",
            }

        agg = _aggregate_usage(by_agent)
        return {
            "includes_manager": include_manager,
            "routing_mode": mode,
            "manager_llm_enabled": manager_llm_enabled,
            "totals": agg.get("totals"),
            "by_model": agg.get("by_model"),
            "by_agent": by_agent,
        }

    groupchat = None
    manager = None
    manager_llm_enabled = False
    try:
        # StateFlow-style orchestration:
        # - Progress through stages (fundamentals -> technicals -> strategy)
        # - If a tool call is pending, hand control back to AG2's built-in routing
        #   so `func_call_filter` can select the executor.
        # - After tool execution (last_speaker == user), return to the requesting analyst.
        # - Terminate after strategy stage completes to avoid max_round loops.
        speaker_order = [finance_reporting_analyst, technical_analyst, strategy_agent]
        analyst_names = {a.name for a in speaker_order}

        stage_order = [finance_reporting_analyst, technical_analyst, strategy_agent]
        stage_idx = 0
        last_requester_name: str | None = None

        def _pending_tool_call(gc) -> bool:
            """Detect whether the most recent message is requesting a tool/function call.

            In AG2/AutoGen 0.10.3, tool calls are represented via function_call/tool_calls
            fields on the latest message. If we detect that, we must hand control back
            to the built-in selection ("auto") so func_call_filter can route execution
            to the tool executor (supervisor).
            """
            msgs = getattr(gc, "messages", None) or []
            if not msgs:
                return False
            last = msgs[-1] if isinstance(msgs[-1], dict) else None
            if not last:
                return False
            # Mirror AutoGen's own check (it uses key presence, not truthiness).
            # Some flows may include the key with a falsey/None value.
            if "function_call" in last or "tool_calls" in last:
                return True

            # Fallback heuristic (some adapters store tool intent in content).
            content = str(last.get("content", "") or "")
            return "Suggested tool call" in content

        def _select_next_speaker(last_speaker, groupchat):
            nonlocal stage_idx, last_requester_name

            # Tool-call routing: let AG2 pick the executor via func_call_filter.
            if _pending_tool_call(groupchat):
                msgs = getattr(groupchat, "messages", None) or []
                last = msgs[-1] if msgs and isinstance(msgs[-1], dict) else None
                if last:
                    nm = last.get("name")
                    if nm in analyst_names:
                        last_requester_name = nm
                return "auto"

            current_stage_agent = stage_order[min(stage_idx, len(stage_order) - 1)]

            # After tool execution, always return to the requesting analyst (or current stage).
            if getattr(last_speaker, "name", None) == getattr(user, "name", None):
                if last_requester_name in analyst_names:
                    for a in speaker_order:
                        if a.name == last_requester_name:
                            return a
                return current_stage_agent

            # First assistant turn after the initial user message.
            if last_speaker is None or getattr(last_speaker, "name", None) == getattr(user, "name", None):
                return current_stage_agent

            # Stage progression: only advance when the stage agent produced a real (non-empty) message.
            if last_speaker is current_stage_agent:
                msgs = getattr(groupchat, "messages", None) or []
                last = msgs[-1] if msgs and isinstance(msgs[-1], dict) else None
                content = last.get("content") if last else None
                if not _is_empty(content):
                    if stage_idx >= len(stage_order) - 1:
                        return None
                    stage_idx += 1
                return stage_order[min(stage_idx, len(stage_order) - 1)]

            # If a non-stage agent somehow spoke, steer back to the current stage.
            return current_stage_agent

        mode = _routing_mode()

        if mode == "nondeterministic":
            # Non-deterministic mode: let AG2/manager handle speaker selection.
            # This may increase cost due to manager LLM calls.
            groupchat = autogen.GroupChat(
                agents=[user, finance_reporting_analyst, technical_analyst, strategy_agent],
                messages=[],
                max_round=12,
                allow_repeat_speaker=False,
                speaker_selection_method="auto",
            )
        else:
            # Deterministic StateFlow mode (default): custom stage router.
            groupchat = autogen.GroupChat(
                # Include supervisor in the chat so it can execute registered tools,
                # but speaker selection will prevent it from being chosen to speak.
                agents=[user, finance_reporting_analyst, technical_analyst, strategy_agent],
                messages=[],
                max_round=12,
                allow_repeat_speaker=False,
                speaker_selection_method=_select_next_speaker,
            )

        # Manager LLM usage:
        # - deterministic mode default: OFF (cheapest)
        # - nondeterministic mode default: ON (to allow LLM-based selection)
        manager_llm_enabled = _env_flag("AG2_MANAGER_LLM", "1" if mode == "nondeterministic" else "0")
        manager_llm_config = AgentConfig.get_llm_runtime_config(timeout=180) if manager_llm_enabled else False
        manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=manager_llm_config)

        result = None
        max_attempts = 6  # allows a few retries for transient 429s

        attempt = 0
        while attempt < max_attempts:
            attempt += 1
            try:
                # Initiate from the supervisor so the first turn is a true user message.
                result = user.initiate_chat(manager, message=user_request)
                break
            except Exception as e:
                msg = str(e)

                is_insufficient_quota = _looks_like_insufficient_quota(msg)
                is_rate_limit = _looks_like_rate_limit(msg) and not is_insufficient_quota
                is_other_transient = (
                    ("Error code: 500" in msg)
                    or ("model_error" in msg)
                    or ("produced invalid content" in msg)
                )

                # Non-retryable: quota is exhausted.
                if is_insufficient_quota:
                    raise RuntimeError(
                        "OpenAI returned insufficient_quota (no remaining credits/quota). "
                        "Add billing/credits, use a different API key, or point OPENAI_BASE_URL to another provider."
                    ) from e

                # If its not retryable, fail fast.
                if not (is_rate_limit or is_other_transient):
                    raise

                if attempt >= max_attempts:
                    raise

                time.sleep(_backoff_seconds(attempt, msg))

        report = None

        # Prefer consolidating from result.chat_history when available.
        if result is not None and hasattr(result, "chat_history") and result.chat_history:
            report = _consolidate(result.chat_history)

        # Fallback: consolidate from groupchat.messages (should exist even if result is odd).
        if report is None and getattr(groupchat, "messages", None):
            report = _consolidate(groupchat.messages)

        if report is None and result is not None and hasattr(result, "summary") and not _is_empty(result.summary):
            report = str(result.summary)

        if report is None:
            report = "Analysis completed, but no usable message was found (empty/None)."

        return {
            "result": report,
            "usage": _get_usage_a1(),
        }
    except Exception as e:
        # If the run died mid-way (e.g., model_error 500), attempt to salvage partial progress.
        try:
            if groupchat is not None and getattr(groupchat, "messages", None):
                msg = str(e)
                if _looks_like_insufficient_quota(msg):
                    note = (
                        "Run interrupted because the OpenAI account/org has no remaining credits/quota "
                        "(insufficient_quota). Add billing/credits, use a different API key, or switch providers."
                    )
                else:
                    note = f"Run interrupted due to an upstream model error. Showing partial results. ({msg})"
                return {
                    "result": _consolidate(
                        groupchat.messages,
                        note=note,
                    ),
                    "usage": _get_usage_a1(),
                }
        except Exception:
            pass
        return {
            "result": f"Error during analysis: {str(e)}",
            "usage": _get_usage_a1(),
        }