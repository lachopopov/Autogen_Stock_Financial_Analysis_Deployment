import autogen
from tools import FinanceTools
from autogen import register_for_llm  # Import for AG2 0.10.3
from agent_config import AgentConfig

def orchestrate_agents(user_request, finance_reporting_analyst, technical_analyst, strategy_agent, user):
    try:
        # Register tools with register_for_llm (auto-generates schema)
        register_for_llm(
            FinanceTools.finance_data_fetch,
            caller=finance_reporting_analyst,
            executor=user
        )
        register_for_llm(
            FinanceTools.technical_analysis_tool,
            caller=technical_analyst,
            executor=user
        )
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

        groupchat = autogen.GroupChat(
            agents=[user, finance_reporting_analyst, technical_analyst, strategy_agent],
            messages=[],
            max_round=9,
            speaker_selection_method="round_robin"
        )

        manager = autogen.GroupChatManager(
            groupchat=groupchat,
            llm_config={"config_list": AgentConfig.get_llm_config(), "timeout": 280, "temperature": 0.5},
        )

        result = user.initiate_chat(manager, message=user_request)

        if hasattr(result, "chat_history") and result.chat_history:
            for msg in reversed(result.chat_history):
                if msg.get("name") in ["user", "manager"]:
                    return msg.get("content", "No final output found.")
        if hasattr(result, 'summary'):
            return result.summary
        return "Analysis completed successfully. No detailed result was returned."
    except Exception as e:
        return f"Error during analysis: {str(e)}"