import re

from langchain_core.messages import HumanMessage, RemoveMessage

# Import tools from separate utility files
from tradingagents.agents.utils.core_stock_tools import (
    get_stock_data
)
from tradingagents.agents.utils.technical_indicators_tools import (
    get_indicators
)
from tradingagents.agents.utils.fundamental_data_tools import (
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement
)
from tradingagents.agents.utils.news_data_tools import (
    get_news,
    get_insider_transactions,
    get_global_news
)
from tradingagents.agents.utils.signal_data_tools import (
    get_profit_forecast,
    get_hot_stocks,
    get_northbound_flow,
    get_concept_blocks,
    get_fund_flow,
    get_dragon_tiger_board,
    get_lockup_expiry,
    get_industry_comparison,
)


_PROMPT_VARIABLE = re.compile(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}")
_REPORT_FIELDS = (
    "market_report", "sentiment_report", "news_report", "fundamentals_report",
    "policy_report", "hot_money_report", "lockup_report", "data_quality_summary",
)


def render_prompt_template(template: str, state: dict | None = None,
                           tool_names: list[str] | None = None) -> str:
    """安全渲染角色提示词变量，缺少变量时阻止请求进入模型。"""
    state = state or {}
    reports = "\n\n".join(
        f"## {field}\n{state.get(field, '[暂无数据]')}" for field in _REPORT_FIELDS
        if state.get(field)
    ) or "[暂无上游分析报告]"
    messages = state.get("messages") or []
    message_text = "\n".join(
        str(getattr(message, "content", message)) for message in messages[-20:]
    ) or "[暂无消息历史]"
    investment_round = (state.get("investment_debate_state") or {}).get("count")
    risk_round = (state.get("risk_debate_state") or {}).get("count")
    values = {
        "ticker": state.get("company_of_interest") or state.get("ticker"),
        "date": state.get("trade_date"),
        "current_date": state.get("trade_date"),
        "reports": reports,
        "messages": message_text,
        "tool_names": ", ".join(tool_names or []) or "[当前角色不使用工具]",
        "debate_round": risk_round if risk_round is not None else investment_round,
    }

    def replace(match):
        key = match.group(1)
        value = values.get(key)
        if value is None:
            raise ValueError(f"提示词变量缺少运行时值: {key}")
        return str(value)

    return _PROMPT_VARIABLE.sub(replace, template)


def get_prompt_override(role: str, default: str, *, state: dict | None = None,
                        tool_names: list[str] | None = None) -> str:
    """读取作业快照中的角色提示词；未配置时返回源码默认提示词。"""
    from tradingagents.dataflows.config import get_config

    configured = get_config().get("prompt_overrides") or {}
    if isinstance(configured, list):
        overrides = {
            item.get("roleKey") or item.get("role") or item.get("templateKey"): item.get("content")
            for item in configured
            if isinstance(item, dict)
        }
    else:
        overrides = configured if isinstance(configured, dict) else {}
    value = overrides.get(role)
    if not isinstance(value, str) or not value.strip():
        return default
    return render_prompt_template(value, state=state, tool_names=tool_names)


def get_language_instruction() -> str:
    """Return a prompt instruction for the configured output language.

    Returns empty string when English (default), so no extra tokens are used.
    Only applied to user-facing agents (analysts, portfolio manager).
    Internal debate agents stay in English for reasoning quality.
    """
    from tradingagents.dataflows.config import get_config
    lang = get_config().get("output_language", "English")
    if lang.strip().lower() == "english":
        return ""
    return f" Write your entire response in {lang}."


def build_instrument_context(ticker: str) -> str:
    """Describe the exact instrument so agents preserve exchange-qualified tickers."""
    return (
        f"The instrument to analyze is `{ticker}`. "
        "Use this exact ticker in every tool call, report, and recommendation, "
        "preserving any exchange suffix (e.g. `.TO`, `.L`, `.HK`, `.T`). "
        "When a tool argument is named `ticker`, pass only this ticker value; "
        "do not pass company names, sectors, concepts, or search keywords."
    )

def create_msg_delete():
    def delete_messages(state):
        """Clear messages and add placeholder for Anthropic compatibility"""
        messages = state["messages"]

        # Remove all messages
        removal_operations = [RemoveMessage(id=m.id) for m in messages]

        # Add a minimal placeholder message
        placeholder = HumanMessage(content="Continue")

        return {"messages": removal_operations + [placeholder]}

    return delete_messages


        
