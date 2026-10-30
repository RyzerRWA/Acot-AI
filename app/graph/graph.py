"""
The ACOT agent as a LangGraph graph.

    model -> (tool calls?) -> tools -> model ...        the model picks read-only tools until it can answer
    model -> (final text)  -> check -> END              numbers in the answer are checked against the tool results
                              check -> model (once)     an answer quoting a number no tool returned is sent back once

Nothing here talks to the database or the model directly: the model call and the tools are passed in, so the
graph can be tested with a scripted model and no credits.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable, TypedDict

from langgraph.graph import END, StateGraph

GIVE_UP = "I couldn't pin that down. Could you rephrase it or narrow it to an area or a budget?"

_TAG = re.compile(r"\[\[\d{1,12}\]\]")
_NUMBER = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(million|thousand|mn|m|k)?\b", re.I)
_SCALE = {"million": 1e6, "mn": 1e6, "m": 1e6, "thousand": 1e3, "k": 1e3}
MIN_CHECKED = 10_000   # only money-sized figures are checked; years, bedrooms and small counts are not


class AgentState(TypedDict, total=False):
    messages: list
    calls: list          # tool calls the model asked for in the last step
    seen: dict           # every card a tool returned this turn
    used: list           # tools called
    tokens: int
    model: str
    rounds: int
    text: str            # the model's final answer
    retried: bool
    retry: bool
    unsupported: list


def _compact(value):
    """Drop empty fields from tool output before it is sent to the model: fewer tokens, same facts."""
    if isinstance(value, dict):
        return {k: _compact(v) for k, v in value.items() if v is not None and v != "" and v != [] and v != {}}
    if isinstance(value, list):
        return [_compact(v) for v in value]
    return value


def _numbers(text: str, scale: bool = True) -> list[float]:
    found = []
    for raw, suffix in _NUMBER.findall(text or ""):
        try:
            value = float(raw.replace(",", ""))
        except ValueError:
            continue
        found.append(value)
        if scale and suffix:
            found.append(value * _SCALE[suffix.lower()])
    return found


def unsupported_numbers(answer: str, messages: list) -> list[float]:
    """Money-sized figures in the answer that appear nowhere in the tool results, the conversation or the profile."""
    allowed: list[float] = []
    for index, message in enumerate(messages):
        if index == 0 and message.get("role") == "system":
            continue  # the instructions contain example figures
        allowed += _numbers(str(message.get("content") or ""))
    bad = []
    for value in _numbers(_TAG.sub("", answer), scale=False):
        if value < MIN_CHECKED or 1900 <= value <= 2100:
            continue
        if not any(abs(value - known) <= max(1.0, known * 0.005) for known in allowed):
            bad.append(value)
    # a figure written as "1.5 million" is read above as 1.5, which is below the limit and so never flagged
    for raw, suffix in _NUMBER.findall(_TAG.sub("", answer)):
        if suffix and float(raw.replace(",", "")) * _SCALE[suffix.lower()] >= MIN_CHECKED:
            value = float(raw.replace(",", "")) * _SCALE[suffix.lower()]
            if not any(abs(value - known) <= max(1.0, known * 0.02) for known in allowed):
                bad.append(value)
    return list(dict.fromkeys(bad))


def build_agent_graph(
    chat: Callable[[list], tuple[str, Any]],
    run_tool: Callable[[str, Any], Any],
    tool_error: type,
    max_rounds: int = 5,
    max_tool_json: int = 9_000,
    log=None,
    needs_grounding: Callable[[str], bool] | None = None,
):
    """chat(messages) -> (model_name, response); run_tool(name, args) -> ToolResult; tool_error: bad-argument exception."""

    def model_node(state: AgentState) -> dict:
        model, response = chat(state["messages"])
        tokens = state.get("tokens", 0) + (getattr(response.usage, "total_tokens", 0) or 0)
        message = response.choices[0].message
        base = {"model": model, "tokens": tokens, "rounds": state.get("rounds", 0) + 1}
        if not message.tool_calls:
            return {**base, "calls": [], "text": message.content or ""}
        assistant = {
            "role": "assistant", "content": message.content,
            "tool_calls": [{"id": c.id, "type": "function",
                            "function": {"name": c.function.name, "arguments": c.function.arguments}}
                           for c in message.tool_calls],
        }
        calls = [(c.id, c.function.name, c.function.arguments) for c in message.tool_calls]
        return {**base, "calls": calls, "messages": [*state["messages"], assistant]}

    def tools_node(state: AgentState) -> dict:
        messages, seen, used = list(state["messages"]), dict(state.get("seen", {})), list(state.get("used", []))
        for call_id, name, arguments in state["calls"]:
            try:
                result = run_tool(name, json.loads(arguments or "{}"))
                output = result.output
                used.append({"name": name, "ok": True})
                for card in result.cards:
                    seen[card["listing_id"]] = card
            except (tool_error, json.JSONDecodeError) as exc:
                used.append({"name": name, "ok": False})   # bad arguments go back so the model can correct itself
                output = {"error": str(exc)}
            except Exception:
                if log:
                    log.exception("tool %s failed", name)
                used.append({"name": name, "ok": False})
                output = {"error": "That lookup failed."}
            messages.append({"role": "tool", "tool_call_id": call_id,
                             "content": json.dumps(_compact(output), default=str,
                                                   separators=(",", ":"))[:max_tool_json]})
        return {"messages": messages, "seen": seen, "used": used, "calls": []}

    def give_up_node(state: AgentState) -> dict:
        return {"text": GIVE_UP, "calls": []}

    def check_node(state: AgentState) -> dict:
        text = state.get("text", "")

        # A question about a real place, project or developer answered from memory, with no tool and no
        # "General guidance" label, is sent back once to use the data.
        if needs_grounding and not state.get("used") and not state.get("retried") \
                and "general guidance" not in text.lower():
            question = next((str(m.get("content") or "") for m in reversed(state["messages"])
                             if m.get("role") == "user"), "")
            if needs_grounding(question):
                note = ("You answered a question about a specific area, project or developer without calling a "
                        "tool. Call the right tool now (market_stats or find_areas for an area, get_project for "
                        "a project, search_projects for a developer), then answer from its result. Put anything "
                        "from general knowledge after the ACOT data, starting with \"General guidance:\". "
                        "Do not mention this note.")
                return {"retried": True, "retry": True, "unsupported": [],
                        "messages": [*state["messages"], {"role": "assistant", "content": text},
                                     {"role": "system", "content": note}]}

        bad = unsupported_numbers(text, state["messages"])
        if not bad or state.get("retried"):
            return {"retry": False, "unsupported": bad}
        shown = ", ".join(f"{v:,.0f}" for v in bad[:5])
        note = ("Your reply quoted figures that no tool returned (" + shown + "). Answer again using only figures "
                "from the tool results; call a tool if you need one. Do not mention this note.")
        return {"retried": True, "retry": True, "unsupported": bad,
                "messages": [*state["messages"], {"role": "assistant", "content": state.get("text", "")},
                             {"role": "system", "content": note}]}

    def after_model(state: AgentState) -> str:
        if not state.get("calls"):
            return "check"
        return "give_up" if state.get("rounds", 0) >= max_rounds else "tools"

    def after_check(state: AgentState) -> str:
        return "model" if state.get("retry") else END

    graph = StateGraph(AgentState)
    graph.add_node("model", model_node)
    graph.add_node("tools", tools_node)
    graph.add_node("check", check_node)
    graph.add_node("give_up", give_up_node)
    graph.set_entry_point("model")
    graph.add_conditional_edges("model", after_model, {"tools": "tools", "check": "check", "give_up": "give_up"})
    graph.add_edge("tools", "model")
    graph.add_conditional_edges("check", after_check, {"model": "model", END: END})
    graph.add_edge("give_up", END)
    return graph.compile()
