"""
Live check: does ACOT understand questions typed the way real people type them
(typos, slang, no area, mixed language, several asks in one line)?
Costs some AI credits.   Run:  python scripts/eval_messy.py [out.json]

expect="results": should answer with real data, not "nothing found".
expect="ask":     too vague to search; should ask back (e.g. which area / budget).
"""
import asyncio
import io
import json
import os
import re
import sys
import time
from contextlib import redirect_stderr, redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.memory.conversation_memory import ConversationMemory  # noqa: E402
from run import initialize_acot, stream_acot  # noqa: E402

CASES = [
    # spelling mistakes
    (["show me projects in jumeira vilage circle"], "results"),
    (["propertys in dubai marena"], "results"),
    (["bussiness bay projcts"], "results"),
    (["downtwon dubai apartmnts price"], "results"),
    (["any thing in palm jumera"], "results"),
    (["dubai hils estate villas"], "results"),
    (["projects in meydaan"], "results"),
    (["arjun projects"], "results"),
    # slang / short / lowercase chat style
    (["jvc 2bhk"], "results"),
    (["2 bhk marina rent?"], "results"),
    (["studio jlt cheap"], "results"),
    (["wat r the best projects in jvc"], "results"),
    (["best properties in jvc"], "results"),
    (["good investment in business bay?"], "results"),
    (["need villa dubai hills budget 5m"], "results"),
    (["1bed under 1m in jvc"], "results"),
    (["rent for 1bhk downtown how much"], "results"),
    (["ready to move flats in dubai marina"], "results"),
    (["projects handing over 2027 in jvc"], "results"),
    (["emaar projects in dubai hills"], "results"),
    # mixed language (Hinglish / Arabic words)
    (["marina mein 2bhk chahiye rent pe"], "results"),
    (["jvc me sasta flat dikhao"], "results"),
    (["shu fi apartments in business bay"], "results"),
    # several asks in one line
    (["price of 23 marina and is it ready and payment plan?"], "results"),
    (["compare jvc and business bay which is better for rent yield"], "results"),
    (["show jvc projects also rent prices there"], "results"),
    # vague: should ask back, not guess or say nothing found
    (["2bhk under 1M?"], "ask"),
    (["good place for family"], "ask"),
    (["something cheap near metro"], "ask"),
    (["i want to invest"], "ask"),
    (["show me apartments"], "ask"),
    (["whats the price"], "ask"),
    # messy follow-ups
    (["projects in jvc", "wich one cheapest"], "results"),
    (["show me projects in business bay", "any 2bhk in those"], "results"),
    (["dubai marina rentals", "and jlt?"], "results"),
    (["projects in jvc", "ok what abt marina"], "results"),
]

# Phrases that mean the AI gave up / found nothing.
FAIL = re.compile(
    r"\b(no (?:matching |such )?(?:listings|projects|results|properties|records|data)|"
    r"couldn'?t find|could not find|not find any|don'?t have (?:any )?(?:data|information|records)|"
    r"no information|nothing (?:found|matched)|unable to find|not available in (?:our|the) (?:data|records))\b",
    re.I,
)
ASK = re.compile(
    r"\b(which (?:area|community|location|project|property|neighbou?rhood)|what(?:'s| is) your budget|"
    r"specify (?:which|the)|share your budget|(?:your|a) budget|preferred area|specific area|"
    r"which part|where (?:in dubai|would you)|could you (?:tell|share|let)|do you have a (?:preferred|specific)|"
    r"any (?:preferred|specific) (?:area|community|location))",
    re.I,
)


def judge(answer, expect):
    text = answer or ""
    if not text.strip():
        return "FAIL: empty"
    if expect == "ask":
        return "ok" if ASK.search(text) else "FAIL: did not ask back"
    return "FAIL: said nothing found" if FAIL.search(text) else "ok"


def fresh(components):
    components["conversation_memory"] = ConversationMemory(
        llm_client=components["query_planner"]._llm_client
    )
    for key in ("agent_history", "profile", "memory_loaded"):
        components.pop(key, None)


async def one_turn(question, components):
    """Run a turn the way the app does in voice mode: written answer, speech, chips, cards."""
    out = {"speech": "", "followups": [], "response": {}}
    async for kind, event in stream_acot(question, components, mode="voice"):
        if kind == "speech":
            out["speech"] += event["text"]
        elif kind == "followups":
            out["followups"] = event["questions"]
        elif kind == "done":
            out["response"] = event
    return out


def card_counts(response):
    """How many of each card list the app would render."""
    counts = {}
    for key, value in response.items():
        if isinstance(value, list) and value and isinstance(value[0], dict):
            counts[key] = len(value)
        elif isinstance(value, dict):
            for sub, items in value.items():
                if isinstance(items, list) and items and isinstance(items[0], dict):
                    counts[f"{key}.{sub}"] = len(items)
    return counts


async def run_all(components):
    rows = []
    for turns, expect in CASES:
        fresh(components)
        out, started, err = {"speech": "", "followups": [], "response": {}}, time.perf_counter(), None
        for turn in turns:
            try:
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    out = await one_turn(turn, components)
            except Exception as exc:  # keep going; record it
                err = str(exc)
        answer = out["response"].get("answer", "")
        verdict = f"FAIL: error {err}" if err else judge(answer, expect)
        rows.append({
            "turns": turns, "expect": expect, "verdict": verdict,
            "seconds": round(time.perf_counter() - started, 1), "answer": answer,
            "speech": out["speech"], "followups": out["followups"],
            "cards": card_counts(out["response"]),
            "response_keys": sorted(out["response"].keys()),
        })
        print(f"[{verdict[:4]}] {round(time.perf_counter() - started, 1):>5}s  {' -> '.join(turns)}  ({verdict})", flush=True)
    return rows


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "messy_results.json"
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        components = initialize_acot()
    only = sys.argv[2] if len(sys.argv) > 2 else ""  # optional: run only cases containing this text
    if only:
        CASES[:] = [c for c in CASES if any(re.search(only, t) for t in c[0])]
    rows = asyncio.run(run_all(components))
    ok = sum(r["verdict"] == "ok" for r in rows)
    print(f"\n{ok}/{len(rows)} passed")
    with open(out, "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
