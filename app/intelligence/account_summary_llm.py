import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

API_KEY = os.getenv("SUMMARY_AICREDITS_API_KEY")
BASE_URL = os.getenv("SUMMARY_AICREDITS_BASE_URL")
MODEL = os.getenv("SUMMARY_AICREDITS_MODEL", "z-ai/glm-5.3")

if not API_KEY or not BASE_URL:
    raise RuntimeError("Summary AI Credits configuration is missing.")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


PROMPT_BEHAVIOUR = """
You are ACOT's account-level real-estate behavior analyst.

Analyze ONLY the supplied account intelligence derived from USER QUESTIONS.

Write the summary directly about the user.

IMPORTANT:
- Start the summary with "The user..." or naturally describe the user's
  interests.
- Do NOT start with "This account..."
- Do NOT start with "This user shows..."
- Do NOT mention question counts.
- Do NOT mention conversation counts.
- Do NOT mention raw statistics or numbers.
- Focus on what the user has a strong interest in.
- Describe clear preferences and recurring research behavior.
- Use natural phrases such as:
  "The user has a strong interest in..."
  "The user has a clear preference for..."
  "The user frequently explores..."
  "The user appears particularly interested in..."
  "The user often compares..."
- Do not analyze assistant answers.
- Do not invent facts.
- Do not overstate conclusions.

Focus on:
locations, bedroom preferences, investment/rental yield,
prices, handover dates, amenities, comparisons, and recent interests.
"""


def generate_account_summary(intelligence: dict) -> str:
    """Return the raw human-readable account summary (CLI/debug use)."""

    prompt = f"""
{PROMPT_BEHAVIOUR}
FORMAT:

Summary:
Write one natural paragraph describing the user's main
real-estate interests and research behavior.
AcotScore:
give a single number between 0 and 10 that represents the user's overall
engagement and interest level in real estate topics.

Key Insights:
Write 4-6 concise points describing the strongest observed interests.

ACCOUNT INTELLIGENCE:
{_compact_intelligence(intelligence)}
"""

    content = _complete(prompt)

    if content:
        return content.strip()

    raise RuntimeError("GLM returned no final content.")


def generate_account_profile(intelligence: dict) -> dict:
    """
    Return the structured account AI summary.

    Shape:
        {
            "summary": str,
            "acot_score": float | None,
            "key_insights": list[str]
        }
    """

    prompt = f"""
{PROMPT_BEHAVIOUR}
OUTPUT FORMAT:

Return ONLY a JSON object, no markdown fences and no commentary.

{{
  "summary": "one natural paragraph describing the user's main real-estate
              interests and research behavior",
  "acot_score": <number between 0 and 10>,
  "key_insights": ["4 to 6 concise points describing the strongest observed
                    interests"]
}}

RULES:
- "acot_score" is a single number between 0 and 10 reflecting overall
  engagement and interest level in real estate topics.
- "key_insights" holds 4 to 6 short strings, each one observation.
- Never mention counts, statistics or raw numbers in the summary.

ACCOUNT INTELLIGENCE:
{_compact_intelligence(intelligence)}
"""

    content = _complete(prompt)

    if not content:
        raise RuntimeError("GLM returned no final content.")

    return _parse_profile(content, intelligence)


# ============================================================
# LLM CALL
# ============================================================

def _complete(prompt: str) -> str | None:

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1200,
        reasoning_effort="low",
    )

    content = response.choices[0].message.content

    return content.strip() if content else None


# ============================================================
# PROMPT PAYLOAD
# ============================================================

def _compact_intelligence(intelligence: dict) -> str:
    """
    Keep the prompt payload focused on behavioural signals.

    Statistics and full question text are dropped so the model reasons
    about behaviour instead of copying raw numbers.
    """

    if not isinstance(intelligence, dict):
        return ""

    payload = {
        "interests": [
            {
                "topic": item.get("topic"),
                "strength": item.get("strength"),
                "conversation_count": item.get("conversation_count"),
            }
            for item in intelligence.get("interests", []) or []
            if isinstance(item, dict)
        ],
        "recurring_questions": [
            item.get("question")
            for item in intelligence.get("repeated_questions", []) or []
            if isinstance(item, dict)
        ][:10],
        "latest_question_at": intelligence.get("latest_question_at"),
    }

    return json.dumps(payload, indent=2, default=str)


# ============================================================
# RESPONSE PARSING
# ============================================================

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def _parse_profile(content: str, intelligence: dict) -> dict:
    """
    Parse the LLM output into the structured profile shape.

    JSON is preferred. The labelled-text format produced by
    generate_account_summary is used as a fallback so the endpoint keeps
    working if the model ignores the JSON instruction.
    """

    raw = _FENCE.sub("", content.strip())
    candidate = _first_json_object(raw)

    if candidate is not None:
        parsed = _normalise_json(candidate)
        if parsed.get("summary"):
            return parsed

    return _parse_labelled_text(raw)


def _first_json_object(raw: str) -> dict | None:

    start = raw.find("{")
    end = raw.rfind("}")

    if start < 0 or end <= start:
        return None

    try:
        value = json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return None

    return value if isinstance(value, dict) else None


def _normalise_json(payload: dict) -> dict:

    summary = payload.get("summary")
    score = payload.get("acot_score", payload.get("acotScore"))
    insights = payload.get("key_insights", payload.get("keyInsights"))

    return {
        "summary": summary.strip() if isinstance(summary, str) else "",
        "acot_score": _clean_score(score),
        "key_insights": _clean_insights(insights),
    }


def _parse_labelled_text(raw: str) -> dict:

    summary, score_text, insights_text = _split_sections(raw)

    return {
        "summary": summary,
        "acot_score": _clean_score(score_text),
        "key_insights": _clean_insights(insights_text.splitlines()),
    }


_SECTION = re.compile(
    r"^\s*(?:\*\*)?(summary|acot\s*score|acotscore|key\s*insights)"
    r"(?:\*\*)?\s*:\s*",
    re.IGNORECASE,
)


def _split_sections(raw: str) -> tuple[str, str, str]:

    sections: dict[str, list[str]] = {
        "summary": [],
        "score": [],
        "insights": [],
    }

    current = None

    for line in raw.splitlines():

        match = _SECTION.match(line)
        remainder = ""

        if match:
            label = match.group(1).lower().replace(" ", "")
            current = (
                "score"
                if label.startswith("acot")
                else "summary"
                if label.startswith("summ")
                else "insights"
            )
            remainder = line[match.end():].strip()
        elif current:
            remainder = line.strip()

        if current and remainder:
            sections[current].append(remainder)

    return (
        " ".join(sections["summary"]).strip(),
        " ".join(sections["score"]).strip(),
        "\n".join(sections["insights"]),
    )


_BULLET = re.compile(r"^\s*(?:[-*\u2022\u2013]|\d+[.)])\s*")
_SCORE = re.compile(r"-?\d+(?:\.\d+)?")


def _clean_score(value: unknown) -> float | None:

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
    elif isinstance(value, str):
        match = _SCORE.search(value.replace(",", "."))
        number = float(match.group()) if match else None
    else:
        return None

    if number is None:
        return None

    return round(max(0.0, min(10.0, number)), 1)


def _clean_insights(value: unknown) -> list[str]:

    if isinstance(value, str):
        lines: list[str] = value.splitlines()
    elif isinstance(value, list):
        lines = [str(item) for item in value]
    else:
        return []

    insights: list[str] = []

    for line in lines:

        cleaned = _BULLET.sub("", line).strip()

        if cleaned:
            insights.append(cleaned)

    return insights[:6]