from app.intelligence.account_summary_llm import (
    _compact_intelligence,
    _parse_profile,
)


intelligence = {
    "user_id": "21005862-7337-4deb-b7e7-1ad3f7b0d11b",
    "statistics": {
        "total_user_questions": 153,
        "meaningful_questions": 118,
        "ignored_noise": 35,
        "total_conversations": 51,
        "meaningful_conversations": 44,
    },
    "interests": [
        {
            "topic": "location",
            "question_count": 39,
            "conversation_count": 26,
            "strength": "very_high",
        },
        {
            "topic": "bedroom",
            "question_count": 29,
            "conversation_count": 17,
            "strength": "very_high",
        },
    ],
    "repeated_questions": [
        {
            "question": "show me project in jumeirah village circle",
            "count": 38,
            "conversation_count": 26,
        },
        {
            "question": "which of these have 2 bedroom options",
            "count": 24,
            "conversation_count": 17,
        },
    ],
    "latest_question_at": "2026-10-05T06:27:01.286+00:00",
}


# The labelled-text shape produced by generate_account_summary.
labelled = """Summary:
The user has a strong interest in Dubai off-plan properties.

AcotScore:
9

Key Insights:
- Very strong focus on Jumeirah Village Circle.
- Clear preference for 2-bedroom units.
- **Strong investment** orientation.
"""

parsed = _parse_profile(labelled, intelligence)

assert parsed["summary"].startswith("The user has a strong interest")
assert "Jumeirah" not in parsed["summary"]
assert parsed["acot_score"] == 9.0
assert len(parsed["key_insights"]) == 3
assert "Strong investment orientation." in parsed["key_insights"]


# The JSON shape requested by generate_account_profile.
fenced = """```json
{
  "summary": "The user focuses on Jumeirah Village Circle.",
  "acot_score": 8.5,
  "key_insights": ["a", "b", "c", "d", "e", "f", "g"]
}
```"""

parsed = _parse_profile(fenced, intelligence)

assert parsed["summary"] == "The user focuses on Jumeirah Village Circle."
assert parsed["acot_score"] == 8.5
assert len(parsed["key_insights"]) == 6


# A model that ignores the requested key names.
camel = '{"summary": "S", "acotScore": "7/10", "keyInsights": ["a", "b"]}'

parsed = _parse_profile(camel, intelligence)

assert parsed["acot_score"] == 7.0
assert parsed["key_insights"] == ["a", "b"]


# Score bounds.
assert _parse_profile('{"summary":"S","acot_score":99}', intelligence)["acot_score"] == 10.0
assert _parse_profile('{"summary":"S","acot_score":-4}', intelligence)["acot_score"] == 0.0
assert _parse_profile('{"summary":"S"}', intelligence)["acot_score"] is None


# Unlabelled output still yields a renderable summary.
garbage = _parse_profile("I cannot help with that.", intelligence)

assert garbage["summary"] == "I cannot help with that."
assert garbage["acot_score"] is None


# The prompt payload must not expose raw counts.
payload = _compact_intelligence(intelligence)

assert "statistics" not in payload
assert "question_count" not in payload
assert '"topic": "location"' in payload
assert "show me project in jumeirah village circle" in payload


if __name__ == "__main__":
    print("Account summary parsing checks passed.")