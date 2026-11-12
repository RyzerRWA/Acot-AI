from app.intelligence.ai_analysis_engine import AIAnalysisEngine


def test_parse_valid_json():
    parsed = AIAnalysisEngine._parse_llm_json(
        '{"summary": "ok", "key_facts": [], "confidence": "High"}'
    )
    assert parsed["summary"] == "ok"
    assert parsed["confidence"] == "High"


def test_parse_trailing_comma_and_unquoted_keys():
    raw = """
    {
      "summary": "JVC 3-bed options",
      "key_facts": ["Budget AED 1.8M",],
      positive_factors: ["Recorded sales"],
      "considerations": [],
      "ai_solution": "Keep the same area and bedrooms.",
      "confidence": "Medium",
    }
    """
    parsed = AIAnalysisEngine._parse_llm_json(raw)
    assert parsed is not None
    assert parsed["summary"] == "JVC 3-bed options"
    assert parsed["positive_factors"] == ["Recorded sales"]
    assert parsed["confidence"] == "Medium"


def test_parse_markdown_fence_and_prose():
    raw = """Here is the analysis:
```json
{"summary": "More JVC matches", "ai_solution": "Show the next projects."}
```
Thanks.
"""
    parsed = AIAnalysisEngine._parse_llm_json(raw)
    assert parsed["summary"] == "More JVC matches"


def test_parse_truncated_string():
    raw = '{"summary": "More options in Jumeirah Village Circle", "ai_solution": "Keep going'
    parsed = AIAnalysisEngine._parse_llm_json(raw)
    assert parsed is not None
    assert "Jumeirah Village Circle" in parsed["summary"]
