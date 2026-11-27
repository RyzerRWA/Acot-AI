"""ACOT QA R17: a failed summary generation never becomes a 500 for the Admin page."""
from app.intelligence import account_summary_service as svc

SIGNAL = {"statistics": {"meaningful_questions": 3}, "latest_question_at": "2026-11-20"}


def _fail(*_args, **_kwargs):
    raise RuntimeError("model timed out")


def test_failure_without_history_says_so(monkeypatch):
    svc.clear_summary_cache()
    monkeypatch.setattr(svc, "analyze_questions", lambda _uid: SIGNAL)
    monkeypatch.setattr(svc, "_get_user_plan", lambda _uid: "freeplan")
    monkeypatch.setattr(svc, "_build_payload", _fail)
    payload = svc.get_account_summary("u1", refresh=True)
    assert payload["available"] is False
    assert payload["acot_score"] is None  # qualification skips a missing score, it never overwrites a lead
    assert "could not be generated" in payload["error"]


def test_failure_shows_last_good_summary(monkeypatch):
    svc.clear_summary_cache()
    svc._write_cache("u2:freeplan:old", {"user_id": "u2", "available": True, "summary": "Ready 2-bed in JLT.", "acot_score": 4.0})
    monkeypatch.setattr(svc, "analyze_questions", lambda _uid: SIGNAL)
    monkeypatch.setattr(svc, "_get_user_plan", lambda _uid: "freeplan")
    monkeypatch.setattr(svc, "_build_payload", _fail)
    payload = svc.get_account_summary("u2", refresh=True)
    assert payload["summary"] == "Ready 2-bed in JLT."
    assert payload["stale"] is True
