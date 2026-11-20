"""Casual-speech turn-act gate — acknowledgements must not hit project data."""

from app.answers.turn_act import casual_kind, classify_turn_act, is_casual_speech
from run import _classify_small_talk, _shortlist_recommend_turn, _small_talk_answer


def test_casual_phrases_detected():
    for text in (
        "ok",
        "okay",
        "cool",
        "great",
        "Ok that's cool",
        "ok that is cool",
        "thanks",
        "thank you",
        "got it",
        "sounds good",
        "awesome",
        "perfect",
    ):
        assert is_casual_speech(text), text
        assert classify_turn_act(text) == "casual", text
        assert _classify_small_talk(text) in {"ack", "thanks", "greeting", "bye"}, text


def test_real_asks_are_not_casual():
    for text in (
        "ok show me more",
        "show me projects in Marina",
        "which of these is best for a family?",
        "suggest me the best project in Dubai Marina for living family",
        "top agents for JVC",
        "Show me the recent transaction details",
        "what about Dubai Marina",
        "same but 2 bedrooms",
    ):
        assert not is_casual_speech(text), text
        assert _classify_small_talk(text) is None, text


def test_casual_ack_answer_has_no_project_dump():
    answer = _small_talk_answer(casual_kind("Ok that's cool"))
    lower = answer.lower()
    assert "glad" in lower or "okay" in lower or "welcome" in lower
    assert "i found" not in lower
    assert "on-screen projects were ranked" not in lower
    assert "acot recommend" not in lower


def test_casual_does_not_shortlist_or_followup_continue():
    from app.memory.conversation_memory import ConversationMemory
    import asyncio
    from app.answers.followup_resolve import resolve_followup

    memory = ConversationMemory(llm_client=None)
    memory.set_dialogue_state({
        "action": "recommend",
        "goal": "family",
        "kind": "filtered",
        "place": "Dubai Marina",
        "beds": 3,
    })
    memory.active_candidates = [
        {"name": "Ocean Heights", "entity_type": "project", "community": "Dubai Marina"},
        {"name": "Marina Pinnacle", "entity_type": "project", "community": "Dubai Marina"},
    ]

    assert _shortlist_recommend_turn("Ok that's cool", memory) is None

    async def _run():
        return await resolve_followup(
            "Ok that's cool",
            memory.get_dialogue_state(),
            llm_generate=None,
        )

    resolution = asyncio.run(_run())
    assert resolution is None or resolution.get("kind") == "new"
