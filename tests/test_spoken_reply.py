import asyncio

import run
from app.answers.spoken import (
    SpokenReply,
    recent_history,
    strip_stock_opener,
    build_spoken_prompt,
    spoken_facts,
    start_spoken_reply,
)


class FakeLLM:
    def __init__(self, pieces=("Yes! ", "I found 2 projects."), fail=False):
        self.pieces = pieces
        self.fail = fail
        self.prompts = []

    async def stream(self, prompt, max_output_tokens=None):
        self.prompts.append(prompt)
        if self.fail:
            raise RuntimeError("model down")
        for piece in self.pieces:
            await asyncio.sleep(0)
            yield piece


class FakeRagChain:
    def __init__(self, llm):
        self.llm = llm


def prepared_with(**overrides):
    base = {
        "question": "Show me projects in Jumeirah Village Circle",
        "standalone_question": "Show me projects in Jumeirah Village Circle",
        "question_type": "browse",
        "structured_summary": {
            "projects": [{"name": "Binghatti Corner"}, {"name": "Belgravia Heights"}],
            "listings": [],
        },
        "use_fast": True,
        "fast_answer": "Here are 2 projects in JVC: Binghatti Corner, Belgravia Heights.",
        "context": None,
    }
    base.update(overrides)
    return base


def test_facts_include_counts_names_and_screen_text():
    facts = spoken_facts(prepared_with())
    assert "projects shown on screen: 2" in facts
    assert "Binghatti Corner, Belgravia Heights" in facts
    assert "Here are 2 projects in JVC" in facts
    assert "listings" not in facts  # empty groups are left out


def test_facts_without_data_say_so():
    assert "No data" in spoken_facts({"structured_summary": {}})


def test_prompt_matches_language_and_name_rules():
    english = build_spoken_prompt("Show me villas", "- facts", user_name="Charan")
    assert "Reply in English only" in english
    assert "Indian, British, American" in english
    assert "Charan" in english and "at most once" in english
    assert "let me check" in english  # tells the model not to repeat the filler
    assert '"Good question"' in english and 'Do not open with "Sure"' in english
    assert "cut off or unfinished" in english  # "Hello. So, uh," must not get a "no listings" reply
    assert "add ONE closely related fact" in english  # "What is the property type of X?": answer + one useful fact
    assert "never add details that are not in them" in english
    assert "answer only the last thing they asked" in english

    # Arabic input still gets an English-only spoken reply.
    mixed = build_spoken_prompt("أرني الفلل في دبي", "- facts")
    assert "Reply in English only" in mixed
    assert "Arabic" in mixed and "Reply in Arabic" not in mixed
    assert "Do not invent a name" in mixed


def test_small_talk_is_spoken_as_is_without_a_model_call():
    async def scenario():
        llm = FakeLLM()
        reply = start_spoken_reply(llm, "hi", prepared_with(small_talk="greeting", fast_answer="Hello Charan!"))
        text = "".join([t async for t in reply.rest()])
        return text, llm.prompts

    text, prompts = asyncio.run(scenario())
    assert text == "Hello Charan!"
    assert prompts == []


def test_failed_spoken_reply_ends_cleanly():
    async def scenario():
        reply = SpokenReply(FakeLLM(fail=True).stream("x"))
        text = "".join([t async for t in reply.rest()])
        return text, reply.ok

    assert asyncio.run(scenario()) == ("", False)


def _collect(mode, monkeypatch, llm):
    prepared = prepared_with(rag_chain=FakeRagChain(llm))

    async def fake_prepare(question, components):
        return prepared

    monkeypatch.setattr(run, "_prepare_acot_turn", fake_prepare)
    monkeypatch.setattr(run, "_stream_sources", lambda p: [])
    monkeypatch.setattr(run, "_followup_questions", lambda p: [])
    monkeypatch.setattr(run, "_finalize_acot_turn", lambda p, a, s: {"answer": a})

    async def scenario():
        return [e async for e in run.stream_acot("Show me projects in JVC", {}, mode=mode)]

    return asyncio.run(scenario())


def test_voice_mode_streams_speech_before_done(monkeypatch):
    events = _collect("voice", monkeypatch, FakeLLM())
    names = [name for name, _ in events]
    speech = "".join(data["text"] for name, data in events if name == "speech")

    assert speech == "I found 2 projects."  # the "Yes!" opener is dropped
    assert names.index("speech_done") < names.index("done")
    assert dict(events)["speech_done"]["ok"] is True
    assert dict(events)["done"]["answer"].startswith("Here are 2 projects")


def test_text_mode_has_no_speech_and_no_extra_model_call(monkeypatch):
    llm = FakeLLM()
    events = _collect("text", monkeypatch, llm)
    assert not [name for name, _ in events if name.startswith("speech")]
    assert llm.prompts == []


def test_voice_mode_still_answers_when_spoken_reply_fails(monkeypatch):
    events = _collect("voice", monkeypatch, FakeLLM(fail=True))
    assert dict(events)["speech_done"]["ok"] is False
    assert "done" in dict(events)


def test_stock_openers_are_dropped():
    assert strip_stock_opener("Good question! In JVC we have 24 projects.") == "In JVC we have 24 projects."
    assert strip_stock_opener("That's a great question. Here you go.") == "Here you go."
    assert strip_stock_opener("Sure! I found 5 projects.") == "I found 5 projects."
    assert strip_stock_opener("Yes, there are 3 villas.") == "There are 3 villas."
    assert strip_stock_opener("Sure! Good question, there are 2.") == "There are 2."
    assert strip_stock_opener("Surely the best area is JVC.") == "Surely the best area is JVC."
    assert strip_stock_opener("In JVC there are 5 projects.") == "In JVC there are 5 projects."


def test_opener_is_removed_even_when_split_across_tokens():
    async def scenario():
        llm = FakeLLM(pieces=("Su", "re! Good ", "ques", "tion! ", "In JVC we have 24 projects right now. ", "Want more?"))
        return "".join([t async for t in start_spoken_reply(llm, "JVC projects", prepared_with()).rest()])

    assert asyncio.run(scenario()) == "In JVC we have 24 projects right now. Want more?"


class FakeMemory:
    def __init__(self, turns):
        self.turns = turns


def test_spoken_reply_sees_the_conversation_so_far():
    memory = FakeMemory([
        {"user_question": "Show me projects in JVC", "answer": "Here are 5 projects: Serenz, Dawn by Binghatti, Azizi Ruby."},
        {"user_question": "", "answer": ""},
    ])
    history = recent_history({"conversation_memory": memory})
    assert history == "User: Show me projects in JVC\nAcot: Here are 5 projects: Serenz, Dawn by Binghatti, Azizi Ruby."

    prompt = build_spoken_prompt("what about the second one?", "- facts", history=history, meaning="Tell me about Dawn by Binghatti")
    assert "Conversation so far:\nUser: Show me projects in JVC" in prompt
    assert "What the user means: Tell me about Dawn by Binghatti" in prompt

    first = build_spoken_prompt("hi", "- facts")
    assert "(this is the first question)" in first and "What the user means" not in first


def test_amenity_codes_become_names_for_the_model():
    from app.answers.amenities import amenity_labels, strip_amenity_codes

    assert amenity_labels(["PR", "SP", "SY", "SE", "PG"]) == [
        "Children's play area", "Shared pool", "Shared gym", "Security", "Private garden",
    ]
    assert amenity_labels("SP, XX") == ["Shared pool"]  # unknown codes are dropped, never guessed
    assert amenity_labels(None) == []
    assert strip_amenity_codes("PR (Private Parking), SP (Swimming Pool)") == "Private Parking, Swimming Pool"


def test_spoken_facts_drop_amenity_codes():
    facts = spoken_facts(prepared_with(
        fast_answer="Azizi Ruby includes:\n- PR (Private Parking)\n- SE (Steam Room)",
    ))
    assert "PR" not in facts and "SE" not in facts
    assert "Private Parking" in facts and "Steam Room" in facts
