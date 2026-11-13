from app.retrieval.hybrid.entity_resolver import EntityResolver

PROJECTS = [
    {"id": 233000511, "name": "Dawn by Binghatti", "city": "Dubai", "community": "Jumeirah Village Circle"},
    {"id": 1, "name": "Samana South Haven 2", "city": "Dubai", "community": "Dubai Land"},
    {"id": 2, "name": "One By Binghatti", "city": "Dubai", "community": "Business Bay"},
]


class Catalog(EntityResolver):
    def _catalog(self):
        return [], PROJECTS

    def _listing_sub_communities(self):
        return []


def test_exact_community_beats_a_longer_sub_community_name():
    class Places(EntityResolver):
        def _catalog(self):
            return [{"id": 1, "name": "Dubai Marina", "city": "Dubai"}], []

        def _listing_sub_communities(self):
            return [{"name": "Dubai Marina Mall", "community": "Dubai Marina"}]

    resolver = Places(object())
    area = resolver.resolve("Show me properties in Dubai Marina").to_dict()
    assert area["entity_type"] == "community"
    assert area["name"] == "Dubai Marina"

    mall = resolver.resolve("Show me properties in Dubai Marina Mall").to_dict()
    assert mall["entity_type"] == "sub_community"
    assert mall["name"] == "Dubai Marina Mall"


def test_spoken_question_with_fillers_needs_the_planner_name():
    resolver = Catalog(object())
    question = "So, uh, explain about Dawn by benatti."  # speech-to-text misheard "Binghatti"

    without = resolver.resolve(question).to_dict()
    assert without["name"] != "Dawn by Binghatti"  # the whole sentence alone is too noisy

    resolved = resolver.resolve(question, candidates_from_planner=["dawn by benatti"]).to_dict()
    assert resolved["name"] == "Dawn by Binghatti"
    assert resolved["entity_id"] == 233000511


def test_spoken_fillers_are_not_read_as_a_list_of_names():
    from app.retrieval.hybrid.question_text import interpret_search, prepare_user_question

    assert prepare_user_question("So, uh, explain about Dawn by benatti.") == "explain about dawn by benatti"
    assert prepare_user_question("Okay, um, show me villas in Dubai Hills") == "show me villas in dubai hills"
    # Real lists still work.
    assert interpret_search("So, uh, explain about Dawn by benatti.").get("kind") != "multi_name"
    assert interpret_search("Serenz, Azizi Ruby, Samana Waves").get("kind") == "multi_name"
    # "um"/"uh" inside real words are untouched.
    assert "umm suqeim" in prepare_user_question("Um, Umm Suqeim villas")
    assert prepare_user_question("Show me Mudon villas") == "show me mudon villas"


def test_voice_commas_and_show_about_work_for_any_place():
    """Voice commas and 'show about X properties' must work for every place.

    No hard-coded community list: the place phrase is normalized, then retrieval
    matches it against the live community catalog.
    """
    from app.retrieval.hybrid.question_text import interpret_search, prepare_user_question, subject_for

    cases = (
        (
            "Can you show about the Jumeirah, Village, Circle properties?",
            "jumeirah village circle",
            "Jumeirah Village Circle",
        ),
        (
            "show me the properties in Business, Bay",
            "business bay",
            "Business Bay",
        ),
        (
            "Can you show about the Dubai Marina properties?",
            "dubai marina",
            "Dubai Marina",
        ),
        (
            "show me properties in Palm, Jumeirah",
            "palm jumeirah",
            "Palm Jumeirah",
        ),
        (
            "tell me about Motor City projects",
            "motor city",
            "Motor City",
        ),
    )

    for question, place, label in cases:
        prepared = prepare_user_question(question)
        assert place in prepared, (question, prepared)
        assert ",".join(place.split()) not in prepared.replace(" ", ""), (question, prepared)

        shape = interpret_search(question)
        assert shape.get("kind") == "community", (question, shape)
        assert shape.get("community") == place, (question, shape)
        assert subject_for(shape) == (label, "place"), question

    # Project-by-developer stays a name pattern, not a fake place.
    by_dev = interpret_search("show me Dawn by Binghatti projects")
    assert by_dev.get("kind") == "name_pattern", by_dev

    # Real multi-project lists must keep their commas.
    assert interpret_search("Serenz, Azizi Ruby, Samana Waves").get("kind") == "multi_name"
