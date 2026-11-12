"""Short, friendly spoken replies for voice mode.

The chat answer is written to be read (names, sqft, tables). In voice mode the app
also needs something that sounds like a person talking, so this writes 2-4 warm
sentences from the same facts the chat answer uses.
"""

import asyncio
import re

from app.answers.amenities import strip_amenity_codes

# Models love opening with these even when told not to; drop them before speaking.
# "Sure!" / "Yes," need punctuation after them, so "Yes, there are…" keeps its meaning
# as "There are…" and "Sure thing" style phrases are left alone.
STOCK_OPENER = re.compile(
    r"^\s*(?:(?:sure|of course|absolutely|certainly|okay|ok|yes|great)[!.,]+\s*)?"
    r"(?:(?:(?:that'?s|what)\s+an?\s+|an?\s+)?(?:good|great|excellent|nice|interesting)\s+question[\s!.,:;-]*)?",
    re.IGNORECASE,
)
OPENER_LOOKAHEAD = 40
MAX_FACT_CHARS = 3000
MAX_SPOKEN_TOKENS = 180
SPOKEN_TIMEOUT_S = 8

_GROUPS = (
    ("projects", "projects"),
    ("listings", "listings"),
    ("community", "communities"),
    ("sub_communities", "sub-communities"),
    ("brokers", "agents"),
)


def _record_name(record):
    if not isinstance(record, dict):
        return ""
    for key in ("name", "title", "project_name"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def spoken_facts(prepared):
    """What the spoken reply may talk about: counts, top names and the on-screen answer."""
    summary = prepared.get("structured_summary") or {}
    lines = []
    # Investment questions: the market signals come first so the reply is built around them.
    for community in summary.get("community") or []:
        signals = community.get("signals") if isinstance(community, dict) else None
        if signals:
            lines.append(f"- Market signals for {community.get('name')} (sales and rent records, shown on screen): {signals}")
    for key, label in _GROUPS:
        items = summary.get(key) or []
        if not isinstance(items, list) or not items:
            continue
        names = [n for n in (_record_name(item) for item in items[:5]) if n]
        line = f"- {label} shown on screen: {len(items)}"
        if names:
            line += f" (first ones: {', '.join(names)})"
        lines.append(line)
    titles = [str(d.get("document_title") or d.get("title") or "") for d in (prepared.get("documents") or []) if isinstance(d, dict)]
    if any(titles):
        lines.append(f"- Documents shown on screen as links: {', '.join(t for t in titles if t)}")
    if summary.get("filter_no_match"):
        lines.append("- Nothing matched the user's exact filters.")

    shown = prepared.get("fast_answer") or prepared.get("context") or ""
    shown = str(shown).strip()[:MAX_FACT_CHARS]
    if shown:
        lines.append("- What the screen shows:\n" + strip_amenity_codes(shown))
    return strip_amenity_codes("\n".join(lines)) or "- No data was found for this question."


HISTORY_TURNS = 3
HISTORY_ANSWER_CHARS = 300


def recent_history(prepared):
    """The last few turns of this chat, so follow-ups like "the second one" make sense."""
    memory = prepared.get("conversation_memory")
    turns = list(getattr(memory, "turns", None) or [])[-HISTORY_TURNS:]
    lines = []
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        asked = str(turn.get("user_question") or "").strip()
        answered = strip_amenity_codes(
            " ".join(str(turn.get("answer") or "").split())
        )[:HISTORY_ANSWER_CHARS]
        if asked:
            lines.append(f"User: {asked}")
        if answered:
            lines.append(f"Acot: {answered}")
    return "\n".join(lines)


def build_spoken_prompt(question, facts, user_name=None, history="", meaning="", language=None):
    # The voice-mode language picker decides; without it, Hindi script means Hindi.
    # Decided here, not by the model, so earlier turns can't pull the reply into another language.
    hindi = language == "hi" if language else bool(re.search(r"[\u0900-\u097F]", question or ""))
    language_rule = (
        "- Reply in Hindi only, even if the user spoke English or another language, or earlier messages were in English. Never use Arabic or any other language.\n"
        '- Write every Hindi word in Devanagari letters ("आपको", "है", "लगभग"), never in English letters ("aapko", "hai", "lagbhag").\n'
        '- Talk the way people in India really speak: friendly, everyday Hinglish. Use the common English words people say anyway (price, project, apartment, location, deal, screen) instead of formal Hindi like "मूल्य", "परियोजना" or "उपलब्ध". Use "आप". Keep project, area and developer names in English letters, and say prices like "लगभग 12 लाख दिरहम".'
        if hindi
        else "- Reply in English only, even if earlier messages were in Hindi or another language. Never use Arabic, Hindi or any other language.\n"
        "- The user may speak any variety of English (Indian, British, American, Australian, and so on). Understand their wording and answer in clear, natural English."
    )
    name_rule = (
        f"- You may use the user's first name ({user_name}) at most once, and only if it feels natural."
        if user_name
        else "- Do not invent a name for the user."
    )
    return f"""You are Acot, a warm and friendly Dubai real estate assistant talking out loud to a user.
They asked a question by voice. The full detailed answer is already on their screen.
Write what you SAY to them now.

Rules:
{language_rule}
- First work out exactly what was asked, and answer only that:
  - One detail (like the property type, price, handover, bedrooms, amenities, location or developer of a project): answer it first and fully, then add ONE closely related fact the user would likely want next, in 1 to 2 natural sentences. For example: "It's an apartment project, with studios up to three-bedroom units starting from about 618 thousand dirhams." Not a full overview, and no follow-up question.
  - A broad question ("show me projects in JVC", "tell me about a project"): 2 to 4 short sentences, about 60 words at most.
  - A comparison of projects or areas ("compare A and B", "A vs B"): 2 to 3 short sentences with the two or three biggest differences (like price, handover or size) and which suits what kind of buyer, then say the full comparison is on their screen. Do not go through every detail.
- If the user corrects themselves mid-sentence ("how many... sorry, what is the property type"), answer only the last thing they asked.
- Sound like a friendly person chatting on a call, not a report: warm, relaxed and human.
  - Use contractions ("it's", "there's", "you'll") and short, easy sentences.
  - A small natural reaction is welcome when it fits, like "JVC's a great pick for families" or "that's a nice budget for Marina", but only one, and only if the facts support it.
- Start straight with the answer, like "In Jumeirah Village Circle there are 5 projects…". Do not open with "Sure", "Yes", "Of course", "Absolutely", "Okay" or "Good question", and never say "let me check" or "one moment".
- For broad questions with results, mention how many there are and 2 or 3 names.
- Say numbers the way people speak: "about 1.2 million dirhams", "around 7 percent". No sqft lists, IDs, links, markdown, emojis, bullet points or tables.
- Never say a two-letter amenity code. Say only the amenity name, for example "private parking", not "P R" or "PR private parking".
- Point to the screen for the details, for example "I've put them all on your screen".
- If the facts include market signals (an investment question), you must build the reply on them: say how the price per sq ft moved over the last year and the rough rental yield, in plain words ("prices are up about 3 percent on last year, and rents work out to roughly 6 percent a year"). Then say the details are on screen and end with "this isn't financial advice". Never tell them to buy or not to buy.
- If the user asks for a brochure, floor plan or other document, never read it out. If the facts list documents shown on screen, say you've put the brochure on their screen. If none is listed, say you couldn't find one for that project.
- For broad questions you may end with one short, friendly follow-up question.
- If the user's message is only a greeting, thanks, or sounds cut off or unfinished (for example "Hello. So, uh,"), do not talk about search results or listings: reply in one short, warm sentence and ask what they would like to know about Dubai property.
- Otherwise use only the facts below, and never add details that are not in them (no guessed amenities, prices or dates). Never say how near or far something is from a metro, school, beach or park unless the facts say it. Only name projects, listings and areas that the facts say are shown on screen. If the facts say NO EXACT MATCH, say that first ("I couldn't find Emaar projects in Dubai Hills"), then describe the fallback as what it is (other projects there, or rentals). If the facts have read_as, use the corrected place name. If a real search found nothing, say so kindly and suggest widening the search or a nearby area.
{name_rule}

- Use the conversation so far to understand follow-ups (like "the second one" or "what about villas?") and don't repeat what you already said.

Conversation so far:
{history or "(this is the first question)"}

User's question: {question}{f"""
What the user means: {meaning}""" if meaning and meaning.strip().lower() != question.strip().lower() else ""}

Facts:
{facts}

What you say{" (in Hindi, written in Devanagari letters)" if hindi else ""}:"""


def strip_stock_opener(text):
    stripped = STOCK_OPENER.sub("", text, count=1)
    return stripped[:1].upper() + stripped[1:] if stripped != text else text


async def stream_spoken_reply(llm, question, prepared, user_name=None, language=None):
    """Yield the spoken reply as the model writes it, minus any "Good question" opener."""
    prompt = build_spoken_prompt(
        question,
        spoken_facts(prepared),
        user_name,
        history=recent_history(prepared),
        meaning=str(prepared.get("standalone_question") or ""),
        language=language,
    )
    head = ""
    async for delta in llm.stream(prompt, max_output_tokens=MAX_SPOKEN_TOKENS):
        if not delta:
            continue
        if head is None:
            yield delta
            continue
        # Hold back the first few words until we can see whether they are an opener.
        head += delta
        if len(head) >= OPENER_LOOKAHEAD:
            yield strip_stock_opener(head)
            head = None
    if head:
        yield strip_stock_opener(head)


async def _fixed(text):
    yield text


class SpokenReply:
    """Writes the spoken reply in the background while the chat answer streams.

    The caller picks up finished text with drain() between chat deltas, then
    rest() at the end. A failure or timeout here never breaks the chat answer.
    """

    def __init__(self, source, timeout=SPOKEN_TIMEOUT_S):
        self.ok = False
        self._queue = asyncio.Queue()
        self._closed = False
        self._task = asyncio.create_task(self._run(source, timeout))

    async def _run(self, source, timeout):
        try:
            async with asyncio.timeout(timeout):
                async for delta in source:
                    self.ok = True
                    self._queue.put_nowait(delta)
        except Exception as exc:  # includes TimeoutError
            print(f"Spoken reply unavailable: {exc!r}")
        finally:
            self._queue.put_nowait(None)

    def drain(self):
        out = []
        while not self._closed and not self._queue.empty():
            item = self._queue.get_nowait()
            if item is None:
                self._closed = True
            else:
                out.append(item)
        return out

    async def rest(self):
        while not self._closed:
            item = await self._queue.get()
            if item is None:
                self._closed = True
            else:
                yield item

    def cancel(self):
        self._task.cancel()


def start_spoken_reply(llm, question, prepared, user_name=None, language=None):
    # Greetings and small talk already have a short friendly English answer; say it as is,
    # unless Hindi was picked, then the model says it in Hindi.
    if prepared.get("small_talk") and prepared.get("fast_answer") and language != "hi":
        return SpokenReply(_fixed(prepared["fast_answer"]))
    return SpokenReply(stream_spoken_reply(llm, question, prepared, user_name, language))
