import re
import json
from collections import Counter, defaultdict
from app.intelligence.account_summary_llm import generate_account_summary
from app.database.chat_supabase_client import supabase
from app.intelligence.account_intelligence_store import (
    save_account_intelligence
)


NOISE = {
    "hi", "hello", "hey", "hey acot", "ok", "okay",
    "yes", "no", "bye", "thanks", "thank you",
    "thankyou", "shukriya", "shukriya bhai",
    "who are you", "what can you do"
}


PATTERNS = {
    "location": [
        "jvc", "jumeirah village circle", "dubai marina",
        "al jaddaf", "community", "area", "location", "district"
    ],
    "bedroom": ["bedroom", "bed", "studio"],
    "price": ["price", "prices", "starting price", "cost", "budget"],
    "investment": [
        "investment", "invest", "roi", "yield",
        "rental yield", "rental income", "return"
    ],
    "comparison": [
        "compare", "comparison", "vs", "versus",
        "difference between", "which is better"
    ],
    "amenities": [
        "amenities", "facilities", "gym", "pool",
        "parking", "clubhouse"
    ],
    "handover": ["handover", "completion", "ready"],
    "payment_plan": [
        "payment plan", "installment",
        "down payment", "monthly payment"
    ]
}


def normalize(question: str) -> str:
    q = question.lower().strip()
    q = re.sub(r"^\s*\d+[.)]\s*", "", q)
    q = re.sub(r"https?://\S+", "", q)
    q = re.sub(r"[^\w\s]", " ", q)
    q = re.sub(r"\b(\d+)\s*(?:beds?|bedrooms?)\b", r"\1 bedroom", q)
    q = re.sub(r"\bprojects?\b", "project", q)
    q = re.sub(r"\bproperties\b", "property", q)
    q = re.sub(r"\bthe\s+(project|property)\b", r"\1", q)
    q = re.sub(r"\b(an?|the)\s+(project|property)\b", r"\2", q)
    return re.sub(r"\s+", " ", q).strip()


def is_gibberish(q: str) -> bool:
    if len(q) < 7 or len(q.split()) != 1:
        return False

    vowels = len(re.findall(r"[aeiou]", q))
    consonants = len(re.findall(r"[bcdfghjklmnpqrstvwxyz]", q))

    return vowels <= 1 and consonants >= 5


def is_noise(question: str) -> bool:
    q = normalize(question)

    return (
        not q
        or q in NOISE
        or re.fullmatch(r"h+i+", q) is not None
        or "localhost" in q
        or is_gibberish(q)
    )


def detect_patterns(question: str) -> list[str]:
    q = question.lower()

    return [
        name
        for name, terms in PATTERNS.items()
        if any(term in q for term in terms)
    ]


def strength(count: int, conversations: int) -> str:
    if conversations >= 5 or count >= 10:
        return "very_high"
    if conversations >= 3 or count >= 5:
        return "high"
    if count >= 3:
        return "medium"
    return "low"


def analyze_questions(user_id: str) -> dict:

    rows = (
        supabase.rpc(
            "get_user_questions",
            {"p_user_id": user_id}
        )
        .execute()
        .data or []
    )

    all_conversations = {
        row["conversation_id"] for row in rows
    }

    valid = []

    for row in rows:
        question = (row.get("question") or "").strip()

        if is_noise(question):
            continue

        valid.append({
            "conversation_id": row["conversation_id"],
            "question": question,
            "normalized": normalize(question),
            "created_at": row["created_at"],
            "patterns": detect_patterns(question),
        })

    counts = Counter()
    conversations = defaultdict(set)

    for row in valid:
        for pattern in row["patterns"]:
            counts[pattern] += 1
            conversations[pattern].add(row["conversation_id"])

    interests = [
        {
            "topic": topic,
            "question_count": count,
            "conversation_count": len(conversations[topic]),
            "strength": strength(
                count,
                len(conversations[topic])
            ),
        }
        for topic, count in counts.most_common()
    ]

    groups = defaultdict(list)

    for row in valid:
        groups[row["normalized"]].append(row)

    repeated = []

    for question, items in groups.items():
        if len(items) < 2:
            continue

        repeated.append({
            "question": question,
            "count": len(items),
            "conversation_count": len({
                x["conversation_id"] for x in items
            }),
            "first_seen": items[0]["created_at"],
            "last_seen": items[-1]["created_at"],
        })

    repeated.sort(
        key=lambda x: (
            x["conversation_count"],
            x["count"]
        ),
        reverse=True
    )

    meaningful_conversations = {
        row["conversation_id"] for row in valid
    }

    return {
        "user_id": user_id,
        "statistics": {
            "total_user_questions": len(rows),
            "meaningful_questions": len(valid),
            "ignored_noise": len(rows) - len(valid),
            "total_conversations": len(all_conversations),
            "meaningful_conversations": len(meaningful_conversations),
        },
        "interests": interests,
        "repeated_questions": repeated[:20],
        "latest_question_at": (
            valid[-1]["created_at"] if valid else None
        ),
    }
# Testing the module

if __name__ == "__main__":
    USER_ID = "21005862-7337-4deb-b7e7-1ad3f7b0d11b"

    result = analyze_questions(USER_ID)
    save_account_intelligence(result)

    summary = generate_account_summary(result)

    print(json.dumps(result, indent=2, default=str))
    print("\nACCOUNT AI SUMMARY:\n")
    print(summary)