import json

from app.database.postgres import execute, scalar


def save_account_intelligence(analysis: dict) -> None:
    stats = analysis["statistics"]

    execute(
        "select public.save_account_intelligence(%s, %s::jsonb, %s, %s)",
        (
            analysis["user_id"],
            json.dumps(analysis),
            stats["meaningful_questions"],
            stats["meaningful_conversations"],
        ),
    )


def get_account_intelligence(user_id: str) -> dict | None:
    result = scalar("select public.get_account_intelligence(%s)", (user_id,))

    return result or None
