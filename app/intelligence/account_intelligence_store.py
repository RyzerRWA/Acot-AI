from app.database.postgres_client import get_postgres_client


db = get_postgres_client()


def save_account_intelligence(analysis: dict) -> None:
    stats = analysis["statistics"]

    db.rpc(
        "save_account_intelligence",
        {
            "p_user_id": analysis["user_id"],
            "p_intelligence": analysis,
            "p_question_count": stats["meaningful_questions"],
            "p_conversation_count": stats["meaningful_conversations"],
        },
    ).execute()


def get_account_intelligence(user_id: str) -> dict | None:
    result = (
        db.rpc(
            "get_account_intelligence",
            {"p_user_id": user_id}
        )
        .execute()
        .data
    )

    if isinstance(result, list):
        result = result[0] if result else None

    return result or None