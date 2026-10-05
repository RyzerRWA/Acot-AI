from app.database.chat_supabase_client import supabase


def save_account_intelligence(analysis: dict) -> None:
    stats = analysis["statistics"]

    supabase.rpc(
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
        supabase.rpc(
            "get_account_intelligence",
            {"p_user_id": user_id}
        )
        .execute()
        .data
    )

    return result or None