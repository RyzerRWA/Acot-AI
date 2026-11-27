import os
from dotenv import load_dotenv

load_dotenv()

# All chat/LLM calls go straight to OpenAI (the AICredits gateway was dropped on 2026-11-27).
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = (os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
OPENAI_MODEL = os.getenv("OPENAI_MODEL") or "gpt-6-luna"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_OPENAI_BASE_URL = (
    os.getenv("GEMINI_BASE_URL")
    or "https://generativelanguage.googleapis.com/v1beta/openai/"
).rstrip("/")

LLM_API_KEY = OPENAI_API_KEY or GEMINI_API_KEY
LLM_BASE_URL = OPENAI_BASE_URL if OPENAI_API_KEY else GEMINI_OPENAI_BASE_URL


def llm_chat_extras(model: str) -> dict:
    """GPT-6 Luna reasons by default. Chat Completions tools only work with reasoning off."""
    name = (model or "").lower()
    if any(tag in name for tag in ("gpt-6", "luna", "gpt-5", "o3", "o4")):
        return {"reasoning_effort": "none"}
    return {}