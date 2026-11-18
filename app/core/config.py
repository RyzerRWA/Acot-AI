import os
from dotenv import load_dotenv

load_dotenv()

AICREDITS_API_KEY = os.getenv("AICREDITS_API_KEY")
AICREDITS_BASE_URL = (
    os.getenv("AICREDITS_BASE_URL") or "https://api.aicredits.in/v1"
).rstrip("/")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_OPENAI_BASE_URL = (
    os.getenv("GEMINI_BASE_URL")
    or "https://generativelanguage.googleapis.com/v1beta/openai/"
).rstrip("/")

LLM_API_KEY = AICREDITS_API_KEY or GEMINI_API_KEY
LLM_BASE_URL = AICREDITS_BASE_URL if AICREDITS_API_KEY else GEMINI_OPENAI_BASE_URL


def llm_chat_extras(model: str) -> dict:
    """GPT-6 Luna reasons by default. Chat Completions tools only work with reasoning off."""
    name = (model or "").lower()
    if any(tag in name for tag in ("gpt-6", "luna", "gpt-5", "o3", "o4")):
        return {"reasoning_effort": "none"}
    return {}