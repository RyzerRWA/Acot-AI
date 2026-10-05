import os
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client, Client


ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env.chat")

url = os.getenv("CHAT_SUPABASE_URL")
key = os.getenv("CHAT_SUPABASE_SECRET_KEY")

if not url or not key:
    raise RuntimeError(
        "CHAT_SUPABASE_URL or CHAT_SUPABASE_SECRET_KEY is missing"
    )

supabase: Client = create_client(url, key)