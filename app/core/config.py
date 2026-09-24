import os
from dotenv import load_dotenv

load_dotenv()

AICREDITS_API_KEY = os.getenv("AICREDITS_API_KEY") or os.getenv("OPENROUTER_API_KEY")
AICREDITS_BASE_URL = (
    os.getenv("AICREDITS_BASE_URL") or "https://api.aicredits.in/v1"
).rstrip("/")