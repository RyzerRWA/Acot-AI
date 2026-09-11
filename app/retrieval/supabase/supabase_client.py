import os
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client, Client


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(ENV_FILE)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SECRET_KEY")


if not SUPABASE_URL:
    raise ValueError(
        f"SUPABASE_URL is missing. Checked: {ENV_FILE}"
    )

if not SUPABASE_KEY:
    raise ValueError(
        f"SUPABASE_SECRET_KEY is missing. Checked: {ENV_FILE}"
    )


# ============================================================
# CREATE SUPABASE CLIENT
# ============================================================

supabase: Client = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)