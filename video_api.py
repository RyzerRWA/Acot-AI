"""
ACOT video generation service — separate from the chat API.

Run (from Acot-AI- folder):
  uvicorn video_api:app --host 0.0.0.0 --port 8090

Nest should call VIDEO_API_URL=http://127.0.0.1:8090
Chat stays on :8080 (api:app). Do not mount this router into api.py.
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

from app.Viedo_generation.router1 import router as video_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("acot.video_api")

app = FastAPI(
    title="ACOT Video Generation",
    version="0.1.0",
    docs_url="/docs",
)

# Nest is the only caller in production; CORS is for local debugging only.
_origins = [
    o.strip()
    for o in (os.getenv("VIDEO_CORS_ORIGINS") or "http://localhost:4000,http://127.0.0.1:4000").split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(video_router)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "acot-video",
        "docs": "/docs",
        "health": "/api/video/health",
    }


logger.info("ACOT video API loaded (port 8090 expected)")
