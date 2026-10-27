import asyncio
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from run import initialize_acot, ask_acot, stream_acot, _slim_stream_response
from app.llm.client import LLMQueueTimeoutError
from app.listings.tools import ToolInputError, clean_profile
from app.logging_utils import (
    install_stdout_logger,
    set_request_tag,
)
from app.memory.conversation_memory import ConversationMemory
from app.intelligence.ai_analysis_engine import AIAnalysisEngine
from app.intelligence.account_summary_service import (
    clear_summary_cache,
    get_account_summary,
)
from app.subscription.token_usage import (
    reset_tokens,
    get_tokens,
    use_request_collector,
    release_request_collector,
)


# ============================================================
# FASTAPI APP
# ============================================================

logger = logging.getLogger("acot.api")

# How many full chats may run at once. Each chat still takes several model
# calls, and those stay capped by AICREDITS_MAX_CONCURRENCY. Past this limit
# the API answers "busy" in a couple of seconds instead of holding the
# client until the 110s stream abort.
_CHAT_LIMIT = int(os.getenv("ACOT_MAX_INFLIGHT_CHATS", "40"))
_CHAT_ADMIT_TIMEOUT = float(os.getenv("ACOT_CHAT_ADMIT_TIMEOUT_SECONDS", "2"))
_BUSY_MESSAGE = "ACOT is busy right now, please try again in a moment."
_chat_slots = None


def _chat_gate():
    global _chat_slots
    if _chat_slots is None:
        _chat_slots = asyncio.Semaphore(_CHAT_LIMIT)
    return _chat_slots


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Warm the ACOT pipeline at boot rather than on the first request.
    # Lazy init cost the first user ~11s; doing it here moves that cost to
    # startup and surfaces failures in the logs instead of as a user-facing
    # timeout. Failures are not fatal: get_api_components() retries lazily.
    started = time.perf_counter()
    try:
        # Route every print() in the pipeline through logging once, instead of
        # swapping global sys.stdout on each request.
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )
        install_stdout_logger(logging.getLogger("acot.pipeline"))

        get_api_components()
        print(
            f"[boot] ACOT pipeline warm in "
            f"{int((time.perf_counter() - started) * 1000)}ms",
            flush=True,
        )
    except Exception as exc:
        print(
            f"[boot] ACOT warmup failed ({exc}); will retry on first request",
            flush=True,
        )
    yield


app = FastAPI(
    title="ACOT AI Real Estate Intelligence API",
    description="Backend API for ACOT Dubai Real Estate Intelligence",
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST MODEL
# ============================================================

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    session_id: str = Field(default="default", min_length=1)
    # Optional, from the app: what is known about the user, for personalised answers. For example
    # {"budget_max_aed": 1500000, "goal": "invest", "preferred_areas": ["JVC"], "family_size": 4}.
    profile: dict | None = None


def _apply_profile(components, profile):
    """Merge an app-supplied user profile into the session. Invalid values are ignored, never fatal."""
    if not profile:
        return
    try:
        components.setdefault("profile", {}).update(clean_profile(profile))
    except ToolInputError:
        pass


# ============================================================
# GLOBAL ACOT COMPONENTS
# ============================================================

_api_components = None

# Session components are cached per conversation and were previously kept in a
# plain dict that was never pruned, so every conversation leaked a
# ConversationMemory for the life of the process. It is now an LRU bounded by
# count and idle time.
_api_sessions: "OrderedDict[str, dict]" = OrderedDict()
_api_sessions_lock = threading.Lock()
API_SESSION_MAX = int(os.getenv("ACOT_SESSION_MAX", "500"))
API_SESSION_TTL_SECONDS = int(os.getenv("ACOT_SESSION_TTL", "7200"))


# ============================================================
# INITIALIZE ACOT ONCE
# ============================================================

def get_api_components():

    global _api_components

    if _api_components is None:

        print("Initializing ACOT backend...")

        # No stdout redirect here: sys.stdout is already routed through the
        # logging stream installed at startup, so library chatter lands in the
        # log without mutating global state.
        _api_components = initialize_acot()

        # Reuse the same AICredits client used by QueryPlanner.
        _api_components["ai_analysis_engine"] = AIAnalysisEngine(
            llm=_api_components["query_planner"]._llm_client
        )

        print("ACOT backend initialized successfully.")

    return _api_components


# ============================================================
# SESSION MANAGEMENT
# ============================================================

def _evict_sessions_locked(now: float) -> None:

    """Drop idle sessions, then trim the oldest until within the cap."""

    expired = [
        sid
        for sid, (_, touched) in _api_sessions.items()
        if now - touched > API_SESSION_TTL_SECONDS
    ]
    for sid in expired:
        _api_sessions.pop(sid, None)

    while len(_api_sessions) > API_SESSION_MAX:
        _api_sessions.popitem(last=False)


def get_session_components(session_id: str):

    base_components = get_api_components()
    now = time.time()

    with _api_sessions_lock:
        entry = _api_sessions.get(session_id)

        if entry is not None:
            components, _ = entry
            # Refresh recency and idle timer.
            _api_sessions[session_id] = (components, now)
            _api_sessions.move_to_end(session_id)
            return components

        session_components = dict(base_components)

        session_components["conversation_memory"] = ConversationMemory(
            llm_client=base_components["query_planner"]._llm_client
        )

        _api_sessions[session_id] = (session_components, now)
        _api_sessions.move_to_end(session_id)
        _evict_sessions_locked(now)

        return session_components


def drop_session(session_id: str) -> bool:

    """Free a session's memory. Returns True if an entry was removed."""

    with _api_sessions_lock:
        entry = _api_sessions.pop(session_id, None)

    if entry is None:
        return False

    try:
        entry[0]["conversation_memory"].clear()
    except Exception:
        pass

    return True


def session_cache_stats() -> dict:

    now = time.time()
    with _api_sessions_lock:
        ages = [now - touched for _, touched in _api_sessions.values()]

    return {
        "sessions": len(_api_sessions),
        "max": API_SESSION_MAX,
        "ttl_seconds": API_SESSION_TTL_SECONDS,
        "oldest_idle_seconds": int(max(ages)) if ages else 0,
    }


# ============================================================
# AI ANALYSIS SUMMARY
# ============================================================

def _normalize_analysis_data(data):
    """Normalize retriever project fields for AIAnalysisEngine scoring."""
    if not isinstance(data, dict):
        return {}

    normalized = dict(data)

    projects = []
    for project in data.get("projects", []) or []:
        if not isinstance(project, dict):
            continue

        item = dict(project)

        # AIAnalysisEngine expects developer_name, while the retriever may
        # return developer. Preserve the original field as well.
        if item.get("developer_name") is None and item.get("developer") is not None:
            item["developer_name"] = item.get("developer")

        # AIAnalysisEngine expects bedroom_min / bedroom_max.
        bedrooms = item.get("bedrooms")
        if isinstance(bedrooms, dict):
            if item.get("bedroom_min") is None:
                item["bedroom_min"] = bedrooms.get("min")
            if item.get("bedroom_max") is None:
                item["bedroom_max"] = bedrooms.get("max")

        projects.append(item)

    normalized["projects"] = projects
    return normalized


async def build_ai_summary(question, response, ai_analysis_engine):
    """
    Run the ACOT AI Analysis Engine for analytical queries.

    The engine is used for comparison, ranking, analysis, and investment
    queries. Documents are always initialized locally so a missing document
    result cannot cause a NameError.
    """

    data = response.get("data", {})

    if not isinstance(data, dict):
        data = {}

    # ---------------------------------------------------------
    # Normalize project fields for AIAnalysisEngine
    # ---------------------------------------------------------
    normalized_projects = []

    for project in data.get("projects", []) or []:
        if not isinstance(project, dict):
            continue

        project = dict(project)

        # Your output format may use "developer".
        # AIAnalysisEngine also understands "developer_name".
        if (
            project.get("developer_name") is None
            and project.get("developer") is not None
        ):
            project["developer_name"] = project["developer"]

        # Your output uses:
        # bedrooms: {"min": 3, "max": 4, "label": "3-4"}
        # AIAnalysisEngine scoring also checks bedroom_min/max.
        bedrooms = project.get("bedrooms")

        if isinstance(bedrooms, dict):
            if project.get("bedroom_min") is None:
                project["bedroom_min"] = bedrooms.get("min")

            if project.get("bedroom_max") is None:
                project["bedroom_max"] = bedrooms.get("max")

        normalized_projects.append(project)

    data["projects"] = normalized_projects

    # ---------------------------------------------------------
    # Documents -- ALWAYS initialize this locally
    # ---------------------------------------------------------
    documents = data.get("documents", [])

    if not isinstance(documents, list):
        documents = []

    # ---------------------------------------------------------
    # Determine question type
    # ---------------------------------------------------------
    question_type = (
        response.get("question_type")
        or "analysis"
    )

    supported_types = {
        "comparison",
        "ranking",
        "analysis",
        "investment",
    }

    # Simple search/information queries do not need the
    # AI analysis engine.
    if question_type not in supported_types:
        return {
            "data": data,
            "data_description": {},
            "acot_recommendation": {},
            "acot_score": None,
        }

    # ---------------------------------------------------------
    # Run the actual ACOT AI Analysis Engine
    # ---------------------------------------------------------
    try:
        analysis = await ai_analysis_engine.analyze(
            question=question,
            structured_summary=data,
            documents=documents,
            question_type=question_type,
        )

        if not isinstance(analysis, dict):
            return {
                "data": data,
                "data_description": {},
                "acot_recommendation": {},
                "acot_score": None,
            }

        return analysis

    except Exception as exc:
        print(
            f"AI Analysis Engine error: {exc}"
        )

        # Keep the API alive even if LLM analysis fails.
        return {
            "data": data,
            "data_description": {},
            "acot_recommendation": {
                "summary": (
                    "AI analysis could not be generated. "
                    "The retrieved ACOT data is still available."
                ),
                "positive_factors": [],
                "considerations": [
                    str(exc)
                ],
                "ai_solution": (
                    "Review the retrieved structured "
                    "evidence directly."
                ),
                "confidence": "Low",
            },
            "acot_score": None,
        }
# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "status": "ok",
        "service": "ACOT",
        "message": "ACOT FastAPI backend is running.",
        "docs": "/docs"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
@app.get("/api/healths")
def health_check():

    return {
        "status": "ok",
        "service": "ACOT",
        "message": "ACOT FastAPI backend is running."
    }


# ============================================================
# ASK ACOT
# ============================================================

@app.post("/api/ask")
async def ask_endpoint(request: AskRequest):

    question = request.question.strip()

    # --------------------------------------------------------
    # Validate question
    # --------------------------------------------------------

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    try:
        await asyncio.wait_for(
            _chat_gate().acquire(),
            timeout=_CHAT_ADMIT_TIMEOUT,
        )
    except TimeoutError:
        raise HTTPException(
            status_code=503,
            detail=_BUSY_MESSAGE,
        )

    use_request_collector(request.session_id)
    reset_tokens()

    try:

        endpoint_started = time.perf_counter()

        # ----------------------------------------------------
        # Get session-specific ACOT components
        # ----------------------------------------------------

        components = get_session_components(
            request.session_id
        )
        _apply_profile(components, request.profile)

        # ----------------------------------------------------
        # Run existing ACOT pipeline
        # ----------------------------------------------------

        set_request_tag(request.session_id)

        response = await ask_acot(
            question,
            components
        )

        # ----------------------------------------------------
        # Add short ACOT score + recommendation
        # ----------------------------------------------------

        summary_started = time.perf_counter()
        # Only chatbot (ask) tokens count against the subscription.
        # Snapshot before build_ai_summary so summary LLM usage is excluded.
        chat_tokens = get_tokens()
        response["ai_summary"] = await build_ai_summary(
            question=question,
            response=response,
            ai_analysis_engine=components["ai_analysis_engine"],
        )

        timing = response.setdefault("meta", {}).setdefault("timing_ms", {})
        timing["ai_summary"] = int(
            round((time.perf_counter() - summary_started) * 1000)
        )
        timing["total"] = int(
            round((time.perf_counter() - endpoint_started) * 1000)
        )
        response["tokens_used"] = chat_tokens
        response.setdefault("meta", {})["tokens_used"] = chat_tokens
        print(f"ACOT timing_ms: {timing}")

        # ----------------------------------------------------
        # Return complete ACOT response
        # ----------------------------------------------------

        release_request_collector()
        return response

    except LLMQueueTimeoutError as exc:

        release_request_collector()
        raise HTTPException(
            status_code=503,
            detail=str(exc)
        )

    except Exception as exc:

        release_request_collector()
        raise HTTPException(
            status_code=500,
            detail=f"ACOT processing error: {str(exc)}"
        )

    finally:
        _chat_gate().release()

def _sse_data(payload):
    body = json.dumps(
        payload,
        ensure_ascii=False,
        default=str,
    )
    return f"data: {body}\n\n"


@app.post("/api/ask/stream")
async def ask_stream_endpoint(request: AskRequest):

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    async def events():
        try:
            await asyncio.wait_for(
                _chat_gate().acquire(),
                timeout=_CHAT_ADMIT_TIMEOUT,
            )
        except TimeoutError:
            yield _sse_data({
                "type": "error",
                "message": _BUSY_MESSAGE,
            })
            return

        try:
            use_request_collector(request.session_id)
            reset_tokens()
            components = get_session_components(
                request.session_id
            )
            _apply_profile(components, request.profile)

            chat_tokens = 0

            async for event_name, data in stream_acot(
                question,
                components
            ):
                if event_name == "meta" and isinstance(data, dict):
                    data = dict(data)
                    data["session_id"] = request.session_id

                if event_name == "done" and isinstance(data, dict):
                    data = dict(data)
                    # Only chatbot (ask) tokens count against the subscription.
                    # Snapshot before build_ai_summary so summary LLM usage is excluded.
                    chat_tokens = get_tokens()
                    # Send the answer first. Listing and search questions get an
                    # empty summary with no model call. Comparison, ranking,
                    # analysis, and investment still wait for the score, because
                    # that score is part of the saved answer.
                    question_type = data.get("question_type") or ""
                    if question_type not in {
                        "comparison",
                        "ranking",
                        "analysis",
                        "investment",
                    }:
                        data["ai_summary"] = {
                            "data": data.get("data") if isinstance(data.get("data"), dict) else {},
                            "data_description": {},
                            "acot_recommendation": {},
                            "acot_score": None,
                        }
                    else:
                        data["ai_summary"] = await build_ai_summary(
                            question=question,
                            response=data,
                            ai_analysis_engine=components["ai_analysis_engine"],
                        )
                    data = _slim_stream_response(data)
                    yield _sse_data({
                        "type": "done",
                        "response": data,
                    })
                    yield _sse_data({
                        "type": "usage",
                        "tokens_used": chat_tokens,
                    })
                    continue

                if isinstance(data, dict) and data.get("type"):
                    yield _sse_data(data)
                    continue

                yield _sse_data({
                    "type": event_name,
                    **(data if isinstance(data, dict) else {"text": data}),
                })

        except LLMQueueTimeoutError as exc:
            yield _sse_data(
                {
                    "type": "error",
                    "message": str(exc),
                },
            )
        except Exception as exc:
            error_tokens = chat_tokens if chat_tokens else get_tokens()
            if error_tokens > 0:
                yield _sse_data({
                    "type": "usage",
                    "tokens_used": error_tokens,
                })
            yield _sse_data(
                {
                    "type": "error",
                    "message": (
                        f"ACOT processing error: {str(exc)}"
                    )
                },
            )
        finally:
            release_request_collector()
            _chat_gate().release()

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================
# ACCOUNT AI SUMMARY
# ============================================================

@app.get("/api/account/summary/{user_id}")
def account_summary(
    user_id: str,
    refresh: bool = False
):

    user = user_id.strip()

    if not user:

        raise HTTPException(
            status_code=400,
            detail="user_id is required."
        )

    try:

        return get_account_summary(user, refresh=refresh)

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Account summary error: {str(exc)}"
        )


@app.post("/api/account/summary/refresh/{user_id}")
def account_summary_refresh(user_id: str):

    clear_summary_cache()

    return account_summary(user_id, refresh=True)


# ============================================================
# CLEAR CONVERSATION SESSION
# ============================================================

@app.delete("/api/session/{session_id}")
def clear_session(session_id: str):

    # Previously this called get_session_components(), which *created* an
    # entry when none existed and never removed it. Closing a conversation
    # therefore leaked memory instead of freeing it.
    removed = drop_session(session_id)

    return {
        "status": "success",
        "session_id": session_id,
        "removed": removed,
        "message": "Conversation memory cleared.",
    }


@app.get("/api/session/stats")
def session_stats():

    return session_cache_stats()