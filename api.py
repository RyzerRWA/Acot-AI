import io
from contextlib import redirect_stdout, redirect_stderr

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from run import initialize_acot, ask_acot
from app.memory.conversation_memory import ConversationMemory
from app.intelligence.ai_analysis_engine import AIAnalysisEngine


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="ACOT AI Real Estate Intelligence API",
    description="Backend API for ACOT Dubai Real Estate Intelligence",
    version="1.0.0",
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


# ============================================================
# GLOBAL ACOT COMPONENTS
# ============================================================

_api_components = None
_api_sessions = {}


# ============================================================
# INITIALIZE ACOT ONCE
# ============================================================

def get_api_components():

    global _api_components

    if _api_components is None:

        print("Initializing ACOT backend...")

        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            _api_components = initialize_acot()

        # Reuse the same OpenRouter-backed LLM client used by QueryPlanner.
        _api_components["ai_analysis_engine"] = AIAnalysisEngine(
            llm=_api_components["query_planner"]._llm_client
        )

        print("ACOT backend initialized successfully.")

    return _api_components


# ============================================================
# SESSION MANAGEMENT
# ============================================================

def get_session_components(session_id: str):

    base_components = get_api_components()

    if session_id not in _api_sessions:

        session_components = dict(base_components)

        session_components["conversation_memory"] = ConversationMemory(
            llm_client=base_components["query_planner"]._llm_client
        )

        _api_sessions[session_id] = session_components

    return _api_sessions[session_id]


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


def build_ai_summary(question, response, ai_analysis_engine):
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

        # Your Supabase/output format may use "developer".
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
        analysis = ai_analysis_engine.analyze(
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

@app.get("/api/health")
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
def ask_endpoint(request: AskRequest):

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

        # ----------------------------------------------------
        # Get session-specific ACOT components
        # ----------------------------------------------------

        components = get_session_components(
            request.session_id
        )

        # ----------------------------------------------------
        # Run existing ACOT pipeline
        # ----------------------------------------------------

        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):

            response = ask_acot(
                question,
                components
            )

        # ----------------------------------------------------
        # Add short ACOT score + recommendation
        # ----------------------------------------------------

        response["ai_summary"] = build_ai_summary(
            question=question,
            response=response,
            ai_analysis_engine=components["ai_analysis_engine"],
        )

        # ----------------------------------------------------
        # Return complete ACOT response
        # ----------------------------------------------------

        return response

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"ACOT processing error: {str(exc)}"
        )


# ============================================================
# CLEAR CONVERSATION SESSION
# ============================================================

@app.delete("/api/session/{session_id}")
def clear_session(session_id: str):

    components = get_session_components(
        session_id
    )

    components["conversation_memory"].clear()

    return {
        "status": "success",
        "session_id": session_id,
        "message": "Conversation memory cleared."
    }