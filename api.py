import io
from contextlib import redirect_stdout, redirect_stderr

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from run import initialize_acot, ask_acot
from app.memory.conversation_memory import ConversationMemory


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
# BUILD SHORT AI SUMMARY
# ============================================================

def build_ai_summary(response):

    data = response.get("data", {})

    investment_analysis = data.get(
        "investment_analysis"
    )

    # --------------------------------------------------------
    # Normal search / non-investment question
    # --------------------------------------------------------

    if not investment_analysis:

        return {
            "acot_score": None,
            "recommendation": (
                "Investment analysis is not applicable to this query."
            )
        }

    # --------------------------------------------------------
    # Get existing ACOT investment score
    # --------------------------------------------------------

    score = investment_analysis.get(
        "investment_score"
    )

    verdict = investment_analysis.get(
        "investment_verdict"
    )

    confidence = investment_analysis.get(
        "confidence"
    )

    # --------------------------------------------------------
    # Build short recommendation
    # --------------------------------------------------------

    if score == 0 and confidence == "Low":

        recommendation = (
            "Insufficient data for an investment decision."
        )

    elif verdict:

        recommendation = verdict

    else:

        recommendation = (
            "Investment recommendation is unavailable."
        )

    # --------------------------------------------------------
    # Final AI summary
    # --------------------------------------------------------

    return {
        "acot_score": score,
        "recommendation": recommendation
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
            response
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