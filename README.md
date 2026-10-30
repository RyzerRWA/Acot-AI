# ACOT AI

ACOT AI is a Python service for grounded Dubai real-estate search, project and area comparisons, rental-listing discovery, and document-based answers. It combines structured PostgreSQL records and document retrieval with language-model responses. Answers should reflect the evidence retrieved for each request; they are not a substitute for professional financial or property advice.

## What it does

- **Project and community search**: retrieves project, developer, location, price, bedroom, size, status, and handover fields from PostgreSQL.
- **Grounded project comparisons**: resolves named projects independently and builds a side-by-side Markdown table for available price, size, bedroom, and handover details. Missing values are not invented; when the records do not support a clear preference, the answer says so.
- **Area comparisons**: handles comparisons between recognized Dubai communities using available project and rental aggregates, with a small set of explicitly general-knowledge area descriptions.
- **Rental search and market summaries**: answers rental questions using read-only tools over the listings data. The agent does not author SQL; tool arguments are validated and database queries are parameterized.
- **Document and brochure questions**: retrieves relevant document chunks with embedding similarity and supplies them as evidence to the answer generator.
- **Investment analysis**: computes deterministic summaries from available records, keeping dataset limitations distinct from financial recommendations.
- **Multi-turn chat**: maintains short-term conversation context for follow-ups such as “their prices,” “which of these,” and “compare the first and second.”
- **HTTP and streaming APIs**: exposes JSON chat, server-sent events (SSE), health checks, session management, and account-summary endpoints.

## Request flow

```mermaid
flowchart TD
    U[Question] --> API[FastAPI API or CLI]
    API --> MEM[Conversation memory and query rewrite]
    MEM --> ROUTER{Question route}
    ROUTER -->|Projects / facts| STRUCT[Structured PostgreSQL retrieval]
    ROUTER -->|Brochures / documents| VECTOR[Embedding and document search]
    ROUTER -->|Rentals| TOOLS[Validated read-only listing tools]
    ROUTER -->|Area comparison| AREA[Database-backed area statistics]
    STRUCT --> CTX[Grounded response context]
    VECTOR --> CTX
    TOOLS --> CTX
    AREA --> OUT[Response]
    CTX --> ANALYSIS[Deterministic analysis when applicable]
    ANALYSIS --> LLM[OpenAI-compatible AICredits model]
    LLM --> OUT
    OUT --> API
    OUT -. session history .-> MEM
```

Not every request uses every component. Simple comparisons and other structured answers can be rendered directly from database rows; questions requiring explanation use retrieved evidence in the language-model prompt.

## Repository layout

```text
.
├── api.py                         # FastAPI application and HTTP/SSE endpoints
├── run.py                         # Application initialization and chat orchestration
├── app/
│   ├── answers/                   # Structured answer formatting and area comparisons
│   ├── core/                      # Environment-backed configuration
│   ├── database/                  # PostgreSQL access and compatibility client
│   ├── graph/                     # LangGraph tools/agent workflow
│   ├── intelligence/              # Investment analysis and account summaries
│   ├── listings/                   # Rental routing, agent, tools, and session store
│   ├── llm/                        # AICredits client, limits, retries, and cache
│   ├── memory/                     # Short-term conversation context
│   ├── rag/                        # Answer chains, embeddings, ingestion, loaders
│   ├── retrieval/
│   │   ├── hybrid/                 # Planning, entity resolution, retrieval, context
│   │   └── postgres/               # Structured, listing, and vector retrievers
│   └── subscription/               # Token usage tracking
├── db/
│   ├── schema.sql                 # Core property, community, and document tables
│   └── chat_schema.sql            # Chat/account SQL objects
├── migrations/                    # Optional pgvector conversion and rollback SQL
├── data/raw/                      # Document catalog and source-data folders
├── docs/ARCHITECTURE.md           # Additional technical architecture notes
├── scripts/                       # Evaluation and safe deployment scripts
├── performance/                   # k6 load-test scripts and report
├── tests/                          # Pytest test suite
├── Docker/Dockerfile              # Container image for the API
├── cloudbuild.yaml                # Cloud Build and Cloud Run deployment
├── requirements.txt
└── pyproject.toml
```

## Main components

| Component | Location | Responsibility |
|---|---|---|
| Application orchestration | `run.py` | Initializes shared services, prepares chat turns, and coordinates retrieval and answer generation. |
| HTTP API | `api.py` | Provides FastAPI routes, session management, account summaries, and streaming. |
| Conversation memory | `app/memory/conversation_memory.py` | Tracks recent turns, entities, and candidate ordering for follow-up questions. |
| Query planning and retrieval | `app/retrieval/hybrid/` | Detects intent, resolves entities, chooses data sources, and prepares evidence. |
| PostgreSQL data access | `app/database/postgres.py` | Provides pooled PostgreSQL access using `DATABASE_URL`. |
| Vector search | `app/retrieval/postgres/pgvector_retriever.py` and `app/rag/embeddings/` | Embeds queries and searches community, project, or document vectors. |
| Rental agent and tools | `app/listings/` | Routes rental questions to an agent with validated, read-only database tools. |
| Structured and area answers | `app/answers/` | Formats database-grounded project comparisons and community comparisons. |
| Investment analysis | `app/intelligence/investment_analyzer.py` | Produces deterministic metrics from supplied structured records. |
| Language-model client | `app/llm/client.py` | Calls the configured AICredits model through an OpenAI-compatible API. |
| Document ingestion | `app/rag/ingestion/` | Fetches and extracts supported sources, cleans and chunks text, creates embeddings, and writes records. |

## Data and database

The application connects directly to PostgreSQL using `DATABASE_URL`. The main schema in [`db/schema.sql`](db/schema.sql) defines:

- `communities` and `sub_communities`: location names, hierarchy, metadata, and embeddings.
- `projects`: project, developer, location, price, size, bedroom range, handover, amenities, URLs, and embeddings.
- `document_chunks`: extracted document passages, source metadata, and embeddings.
- `match_communities`, `match_projects`, and `match_document_chunks`: database functions used for similarity retrieval.

The baseline schema stores embeddings as `real[]` and includes SQL cosine similarity functions. [`migrations/001_pgvector.sql`](migrations/001_pgvector.sql) is an optional migration to `vector(384)` with HNSW indexes; review its prerequisites and back up the database before applying it. A rollback script is provided alongside it.

Rental listing rows are queried by the listings tools and may be provisioned separately from `db/schema.sql`. The account/chat SQL in `db/chat_schema.sql` also depends on the existing account schema in the target database. Apply database scripts only to the intended environment.

## Requirements

- Python 3.10 or later (the Docker image uses Python 3.12).
- PostgreSQL with the ACOT tables populated; use `DATABASE_URL`.
- An AICredits API key and a model available to that key.
- Network access to the configured model endpoint and database.

The embedding model (`all-MiniLM-L6-v2`) may download model files the first time it is initialized. The PostgreSQL extension migration is optional when using the baseline array-based vector functions.

## Local setup

Create and activate a virtual environment, then install the pinned/project dependencies:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Create a root `.env` file with the required settings:

```dotenv
DATABASE_URL=postgresql://USER:PASSWORD@HOST:5432/DATABASE
AICREDITS_API_KEY=your_api_key
```

Do not commit `.env` or paste real credentials into source code, logs, or documentation.

### Configuration

| Variable | Required | Purpose |
|---|---:|---|
| `DATABASE_URL` | Yes | PostgreSQL connection string for project, vector, rental, and related data. |
| `AICREDITS_API_KEY` | Yes for model-backed flows | API key for the OpenAI-compatible AICredits endpoint. |
| `AICREDITS_BASE_URL` | No | AICredits API base URL; defaults to `https://api.aicredits.in/v1`. |
| `AICREDITS_MODEL` | No | Primary model; defaults to `openai/gpt-4o-mini`. |
| `AICREDITS_FALLBACK_MODELS` | No | Comma-separated fallback model names for the rental agent. |
| `AICREDITS_MAX_CONCURRENCY` | No | Concurrent LLM-call limit. |
| `AICREDITS_QUEUE_TIMEOUT_SECONDS` | No | Maximum wait for an available LLM-call slot. |
| `ACOT_MAX_INFLIGHT_CHATS` | No | Maximum admitted chat requests; defaults to `40`. |
| `ACOT_CHAT_ADMIT_TIMEOUT_SECONDS` | No | Time to wait for a chat slot before returning busy. |
| `ACOT_SESSION_MAX` / `ACOT_SESSION_TTL` | No | Maximum cached API sessions and idle session lifetime. |
| `DB_POOL_MAX` | No | PostgreSQL connection-pool maximum. |
| `PHOTOS_BASE_URL` | No | Base URL used when constructing rental listing photo links. |
| `ACOT_PERSIST_SESSIONS` | No | Enables persistent rental-agent session storage when set to a truthy value. |
| `ACOT_USE_AGENT` | No | Enables the optional agent path when set to a truthy value. |
| `SUMMARY_AICREDITS_API_KEY`, `SUMMARY_AICREDITS_BASE_URL`, `SUMMARY_AICREDITS_MODEL` | No | Separate configuration for the account-summary model integration. |

The code loads environment variables from `.env` where configured. Do not copy production credentials into local or example configuration files.

## Run the application

### FastAPI service

```bash
uvicorn api:app --host 0.0.0.0 --port 8080
```

Interactive API documentation is available at `/docs` while the service is running.

### Command-line chat

```bash
python run.py
```

Enter questions at the prompt. Type `exit` or `quit` to end the session.

## HTTP API

The chat request accepts a question and optional conversation/user context:

```json
{
  "question": "Compare Binghatti Etherea and Serenz",
  "session_id": "conversation-123",
  "profile": null,
  "user_name": null
}
```

| Method and route | Purpose |
|---|---|
| `GET /` | Service status and link to API docs. |
| `GET /health` | Health check. |
| `GET /api/healths` | Health-check alias. |
| `POST /api/ask` | Returns a complete grounded answer and structured response data. |
| `POST /api/ask/stream` | Streams answer events using server-sent events. |
| `GET /api/account/summary/{user_id}` | Returns an account summary; accepts optional `refresh` query parameter. |
| `POST /api/account/summary/refresh/{user_id}` | Clears summary cache and refreshes the account summary. |
| `DELETE /api/session/{session_id}` | Clears cached conversation state for a session. |
| `GET /api/session/stats` | Returns session-cache statistics. |

Chat responses include the answer, response type, structured data (projects, listings, comparisons, sources, and analysis as applicable), metadata, and token usage. Refer to the Pydantic request model and response-building code for the current field definitions.

## Example questions

- “Show me projects in Jumeirah Village Circle.”
- “What is the developer and handover date for Binghatti Etherea?”
- “Compare Binghatti Etherea and Serenz. Which has the lower starting price?”
- “Compare Dubai Marina and Business Bay.”
- “Find furnished two-bedroom rentals in Dubai Marina under AED 180,000 per year.”
- “Which of these projects have two-bedroom options?”
- “What amenities are listed in the project brochure?”
- “Summarize investment metrics available for this community.”

The answer depends on the rows and documents retrieved from the configured database. The sample names are query examples, not a guarantee that a particular record is present.

## Documents and embeddings

The ingestion and retrieval code is under `app/rag/`:

1. Detect and load supported input formats (including PDFs, JSON, and URLs).
2. Extract and clean text, then split it into chunks.
3. Generate 384-dimensional embeddings with `all-MiniLM-L6-v2`.
4. Store chunks and metadata in PostgreSQL for later similarity search.

The repository includes ingestion helpers and tests. Run ingestion only after configuring the target database and verifying the input catalog and destination.

## Deployment and performance

- `Docker/Dockerfile` builds the API container and starts `uvicorn api:app`.
- `cloudbuild.yaml` builds and pushes an image, then invokes `scripts/cloud-run-safe-deploy.sh` to deploy to Cloud Run and probe the new revision before promoting the stable image tag.
- `performance/` contains k6 scripts and a load-test report. Review the target URL, account limits, and database/model costs before running load tests.

Deployment requires correctly configured Cloud Build permissions, Artifact Registry, Cloud Run, and service environment variables. Never put secrets in the image or commit them to the repository.

## Tests and evaluation

Install the requirements, then run the test suite from the repository root:

```bash
python -m pytest tests
```

Run a focused test module:

```bash
python -m pytest tests/test_answer_routing.py
python -m pytest tests/test_listings_agent.py
```

The pytest suite covers answer routing, retrieval, embeddings, ingestion, listings tools, and account summaries. Some tests use mocks or fixtures; passing a unit test does not verify connectivity to a live PostgreSQL database or model provider.

The optional live listings evaluation script compares agent answers with database-derived expected results:

```bash
python scripts/eval_listings_agent.py
```

It requires the relevant database and model credentials and may incur provider usage. Review the script before running it against a production environment.

## Grounding and limitations

- Use retrieved ACOT records and document evidence; do not treat model prior knowledge as verified property data.
- Missing database fields remain unknown; they are not silently replaced with fabricated values or zeros.
- Keep project-level starting prices distinct from unit-specific prices and rental listing prices.
- A future handover date alone does not establish a project’s construction status.
- Comparisons summarize recorded fields; scores or relative rankings are not investment returns, financial advice, or guarantees of project quality.
- Results reflect the available database contents and may be incomplete or out of date.

## Additional documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — technical architecture notes.
- [`Question.txt`](Question.txt) — question/evaluation examples.
- [`performance/K6_REPORT.md`](performance/K6_REPORT.md) — recorded load-test report.

## License

This project is proprietary and confidential. All rights reserved by Ryzer RWA / ACOT AI.
