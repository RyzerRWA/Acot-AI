# ACOT Backend — k6 Load Test Report (user range: 100)

**Test date:** 19 Oct 2026 · **k6:** v2.2.0 · **Backend target:** local uvicorn (`127.0.0.1:8000`, FastAPI)
**Scripts:** `performance/k6_load_test.js` (full), `performance/k6_diagnose.js` (attribution)
**Raw exports:** `performance/results/*.json`

---

## 1. Verdict

| Area | Verdict |
|---|---|
| Core ask pipeline correctness under load | ✅ **Good** — 0 backend 500s, 0 empty answers, 0 LLM-queue 503s during the valid run |
| Light endpoints (health/stats/session/summary) | ⚠️ **Depends on concurrent LLM traffic** — trivial endpoints exhibit 1–22s tails while real asks run |
| LLM provider capacity | ⚠️ **At the cap** — AICredits returned HTTP 429 under concurrency (~this is a real risk at scale) |
| Token cost | ℹ️ ~2.7–2.9k tokens per real question; ~178k tokens for 65 asks in the main run |
| Endpoint security | ❌ `/api/ask` is unauthenticated with CORS `*` — anyone can burn your AICredits key |

---

## 2. Test design

Two concurrent scenarios, **peak = 100 VUs**:

| Scenario | VUs | Work | Timeline |
|---|---|---|---|
| `ramp_light` | 0→20→45→**90**→0 | `GET /`, `GET /health`, `GET /api/session/stats`, `GET /api/account/summary/{id}`, `DELETE /api/session/{id}` | 210 s total, 90 s hold at 90 |
| `ask_probe` | 0→4→**10**→0 | Real `POST /api/ask` with rotating Dubai questions (structured / comparison / ranking / documents / small-talk), session reuse per VU | aligned, 10-VU hold |

The ask probe is deliberately capped at 10 VUs so token burn stays bounded while still sampling real LLM queueing/latency. Data came from a run where the server was verifiably healthy for the full window (see §6 for environment caveats).

---

## 3. Results — request volume & success

| Metric | Value |
|---|---|
| Total HTTP requests | **11,590** (avg 55.2 r/s over the 210 s window) |
| Iterations | 2,325 |
| Light-mix failures | 117 / 11,525 (**1.01%**) — *test-artifact, see note* |
| Ask-probe failures | **0 / 65** — no 500s, no LLM-queue 503s, every response had an answer |
| Checks | 16,254 with **99.28%** pass (all failures = the same tail-end DELETE artifact) |
| Tokens spent (chat pipeline) | **177,945** (~2.7k per ask) |

**Note on the 117 failures:** they occurred only at the tail when k6's original ramp-down cut connections abruptly. The server's own access log shows every one of those requests answered `200 OK`; the server only emitted asyncio accept errors (`WinError 64: network name no longer available`) from cancelled sockets. After adding a graceful ramp-down, this failure class disappeared.

---

## 4. Results — latency

### 4.1 `POST /api/ask` (end-to-end, 65 samples)

| p50 | p90 | p95 | p99 | max |
|---|---|---|---|---|
| 9.4 s | 29.1 s | 43.2 s | 51.0 s | 59.1 s |

Breakdown of time **inside the pipeline** (from `meta.timing_ms`):

| Stage | median | p90 | max | Notes |
|---|---|---|---|---|
| rewrite (conversation) | 1.0 s | 7.0 s | **41.7 s** | fully LLM-bound; max suggests queue+retry stacking |
| answer | ~0 (fast path) | 3.3 s | 30.0 s | 0 ms = deterministic/fast structured answers |
| ai_summary (analysis) | 0 | n/a | 14.2 s | only charged for comparison/ranking/analysis types |
| **total** | **5.5 s** | **20.8 s** | **58.9 s** | |

Sources of the long tail: (a) provider-side retries on 429, (b) LLM queueing behind the semaphore, (c) event-loop blocking described in §5.1, (d) sequential Supabase lookups inside comparison questions.

### 4.2 Light mix (90 VUs, per endpoint)

| Endpoint | p95 (clean dataset) | max observed | Comment |
|---|---|---|---|
| `GET /health` | 100–145 ms | **10.6 s** | max tail only during concurrent asks |
| `GET /api/session/stats` | 90–110 ms | 13.2 s | same pattern |
| `DELETE /api/session/{id}` | 0.3–1.6 s | 13.3 s | p95 fine when the loop isn't stalled |
| `GET /` | 0.6 s | **22.7 s** | |
| `GET /api/account/summary/{id}` | **2.3–6.7 s** | 14.5 s | cold-cache first call does Supabase work; see §5.3 |

The sub-200ms p95s prove the handlers themselves are fast; the 10–22s maxima are the **event loop being occupied by sibling ask requests** — the top code finding (§5.1).

---

## 5. Findings & recommended fixes

### 5.1 Event-loop blocking in the retriever — HIGH

`app/retrieval/hybrid/hybrid_retriever.py` (`async def retrieve`, ~lines 2431–2446):

- when **both** structured and documents are needed → correctly uses a `ThreadPoolExecutor`
- when **only one** is needed → `_run_structured()` / `_run_documents()` run **inline on the event loop**, blocking *every* other in-flight request for the full duration (observed 0.9–5.7 s per call; document retrieval also runs the sentence-transformers encode inline — the `Batches: 100%` work)

Fix (minimal):
```python
loop = asyncio.get_running_loop()
elif needs_structured:
    structured_result, timing_ms["structured_retrieval"] = await loop.run_in_executor(
        None, _run_structured
    )
elif needs_documents:
    documents, timing_ms["document_retrieval"] = await loop.run_in_executor(
        None, _run_documents
    )
```
Use the same pooled pattern as the both-needed branch. Expected effect: light-endpoint p95 stays ≤300 ms even under full LLM load, and the 9–13s maxima disappear.

### 5.2 LLM concurrency > provider cap — MEDIUM

`app/llm/client.py` lines 16–27: `AICREDITS_MAX_CONCURRENCY` defaults to **20**, while the comment itself documents the provider cap at ~5. During the test AICredits returned **`HTTP 429 Too Many Requests`** (handled by retry, attempt 2/2), plus one `Connection error` fallback to deterministic planning. 429s inflate latency tails via retry/backoff.

Recommendation: set `AICREDITS_MAX_CONCURRENCY=5` (or whatever the provider contract is), and make the 15s queue timeout (`AICREDITS_QUEUE_TIMEOUT_SECONDS`) an SLO-backed value — the observed rewrite max of 41.7s exceeds it, indicating retries retry *after* acquiring a slot.

### 5.3 Account summary — cold path is slow; warm path fine

`app/intelligence/account_summary_service.py` caches per user for **15 min TTL** (`CACHE_TTL_SECONDS`). The test user's history is actually empty (`available:false` envelope returned), so latency measured mostly the cached/no-history path (still ~2–6.7s p95 when the loop was stalled). Action items:
1. Verify with a user that has real chat history.
2. Consider pre-warming the cache at boot for the current account, consistent with the boot-warm approach used for the ACOT pipeline.

### 5.4 Unauthenticated `/api/ask` + CORS `*` — MEDIUM (security)

`api.py`: every ask consumes AICredits tokens / moves subscription quota (`use_request_collector`, `reset_tokens`) with **no auth**, and CORS allows all origins with `allow_credentials=True` (an invalid combination that browsers reject — either drop credentials or restrict origins). Since requests are cost-bearing, protect `/api/ask` with an auth middleware or per-IP/session rate limit before exposing it.

### 5.5 Single uvicorn worker

The Dockerfile runs `uvicorn api:app` without `--workers`. One event loop serializes all work (worse with §5.1). Once 5.1 is fixed, evaluate `--workers 2` for Cloud Run CPU settings, keeping the LLM semaphore shared correctly (currently process-local).

---

## 6. Environment caveats (honesty section)

- Server instances on this machine were repeatedly killed **right after each k6 run window** (uvicorn processes vanished; no crash logs in-process; detached processes too). Two extra verification runs were therefore partially contaminated (~40–49% connection failures attributable to a dying/replace-in-progress server, *not* to your code). Firmware/machine/OneDrive is actively interfering with files in this repo (project script files were deleted twice mid-session by OneDrive sync). For a stable “true capacity” number, rerun on Cloud Run:
  `k6 run -e BASE_URL=https://your-cloud-run-url performance/k6_load_test.js`
- Data in §3–4 comes from the run where the server was verifiably healthy throughout; conflicting numbers from degraded runs were excluded.

## 7. How to reproduce

```powershell
# Start backend
AIVenv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000

# Full 100-user test (~3.5 min)
k6 run performance/k6_load_test.js

# Per-endpoint sanity / DELETE diagnostics
k6 run performance/k6_diagnose.js
```
