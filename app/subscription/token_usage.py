from contextvars import ContextVar
from threading import Lock

_request_tokens = ContextVar("request_tokens", default=0)

# Token counting is keyed by session_id.
#
# A single global collector was not safe under concurrency: simultaneous
# requests overwrote each other's bucket and cross-billed, with one request
# reporting 3775 tokens while three siblings reported ~950.
#
# Streaming runs parts of the pipeline in a ThreadPoolExecutor. Note that
# ThreadPoolExecutor.submit() does NOT propagate contextvars (unlike
# asyncio.to_thread), so a worker thread sees _current_session as None and its
# add_tokens() calls fall through to the thread-local default counter. That is
# harmless today because the pooled retrieval helpers (structured/document)
# do not call the LLM; if that ever changes, wrap the worker body in
# contextvars.copy_context().run(...) so the session id reaches it.
#
# The mutable counter lives in a shared dict keyed by session. The ContextVar
# carries only the current session id, and is isolated per asyncio task, which
# is what keeps concurrent HTTP requests from cross-billing each other.
_collectors: dict[str, dict] = {}
_collectors_lock = Lock()
_current_session: ContextVar[str | None] = ContextVar(
    "token_session", default=None
)


def use_request_collector(session_id: str) -> dict:
    """Start (or join) the token bucket for this session.

    Ref-counted so two concurrent requests on the same session do not
    release each other's bucket.
    """
    sid = str(session_id)
    with _collectors_lock:
        entry = _collectors.get(sid)
        if entry is None:
            entry = {"tokens": 0, "refs": 0}
            _collectors[sid] = entry
        entry["refs"] += 1
    _current_session.set(sid)
    return entry


def release_request_collector() -> None:
    sid = _current_session.get()
    _current_session.set(None)
    if sid is None:
        return
    with _collectors_lock:
        entry = _collectors.get(sid)
        if entry is None:
            return
        entry["refs"] -= 1
        if entry["refs"] <= 0:
            _collectors.pop(sid, None)


def _active_entry() -> dict | None:
    sid = _current_session.get()
    if sid is None:
        return None
    return _collectors.get(sid)


def reset_tokens() -> None:
    entry = _active_entry()
    if entry is not None:
        with _collectors_lock:
            entry["tokens"] = 0
        return
    _request_tokens.set(0)


def add_tokens(tokens: int) -> None:
    n = int(tokens or 0)
    entry = _active_entry()
    if entry is not None:
        with _collectors_lock:
            entry["tokens"] += n
        return
    _request_tokens.set(_request_tokens.get() + n)


def get_tokens() -> int:
    entry = _active_entry()
    if entry is not None:
        with _collectors_lock:
            return entry["tokens"]
    return _request_tokens.get()