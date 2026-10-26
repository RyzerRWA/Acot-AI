"""Route the pipeline's print() output through the logging system.

The retrieval pipeline reports progress with bare print() calls. The API
previously silenced them per request with

    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):

but contextlib.redirect_stdout swaps the *process-global* sys.stdout. With
concurrent requests one request's redirect captured another request's output,
and the restore clobbered whatever the other request had installed.

Installing this stream once at startup gives the same result without touching
global state per request: every print() is forwarded to a logger, and the
active request is identified from a ContextVar so concurrent lines stay
attributable.
"""

import sys
from contextvars import ContextVar
from typing import Optional

_request_tag: ContextVar[Optional[str]] = ContextVar(
    "acot_request_tag", default=None
)


def set_request_tag(tag: Optional[str]):
    """Tag subsequent output in this context with a request identifier."""
    _request_tag.set(tag)
    return tag


def get_request_tag() -> Optional[str]:
    return _request_tag.get()


class ContextLogStream:
    """Minimal writable stream that turns writes into log records.

    Implements just enough of the file protocol for print() and the
    contextlib redirectors: write(), flush(), and isatty().
    """

    def __init__(self, logger, level: int = 20):
        self._logger = logger
        self._level = level
        self._buffer = ""
        self._stream = sys.__stdout__

    def write(self, data: str) -> int:
        if not data:
            return 0

        # Keep the console copy so interactive CLI runs still show progress.
        try:
            self._stream.write(data)
        except Exception:
            pass

        self._buffer += data

        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._emit(line)

        return len(data)

    def _emit(self, line: str) -> None:
        text = line.strip()
        if not text:
            return
        tag = _request_tag.get()
        if tag:
            self._logger.log(self._level, "[%s] %s", tag, text)
        else:
            self._logger.log(self._level, "%s", text)

    def flush(self) -> None:
        if self._buffer.strip():
            self._emit(self._buffer)
        self._buffer = ""
        try:
            self._stream.flush()
        except Exception:
            pass

    def isatty(self) -> bool:
        try:
            return self._stream.isatty()
        except Exception:
            return False

    def writable(self) -> bool:
        return True

    def fileno(self) -> int:
        return self._stream.fileno()

    @property
    def encoding(self) -> str:
        return getattr(self._stream, "encoding", "utf-8")


def install_stdout_logger(logger, level: int = 20) -> ContextLogStream:
    """Install the stream as sys.stdout/sys.stderr. Returns it for teardown."""
    stream = ContextLogStream(logger, level)
    sys.stdout = stream
    sys.stderr = stream
    return stream