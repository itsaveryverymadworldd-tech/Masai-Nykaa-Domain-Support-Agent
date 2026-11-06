"""Structured JSON-Lines logging (Part 3, Task 12).

Every request produces exactly **one** JSON object on its own line, with a trace
ID and timing information -- the ELK-style shape the brief asks for.

Critical requirement: the logged request text must **not** contain the raw,
unmasked value of any fixed-format PII field that the Task 10 guardrail masks.
The same :func:`nykaa.guardrails.mask_pii` is therefore applied to what is
written to disk as is applied to what the model sees, so a phone number or card
last-4 can never reach the log in the clear.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nykaa import config
from nykaa.guardrails import mask_pii

LOG_PATH = config.DATA_DIR / "requests.jsonl"


def new_trace_id() -> str:
    """Short, unique, log-friendly trace id."""
    return uuid.uuid4().hex[:16]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class RequestLogger:
    """Append-only JSON-Lines logger. Thread-safe for the ASGI threadpool."""

    def __init__(self, path: Path = LOG_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log(
        self,
        *,
        trace_id: str,
        endpoint: str,
        request_text: str,
        status: str,
        duration_ms: float,
        session_id: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Write one masked JSON-Lines entry and return it."""
        masked = mask_pii(request_text)
        entry: dict[str, Any] = {
            "timestamp": _utc_now(),
            "trace_id": trace_id,
            "endpoint": endpoint,
            "status": status,
            "duration_ms": round(duration_ms, 3),
            "session_id": session_id,
            # Only the masked form is ever persisted.
            "request_text": masked.masked,
            "pii_redactions": masked.redactions,
            "pii_masked": masked.changed,
        }
        if extra:
            entry["extra"] = extra

        line = json.dumps(entry, ensure_ascii=False)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        return entry

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def reset(self) -> None:
        if self.path.exists():
            self.path.unlink()


#: Process-wide logger used by the API.
LOGGER = RequestLogger()


@dataclass
class Timer:
    """Context manager measuring elapsed milliseconds."""

    started: float = field(default_factory=time.perf_counter)

    @property
    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.started) * 1000.0

    def __enter__(self) -> "Timer":
        self.started = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        return None
