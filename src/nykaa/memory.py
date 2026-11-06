"""Session memory for multi-turn conversations (Part 2, Task 8).

Uses LangChain's session-based memory as the brief prescribes:
``InMemoryChatMessageHistory`` for the store and ``RunnableWithMessageHistory``
to bind it to a runnable.

Note: ``RunnableWithMessageHistory`` emits a ``LangChainDeprecationWarning``
pointing at LangGraph's persistence layer. As the brief says, this is expected,
the class still functions correctly, and it does not need to be silenced.

Memory is in-process only -- it does not survive a restart, which the brief
explicitly accepts as sufficient.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory


@dataclass
class MemoryStore:
    """One ``InMemoryChatMessageHistory`` per session id."""

    sessions: dict[str, InMemoryChatMessageHistory] = field(default_factory=dict)

    def get(self, session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in self.sessions:
            self.sessions[session_id] = InMemoryChatMessageHistory()
        return self.sessions[session_id]

    def reset(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)

    def turns(self, session_id: str) -> int:
        return len(self.get(session_id).messages)

    def transcript(self, session_id: str) -> list[dict[str, str]]:
        return [
            {"role": getattr(m, "type", "?"), "content": str(m.content)}
            for m in self.get(session_id).messages
        ]


#: Process-wide store, so the FastAPI WebSocket endpoint (Part 3) can share it.
STORE = MemoryStore()


def history_as_text(session_id: str, max_turns: int = 6) -> str:
    """Render recent history as plain text for the crew prompt.

    Only the last ``max_turns`` messages are used, so a long conversation cannot
    grow the prompt without bound (Part 4's runtime budget cap).
    """
    msgs = STORE.get(session_id).messages[-max_turns:]
    if not msgs:
        return "(no earlier conversation)"
    lines = []
    for m in msgs:
        who = "Customer" if getattr(m, "type", "?") == "human" else "Agent"
        lines.append(f"{who}: {m.content}")
    return "\n".join(lines)


def build_conversational_runnable(run_fn):
    """Wrap the crew in a LangChain runnable backed by session memory.

    ``run_fn(question, history_text) -> str`` performs the actual work; this
    function only handles session bookkeeping, so the memory layer stays
    separable from the crew.
    """

    def _respond(payload: dict[str, Any]) -> str:
        question = payload["question"]
        history = payload.get("history") or []
        history_text = "\n".join(
            f"{'Customer' if getattr(m, 'type', '?') == 'human' else 'Agent'}: {m.content}"
            for m in history
        )
        return run_fn(question, history_text)

    base = RunnableLambda(_respond)
    return RunnableWithMessageHistory(
        base,
        lambda session_id: STORE.get(session_id),
        input_messages_key="question",
        history_messages_key="history",
    )


def ask(runnable, session_id: str, question: str) -> str:
    """One conversational turn against ``runnable`` under ``session_id``."""
    return runnable.invoke(
        {"question": question},
        config={"configurable": {"session_id": session_id}},
    )
