"""FastAPI deployment (Part 3, Task 11).

Endpoints
    POST /ask            -- one grounded crew turn, with input guardrails
    POST /add-document   -- add a KB document and re-index both collections
    WS   /ws/chat        -- real-time multi-turn chat with session memory
    GET  /health         -- liveness

The WebSocket endpoint catches ``WebSocketDisconnect`` so a client dropping
mid-conversation never takes the server down for other clients.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from nykaa import config
from nykaa.guardrails import apply_input_guardrails
from nykaa.logging_utils import LOGGER, Timer, new_trace_id
from nykaa.memory import STORE
from nykaa.rag import build_all_collections, get_client, grounded_answer, retrieve
from nykaa.schemas import SupportResponse


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Build both collections on boot so the first request is not slow."""
    build_all_collections(reset=False)
    yield


app = FastAPI(
    title="Nykaa Domain Support Agent",
    description=(
        "Grounded policy + order-status support agent. Runs entirely under "
        "MOCK_LLM: no API keys, no network."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- #
# Request / response models
# --------------------------------------------------------------------------- #


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=config.MAX_QUERY_CHARS)
    record_id: str | None = None
    session_id: str | None = None


class AskResponse(BaseModel):
    trace_id: str
    allowed: bool
    reject_reason: str | None = None
    response: SupportResponse | None = None
    duration_ms: float


class AddDocumentRequest(BaseModel):
    doc_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")
    title: str = Field(min_length=1, max_length=200)
    body: str = Field(min_length=1, max_length=20_000)


class AddDocumentResponse(BaseModel):
    trace_id: str
    doc_id: str
    ok: bool
    message: str
    collection_counts: dict[str, int]
    duration_ms: float


# --------------------------------------------------------------------------- #
# POST /ask
# --------------------------------------------------------------------------- #


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    """One grounded crew turn. Input guardrails run before anything else."""
    trace_id = new_trace_id()
    with Timer() as timer:
        verdict = apply_input_guardrails(req.query)

        if not verdict.allowed:
            LOGGER.log(
                trace_id=trace_id,
                endpoint="/ask",
                request_text=req.query,
                status="rejected_guardrail",
                duration_ms=timer.elapsed_ms,
                session_id=req.session_id,
                extra={"reject_reason": verdict.reject_reason},
            )
            return AskResponse(
                trace_id=trace_id,
                allowed=False,
                reject_reason=verdict.reject_reason,
                duration_ms=round(timer.elapsed_ms, 3),
            )

        # The crew is fed the MASKED query, matching what is logged.
        from nykaa.crew import run_crew

        run = run_crew(verdict.masked_query, req.record_id)
        from nykaa.schemas import parse_response

        parsed = parse_response(run.result)

        LOGGER.log(
            trace_id=trace_id,
            endpoint="/ask",
            request_text=req.query,
            status="ok",
            duration_ms=timer.elapsed_ms,
            session_id=req.session_id,
            extra={
                "record_id": req.record_id,
                "grounded": parsed.grounded,
                "needs_human": parsed.needs_human,
                "tools_invoked": run.tools_invoked,
            },
        )
        return AskResponse(
            trace_id=trace_id,
            allowed=True,
            response=parsed,
            duration_ms=round(timer.elapsed_ms, 3),
        )


# --------------------------------------------------------------------------- #
# POST /add-document
# --------------------------------------------------------------------------- #


@app.post("/add-document", response_model=AddDocumentResponse)
def add_document(req: AddDocumentRequest) -> AddDocumentResponse:
    """Append a KB document and re-index both chunking collections."""
    trace_id = new_trace_id()
    with Timer() as timer:
        path = config.KB_DIR / f"{req.doc_id}.md"
        if path.exists():
            raise HTTPException(
                status_code=409, detail=f"document {req.doc_id!r} already exists"
            )

        path.write_text(f"# {req.title}\n\n{req.body.strip()}\n", encoding="utf-8")
        built = build_all_collections(reset=True)
        counts = {
            config.COLLECTION_FIXED: built["fixed_overlap"].count(),
            config.COLLECTION_SENTENCE: built["sentence"].count(),
        }

        LOGGER.log(
            trace_id=trace_id,
            endpoint="/add-document",
            request_text=f"{req.doc_id}: {req.title}",
            status="ok",
            duration_ms=timer.elapsed_ms,
            extra={"doc_id": req.doc_id, "counts": counts},
        )
        return AddDocumentResponse(
            trace_id=trace_id,
            doc_id=req.doc_id,
            ok=True,
            message="document added and both collections re-indexed",
            collection_counts=counts,
            duration_ms=round(timer.elapsed_ms, 3),
        )


# --------------------------------------------------------------------------- #
# GET /health
# --------------------------------------------------------------------------- #


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "mock_llm": config.MOCK_LLM,
        "collections": [config.COLLECTION_FIXED, config.COLLECTION_SENTENCE],
        "similarity_threshold": config.SIMILARITY_THRESHOLD,
        "escalation_threshold": config.DEFAULT_ESCALATION_THRESHOLD,
    }


# --------------------------------------------------------------------------- #
# WS /ws/chat
# --------------------------------------------------------------------------- #


@app.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket) -> None:
    """Real-time multi-turn chat.

    A client disconnecting mid-conversation is caught and handled: the handler
    returns cleanly, the socket is closed once, and the server keeps serving
    every other client.
    """
    await websocket.accept()
    session_id = websocket.query_params.get("session_id") or f"ws-{new_trace_id()}"
    turn = 0

    try:
        while True:
            try:
                payload = await websocket.receive_json()
            except WebSocketDisconnect:
                raise
            except Exception:
                # Malformed (non-JSON) frame: report it and keep the socket open
                # rather than attempting another blocking receive.
                await websocket.send_json({"error": "expected a JSON object"})
                continue

            query = str(payload.get("query", "")).strip()
            if not query:
                await websocket.send_json({"error": "empty query"})
                continue
            if query.lower() in {"quit", "exit", "bye"}:
                await websocket.send_json({"type": "closing", "session_id": session_id})
                break

            turn += 1
            trace_id = new_trace_id()
            with Timer() as timer:
                verdict = apply_input_guardrails(query)
                if not verdict.allowed:
                    LOGGER.log(
                        trace_id=trace_id,
                        endpoint="/ws/chat",
                        request_text=query,
                        status="rejected_guardrail",
                        duration_ms=timer.elapsed_ms,
                        session_id=session_id,
                        extra={"turn": turn, "reject_reason": verdict.reject_reason},
                    )
                    await websocket.send_json(
                        {
                            "trace_id": trace_id,
                            "turn": turn,
                            "allowed": False,
                            "reject_reason": verdict.reject_reason,
                        }
                    )
                    continue

                from nykaa.crew import run_crew
                from nykaa.schemas import parse_response

                # Session memory: prepend the previous customer turn so
                # elliptical follow-ups resolve, exactly as in Task 8.
                history = STORE.get(session_id)
                prior = None
                for m in reversed(history.messages):
                    if getattr(m, "type", "") == "human":
                        prior = str(m.content)
                        break
                effective = (
                    f"{prior} {verdict.masked_query}"
                    if prior and prior.lower() != verdict.masked_query.lower()
                    else verdict.masked_query
                )

                # NOTE: crewai 1.x refuses a synchronous kickoff() from inside a
                # running event loop ("Agent execution was invoked synchronously
                # from within a running event loop"), so the blocking crew is
                # dispatched to a worker thread instead.
                run = await run_in_threadpool(run_crew, effective)
                parsed = parse_response(run.result)

                history.add_user_message(query)
                history.add_ai_message(parsed.answer)

                LOGGER.log(
                    trace_id=trace_id,
                    endpoint="/ws/chat",
                    request_text=query,
                    status="ok",
                    duration_ms=timer.elapsed_ms,
                    session_id=session_id,
                    extra={
                        "turn": turn,
                        "memory_applied": prior is not None,
                        "tools_invoked": run.tools_invoked,
                    },
                )

            await websocket.send_json(
                {
                    "trace_id": trace_id,
                    "turn": turn,
                    "allowed": True,
                    "session_id": session_id,
                    "response": parsed.model_dump(),
                    "duration_ms": round(timer.elapsed_ms, 3),
                }
            )

    except WebSocketDisconnect:
        # Expected: the client went away mid-conversation. The server must keep
        # running for other clients -- so we return here rather than raising.
        LOGGER.log(
            trace_id=new_trace_id(),
            endpoint="/ws/chat",
            request_text="(client disconnected)",
            status="disconnect",
            duration_ms=0.0,
            session_id=session_id,
            extra={"turn": turn},
        )
        return
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def search(query: str, top_k: int = 3) -> list[dict[str, Any]]:
    """Small helper used by tests and scripts."""
    collection = get_client().get_or_create_collection(config.COLLECTION_SENTENCE)
    return [
        {"doc_id": h.doc_id, "similarity": h.similarity, "text": h.text}
        for h in retrieve(query, collection, top_k=top_k)
    ]


__all__ = ["app", "AskRequest", "AskResponse", "AddDocumentRequest", "grounded_answer"]
