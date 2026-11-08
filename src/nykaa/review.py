"""Autogen review stage (Part 4, Task 14).

A 2-agent ``RoundRobinGroupChat`` team that reviews the crew's draft answer and
either approves it unchanged or revises it:

    Policy-Compliance-Reviewer  ->  Final-Editor (structured verdict)

The team takes the composer's draft **plus the original retrieved context**, so
the reviewers judge groundedness against the same evidence the crew had.

Verified API surface for autogen-agentchat 0.7.5
------------------------------------------------
* ``RoundRobinGroupChat(participants, name, description, termination_condition,
  max_turns, runtime, custom_message_types, emit_team_events)`` -- so the real
  parameter is ``max_turns``, **not** ``max_iterations``.
* ``MaxMessageTermination(max_messages, include_agent_event=False)``.
* ``AssistantAgent(..., output_content_type=<BaseModel>)`` gives the agent a
  Pydantic structured output.
* ``StructuredMessage`` is importable from ``autogen_agentchat.messages``.

Both binding requirements the brief warns about are honoured:

1. ``max_turns=2`` lets **both** agents speak (the alternative,
   ``MaxMessageTermination(3)``, is needed only if you count the initiating task
   message as message 1 -- ``MaxMessageTermination(2)`` would cut the second
   agent off).
2. Determining a structured output needs the field on the agent **and** the team
   constructed with
   ``custom_message_types=[StructuredMessage[VerdictModel]]`` -- omitting the
   second raises ``ValueError: Message type ... is not registered``.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Mapping, Sequence

from autogen_core.models import (
    ChatCompletionClient,
    CreateResult,
    LLMMessage,
    ModelInfo,
    RequestUsage,
)
from pydantic import BaseModel, Field

from nykaa import llm_backend
from nykaa.guardrails import check_groundedness

# --------------------------------------------------------------------------- #
# The verdict model
# --------------------------------------------------------------------------- #


class VerdictModel(BaseModel):
    """Structured output from the Final-Editor agent."""

    approved: bool = Field(
        description="True if the draft is approved unchanged, False if it was revised."
    )
    final_answer: str = Field(description="The answer that should reach the user.")
    reason: str = Field(description="Why the draft was approved or revised.")


# --------------------------------------------------------------------------- #
# Deterministic offline model client for Autogen
# --------------------------------------------------------------------------- #


class MockReviewClient(ChatCompletionClient):
    """Deterministic ``ChatCompletionClient`` for the review team.

    Mirrors :class:`nykaa.mock_llm.MockLLM`: no API key, no network. It receives
    the draft and the retrieved context (passed through by the runner) and
    applies the same groundedness rule a reviewer would.
    """

    def __init__(self) -> None:
        self.calls = 0
        self._usage = RequestUsage(prompt_tokens=0, completion_tokens=0)

    # ---- required abstract surface ---------------------------------------- #

    async def create(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Any] = (),
        tool_choice: Any = "auto",
        json_output: Any = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Any = None,
    ) -> CreateResult:
        self.calls += 1
        prompt_text = "\n".join(str(getattr(m, "content", "")) for m in messages)
        self._usage.prompt_tokens += len(prompt_text) // 4

        payload = _extract_payload(prompt_text)

        if json_output is not None or _wants_json(messages):
            verdict = decide_verdict(payload.get("draft", ""), payload.get("context", []))
            content: Any = verdict.model_dump_json()
        else:
            # The reviewer speaks first: a short compliance note.
            note = compliance_note(payload.get("draft", ""), payload.get("context", []))
            content = note

        self._usage.completion_tokens += len(str(content)) // 4
        return CreateResult(
            content=content,
            finish_reason="stop",
            usage=RequestUsage(prompt_tokens=0, completion_tokens=0),
            cached=False,
        )

    def create_stream(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Any] = (),
        tool_choice: Any = "auto",
        json_output: Any = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Any = None,
    ) -> AsyncGenerator[Any, None]:
        async def _gen():
            result = await self.create(
                messages,
                tools=tools,
                tool_choice=tool_choice,
                json_output=json_output,
                extra_create_args=extra_create_args,
                cancellation_token=cancellation_token,
            )
            yield result

        return _gen()

    async def close(self) -> None:
        return None

    def count_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Any] = ()) -> int:
        return sum(len(str(getattr(m, "content", ""))) for m in messages) // 4

    def remaining_tokens(
        self, messages: Sequence[LLMMessage], *, tools: Sequence[Any] = ()
    ) -> int:
        return 100_000 - self.count_tokens(messages, tools=tools)

    def actual_usage(self) -> RequestUsage:
        return self._usage

    def total_usage(self) -> RequestUsage:
        return self._usage

    @property
    def capabilities(self) -> Any:
        return {"vision": False, "function_calling": False, "json_output": True}

    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo(
            vision=False,
            function_calling=False,
            json_output=True,
            family="mock",
            name="mock-review-llm",
        )


# --------------------------------------------------------------------------- #
# Deterministic review logic
# --------------------------------------------------------------------------- #


def _wants_json(messages: Sequence[LLMMessage]) -> bool:
    text = " ".join(str(getattr(m, "content", "")) for m in messages).lower()
    return "json" in text and ("verdict" in text or "output_content_type" in text)


def _extract_payload(text: str) -> dict[str, Any]:
    """Pull the reviewer's evidence bundle out of the prompt."""
    payload: dict[str, Any] = {"draft": "", "context": []}
    marker = "REVIEW_PAYLOAD:"
    if marker in text:
        raw = text.split(marker, 1)[1].strip()
        # The JSON object is the first balanced object after the marker.
        try:
            obj, _ = json.JSONDecoder().raw_decode(raw[raw.index("{"):])
            if isinstance(obj, dict):
                payload.update(obj)
        except Exception:  # noqa: BLE001 - fall back to empty payload
            pass
    return payload


def compliance_note(draft: str, context: list[str]) -> str:
    """The Policy-Compliance-Reviewer's contribution."""
    verdict = check_groundedness(draft, context)
    if verdict.supported:
        return (
            "COMPLIANCE OK: every claim in the draft is supported by the retrieved "
            "context. Recommend approval without changes."
        )
    return (
        f"COMPLIANCE ISSUE: {len(verdict.unsupported_sentences)} sentence(s) are not "
        "supported by the retrieved context. Recommend the Final-Editor remove them. "
        f"Unsupported: {verdict.unsupported_sentences!r}"
    )


def decide_verdict(draft: str, context: list[str]) -> VerdictModel:
    """Approve the draft unchanged, or revise it by dropping ungrounded claims."""
    verdict = check_groundedness(draft, context)
    if verdict.supported:
        return VerdictModel(
            approved=True,
            final_answer=draft,
            reason=(
                "Every sentence is supported by the retrieved context "
                f"(token coverage {verdict.coverage:.2f}); approved unchanged."
            ),
        )

    kept: list[str] = []
    import re

    for sentence in re.split(r"(?<=[.!?])\s+", draft):
        if not sentence.strip():
            continue
        if sentence.strip() in verdict.unsupported_sentences:
            continue
        kept.append(sentence.strip())
    revised = " ".join(kept).strip()

    return VerdictModel(
        approved=False,
        final_answer=revised,
        reason=(
            f"Revised: removed {len(verdict.unsupported_sentences)} sentence(s) not "
            "supported by the retrieved context "
            f"(coverage {verdict.coverage:.2f})."
        ),
    )


# --------------------------------------------------------------------------- #
# The team
# --------------------------------------------------------------------------- #


@dataclass
class ReviewOutcome:
    draft: str
    verdict: VerdictModel
    transcript: list[dict[str, str]] = field(default_factory=list)
    client_calls: int = 0


def build_review_team(client: Any):
    """Construct the 2-agent Round Robin review team."""
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.messages import StructuredMessage
    from autogen_agentchat.teams import RoundRobinGroupChat

    reviewer = AssistantAgent(
        # NOTE: autogen-agentchat 0.7.5 requires `name` to be a valid Python
        # identifier, so the hyphenated display name lives in `description`.
        # "Policy-Compliance-Reviewer" as `name` raises:
        #   ValueError: The agent name must be a valid Python identifier.
        name="Policy_Compliance_Reviewer",
        description="Policy-Compliance-Reviewer",
        model_client=client,
        system_message=(
            "You check whether a draft support answer is fully supported by the "
            "retrieved policy context. You report compliance OK or the specific "
            "unsupported claims. You do not rewrite the answer."
        ),
    )

    editor = AssistantAgent(
        name="Final_Editor",
        description="Final-Editor",
        model_client=client,
        system_message=(
            "You produce the final verdict as JSON with fields approved, "
            "final_answer and reason. Approve the draft unchanged if it is fully "
            "grounded; otherwise return a revised final_answer with the "
            "unsupported claims removed. Output JSON only."
        ),
        output_content_type=VerdictModel,
    )

    return RoundRobinGroupChat(
        participants=[reviewer, editor],
        # max_turns=2 lets BOTH agents speak. (With MaxMessageTermination you
        # would need (3), because the initiating task message counts as 1.)
        max_turns=2,
        # Required in addition to output_content_type on the agent, or the run
        # raises: ValueError: Message type ... is not registered.
        custom_message_types=[StructuredMessage[VerdictModel]],
    )


async def _run(draft: str, context: list[str]) -> ReviewOutcome:
    from autogen_agentchat.base import TaskResult

    client = llm_backend.build_autogen_client()
    team = build_review_team(client)

    payload = json.dumps({"draft": draft, "context": context})
    task = (
        "Review the following draft answer against the retrieved context.\n"
        "Approve it unchanged if every claim is supported; otherwise revise it.\n\n"
        f"REVIEW_PAYLOAD: {payload}"
    )

    transcript: list[dict[str, str]] = []
    verdict: VerdictModel | None = None

    async for message in team.run_stream(task=task):
        if isinstance(message, TaskResult):
            break
        source = getattr(message, "source", "?")
        content = getattr(message, "content", "")
        if isinstance(content, VerdictModel):
            verdict = content
            rendered = content.model_dump_json()
        else:
            rendered = str(content)
        transcript.append({"source": source, "content": rendered})

    if verdict is None:
        verdict = decide_verdict(draft, context)

    return ReviewOutcome(
        draft=draft,
        verdict=verdict,
        transcript=transcript,
        # MockReviewClient counts calls; a real client does not expose that.
        client_calls=int(getattr(client, "calls", 0) or 0),
    )


def run_review(draft: str, context: list[str]) -> ReviewOutcome:
    """Synchronous entry point."""
    return asyncio.run(_run(draft, context))
