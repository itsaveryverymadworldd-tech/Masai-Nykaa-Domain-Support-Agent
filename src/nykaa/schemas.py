"""Structured output schema (Part 2, Task 9).

Every crew response must conform to :class:`SupportResponse`. The brief specifies
``response_format``; in crewai 1.x that constructor argument no longer exists on
``Task`` -- the equivalent field is ``output_pydantic`` (verified by probing
``Task.model_fields``). Both are used here: ``output_pydantic`` on the composer
task, plus explicit in-code validation of every response, because the brief
requires the response to be *validated in code* rather than merely requested.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field, field_validator

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


class SupportResponse(BaseModel):
    """The one shape every crew response must take."""

    answer: str = Field(description="The grounded answer text shown to the user.")
    grounded: bool = Field(
        description="True only if the answer came from retrieved KB context."
    )
    sources: list[str] = Field(
        default_factory=list, description="Knowledge-base doc ids used."
    )
    top_similarity: float = Field(
        default=0.0, description="Best retrieval similarity behind the answer."
    )
    order_record_id: str | None = Field(
        default=None, description="Order id looked up, if any."
    )
    order_status: str | None = Field(
        default=None, description="Status of the looked-up order, if any."
    )
    escalation_score: float | None = Field(
        default=None, description="Escalation score in [0, 1], if an order was looked up."
    )
    needs_human: bool = Field(
        description="True if the conversation should be handed to a human agent."
    )
    reason: str = Field(description="Why the system answered, refused or escalated.")

    @field_validator("answer")
    @classmethod
    def answer_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("answer must not be empty")
        return v.strip()

    @field_validator("top_similarity")
    @classmethod
    def similarity_in_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"top_similarity must be in [0,1], got {v}")
        return v

    @field_validator("escalation_score")
    @classmethod
    def escalation_in_range(cls, v: float | None) -> float | None:
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError(f"escalation_score must be in [0,1], got {v}")
        return v

    @field_validator("sources")
    @classmethod
    def sources_are_strings(cls, v: list[str]) -> list[str]:
        return [str(s) for s in v]


class SchemaValidationError(ValueError):
    """Raised when a crew response cannot be coerced into SupportResponse."""


def parse_response(raw: Any) -> SupportResponse:
    """Coerce a crew output into :class:`SupportResponse`, validating strictly.

    Accepts a ``SupportResponse`` already, a ``CrewOutput``, a JSON string, or
    text containing a JSON object.
    """
    if isinstance(raw, SupportResponse):
        return raw

    candidate: Any = raw
    if hasattr(raw, "pydantic") and getattr(raw, "pydantic") is not None:
        candidate = raw.pydantic
        if isinstance(candidate, SupportResponse):
            return candidate
    if hasattr(raw, "raw"):
        candidate = raw.raw

    if isinstance(candidate, SupportResponse):
        return candidate

    if isinstance(candidate, dict):
        return SupportResponse.model_validate(candidate)

    text = str(candidate).strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[1] if "\n" in text else text

    try:
        return SupportResponse.model_validate(json.loads(text))
    except Exception:  # noqa: BLE001 - fall through to the embedded-object scan
        pass

    match = _JSON_OBJECT_RE.search(text)
    if match:
        try:
            return SupportResponse.model_validate(json.loads(match.group(0)))
        except Exception as exc:  # noqa: BLE001
            raise SchemaValidationError(
                f"embedded JSON did not validate against SupportResponse: {exc}"
            ) from exc

    raise SchemaValidationError(
        f"no JSON object found in crew output: {text[:200]!r}"
    )
