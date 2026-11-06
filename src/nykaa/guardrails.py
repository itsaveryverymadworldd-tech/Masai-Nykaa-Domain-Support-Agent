"""Guardrails (Part 2, Task 10).

Input side
    * PII masking for the two **fixed-format** fields the brief names: Indian
      mobile numbers and payment-card last-4-digits.
    * Prompt-injection detection.

    The brief is explicit that customer name and delivery address are free text
    with no reliable pattern to match under a keyless, MOCK_LLM-only masker, and
    are **acknowledged as out of scope**. This module says so in code rather
    than pretending to cover them.

Output side
    * A groundedness check that refuses to answer when the retrieved context
      does not support the generated text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# --------------------------------------------------------------------------- #
# Input side -- PII masking
# --------------------------------------------------------------------------- #

#: Indian mobile numbers: optional +91 / 0 prefix, then a 10-digit number
#: starting 6-9. The number itself may be written with a single internal
#: separator ("91234 56789"), which is a common Indian convention.
#: Anchored with lookarounds so it cannot bite a slice out of a longer digit run
#: (e.g. an order id or a card number).
PHONE_RE = re.compile(
    r"(?<!\d)"
    r"(?:(?:\+?91[\s\-]?)|(?:0[\s\-]?))?"
    r"([6-9]\d{4}[\s\-]?\d{5})"
    r"(?!\d)"
)

#: Card last-4 patterns. Ordered most-specific first.
CARD_LAST4_RES: tuple[re.Pattern[str], ...] = (
    # XXXX-XXXX-XXXX-1234  /  **** **** **** 1234
    re.compile(r"\b(?:[xX*]{4}[\s\-]?){3}(\d{4})(?!\d)"),
    # "card ending in 1234", "last 4 digits are 8765", "last four 1234"
    # Allows words between the keyword and the digits ("last 4 digits are 8765").
    re.compile(
        r"(?:last\s*(?:4|four)|ending(?:\s*(?:in|with))?)"
        r"[^\d\n]{0,24}(\d{4})(?!\d)",
        re.IGNORECASE,
    ),
    # "card number 1234", "debit card 4321"
    re.compile(r"\b(?:card|debit|credit)\b[^\d\n]{0,24}(\d{4})(?!\d)", re.IGNORECASE),
)

MASKED_PHONE = "[PHONE_REDACTED]"
MASKED_CARD = "[CARD_LAST4_REDACTED]"

#: Free-text fields the brief places out of scope. Recorded, not implemented.
OUT_OF_SCOPE_PII = ("customer_name", "delivery_address")


@dataclass
class MaskResult:
    """Result of masking one piece of text."""

    original: str
    masked: str
    redactions: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return self.original != self.masked


def mask_pii(text: str) -> MaskResult:
    """Mask fixed-format PII. Idempotent and safe to apply to logs as well.

    Applied identically to what the model sees and what is written to disk, so a
    fixed-format field never reaches a log in the clear.
    """
    redactions: list[str] = []
    masked = text

    def _phone_sub(m: re.Match[str]) -> str:
        redactions.append(f"phone:{m.group(1)[:2]}****{m.group(1)[-2:]}")
        return MASKED_PHONE

    masked = PHONE_RE.sub(_phone_sub, masked)

    for pattern in CARD_LAST4_RES:
        def _card_sub(m: re.Match[str]) -> str:
            redactions.append(f"card_last4:** {m.group(1)[-1]}")
            return MASKED_CARD

        masked = pattern.sub(_card_sub, masked)

    return MaskResult(original=text, masked=masked, redactions=redactions)


# --------------------------------------------------------------------------- #
# Input side -- prompt-injection detection
# --------------------------------------------------------------------------- #

INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("instruction_override", re.compile(
        r"\b(?:ignore|disregard|forget|override)\b[^.]{0,40}\b"
        r"(?:previous|prior|above|earlier|all)\b[^.]{0,20}\b(?:instruction|prompt|rule|direction)s?\b",
        re.IGNORECASE)),
    ("system_prompt_probe", re.compile(
        r"\b(?:reveal|show|print|repeat|output|tell)\b[^.]{0,30}\b"
        r"(?:your\s+)?(?:system\s+prompt|initial\s+prompt|instructions|hidden\s+prompt)\b",
        re.IGNORECASE)),
    ("role_reassignment", re.compile(
        r"\b(?:you\s+are\s+now|act\s+as|pretend\s+to\s+be|from\s+now\s+on\s+you)\b",
        re.IGNORECASE)),
    ("guardrail_bypass", re.compile(
        r"\b(?:jailbreak|DAN\s+mode|developer\s+mode|bypass\s+(?:the\s+)?"
        r"(?:guardrail|filter|safety|restriction)s?)\b",
        re.IGNORECASE)),
    ("data_exfiltration", re.compile(
        r"\b(?:dump|list|export|give\s+me)\b[^.]{0,30}\b"
        r"(?:all\s+)?(?:customer|order|user)\s*(?:data|records|details|list)\b",
        re.IGNORECASE)),
)


@dataclass
class InjectionVerdict:
    detected: bool
    rule: str | None = None
    matched_text: str | None = None
    detail: str = ""


def detect_injection(text: str) -> InjectionVerdict:
    """Detect prompt-injection attempts, returning the rule that fired."""
    for name, pattern in INJECTION_PATTERNS:
        m = pattern.search(text)
        if m:
            return InjectionVerdict(
                detected=True,
                rule=name,
                matched_text=m.group(0),
                detail=f"input matched the {name!r} injection rule",
            )
    return InjectionVerdict(detected=False)


# --------------------------------------------------------------------------- #
# Output side -- groundedness against retrieved context
# --------------------------------------------------------------------------- #

_STOPWORDS = frozenset(
    """a an and are as at be by for from has have in is it its of on or that the
    to was were will with this these those you your we our they their""".split()
)


def _content_tokens(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {t for t in tokens if t not in _STOPWORDS and len(t) > 2}


@dataclass
class GroundednessVerdict:
    supported: bool
    coverage: float
    unsupported_sentences: list[str] = field(default_factory=list)
    detail: str = ""


def check_groundedness(
    answer: str,
    contexts: list[str],
    min_coverage: float = 0.75,
) -> GroundednessVerdict:
    """Refuse an answer whose content is not supported by the retrieved context.

    Each sentence of the answer must have at least ``min_coverage`` of its
    content tokens present in the concatenated retrieved context. This is the
    output-side guard the brief requires: it catches an ungrounded claim even
    when retrieval similarity looked fine.
    """
    if not contexts:
        return GroundednessVerdict(
            supported=False,
            coverage=0.0,
            unsupported_sentences=[answer],
            detail="no retrieved context was supplied",
        )

    # Strip the generator's provenance footer before scoring. It is metadata
    # about the retrieval, not a claim about policy, so scoring it would flag
    # every legitimate answer as ungrounded.
    answer = re.split(r"\(Answered from:", answer)[0].strip()

    context_tokens = _content_tokens(" ".join(contexts))
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
    if not sentences:
        return GroundednessVerdict(
            supported=False, coverage=0.0, detail="answer was empty"
        )

    unsupported: list[str] = []
    coverages: list[float] = []
    for sentence in sentences:
        # Skip the trailing provenance footer our generator appends.
        if sentence.startswith("(Answered from:"):
            continue
        tokens = _content_tokens(sentence)
        if not tokens:
            continue
        covered = len(tokens & context_tokens) / len(tokens)
        coverages.append(covered)
        if covered < min_coverage:
            unsupported.append(sentence)

    overall = sum(coverages) / len(coverages) if coverages else 0.0
    return GroundednessVerdict(
        supported=not unsupported,
        coverage=round(overall, 4),
        unsupported_sentences=unsupported,
        detail=(
            "every sentence is supported by the retrieved context"
            if not unsupported
            else f"{len(unsupported)} sentence(s) fall below "
            f"{min_coverage:.0%} token coverage of the retrieved context"
        ),
    )


# --------------------------------------------------------------------------- #
# Combined input gate
# --------------------------------------------------------------------------- #


@dataclass
class InputVerdict:
    allowed: bool
    masked_query: str
    mask: MaskResult
    injection: InjectionVerdict
    reject_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reject_reason": self.reject_reason,
            "redactions": self.mask.redactions,
            "injection_rule": self.injection.rule,
            "masked_query": self.masked_query,
        }


def apply_input_guardrails(text: str) -> InputVerdict:
    """Mask fixed-format PII, then screen for prompt injection."""
    mask = mask_pii(text)
    injection = detect_injection(mask.masked)
    if injection.detected:
        return InputVerdict(
            allowed=False,
            masked_query=mask.masked,
            mask=mask,
            injection=injection,
            reject_reason=injection.detail,
        )
    return InputVerdict(
        allowed=True, masked_query=mask.masked, mask=mask, injection=injection
    )
