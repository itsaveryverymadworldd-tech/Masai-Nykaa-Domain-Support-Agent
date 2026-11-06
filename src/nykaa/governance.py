"""Four-layer AI-governance model (Part 4, Task 15).

Application layer
    * **Least autonomy, enforced.** ``check_order_status`` has exactly one owner
      (the Lookup Agent). :func:`enforce_tool_ownership` raises when any other
      agent is wired to it, so the restriction is a guard rather than a comment.
    * **Risk classification** for this system, with justification.

Runtime layer
    * A per-request token/cost budget cap. An oversized request is **rejected**
      with a clear reason instead of silently exceeding the budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nykaa import config

# --------------------------------------------------------------------------- #
# Application layer -- least autonomy
# --------------------------------------------------------------------------- #

#: The only agent permitted to hold each privileged tool.
TOOL_OWNERS: dict[str, str] = {
    "check_order_status": "Order Lookup Specialist",
}

#: Tools that every agent may hold (none are privileged).
UNRESTRICTED_TOOLS: frozenset[str] = frozenset({"rag_lookup"})


class LeastAutonomyViolation(PermissionError):
    """Raised when a privileged tool is wired to an agent that does not own it."""


def enforce_tool_ownership(agent_role: str, tool_name: str) -> None:
    """Raise if ``agent_role`` is not the declared owner of ``tool_name``.

    Wiring a privileged tool to any other agent is refused **before** the crew is
    constructed, so the restriction cannot be bypassed by a future edit that
    forgets about it.
    """
    owner = TOOL_OWNERS.get(tool_name)
    if owner is None:
        return  # not a privileged tool
    if agent_role != owner:
        raise LeastAutonomyViolation(
            f"least-autonomy violation: {tool_name!r} is owned by {owner!r} and "
            f"must not be wired to {agent_role!r}"
        )


def check_agent_tools(agent: Any) -> list[str]:
    """Validate every tool on ``agent``; returns the roles/tools checked."""
    role = getattr(agent, "role", "(unknown)")
    checked: list[str] = []
    for tool in getattr(agent, "tools", None) or []:
        name = getattr(tool, "name", "")
        enforce_tool_ownership(role, name)
        checked.append(name)
    return checked


# --------------------------------------------------------------------------- #
# Application layer -- risk classification
# --------------------------------------------------------------------------- #

RISK_SCHEME = {
    "Low": "summarisation, transcription",
    "Medium": "code generation, customer support tickets",
    "High": "medical data, hiring decisions, financial data",
}

RISK_LEVEL = "Medium"
RISK_JUSTIFICATION = (
    "This system is classified **Medium** risk. It handles customer support "
    "tickets: it answers policy questions and reports the status of one order at "
    "a time, which is squarely the 'customer support tickets' example the brief "
    "places at Medium. It is not Low risk, because the answers drive real "
    "customer expectations about refunds, returns and delivery windows -- a "
    "confidently wrong answer has a direct commercial cost, which is why the "
    "output-side groundedness guard exists. It is not High risk, because it "
    "handles no medical data, makes no hiring decisions, and touches no financial "
    "data beyond the value already printed on the customer's own order: it never "
    "sees a full card number (only the last four, masked on input), never "
    "initiates a payment, and cannot change an order's state. The Medium "
    "classification is what justifies the operational controls actually "
    "implemented here: an independent second-agent review stage before any answer "
    "is released, least-autonomy tool wiring, and a hard per-request budget cap."
)


# --------------------------------------------------------------------------- #
# Runtime layer -- token / cost budget
# --------------------------------------------------------------------------- #

#: Rough prompt overhead per crew turn (system prompts for 3 agents plus the
#: orchestration scaffolding), in tokens.
PROMPT_OVERHEAD_TOKENS = 900

#: Cheap, dependency-free token estimate. Deliberately conservative.
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Estimate token count from character length."""
    return max(1, len(text) // CHARS_PER_TOKEN)


@dataclass
class BudgetVerdict:
    allowed: bool
    estimated_tokens: int
    budget_tokens: int
    reason: str
    query_chars: int
    max_query_chars: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "estimated_tokens": self.estimated_tokens,
            "budget_tokens": self.budget_tokens,
            "query_chars": self.query_chars,
            "max_query_chars": self.max_query_chars,
            "reason": self.reason,
        }


def check_request_budget(query: str, retrieved_context: str = "") -> BudgetVerdict:
    """Enforce the per-request token budget.

    An oversized request is **rejected** with an explicit reason. It is never
    truncated and silently sent anyway -- silently exceeding the budget is the
    failure mode this guard exists to prevent.
    """
    query_chars = len(query)
    estimated = (
        PROMPT_OVERHEAD_TOKENS
        + estimate_tokens(query)
        + estimate_tokens(retrieved_context)
    )
    budget = config.MAX_TOKENS_PER_REQUEST

    if query_chars > config.MAX_QUERY_CHARS:
        return BudgetVerdict(
            allowed=False,
            estimated_tokens=estimated,
            budget_tokens=budget,
            reason=(
                f"query is {query_chars} characters, over the "
                f"{config.MAX_QUERY_CHARS}-character limit"
            ),
            query_chars=query_chars,
            max_query_chars=config.MAX_QUERY_CHARS,
        )

    if estimated > budget:
        return BudgetVerdict(
            allowed=False,
            estimated_tokens=estimated,
            budget_tokens=budget,
            reason=(
                f"estimated {estimated} tokens exceeds the {budget}-token "
                "per-request budget"
            ),
            query_chars=query_chars,
            max_query_chars=config.MAX_QUERY_CHARS,
        )

    return BudgetVerdict(
        allowed=True,
        estimated_tokens=estimated,
        budget_tokens=budget,
        reason="within budget",
        query_chars=query_chars,
        max_query_chars=config.MAX_QUERY_CHARS,
    )
