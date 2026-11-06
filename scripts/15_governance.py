"""Task 15 -- the four-layer AI-governance model.

Application layer: least autonomy (enforced, not just asserted) and a risk
classification with justification.
Runtime layer: a per-request budget cap that rejects an oversized request.

Run with::

    uv run python scripts/15_governance.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from nykaa import config  # noqa: E402
from nykaa.crew import assert_least_autonomy, build_agents  # noqa: E402
from nykaa.governance import (  # noqa: E402
    RISK_JUSTIFICATION,
    RISK_LEVEL,
    RISK_SCHEME,
    TOOL_OWNERS,
    LeastAutonomyViolation,
    check_agent_tools,
    check_request_budget,
    enforce_tool_ownership,
)
from nykaa.tools import build_check_order_status_tool  # noqa: E402


def main() -> int:
    out: list[str] = []
    failures: list[str] = []

    def say(line: str = "") -> None:
        print(line)
        out.append(line)

    say("=" * 78)
    say("TASK 15 -- FOUR-LAYER AI GOVERNANCE")
    say("=" * 78)
    say()

    # ================================================================== #
    say("#" * 78)
    say("# APPLICATION LAYER -- LEAST AUTONOMY")
    say("#" * 78)
    say("privileged tool ownership table:")
    for tool_name, owner in TOOL_OWNERS.items():
        say(f"    {tool_name:<22} -> owned by {owner!r}")
    say()

    la = assert_least_autonomy()
    say("actual wiring in the built crew:")
    say(f"    agents                            : {la['agents_total']}")
    say(f"    holders of check_order_status     : {la['check_order_status_holders']}")
    say(f"    exactly one holder                : {la['enforced']}")
    if not la["enforced"]:
        failures.append("least autonomy not enforced in the built crew")
    say()

    say("enforcement is a GUARD, not a comment -- attempting a bad wiring:")
    # 1. The permitted wiring passes.
    try:
        enforce_tool_ownership("Order Lookup Specialist", "check_order_status")
        say("    [OK]   Order Lookup Specialist -> check_order_status : allowed")
    except LeastAutonomyViolation as exc:
        failures.append(f"permitted wiring was blocked: {exc}")
        say(f"    [FAIL] {exc}")

    # 2. Wiring the same tool to any other agent is refused.
    for intruder in ("Policy Retrieval Specialist", "Response Composer"):
        try:
            enforce_tool_ownership(intruder, "check_order_status")
            failures.append(f"{intruder} was allowed to take check_order_status")
            say(f"    [FAIL] {intruder} -> check_order_status : ALLOWED (should be blocked)")
        except LeastAutonomyViolation as exc:
            say(f"    [OK]   {intruder} -> check_order_status : BLOCKED")
            say(f"           {exc}")

    # 3. Validate every tool on every real agent.
    say()
    say("validating every tool on every agent in the real crew:")
    from nykaa.mock_llm import MockLLM

    agents = build_agents(MockLLM(model="mock-llm"))
    for agent in agents:
        try:
            checked = check_agent_tools(agent)
            say(f"    [OK] {agent.role:<30} tools={checked}")
        except LeastAutonomyViolation as exc:
            failures.append(str(exc))
            say(f"    [FAIL] {agent.role}: {exc}")
    say()
    say("one-paragraph explanation of the guard:")
    say("  Only the Lookup Agent may call check_order_status. This is enforced in")
    say("  two independent places rather than by convention: the crew builder")
    say("  constructs the tool once and attaches it to the Lookup Agent alone, and")
    say("  enforce_tool_ownership() re-checks every agent's tool list before the")
    say("  crew is built, raising LeastAutonomyViolation if a privileged tool is")
    say("  ever wired to a non-owner. The second check is what makes this a guard")
    say("  rather than a comment: a future edit that adds the tool to the Composer")
    say("  for convenience would fail loudly at construction time instead of")
    say("  silently widening the blast radius. The Retrieval and Composer agents")
    say("  are therefore structurally incapable of reading order records, so a")
    say("  prompt-injection attempt aimed at the retrieval path cannot reach them.")
    say()

    # ================================================================== #
    say("#" * 78)
    say("# APPLICATION LAYER -- RISK CLASSIFICATION")
    say("#" * 78)
    for level, examples in RISK_SCHEME.items():
        marker = " <== THIS SYSTEM" if level == RISK_LEVEL else ""
        say(f"    {level:<7} {examples}{marker}")
    say()
    say(f"CLASSIFICATION: {RISK_LEVEL}")
    say()
    say("justification:")
    for line in RISK_JUSTIFICATION.split(". "):
        if line.strip():
            say(f"  {line.strip().rstrip('.')}.")
    say()

    # ================================================================== #
    say("#" * 78)
    say("# RUNTIME LAYER -- PER-REQUEST TOKEN / COST BUDGET")
    say("#" * 78)
    say(f"budget per request : {config.MAX_TOKENS_PER_REQUEST} tokens")
    say(f"max query length   : {config.MAX_QUERY_CHARS} characters")
    say()

    normal = "How long do I have to return a pair of shoes?"
    say("CASE 1 -- a normal in-budget request")
    verdict = check_request_budget(normal)
    say(f"    query chars      : {verdict.query_chars}")
    say(f"    estimated tokens : {verdict.estimated_tokens} / {verdict.budget_tokens}")
    say(f"    allowed          : {verdict.allowed}")
    say(f"    reason           : {verdict.reason}")
    if not verdict.allowed:
        failures.append("a normal request was rejected by the budget cap")
    say()

    say("CASE 2 -- a DELIBERATELY OVERSIZED request (must be REJECTED)")
    oversized = (
        "Please answer the following very long question about our Nykaa order "
        "policy in full detail. " * 60
    )
    say(f"    query chars      : {len(oversized)}")
    verdict2 = check_request_budget(oversized)
    say(f"    estimated tokens : {verdict2.estimated_tokens} / {verdict2.budget_tokens}")
    say(f"    allowed          : {verdict2.allowed}")
    say(f"    reason           : {verdict2.reason}")
    if verdict2.allowed:
        failures.append("the oversized request was NOT rejected")
    else:
        say("    [OK] rejected rather than silently truncated or sent anyway")
    say()

    say("CASE 3 -- normal query, but an OVERSIZED RETRIEVED CONTEXT")
    say("          (the realistic way the token budget binds: the query is short,")
    say("           the assembled prompt is not)")
    big_context = "policy clause text " * 1200
    verdict3 = check_request_budget(normal, big_context)
    say(f"    query chars      : {verdict3.query_chars} "
        f"(limit {verdict3.max_query_chars}, so within the char cap)")
    say(f"    context chars    : {len(big_context)}")
    say(f"    estimated tokens : {verdict3.estimated_tokens} / {verdict3.budget_tokens}")
    say(f"    allowed          : {verdict3.allowed}")
    say(f"    reason           : {verdict3.reason}")
    if verdict3.allowed:
        failures.append("a request over the token budget was not rejected")
    else:
        say("    [OK] rejected on the TOKEN budget, independently of the char cap")
    say()

    say("=" * 78)
    if failures:
        say(f"RESULT: FAILED ({len(failures)})")
        for f in failures:
            say(f"  - {f}")
    else:
        say("RESULT: PASS")
        say("  Application layer: least autonomy enforced (bad wirings blocked),")
        say(f"  risk classified {RISK_LEVEL} with justification.")
        say("  Runtime layer: the oversized request was rejected, not truncated.")
    say("=" * 78)

    config.TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.TRANSCRIPTS_DIR / "15_governance.txt").write_text(
        "\n".join(out) + "\n", encoding="utf-8"
    )
    print("\n[transcript written to transcripts/15_governance.txt]")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
