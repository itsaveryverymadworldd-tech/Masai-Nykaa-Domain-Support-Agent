"""Task 7 -- run the crew and demonstrate both tools actually firing.

Requirement: "Demonstrate both the RAG tool and the lookup tool actually being
invoked on different sample queries" via ``kickoff()``.

Run with::

    uv run python scripts/07_crew_demo.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from nykaa import config  # noqa: E402
from nykaa.crew import assert_least_autonomy, run_crew  # noqa: E402
from nykaa.schemas import SchemaValidationError, parse_response  # noqa: E402


SCENARIOS = [
    {
        "label": "POLICY-ONLY QUERY  (must invoke rag_lookup, not check_order_status)",
        "query": "How long do I have to return a pair of shoes?",
        "record_id": None,
        "expect_tool": "rag_lookup",
        "expect_absent": "check_order_status",
    },
    {
        "label": "ORDER QUERY  (must invoke check_order_status)",
        "query": (
            "What is the current status of order ORD-0037, and should it be "
            "escalated to a human agent?"
        ),
        "record_id": "ORD-0037",
        "expect_tool": "check_order_status",
        "expect_absent": None,
    },
]


def main() -> int:
    out: list[str] = []
    failures: list[str] = []

    def say(line: str = "") -> None:
        print(line)
        out.append(line)

    say("=" * 78)
    say("TASK 7 -- CREWAI CREW, TOOLS DEMONSTRABLY INVOKED VIA kickoff()")
    say("=" * 78)
    say(f"LLM backend     : MOCK_LLM (crewai BaseLLM subclass, zero API keys)")
    say(f"process         : sequential")
    say(f"agents          : 3 (Policy Retrieval / Order Lookup / Response Composer)")
    say(
        f"telemetry       : CREWAI_DISABLE_TELEMETRY="
        f"{os.environ.get('CREWAI_DISABLE_TELEMETRY')!r}, "
        f"OTEL_SDK_DISABLED={os.environ.get('OTEL_SDK_DISABLED')!r}"
    )
    say()

    say("-" * 78)
    say("LEAST-AUTONOMY WIRING CHECK (Part 4, Application layer)")
    say("-" * 78)
    la = assert_least_autonomy()
    say(f"  agents total                     : {la['agents_total']}")
    say(f"  agents holding check_order_status: {la['check_order_status_holders']}")
    say(f"  exactly one holder (Lookup Agent): {la['enforced']}")
    if not la["enforced"]:
        failures.append("least-autonomy wiring check failed")
    say()

    for sc in SCENARIOS:
        say("=" * 78)
        say(sc["label"])
        say("=" * 78)
        say(f"query      : {sc['query']}")
        say(f"record_id  : {sc['record_id']}")
        say()

        run = run_crew(sc["query"], sc["record_id"])

        say(f"llm calls  : {run.llm_calls}")
        say("tool dispatch log (every entry is routed by DECLARED SCHEMA, never by tool name):")
        for entry in run.action_log:
            if entry.get("tool"):
                say(f"    agent={entry['agent']!r}")
                say(f"      tool      = {entry['tool']}")
                say(f"      arguments = {json.dumps(entry['arguments'])}")
                say(f"      schema required keys = {entry['schema_required_keys']}")
                say(f"      dispatched by = {entry['dispatched_by']}")
            else:
                say(f"    agent={entry['agent']!r} -> NO TOOL CALL")
                say(f"      reason    = {entry.get('skipped_reason')}")
        say()
        say(f"tools invoked on this query: {run.tools_invoked}")

        if sc["expect_tool"] and sc["expect_tool"] not in run.tools_invoked:
            failures.append(
                f"{sc['label']}: expected {sc['expect_tool']} to be invoked"
            )
        if sc["expect_absent"] and sc["expect_absent"] in run.tools_invoked:
            failures.append(
                f"{sc['label']}: {sc['expect_absent']} must NOT be invoked"
            )
        say()

        say("FINAL CREW OUTPUT (validated against the Pydantic schema):")
        try:
            parsed = parse_response(run.result)
            say(parsed.model_dump_json(indent=2))
            say()
            say("  [OK] response validates against SupportResponse")
        except SchemaValidationError as exc:
            failures.append(f"{sc['label']}: schema validation failed: {exc}")
            say(f"  [FAIL] {exc}")
            say(f"  raw: {str(run.result)[:400]}")
        say()

    say("=" * 78)
    if failures:
        say(f"RESULT: FAILED ({len(failures)})")
        for f in failures:
            say(f"  - {f}")
    else:
        say("RESULT: PASS")
        say("  rag_lookup was invoked on the policy-only query.")
        say("  check_order_status was invoked on the order query.")
        say("  check_order_status was NOT invoked when no record id was supplied.")
        say("  Every crew response validated against the SupportResponse schema.")
        say("  check_order_status is wired to exactly one agent (least autonomy).")
    say("=" * 78)

    config.TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.TRANSCRIPTS_DIR / "07_crew_demo.txt").write_text(
        "\n".join(out) + "\n", encoding="utf-8"
    )
    print("\n[transcript written to transcripts/07_crew_demo.txt]")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
