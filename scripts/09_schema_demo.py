"""Task 9 -- every crew response conforms to the declared Pydantic schema.

The brief requires a BaseModel "that every crew response must conform to
(response_format), and validate each response against it in code".

Two things are shown:
  a) real crew responses validate against SupportResponse
  b) malformed responses are actually REJECTED -- a schema that never rejects
     anything proves nothing

Run with::

    uv run python scripts/09_schema_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from pydantic import ValidationError  # noqa: E402

from nykaa import config  # noqa: E402
from nykaa.crew import run_crew  # noqa: E402
from nykaa.schemas import SupportResponse, parse_response  # noqa: E402

LIVE_QUERIES = [
    ("How long do I have to return a pair of shoes?", None),
    ("When will I get my money back for a cash on delivery order?", None),
    ("What is the status of order ORD-0037?", "ORD-0037"),
]

# Deliberately malformed payloads -- each must be rejected.
NEGATIVE_CASES = [
    (
        "empty answer",
        {"answer": "", "grounded": True, "needs_human": False, "reason": "x"},
    ),
    (
        "missing required field 'needs_human'",
        {"answer": "ok", "grounded": True, "reason": "x"},
    ),
    (
        "escalation_score out of range",
        {
            "answer": "ok",
            "grounded": True,
            "needs_human": False,
            "reason": "x",
            "escalation_score": 7.5,
        },
    ),
    (
        "top_similarity out of range",
        {
            "answer": "ok",
            "grounded": True,
            "needs_human": False,
            "reason": "x",
            "top_similarity": -0.4,
        },
    ),
    (
        "wrong type for grounded",
        {
            "answer": "ok",
            "grounded": "yes please",
            "needs_human": False,
            "reason": "x",
        },
    ),
]


def main() -> int:
    out: list[str] = []
    failures: list[str] = []

    def say(line: str = "") -> None:
        print(line)
        out.append(line)

    say("=" * 78)
    say("TASK 9 -- STRUCTURED OUTPUT SCHEMA (Pydantic), VALIDATED IN CODE")
    say("=" * 78)
    say("schema fields:")
    for name, field in SupportResponse.model_fields.items():
        say(f"    {name:<20} {field.annotation}")
    say()
    say("NOTE: the brief names `response_format`; in crewai 1.x that argument no")
    say("longer exists on Task. `Task.model_fields` shows the equivalent is")
    say("`output_pydantic`, which is what crew.py wires onto the composer task.")
    say()

    say("#" * 78)
    say("# LIVE CREW RESPONSES")
    say("#" * 78)
    for query, record_id in LIVE_QUERIES:
        run = run_crew(query, record_id)
        say()
        say(f"QUERY : {query}" + (f"   (record_id={record_id})" if record_id else ""))
        try:
            parsed = parse_response(run.result)
            say(f"  [OK] validated: grounded={parsed.grounded} "
                f"needs_human={parsed.needs_human} "
                f"sources={parsed.sources} "
                f"escalation_score={parsed.escalation_score}")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"live response failed validation for {query!r}: {exc}")
            say(f"  [FAIL] {exc}")
    say()

    say("#" * 78)
    say("# NEGATIVE CASES -- these MUST be rejected")
    say("#" * 78)
    for label, payload in NEGATIVE_CASES:
        try:
            SupportResponse.model_validate(payload)
            failures.append(f"negative case accepted but should have been rejected: {label}")
            say(f"  [FAIL] ACCEPTED  {label}")
        except ValidationError as exc:
            first = exc.errors()[0]
            say(f"  [OK]   rejected  {label}")
            say(f"         -> {first['loc']}: {first['msg']}")
    say()

    say("=" * 78)
    if failures:
        say(f"RESULT: FAILED ({len(failures)})")
        for f in failures:
            say(f"  - {f}")
    else:
        say("RESULT: PASS")
        say(f"  {len(LIVE_QUERIES)}/{len(LIVE_QUERIES)} live crew responses validated against "
            f"SupportResponse.")
        say(f"  {len(NEGATIVE_CASES)}/{len(NEGATIVE_CASES)} malformed payloads were "
            f"rejected with a Pydantic ValidationError.")
    say("=" * 78)

    config.TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.TRANSCRIPTS_DIR / "09_schema_demo.txt").write_text(
        "\n".join(out) + "\n", encoding="utf-8"
    )
    print("\n[transcript written to transcripts/09_schema_demo.txt]")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
