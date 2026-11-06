"""Task 14 -- Autogen Round Robin review stage.

Demonstrates BOTH required outcomes:
  a) the review stage approves a grounded draft UNCHANGED
  b) the review stage REVISES a draft carrying a deliberately injected,
     unsupported claim

Run with::

    uv run python scripts/14_autogen_review.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from nykaa import config  # noqa: E402
from nykaa.rag import build_all_collections, grounded_answer  # noqa: E402
from nykaa.review import (  # noqa: E402
    VerdictModel,
    build_review_team,
    run_review,
)

QUERY = "How long do I have to return a pair of shoes?"

INJECTED_CLAIM = (
    " Nykaa also offers a lifetime return window of 365 days on all Electronics "
    "purchased through the mobile app, and a full cash refund at any store."
)


def main() -> int:
    out: list[str] = []
    failures: list[str] = []

    def say(line: str = "") -> None:
        print(line)
        out.append(line)

    built = build_all_collections(reset=False)
    collection = built["sentence"]
    source = grounded_answer(QUERY, collection)
    context = source.contexts

    say("=" * 78)
    say("TASK 14 -- AUTOGEN ROUND ROBIN REVIEW STAGE")
    say("=" * 78)
    say("team            : RoundRobinGroupChat")
    say("agents          : Policy-Compliance-Reviewer, Final-Editor")
    say("binding         : max_turns=2  (NOT max_iterations)")
    say(f"                  (MaxMessageTermination would need (3), because the")
    say(f"                   initiating task message counts as message 1)")
    say("custom_message_types=[StructuredMessage[VerdictModel]]  <-- required,")
    say("                  omitting it raises 'Message type ... is not registered'")
    say("structured out  : output_content_type=VerdictModel on Final-Editor")
    say("verdict fields  : approved: bool, final_answer: str, reason: str")
    say()
    say(f"query           : {QUERY}")
    say(f"retrieved docs  : {[h.doc_id for h in source.hits]}")
    say(f"context chunks  : {len(context)}")
    say()

    # ------------------------------------------------------------------ #
    say("#" * 78)
    say("# CASE A -- grounded draft  (expect: approved UNCHANGED)")
    say("#" * 78)
    outcome_a = run_review(source.answer, context)
    say(f"client calls    : {outcome_a.client_calls}")
    say("round-robin transcript:")
    for m in outcome_a.transcript:
        say(f"    [{m['source']}]  {m['content'][:150]}")
    say()
    say(f"VERDICT approved     : {outcome_a.verdict.approved}")
    say(f"VERDICT reason       : {outcome_a.verdict.reason}")
    say(f"final_answer unchanged: {outcome_a.verdict.final_answer.strip() == source.answer.strip()}")
    if not outcome_a.verdict.approved:
        failures.append("Case A: grounded draft was not approved")
    if outcome_a.verdict.final_answer.strip() != source.answer.strip():
        failures.append("Case A: approved draft was modified")
    spoke = {m["source"] for m in outcome_a.transcript}
    if len(spoke) < 2:
        failures.append(f"Case A: only {spoke} spoke -- max_turns cut an agent off")
    say()

    # ------------------------------------------------------------------ #
    say("#" * 78)
    say("# CASE B -- draft with a DELIBERATELY INJECTED ungrounded claim")
    say("#" * 78)
    say(f"injected claim  :{INJECTED_CLAIM}")
    # Inject into the draft BODY, before the provenance footer -- realistic
    # placement for an ungrounded claim in a draft answer.
    marker = "(Answered from:"
    if marker in source.answer:
        head, tail = source.answer.split(marker, 1)
        poisoned = head.rstrip() + INJECTED_CLAIM + "\n\n" + marker + tail
    else:
        poisoned = source.answer + INJECTED_CLAIM
    outcome_b = run_review(poisoned, context)
    say(f"client calls    : {outcome_b.client_calls}")
    say("round-robin transcript:")
    for m in outcome_b.transcript:
        say(f"    [{m['source']}]  {m['content'][:180]}")
    say()
    say(f"VERDICT approved      : {outcome_b.verdict.approved}")
    say(f"VERDICT reason        : {outcome_b.verdict.reason}")
    say(f"injected claim still present in final_answer: "
        f"{'365 days' in outcome_b.verdict.final_answer}")
    say(f"final_answer (first 300 chars): {outcome_b.verdict.final_answer[:300]}")
    if outcome_b.verdict.approved:
        failures.append("Case B: ungrounded draft was approved instead of revised")
    if "365 days" in outcome_b.verdict.final_answer:
        failures.append("Case B: injected claim survived the revision")
    if not outcome_b.verdict.reason:
        failures.append("Case B: no reason given for the revision")
    spoke_b = {m["source"] for m in outcome_b.transcript}
    if len(spoke_b) < 2:
        failures.append(f"Case B: only {spoke_b} spoke")
    say()

    # ------------------------------------------------------------------ #
    say("#" * 78)
    say("# STRUCTURED VERDICT VALIDATION (both cases)")
    say("#" * 78)
    for label, outcome in (("A", outcome_a), ("B", outcome_b)):
        try:
            reparsed = VerdictModel.model_validate_json(outcome.verdict.model_dump_json())
            say(f"  case {label}: [OK] VerdictModel round-trips "
                f"(approved={reparsed.approved}, "
                f"final_answer={len(reparsed.final_answer)} chars, "
                f"reason={len(reparsed.reason)} chars)")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"case {label}: verdict did not validate: {exc}")
            say(f"  case {label}: [FAIL] {exc}")
    say()

    say("=" * 78)
    if failures:
        say(f"RESULT: FAILED ({len(failures)})")
        for f in failures:
            say(f"  - {f}")
    else:
        say("RESULT: PASS")
        say("  Case A: grounded draft approved UNCHANGED.")
        say("  Case B: ungrounded draft REVISED, injected claim removed.")
        say("  Both agents spoke in both cases (max_turns=2).")
        say("  Both verdicts are valid Pydantic VerdictModel instances.")
    say("=" * 78)

    config.TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.TRANSCRIPTS_DIR / "14_autogen_review.txt").write_text(
        "\n".join(out) + "\n", encoding="utf-8"
    )
    print("\n[transcript written to transcripts/14_autogen_review.txt]")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
