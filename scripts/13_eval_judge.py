"""Task 13 -- evaluate the crew on 15 queries with four scored properties.

Runs the full crew over the evaluation set and scores Accuracy, Grounding,
Completeness and Safety for every query, then averages each across all 15.

Run with::

    uv run python scripts/13_eval_judge.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from nykaa import config  # noqa: E402
from nykaa.crew import run_crew  # noqa: E402
from nykaa.eval_set import EVAL_SET, REQUIRED_TOPICS  # noqa: E402
from nykaa.judge import (  # noqa: E402
    averages,
    build_judge_prompt,
    judge_case,
)
from nykaa.rag import build_all_collections, grounded_answer, retrieve  # noqa: E402
from nykaa.schemas import parse_response  # noqa: E402


def main() -> int:
    out: list[str] = []
    failures: list[str] = []

    def say(line: str = "") -> None:
        print(line)
        out.append(line)

    built = build_all_collections(reset=False)
    collection = built["sentence"]

    say("=" * 78)
    say("TASK 13 -- LLM-AS-JUDGE EVALUATION (MOCK_LLM), 15 QUERIES")
    say("=" * 78)
    say(f"evaluation set  : {len(EVAL_SET)} queries")
    say(f"  in-scope      : {sum(1 for c in EVAL_SET if c.kind == 'in_scope')}")
    say(f"  order lookup  : {sum(1 for c in EVAL_SET if c.kind == 'order')}")
    say(f"  out-of-scope  : {sum(1 for c in EVAL_SET if c.kind == 'out_of_scope')}")
    say(f"judge           : deterministic rubric evaluator under MOCK_LLM")
    say()

    # Coverage check: every required KB topic must be touched.
    covered = {c.topic for c in EVAL_SET}
    missing = REQUIRED_TOPICS - covered
    say(f"required KB topics covered : {len(REQUIRED_TOPICS & covered)}/{len(REQUIRED_TOPICS)}")
    if missing:
        failures.append(f"evaluation set does not cover: {sorted(missing)}")
        for m in sorted(missing):
            say(f"  MISSING: {m}")
    else:
        say("  [OK] every required knowledge-base topic has at least one query")
    say()

    scores = []
    say("#" * 78)
    say("# PER-QUERY RESULTS")
    say("#" * 78)

    for case in EVAL_SET:
        run = run_crew(case.query, case.record_id)
        parsed = parse_response(run.result)
        hits = retrieve(case.query, collection, top_k=3)
        doc_ids = []
        for h in hits:
            if h.doc_id not in doc_ids:
                doc_ids.append(h.doc_id)

        # Judge against the SAME context the rag_lookup tool actually used,
        # which includes document expansion -- otherwise a complete answer gets
        # scored as ungrounded because its sibling chunks are not in top-3.
        direct = grounded_answer(case.query, collection)
        contexts = direct.contexts if direct.answered else [h.text for h in hits]

        score = judge_case(
            case=case,
            answer=parsed.answer,
            contexts=contexts,
            retrieved_doc_ids=doc_ids,
            answered=parsed.grounded,
            top_similarity=parsed.top_similarity,
            tool_invoked="check_order_status" in run.tools_invoked,
            order_record_id=parsed.order_record_id,
        )
        scores.append(score)

        say()
        say(f"{case.case_id}  [{case.kind}]  {case.topic}")
        say(f"    query         : {case.query}")
        say(f"    answered      : {parsed.grounded}   top-1 similarity: {parsed.top_similarity:.4f}")
        say(f"    retrieved     : {doc_ids}")
        if case.expected_facts:
            present = [f for f in case.expected_facts if f.lower() in parsed.answer.lower()]
            say(f"    expected facts: {list(case.expected_facts)} -> present {present}")
        say(f"    tools invoked : {run.tools_invoked}")
        say(f"    SCORES  accuracy={score.accuracy:.2f}  grounding={score.grounding:.2f}"
            f"  completeness={score.completeness:.2f}  safety={score.safety:.2f}"
            f"   (mean {score.mean:.3f})")
        for note in score.notes:
            say(f"      - {note}")

    # ------------------------------------------------------------------ #
    say()
    say("=" * 78)
    say("SUMMARY TABLE -- all four scores per query")
    say("=" * 78)
    say(f"{'case':<6}{'kind':<14}{'accuracy':>10}{'grounding':>11}"
        f"{'complete':>10}{'safety':>9}{'mean':>8}")
    say("-" * 78)
    for s in scores:
        say(f"{s.case_id:<6}{s.kind:<14}{s.accuracy:>10.2f}{s.grounding:>11.2f}"
            f"{s.completeness:>10.2f}{s.safety:>9.2f}{s.mean:>8.3f}")
    say("-" * 78)

    agg = averages(scores)
    say(f"{'AVERAGE':<20}{agg['accuracy']:>10.4f}{agg['grounding']:>11.4f}"
        f"{agg['completeness']:>10.4f}{agg['safety']:>9.4f}{agg['mean']:>8.4f}")
    say()
    say("four averages across all 15 queries:")
    say(f"  Accuracy     {agg['accuracy']:.4f}")
    say(f"  Grounding    {agg['grounding']:.4f}")
    say(f"  Completeness {agg['completeness']:.4f}")
    say(f"  Safety       {agg['safety']:.4f}")
    say()

    say("the judge prompt that would be sent to a real LLM (first case, truncated):")
    first = EVAL_SET[0]
    prompt = build_judge_prompt(
        first, "(answer text)", ["01_return_window"], True
    )
    for line in prompt.strip().splitlines()[:8]:
        say(f"    {line}")
    say("    ...")
    say()

    say("=" * 78)
    if failures:
        say(f"RESULT: FAILED ({len(failures)})")
        for f in failures:
            say(f"  - {f}")
    else:
        say("RESULT: PASS")
        say(f"  {len(scores)}/{len(EVAL_SET)} queries scored on all four properties.")
        say(f"  Averages: accuracy {agg['accuracy']:.4f}, grounding {agg['grounding']:.4f}, "
            f"completeness {agg['completeness']:.4f}, safety {agg['safety']:.4f}")
    say("=" * 78)

    config.TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.TRANSCRIPTS_DIR / "13_eval_judge.txt").write_text(
        "\n".join(out) + "\n", encoding="utf-8"
    )
    (config.TRANSCRIPTS_DIR / "13_eval_scores.json").write_text(
        json.dumps(
            {"per_query": [s.to_dict() for s in scores], "averages": agg}, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    print("\n[transcript written to transcripts/13_eval_judge.txt]")
    print("[scores written to transcripts/13_eval_scores.json]")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
