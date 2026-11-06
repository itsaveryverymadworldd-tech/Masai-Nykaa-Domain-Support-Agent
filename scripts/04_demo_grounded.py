"""Task 4 -- demonstrate grounded generation and the "I don't know" fallback.

Runs every in-scope query (must be answered from retrieved context only) plus
deliberately out-of-scope queries that MUST trigger the refusal path.

The prohibition on a preset threshold is the point of this pair of scripts:
``scripts/03_calibrate_threshold.py`` measures the clusters, this one proves the
resulting threshold actually separates them at runtime.

Run with::

    uv run python scripts/04_demo_grounded.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nykaa import config  # noqa: E402
from nykaa.queries import IN_SCOPE, OUT_OF_SCOPE, TOP_K  # noqa: E402
from nykaa.rag import build_all_collections, grounded_answer  # noqa: E402


def main() -> int:
    out: list[str] = []

    def say(line: str = "") -> None:
        print(line)
        out.append(line)

    built = build_all_collections(reset=True)
    collection = built["sentence"]  # deployed strategy (see Task 5)
    threshold = config.SIMILARITY_THRESHOLD

    say("=" * 78)
    say("TASK 4 -- GROUNDED GENERATION + 'I DON'T KNOW' FALLBACK")
    say("=" * 78)
    say(f"collection      : {config.COLLECTION_SENTENCE}")
    say(f"threshold       : {threshold:.4f}  (empirically calibrated, not a preset)")
    say(f"top-k           : {TOP_K}")
    say(f"queries         : {len(IN_SCOPE)} in-scope + {len(OUT_OF_SCOPE)} out-of-scope")
    say()

    failures: list[str] = []

    say("#" * 78)
    say(f"# IN-SCOPE QUERIES  (must be ANSWERED from retrieved context)")
    say("#" * 78)
    for q in IN_SCOPE:
        ans = grounded_answer(q.text, collection, threshold=threshold, top_k=TOP_K)
        say()
        say(f"QUERY        : {q.text}")
        say(f"topic        : {q.topic}")
        say(f"gold doc     : {', '.join(sorted(q.gold_doc_ids))}")
        say(f"top-1 sim    : {ans.top_similarity:.4f}  vs threshold {threshold:.4f}")
        say(f"answered     : {ans.answered}")
        say(f"retrieved    : {', '.join(h.doc_id + ' (' + format(h.similarity, '.3f') + ')' for h in ans.hits)}")
        hit = q.gold_doc_ids & {h.doc_id for h in ans.hits}
        say(f"gold in top-{TOP_K} : {bool(hit)}")
        if not ans.answered:
            failures.append(f"in-scope query wrongly refused: {q.text!r}")
        if not hit:
            failures.append(f"in-scope query did not retrieve its gold doc: {q.text!r}")
        say(f"ANSWER       : {ans.answer.splitlines()[0][:300]}")
        say("-" * 78)

    say()
    say("#" * 78)
    say(f"# OUT-OF-SCOPE QUERIES  (must trigger the FALLBACK / refusal)")
    say("#" * 78)
    for raw in OUT_OF_SCOPE:
        ans = grounded_answer(raw, collection, threshold=threshold, top_k=TOP_K)
        say()
        say(f"QUERY        : {raw}")
        say(f"top-1 sim    : {ans.top_similarity:.4f}  vs threshold {threshold:.4f}")
        say(f"answered     : {ans.answered}")
        say(f"nearest doc  : {ans.hits[0].doc_id if ans.hits else '(none)'}"
            f"  (the forced nearest neighbour, similarity {ans.top_similarity:.4f})")
        say(f"RESPONSE     : {ans.answer.splitlines()[0][:160]}")
        if ans.answered:
            failures.append(f"out-of-scope query wrongly ANSWERED: {raw!r}")
        ok = "FALLBACK CORRECT" if not ans.answered else "FAILED TO FALL BACK"
        say(f"VERDICT      : {ok}")
        say("-" * 78)

    say()
    say("=" * 78)
    if failures:
        say(f"RESULT: FAILED ({len(failures)} problem(s))")
        for f in failures:
            say(f"  - {f}")
    else:
        say("RESULT: PASS")
        say(f"  {len(IN_SCOPE)}/{len(IN_SCOPE)} in-scope queries answered from retrieved context,")
        say("  every gold document retrieved inside top-3.")
        say(f"  {len(OUT_OF_SCOPE)}/{len(OUT_OF_SCOPE)} out-of-scope queries correctly triggered")
        say(f"  the 'I don't know' fallback (highest out-of-scope similarity "
            f"{max(grounded_answer(q, collection, threshold=threshold, top_k=TOP_K).top_similarity for q in OUT_OF_SCOPE):.4f}"
            f" < threshold {threshold:.4f}).")
    say("=" * 78)

    config.TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.TRANSCRIPTS_DIR / "04_demo_grounded.txt").write_text(
        "\n".join(out) + "\n", encoding="utf-8"
    )
    print("\n[transcript written to transcripts/04_demo_grounded.txt]")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
