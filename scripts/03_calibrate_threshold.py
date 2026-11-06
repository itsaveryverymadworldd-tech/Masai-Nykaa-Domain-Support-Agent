"""Task 4 -- calibrate the "I don't know" threshold empirically.

The brief explicitly rejects tutorial presets (0.5 / 0.6 / 0.7) because they do
not reliably separate short policy-sentence embeddings from unrelated queries.
So we measure instead:

  1. top-1 cosine similarity for every in-scope query   -> the "answerable" cluster
  2. top-1 cosine similarity for every out-of-scope query -> the "unrelated" cluster
  3. place the threshold in the gap between the two observed clusters

Both collections are measured, since they embed different text and therefore
have different similarity scales.

Run with::

    uv run python scripts/03_calibrate_threshold.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nykaa import config  # noqa: E402
from nykaa.queries import IN_SCOPE, OUT_OF_SCOPE, TOP_K  # noqa: E402
from nykaa.rag import build_all_collections, retrieve  # noqa: E402


def measure(collection, queries: list[str]) -> list[float]:
    """Top-1 similarity for each query against ``collection``."""
    tops: list[float] = []
    for q in queries:
        hits = retrieve(q, collection, top_k=TOP_K)
        tops.append(hits[0].similarity if hits else 0.0)
    return tops


def main() -> int:
    out: list[str] = []

    def say(line: str = "") -> None:
        print(line)
        out.append(line)

    built = build_all_collections(reset=True)
    collections = {
        "A/fixed_overlap": built["fixed_overlap"],
        "B/sentence": built["sentence"],
    }

    say("=" * 78)
    say("TASK 4 -- EMPIRICAL CALIBRATION OF THE 'I DON'T KNOW' THRESHOLD")
    say("=" * 78)
    say("No preset is used. The threshold is derived from the two clusters below.")
    say()

    recommendations: dict[str, float] = {}
    summary: dict[str, dict[str, float]] = {}

    for label, coll in collections.items():
        in_tops = measure(coll, [q.text for q in IN_SCOPE])
        out_tops = measure(coll, OUT_OF_SCOPE)

        say("-" * 78)
        say(f"COLLECTION {label}")
        say("-" * 78)
        say("  IN-SCOPE (must be answerable):")
        for q, s in zip(IN_SCOPE, in_tops):
            say(f"    {s:.4f}   {q.text}")
        say("  OUT-OF-SCOPE (must fall back):")
        for q, s in zip(OUT_OF_SCOPE, out_tops):
            say(f"    {s:.4f}   {q}")
        say()

        lo_in, hi_out = min(in_tops), max(out_tops)
        say(f"  lowest  in-scope  top-1 : {lo_in:.4f}")
        say(f"  highest out-scope top-1 : {hi_out:.4f}")
        say(f"  gap between clusters    : {lo_in - hi_out:+.4f}")

        if lo_in <= hi_out:
            say("  !! clusters OVERLAP -- no threshold can separate them cleanly.")
            say("     Fix the retrieval, not the number.")
            recommendations[label] = (lo_in + hi_out) / 2
        else:
            # Midpoint of the gap: maximal margin from both clusters.
            thr = (lo_in + hi_out) / 2
            say(f"  clusters separate. Midpoint threshold = {thr:.4f}")
            recommendations[label] = thr
        say()

        summary[label] = {
            "min_in_scope": lo_in,
            "max_in_scope": max(in_tops),
            "max_out_of_scope": hi_out,
            "gap": lo_in - hi_out,
            "threshold": recommendations[label],
        }

    say("=" * 78)
    say("RECOMMENDED THRESHOLDS")
    say("=" * 78)
    for label, thr in recommendations.items():
        say(f"  {label:<18} {thr:.4f}")
    # The deployed threshold is the more conservative of the two: a refusal is
    # cheaper than a confident wrong answer in a customer-support setting.
    deployed = max(recommendations.values())
    say()
    say(f"  DEPLOYED (conservative = max of the two): {deployed:.4f}")
    say()
    say("Measured values for README.md:")
    for label, s in summary.items():
        say(
            f"  {label}: in-scope top-1 {s['min_in_scope']:.4f}-"
            f"{s['max_in_scope']:.4f}, "
            f"out-of-scope top-1 max {s['max_out_of_scope']:.4f}, "
            f"threshold {s['threshold']:.4f}"
        )
    say("=" * 78)

    config.TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.TRANSCRIPTS_DIR / "04_calibrate_threshold.txt").write_text(
        "\n".join(out) + "\n", encoding="utf-8"
    )
    print(f"\n[transcript written to transcripts/04_calibrate_threshold.txt]")
    print(f"\n>>> SET config.SIMILARITY_THRESHOLD = {deployed:.4f} <<<")
    return 0


if __name__ == "__main__":
    sys.exit(main())
