"""Task 16 -- response caching for grounded generation.

Demonstrates a real cache hit on a repeated identical query, with before/after
evidence: an underlying-call counter plus a timing comparison.

Run with::

    uv run python scripts/16_caching.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from nykaa import config  # noqa: E402
from nykaa.cache import CachedGenerator, ResponseCache, normalize_query  # noqa: E402
from nykaa.rag import build_all_collections, get_client, grounded_answer  # noqa: E402

QUERY = "How long do I have to return a pair of shoes?"


def main() -> int:
    out: list[str] = []
    failures: list[str] = []

    def say(line: str = "") -> None:
        print(line)
        out.append(line)

    built = build_all_collections(reset=False)
    collection = built["sentence"]

    def generator(q: str):
        """The grounded-generation step being cached."""
        return grounded_answer(q, collection)

    cached = CachedGenerator(generator)

    say("=" * 78)
    say("TASK 16 -- RESPONSE CACHING (in-memory, keyed by normalised query)")
    say("=" * 78)
    say(f"cache key  : normalised query text")
    say(f"normalising: {normalize_query(QUERY)!r}")
    say(f"scope      : in-process only")
    say()

    say("#" * 78)
    say("# BEFORE / AFTER -- the SAME query issued twice")
    say("#" * 78)

    t0 = time.perf_counter()
    first = cached(QUERY)
    t_first = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    second = cached(QUERY)
    t_second = (time.perf_counter() - t1) * 1000

    say()
    say(f"call 1 (cache MISS expected)")
    say(f"    underlying generator calls so far : {cached.calls}")
    say(f"    elapsed                           : {t_first:.2f} ms")
    say(f"    top_similarity                    : {first.top_similarity:.4f}")
    say()
    say(f"call 2 (cache HIT expected)")
    say(f"    underlying generator calls so far : {cached.calls}")
    say(f"    elapsed                           : {t_second:.2f} ms")
    say(f"    top_similarity                    : {second.top_similarity:.4f}")
    say()

    say("EVIDENCE:")
    say(f"    underlying generator call count : {cached.calls}  (must be 1, not 2)")
    say(f"    cache hits                      : {cached.cache.stats.hits}")
    say(f"    cache misses                    : {cached.cache.stats.misses}")
    report = cached.report()
    say(f"    hit rate                        : {report['hit_rate']:.4f}")
    say(f"    identical result object returned: {first is second}")
    say(f"    speedup on the hit              : "
        f"{t_first / t_second:.1f}x ({t_first:.2f} ms -> {t_second:.2f} ms)")

    if cached.calls != 1:
        failures.append(f"expected 1 underlying call, got {cached.calls}")
    if cached.cache.stats.hits != 1:
        failures.append(f"expected 1 cache hit, got {cached.cache.stats.hits}")
    if first is not second:
        failures.append("cache hit did not return the cached object")
    say()

    say("#" * 78)
    say("# NORMALISATION -- a trivially different phrasing still hits")
    say("#" * 78)
    variant = "  HOW long do I have to return a pair of shoes???  "
    say(f"variant query : {variant!r}")
    say(f"normalised    : {normalize_query(variant)!r}")
    say(f"same key as original: {normalize_query(variant) == normalize_query(QUERY)}")
    before = cached.calls
    cached(variant)
    say(f"underlying calls after the variant : {cached.calls} (was {before})")
    if cached.calls != before:
        failures.append("normalised variant did not produce a cache hit")
    else:
        say("    [OK] normalisation produced a hit; no redundant generation")
    say()

    say("#" * 78)
    say("# A DIFFERENT query must NOT hit")
    say("#" * 78)
    other = "What is the warranty on a laptop I bought?"
    before = cached.calls
    cached(other)
    say(f"query         : {other}")
    say(f"underlying calls after it : {cached.calls} (was {before})")
    if cached.calls != before + 1:
        failures.append("a different query incorrectly hit the cache")
    else:
        say("    [OK] cache miss, generator ran")
    say()

    say("FINAL CACHE REPORT")
    for k, v in cached.report().items():
        say(f"    {k:<20} {v}")
    say()

    say("=" * 78)
    if failures:
        say(f"RESULT: FAILED ({len(failures)})")
        for f in failures:
            say(f"  - {f}")
    else:
        say("RESULT: PASS")
        say("  A repeated identical query produced a real cache hit, avoiding a")
        say(f"  redundant grounded-generation call: {cached.cache.stats.hits} hits across "
            f"{cached.cache.stats.lookups} lookups left the underlying generator at")
        say(f"  only {cached.calls} call(s), with a {t_first / t_second:.1f}x speedup on the hit.")
    say("=" * 78)

    config.TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.TRANSCRIPTS_DIR / "16_caching.txt").write_text(
        "\n".join(out) + "\n", encoding="utf-8"
    )
    print("\n[transcript written to transcripts/16_caching.txt]")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
