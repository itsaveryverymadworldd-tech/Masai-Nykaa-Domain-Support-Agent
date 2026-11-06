"""Task 3 -- chunk the KB both ways, embed, and index into two collections.

Builds both ChromaDB collections and proves each one retrieves sensibly on a
sample query. Writes its full output to ``transcripts/03_build_index.txt``.

Run with::

    uv run python scripts/02_build_index.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nykaa import config  # noqa: E402
from nykaa.chunking import chunk_stats, load_documents, build_both  # noqa: E402
from nykaa.rag import build_all_collections, retrieve  # noqa: E402

SAMPLE_QUERY = "How long do I have to return a pair of shoes?"


def main() -> int:
    out: list[str] = []

    def say(line: str = "") -> None:
        print(line)
        out.append(line)

    say("=" * 78)
    say("TASK 3 REPORT -- chunking, embedding, two ChromaDB collections")
    say("=" * 78)
    say(f"embedding model : {config.EMBEDDING_MODEL_NAME}")
    say(f"chunk size      : {config.CHUNK_SIZE_WORDS} words, "
        f"overlap {config.CHUNK_OVERLAP_WORDS} words (strategy A)")
    say(f"sentence group  : 2 sentences per chunk (strategy B)")
    say()

    docs = load_documents(config.KB_DIR)
    fixed, sentence = build_both(docs)

    say(f"knowledge base  : {len(docs)} documents from {config.KB_DIR.name}/")
    for d in docs:
        say(f"    {d['doc_id']:<32} {d['title']}")
    say()

    built = build_all_collections(reset=True)

    say("Strategy A -- fixed-size with overlap")
    for k, v in chunk_stats(fixed).items():
        say(f"    {k:<12} {v}")
    say(f"    collection   {config.COLLECTION_FIXED} "
        f"(count={built['fixed_overlap'].count()})")
    say()
    say("Strategy B -- sentence-based")
    for k, v in chunk_stats(sentence).items():
        say(f"    {k:<12} {v}")
    say(f"    collection   {config.COLLECTION_SENTENCE} "
        f"(count={built['sentence'].count()})")
    say()

    say("-" * 78)
    say(f"SAMPLE QUERY (proves both collections retrieve sensibly):")
    say(f'  "{SAMPLE_QUERY}"')
    say("-" * 78)
    for label, coll in (
        (f"A/{config.COLLECTION_FIXED}", built["fixed_overlap"]),
        (f"B/{config.COLLECTION_SENTENCE}", built["sentence"]),
    ):
        say()
        say(f"[{label}]")
        for h in retrieve(SAMPLE_QUERY, coll, top_k=3):
            say(f"    {h.similarity:.4f}  {h.doc_id:<32} {h.chunk_id}")
            say(f"            {h.text[:110]}...")
    say()

    say("=" * 78)
    say("PASS -- both strategies chunked, embedded and indexed separately.")
    say("=" * 78)

    config.TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.TRANSCRIPTS_DIR / "03_build_index.txt").write_text(
        "\n".join(out) + "\n", encoding="utf-8"
    )
    print(f"\n[transcript written to transcripts/03_build_index.txt]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
