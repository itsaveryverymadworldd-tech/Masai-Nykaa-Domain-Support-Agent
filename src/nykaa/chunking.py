"""Two chunking strategies over the knowledge base (Part 1, Task 3).

The brief requires the KB to be chunked BOTH as fixed-size-with-overlap AND as
sentence-based chunks, and each strategy to be indexed in its own ChromaDB
collection. This module is the single source of truth for both strategies.

Strategy A -- fixed-size with overlap
    Sliding word window of ``size`` words advancing by ``size - overlap``.
    Deterministic, ignores linguistic structure, produces uniform chunk sizes.

Strategy B -- sentence-based
    Groups whole sentences into chunks of at most ``SENTENCES_PER_CHUNK``
    sentences. Never splits a sentence, so every chunk is a set of complete
    claims -- which matters for a policy KB where a half-sentence like
    "unless the item arrived damaged" carries the opposite meaning alone.

Every chunk carries ``doc_id`` so retrieval scores can be mapped back to parent
documents and de-duplicated before precision/recall is computed (Task 5).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

# --------------------------------------------------------------------------- #
# Tunables -- quoted in README.md
# --------------------------------------------------------------------------- #

CHUNK_SIZE_WORDS = 90
CHUNK_OVERLAP_WORDS = 25
SENTENCES_PER_CHUNK = 2


@dataclass(frozen=True)
class Chunk:
    """A single retrievable unit of text."""

    chunk_id: str
    doc_id: str
    doc_title: str
    text: str
    strategy: str
    position: int

    def metadata(self) -> dict[str, Any]:
        """ChromaDB metadata -- flat scalars only."""
        return {
            "doc_id": self.doc_id,
            "doc_title": self.doc_title,
            "strategy": self.strategy,
            "position": self.position,
            "n_words": len(self.text.split()),
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Knowledge-base loading
# --------------------------------------------------------------------------- #

_TITLE_RE = re.compile(r"^#\s+(.*)$", re.MULTILINE)


def load_documents(kb_dir: Path) -> list[dict[str, str]]:
    """Load every ``*.md`` file in ``kb_dir`` as a document.

    Returns documents sorted by ``doc_id`` (the filename stem) so chunk order --
    and therefore every downstream number -- is stable across runs.
    """
    docs: list[dict[str, str]] = []
    for path in sorted(kb_dir.glob("*.md")):
        raw = path.read_text(encoding="utf-8").strip()
        m = _TITLE_RE.search(raw)
        title = m.group(1).strip() if m else path.stem
        # Drop the markdown heading; the title is stored separately in metadata.
        body = _TITLE_RE.sub("", raw, count=1).strip()
        docs.append({"doc_id": path.stem, "title": title, "body": body})
    return docs


# --------------------------------------------------------------------------- #
# Strategy A -- fixed-size with overlap
# --------------------------------------------------------------------------- #


def chunk_fixed_overlap(
    docs: Iterable[dict[str, str]],
    size: int = CHUNK_SIZE_WORDS,
    overlap: int = CHUNK_OVERLAP_WORDS,
) -> list[Chunk]:
    """Sliding word window. Always emits at least one chunk per document."""
    if overlap >= size:
        raise ValueError(f"overlap ({overlap}) must be smaller than size ({size})")

    step = size - overlap
    chunks: list[Chunk] = []
    for doc in docs:
        words = doc["body"].split()
        if not words:
            continue
        position = 0
        start = 0
        while start < len(words):
            window = words[start : start + size]
            chunks.append(
                Chunk(
                    chunk_id=f"{doc['doc_id']}::fixed::{position}",
                    doc_id=doc["doc_id"],
                    doc_title=doc["title"],
                    text=" ".join(window),
                    strategy="fixed_overlap",
                    position=position,
                )
            )
            position += 1
            if start + size >= len(words):
                break
            start += step
    return chunks


# --------------------------------------------------------------------------- #
# Strategy B -- sentence-based
# --------------------------------------------------------------------------- #

#: Split on ., ! or ? followed by whitespace and a capital/quote.
#: Deliberately simple: it does not try to be a full sentence tokenizer, but it
#: does guard the one abbreviation this KB actually uses ("Rs.10,000") by
#: requiring the next character to be upper-case or a quote.
_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z"\u201c(])')


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentences, keeping terminal punctuation."""
    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(text.strip())]
    return [p for p in parts if p]


def chunk_sentence_based(
    docs: Iterable[dict[str, str]],
    sentences_per_chunk: int = SENTENCES_PER_CHUNK,
) -> list[Chunk]:
    """Group whole sentences into chunks. Never splits mid-sentence."""
    chunks: list[Chunk] = []
    for doc in docs:
        sentences = split_sentences(doc["body"])
        if not sentences:
            continue
        position = 0
        for start in range(0, len(sentences), sentences_per_chunk):
            group = sentences[start : start + sentences_per_chunk]
            chunks.append(
                Chunk(
                    chunk_id=f"{doc['doc_id']}::sent::{position}",
                    doc_id=doc["doc_id"],
                    doc_title=doc["title"],
                    text=" ".join(group),
                    strategy="sentence",
                    position=position,
                )
            )
            position += 1
    return chunks


# --------------------------------------------------------------------------- #
# Convenience
# --------------------------------------------------------------------------- #


def build_both(docs: Iterable[dict[str, str]]) -> tuple[list[Chunk], list[Chunk]]:
    """Return ``(fixed_overlap_chunks, sentence_chunks)``."""
    docs = list(docs)
    return chunk_fixed_overlap(docs), chunk_sentence_based(docs)


def chunk_stats(chunks: list[Chunk]) -> dict[str, Any]:
    """Summary numbers for the transcript."""
    lengths = [len(c.text.split()) for c in chunks]
    return {
        "n_chunks": len(chunks),
        "n_docs": len({c.doc_id for c in chunks}),
        "min_words": min(lengths),
        "max_words": max(lengths),
        "mean_words": round(sum(lengths) / len(lengths), 1),
        "strategy": chunks[0].strategy if chunks else "(none)",
    }
