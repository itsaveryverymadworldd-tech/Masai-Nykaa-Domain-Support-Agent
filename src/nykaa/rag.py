"""RAG core: embeddings, the two ChromaDB collections, retrieval, generation.

Part 1, Tasks 3-5.

Two chunking strategies (see :mod:`nykaa.chunking`) are embedded with a local
SentenceTransformers model and indexed into **two separate** ChromaDB
collections, so Task 5 can score them independently on the same queries.

Grounded generation is deliberately extractive: under ``MOCK_LLM`` there is no
model to judge groundedness, so the only honest signal is retrieval similarity.
When the best similarity falls below the calibrated threshold, the system
refuses rather than answering -- that is the "I don't know" fallback, and the
threshold is measured empirically in ``scripts/03_calibrate_threshold.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Sequence

import chromadb

from nykaa import config
from nykaa.chunking import Chunk, load_documents, build_both


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=1)
def get_encoder(model_name: str = config.EMBEDDING_MODEL_NAME):
    """Load the local SentenceTransformers encoder once per process.

    ``normalize_embeddings=True`` means cosine similarity is a plain dot product
    and the ChromaDB ``cosine`` space distance (``1 - similarity``) is exact.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed(texts: Sequence[str]) -> list[list[float]]:
    """Embed texts as unit-normalised vectors."""
    vectors = get_encoder().encode(
        list(texts), normalize_embeddings=True, show_progress_bar=False
    )
    return [v.tolist() for v in vectors]


# --------------------------------------------------------------------------- #
# ChromaDB
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=1)
def get_client() -> chromadb.ClientAPI:
    """Persistent Chroma client rooted at ``data/chroma``."""
    return chromadb.PersistentClient(path=str(config.CHROMA_DIR))


def build_collection(name: str, chunks: list[Chunk], reset: bool = True):
    """Embed ``chunks`` and upsert them into the named collection."""
    client = get_client()
    if reset:
        try:
            client.delete_collection(name)
        except Exception:  # noqa: BLE001 - absent collection is fine
            pass
    collection = client.get_or_create_collection(
        name=name, metadata={"hnsw:space": "cosine"}
    )
    if chunks:
        collection.upsert(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[c.metadata() for c in chunks],
            embeddings=embed([c.text for c in chunks]),
        )
    return collection


def build_all_collections(reset: bool = True) -> dict[str, Any]:
    """Build BOTH collections from the KB and return them with their chunks."""
    docs = load_documents(config.KB_DIR)
    fixed, sentence = build_both(docs)
    return {
        "docs": docs,
        "fixed_overlap": build_collection(config.COLLECTION_FIXED, fixed, reset),
        "sentence": build_collection(config.COLLECTION_SENTENCE, sentence, reset),
        "fixed_chunks": fixed,
        "sentence_chunks": sentence,
    }


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #


@dataclass
class Hit:
    """One retrieved chunk with its similarity."""

    chunk_id: str
    doc_id: str
    doc_title: str
    text: str
    similarity: float


def retrieve(
    query: str,
    collection,
    top_k: int = config.DEFAULT_TOP_K,
) -> list[Hit]:
    """Return the top-k chunks for ``query``, highest similarity first.

    ChromaDB's ``cosine`` space reports ``distance = 1 - cosine_similarity``;
    this converts back so the returned number means what it says.
    """
    res = collection.query(
        query_embeddings=embed([query]),
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    hits: list[Hit] = []
    for cid, doc, meta, dist in zip(
        res["ids"][0],
        res["documents"][0],
        res["metadatas"][0],
        res["distances"][0],
    ):
        hits.append(
            Hit(
                chunk_id=cid,
                doc_id=meta["doc_id"],
                doc_title=meta["doc_title"],
                text=doc,
                similarity=round(1.0 - float(dist), 6),
            )
        )
    return hits


# --------------------------------------------------------------------------- #
# Grounded generation
# --------------------------------------------------------------------------- #

FALLBACK_MESSAGE = (
    "I don't have that in the Nykaa policy knowledge base, so I can't answer it "
    "reliably. I'd rather say I don't know than guess. Let me route this to a "
    "human support agent."
)


def chunks_for_doc(collection, doc_id: str) -> list[tuple[str, str]]:
    """Every ``(chunk_id, text)`` belonging to ``doc_id``, in stored order."""
    res = collection.get(where={"doc_id": doc_id}, include=["documents", "metadatas"])
    pairs = list(zip(res["ids"], res["documents"], res["metadatas"]))
    pairs.sort(key=lambda p: p[2].get("position", 0))
    return [(cid, text) for cid, text, _ in pairs]


@dataclass
class GroundedAnswer:
    """Result of grounded generation, including the refusal path."""

    query: str
    answered: bool
    answer: str
    top_similarity: float
    threshold: float
    hits: list[Hit]
    contexts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "answered": self.answered,
            "answer": self.answer,
            "top_similarity": self.top_similarity,
            "threshold": self.threshold,
            "sources": [
                {
                    "chunk_id": h.chunk_id,
                    "doc_id": h.doc_id,
                    "similarity": h.similarity,
                }
                for h in self.hits
            ],
        }


def grounded_answer(
    query: str,
    collection,
    threshold: float | None = None,
    top_k: int = config.DEFAULT_TOP_K,
) -> GroundedAnswer:
    """Answer ``query`` using ONLY retrieved context, or refuse.

    Refusal is driven purely by the top-1 similarity against ``threshold``,
    which must be calibrated empirically (Task 4) rather than preset.
    """
    threshold = config.SIMILARITY_THRESHOLD if threshold is None else threshold
    hits = retrieve(query, collection, top_k=top_k)
    top = hits[0].similarity if hits else 0.0

    if not hits or top < threshold:
        return GroundedAnswer(
            query=query,
            answered=False,
            answer=FALLBACK_MESSAGE,
            top_similarity=top,
            threshold=threshold,
            hits=hits,
        )

    # Extractive composition. Start from the chunks that cleared the threshold,
    # then apply **document expansion**: if the top match belongs to a document,
    # that document's other chunks are pulled in too. Without this, a question
    # whose answer lives in a sibling chunk of the right document is answered
    # incompletely -- the evaluation set showed exactly that (the correct
    # document ranked first while the expected fact sat in its 4th-ranked
    # sibling chunk).
    kept = [h for h in hits if h.similarity >= threshold]
    used: list[str] = [h.text for h in kept]
    used_ids = {h.chunk_id for h in kept}

    for doc_id in dict.fromkeys(h.doc_id for h in kept):
        for cid, text in chunks_for_doc(collection, doc_id):
            if cid in used_ids:
                continue
            used.append(text)
            used_ids.add(cid)

    titles = sorted({h.doc_title for h in kept})
    answer = (
        f"{' '.join(used)}\n\n(Answered from: {', '.join(titles)}. "
        f"Top match similarity {top:.3f} >= threshold {threshold:.3f}.)"
    )
    return GroundedAnswer(
        query=query,
        answered=True,
        answer=answer,
        top_similarity=top,
        threshold=threshold,
        hits=hits,
        contexts=used,
    )
