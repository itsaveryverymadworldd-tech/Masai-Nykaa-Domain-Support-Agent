"""One-time download of the local embedding model.

The rest of the project runs with ``HF_HUB_OFFLINE=1`` and
``TRANSFORMERS_OFFLINE=1`` so that no transcript can depend on a network call.
This script is the single, explicit exception: it fetches the model once and
caches it locally, after which everything else works offline.

Run with::

    uv run python scripts/01_fetch_model.py
"""

from __future__ import annotations

import os
import sys

# This is the ONLY place in the project that is allowed to touch the network.
# Clear the offline flags that src/nykaa/config.py would otherwise set.
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main() -> int:
    print(f"Fetching embedding model: {MODEL_NAME}")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME)
    dim = model.get_sentence_embedding_dimension()
    print(f"  embedding dimension : {dim}")

    # Prove it actually encodes, and prove the encoder is deterministic --
    # reproducibility of every retrieval number downstream depends on that.
    probe = ["return window for footwear", "reverse pickup eligibility"]
    a = model.encode(probe, normalize_embeddings=True)
    b = model.encode(probe, normalize_embeddings=True)
    import numpy as np

    if not np.allclose(a, b):
        print("  ERROR: embeddings are not deterministic between calls")
        return 1

    print(f"  encoded {len(probe)} probe sentences, deterministic = True")
    print(f"  cache location : {os.environ.get('HF_HOME', '~/.cache/huggingface')}")
    print()
    print("OK - model cached locally. The project now runs fully offline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
