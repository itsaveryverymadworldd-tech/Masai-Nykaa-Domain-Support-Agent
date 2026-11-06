"""Response caching for grounded generation (Part 4, Task 16).

An in-memory cache keyed by **normalised query text**, so trivially different
phrasings of the same question share an entry. Every hit avoids a redundant
grounded-generation call, which is what the before/after evidence demonstrates.

Deliberately in-process only: the brief asks for an in-memory cache, and a
process-local cache is the honest scope for it.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass, field
from typing import Any, Callable

_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_query(query: str) -> str:
    """Canonical cache key: lower-case, punctuation-stripped, whitespace-collapsed.

    Full stop only -- no stemming or synonyms. That keeps the key predictable and
    the hit/miss behaviour explainable in the transcript, which matters more here
    than maximising the hit rate.
    """
    lowered = query.strip().lower()
    depunctuated = lowered.translate(_PUNCT_TABLE)
    return _WHITESPACE_RE.sub(" ", depunctuated).strip()


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0

    @property
    def lookups(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0


class ResponseCache:
    """In-memory query -> result cache with hit/miss accounting."""

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}
        self.stats = CacheStats()

    def get(self, query: str) -> Any | None:
        key = normalize_query(query)
        if key in self._store:
            self.stats.hits += 1
            return self._store[key]
        self.stats.misses += 1
        return None

    def put(self, query: str, value: Any) -> None:
        self._store[normalize_query(query)] = value

    def clear(self) -> None:
        self._store.clear()
        self.stats = CacheStats()

    def __len__(self) -> int:
        return len(self._store)

    def keys(self) -> list[str]:
        return list(self._store)


class CachedGenerator:
    """Wraps a grounded-generation callable with :class:`ResponseCache`.

    ``calls`` counts how many times the *underlying* generator actually ran, which
    is the before/after evidence the task asks for: a cache hit must leave that
    counter untouched.
    """

    def __init__(self, generator: Callable[[str], Any], cache: ResponseCache | None = None):
        self._generator = generator
        self.cache = cache or ResponseCache()
        self.calls = 0

    def __call__(self, query: str) -> Any:
        cached = self.cache.get(query)
        if cached is not None:
            return cached
        self.calls += 1
        result = self._generator(query)
        self.cache.put(query, result)
        return result

    @property
    def evaded_calls(self) -> int:
        """Underlying generator calls avoided by cache hits."""
        return self.cache.stats.hits

    def report(self) -> dict[str, Any]:
        return {
            "underlying_calls": self.calls,
            "hits": self.cache.stats.hits,
            "misses": self.cache.stats.misses,
            "lookups": self.cache.stats.lookups,
            "hit_rate": round(self.cache.stats.hit_rate, 4),
            "entries": len(self.cache),
        }
