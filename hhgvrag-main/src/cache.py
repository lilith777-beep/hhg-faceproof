"""
cache.py — R5: Cache-Augmented Generation (CAG).

Two complementary caches, wired into the harness as pre-retrieval and post-answer layers:
1. SemanticResponseCache — embedding-similarity cache for sub-ms hits on repeated/similar queries.
2. SessionContext — per-session conversation history for context-aware query expansion.

Pure-stdlib + numpy. CPU-testable.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class _CacheEntry:
    query_vec: list
    response: object
    timestamp: float
    key: object = None   # non-embedding cache dimensions (e.g. top_k) — must match exactly


class SemanticResponseCache:
    """Embedding-similarity response cache. Sub-ms hits for repeated/similar queries.
    Thread-safe: the harness is shared across concurrent API requests, and an unlocked
    scan racing an eviction can map best_idx onto a DIFFERENT entry -> wrong answer."""

    def __init__(self, threshold: float = 0.92, max_size: int = 256,
                 ttl_s: float = 900.0):
        self.threshold = threshold
        self.max_size = max_size
        self.ttl_s = ttl_s     # an answer cached during a DEGRADED window (rerank timeout
        self._entries: list = []   # -> un-reranked evidence) must not be served forever
        self._lock = threading.Lock()

    def get(self, query_vec: list, key: object = None) -> Optional[object]:
        cutoff = time.monotonic() - self.ttl_s
        with self._lock:
            entries = [e for e in self._entries if e.key == key and e.timestamp >= cutoff]
        if not entries:
            return None
        mat = np.array([e.query_vec for e in entries], dtype=np.float32)
        qv = np.array(query_vec, dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1)
        qn = np.linalg.norm(qv)
        if qn < 1e-9:
            return None
        sims = (mat @ qv) / (norms * qn + 1e-9)
        best_idx = int(np.argmax(sims))
        if sims[best_idx] >= self.threshold:
            resp = entries[best_idx].response
            # deep copy: the cached response's citations list was aliased into every hit —
            # one caller mutating it would corrupt the cache for everyone after
            try:
                return resp.model_copy(deep=True)
            except AttributeError:
                return resp
        return None

    def put(self, query_vec: list, response: object, key: object = None) -> None:
        with self._lock:
            self._entries.append(_CacheEntry(
                query_vec=list(query_vec), response=response,
                timestamp=time.monotonic(), key=key))
            if len(self._entries) > self.max_size:
                self._entries.pop(0)

    @property
    def size(self) -> int:
        return len(self._entries)


class SessionContext:
    """Per-session conversation history for context-aware query expansion (R5c). Thread-safe."""

    def __init__(self, max_turns: int = 5, max_sessions: int = 2048):
        self.max_turns = max_turns
        self.max_sessions = max_sessions   # every novel session_id allocates an entry; the
        self._sessions: dict = {}          # endpoint is public -> bound it (LRU eviction)
        self._lock = threading.Lock()

    def add_turn(self, session_id: str, query: str, answer: str) -> None:
        with self._lock:
            if session_id in self._sessions:
                turns = self._sessions.pop(session_id)     # re-insert -> LRU recency
            else:
                turns = []
            self._sessions[session_id] = turns             # dicts preserve insertion order
            turns.append({"q": query, "a": answer})
            if len(turns) > self.max_turns:
                turns.pop(0)
            while len(self._sessions) > self.max_sessions:
                self._sessions.pop(next(iter(self._sessions)))   # evict least-recent

    def rewrite(self, session_id: str, query: str) -> str:
        """Expand the query with the last turn for topic continuity and coreference."""
        with self._lock:
            turns = self._sessions.get(session_id)
            last = turns[-1] if turns else None
        if last is None:
            return query
        return f"{last['q']} {query}"

    def has_history(self, session_id: str) -> bool:
        with self._lock:
            return bool(self._sessions.get(session_id))

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
