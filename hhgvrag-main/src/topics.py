"""
topics.py — post-embedding topic classification for chunk metadata.

After chunks are embedded, cluster them by dense vector similarity and assign a topic
label to each chunk. The label goes into the Qdrant payload as metadata, enabling
topic-level filtering at query time (router can narrow retrieval to a subject area).

Two backends:
- KeywordTopicClassifier: fast, no model, assigns topic from dominant keywords in each
  cluster. CPU-testable.
- EmbeddingTopicClassifier: clusters pre-computed dense embeddings (k-means), then labels
  each cluster by its most representative terms. Runs at index-build time on Modal.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Sequence

import numpy as np

_WORD = re.compile(r"\w+", re.UNICODE)
_STOP = set(
    "a an the of to in on for and or is are was were be been being this that these those "
    "it its as at by with from into about over under how what when where who whom which why "
    "do does did can could should would will may might i you he she they we me my your "
    "has have had not no also been very more most some any all each such than other but "
    "so if then because while after before between both through during only own same up down".split()
)

TOPIC_SEEDS = {
    "technology": {"software", "computer", "algorithm", "programming", "code", "system",
                   "data", "digital", "internet", "network", "server", "api", "database",
                   "machine", "learning", "model", "neural", "artificial", "intelligence"},
    "science": {"research", "study", "experiment", "theory", "scientific", "biology",
                "physics", "chemistry", "molecular", "cell", "energy", "quantum",
                "particle", "organism", "evolution", "genetic"},
    "geography": {"country", "city", "region", "state", "province", "coast", "river",
                  "mountain", "island", "ocean", "lake", "forest", "climate", "weather",
                  "border", "population", "capital", "territory"},
    "history": {"century", "war", "empire", "king", "dynasty", "ancient", "colonial",
                "independence", "revolution", "historical", "civilization", "era", "period"},
    "culture": {"festival", "tradition", "religion", "temple", "music", "dance", "art",
                "cuisine", "language", "literature", "heritage", "ceremony", "ritual"},
    "health": {"disease", "treatment", "medical", "health", "hospital", "patient",
               "symptom", "drug", "therapy", "clinical", "doctor", "infection"},
    "business": {"company", "market", "industry", "trade", "economic", "financial",
                 "investment", "commerce", "growth", "revenue", "production", "export"},
    "travel": {"tourism", "tourist", "beach", "hotel", "destination", "travel",
               "visit", "attraction", "resort", "heritage", "monument", "sightseeing"},
}


def _extract_keywords(text: str, n: int = 20) -> list:
    words = [w.lower() for w in _WORD.findall(text) if len(w) > 2]
    return [w for w, _ in Counter(w for w in words if w not in _STOP).most_common(n)]


class KeywordTopicClassifier:
    """Assign a topic label to each chunk by keyword overlap with seed vocabularies."""

    def __init__(self, seeds: dict = None):
        self.seeds = seeds or TOPIC_SEEDS

    def classify_one(self, text: str) -> str:
        words = set(w.lower() for w in _WORD.findall(text) if len(w) > 2)
        best, best_score = "general", 0
        for topic, seed in self.seeds.items():
            score = len(words & seed)
            if score > best_score:
                best, best_score = topic, score
        return best

    def classify_batch(self, chunks: Sequence, embeddings=None) -> list:
        return [self.classify_one(c.text) for c in chunks]


class EmbeddingTopicClassifier:
    """Cluster pre-computed embeddings, then label each cluster by its top keywords."""

    def __init__(self, n_topics: int = 8, keyword_fallback: bool = True):
        self.n_topics = n_topics
        self.keyword_clf = KeywordTopicClassifier() if keyword_fallback else None

    def classify_batch(self, chunks: Sequence, embeddings: Sequence) -> list:
        n = len(chunks)
        if n <= self.n_topics:
            if self.keyword_clf:
                return self.keyword_clf.classify_batch(chunks)
            return ["general"] * n

        vecs = np.array([e.dense if hasattr(e, "dense") else list(e) for e in embeddings],
                        dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms < 1e-9, 1.0, norms)
        vecs = vecs / norms

        k = min(self.n_topics, n)
        seeds = [0]
        for _ in range(k - 1):
            sims = vecs @ vecs[seeds].T
            closest = sims.max(axis=1)
            nxt = int(np.argmin(closest))
            if nxt in seeds:
                break
            seeds.append(nxt)
        centers = vecs[seeds].copy()

        for _ in range(12):
            sims = vecs @ centers.T
            labels = sims.argmax(axis=1)
            for j in range(len(centers)):
                members = vecs[labels == j]
                if len(members):
                    c = members.mean(axis=0)
                    norm = np.linalg.norm(c)
                    centers[j] = c / norm if norm > 1e-9 else c

        labels = (vecs @ centers.T).argmax(axis=1)

        cluster_names = {}
        for cluster_id in range(len(centers)):
            member_indices = [i for i, la in enumerate(labels) if la == cluster_id]
            if not member_indices:
                cluster_names[cluster_id] = "general"
                continue
            combined = " ".join(chunks[i].text for i in member_indices[:50])
            if self.keyword_clf:
                cluster_names[cluster_id] = self.keyword_clf.classify_one(combined)
            else:
                kws = _extract_keywords(combined, 3)
                cluster_names[cluster_id] = "_".join(kws) if kws else "general"

        return [cluster_names[int(la)] for la in labels]


def apply_topic_labels(chunks: Sequence, embeddings: Sequence,
                       classifier=None) -> Sequence:
    """Mutate chunk.extra['topic'] in place. Returns the same chunks list."""
    clf = classifier or KeywordTopicClassifier()
    topics = clf.classify_batch(chunks, embeddings)
    topic_counts = Counter(topics)
    for chunk, topic in zip(chunks, topics):
        chunk.extra = {**chunk.extra, "topic": topic}
    print(f"  topic classification: {dict(topic_counts)}")
    return chunks
