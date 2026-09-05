from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .errors import FaceInputError


@dataclass(frozen=True, slots=True)
class VectorHit:
    key: str
    score: float
    rank: int


@dataclass(frozen=True, slots=True)
class MediaVectorHit:
    media_key: str
    score: float
    rank: int
    vector_index: int
    query_index: int
    metadata: dict[str, object]


class ExactCosineIndex:
    """Exact FAISS inner-product search over normalized float32 descriptors."""

    backend = "faiss.IndexFlatIP"

    def __init__(self, dimensions: int) -> None:
        try:
            import faiss
        except ImportError as exc:
            raise FaceInputError(
                "FAISS is required for vector retrieval; run pip install -e '.[dev,vision]'"
            ) from exc
        if dimensions <= 0:
            raise ValueError("vector dimensions must be positive")
        self.faiss = faiss
        self.dimensions = dimensions
        self.index = faiss.IndexFlatIP(dimensions)
        self.keys: list[str] = []

    def add(self, keys: list[str], vectors: list[np.ndarray]) -> None:
        if len(keys) != len(vectors):
            raise ValueError("keys and vectors must have equal length")
        if not vectors:
            return
        matrix = self._matrix(vectors)
        self.faiss.normalize_L2(matrix)
        self.index.add(matrix)
        self.keys.extend(keys)

    def search(self, query: np.ndarray, *, limit: int) -> list[VectorHit]:
        if self.index.ntotal == 0 or limit <= 0:
            return []
        matrix = self._matrix([query])
        self.faiss.normalize_L2(matrix)
        scores, indices = self.index.search(matrix, min(limit, self.index.ntotal))
        return [
            VectorHit(self.keys[int(index)], float(score), rank)
            for rank, (score, index) in enumerate(zip(scores[0], indices[0], strict=True), start=1)
            if index >= 0
        ]

    def _matrix(self, vectors: list[np.ndarray]) -> np.ndarray:
        matrix = np.ascontiguousarray(np.stack(vectors), dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != self.dimensions:
            raise ValueError(
                f"expected vectors with dimension {self.dimensions}, got {matrix.shape}"
            )
        norms = np.linalg.norm(matrix, axis=1)
        if np.any(~np.isfinite(matrix)) or np.any(norms <= 0):
            raise ValueError("vectors must contain finite, non-zero values")
        return matrix


class ExactMediaIndex:
    """Exact cosine retrieval aggregated by distinct opaque media identifiers.

    Every vector occurrence is retained. Search exhausts the exact IndexFlatIP result,
    keeps the best query/vector pair per media item, and applies a deterministic tie break.
    """

    backend = ExactCosineIndex.backend

    def __init__(self, dimensions: int) -> None:
        self.index = ExactCosineIndex(dimensions)
        self.media_keys: list[str] = []
        self.metadata: list[dict[str, object]] = []

    def add(
        self,
        media_keys: list[str],
        vectors: list[np.ndarray],
        metadata: list[dict[str, object]],
    ) -> None:
        if len(media_keys) != len(vectors) or len(vectors) != len(metadata):
            raise ValueError("media keys, vectors, and metadata must have equal length")
        start = len(self.media_keys)
        self.index.add([str(index) for index in range(start, start + len(vectors))], vectors)
        self.media_keys.extend(str(value) for value in media_keys)
        self.metadata.extend(dict(value) for value in metadata)

    def search(self, queries: list[np.ndarray], *, limit: int) -> list[MediaVectorHit]:
        if limit <= 0 or not queries or not self.media_keys:
            return []
        best: dict[str, tuple[float, int, int]] = {}
        for query_index, query in enumerate(queries):
            for hit in self.index.search(query, limit=len(self.media_keys)):
                vector_index = int(hit.key)
                media_key = self.media_keys[vector_index]
                candidate = (hit.score, -query_index, -vector_index)
                current = best.get(media_key)
                if current is None or candidate > (current[0], -current[2], -current[1]):
                    best[media_key] = (hit.score, vector_index, query_index)
        ordered = sorted(best.items(), key=lambda item: (-item[1][0], item[0], item[1][1]))
        return [
            MediaVectorHit(
                media_key=media_key,
                score=score,
                rank=rank,
                vector_index=vector_index,
                query_index=query_index,
                metadata=self.metadata[vector_index],
            )
            for rank, (media_key, (score, vector_index, query_index)) in enumerate(
                ordered[:limit], start=1
            )
        ]
