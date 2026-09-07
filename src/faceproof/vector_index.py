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
        self._add_vectors(vectors)
        self.keys.extend(keys)

    def _add_vectors(self, vectors: list[np.ndarray]) -> None:
        matrix = self._matrix(vectors)
        self.faiss.normalize_L2(matrix)
        self.index.add(matrix)

    def search(self, query: np.ndarray, *, limit: int) -> list[VectorHit]:
        if self.index.ntotal == 0 or limit <= 0:
            return []
        matrix = self._matrix([query])
        self.faiss.normalize_L2(matrix)
        requested = min(limit, self.index.ntotal)
        # Ask for one boundary row so a tie at ``limit`` can be resolved by insertion
        # order rather than by FAISS's implementation-specific heap order. Expand only
        # when the boundary is tied; ordinary limited searches retain their small result.
        fetched = min(requested + 1, self.index.ntotal)
        while True:
            scores, indices = self.index.search(matrix, fetched)
            if fetched == self.index.ntotal or scores[0, requested] < scores[0, requested - 1]:
                break
            fetched = min(self.index.ntotal, max(fetched + 1, fetched * 2))
        rows = sorted(
            (
                (float(score), int(index))
                for score, index in zip(scores[0], indices[0], strict=True)
                if index >= 0
            ),
            key=lambda row: (-row[0], row[1]),
        )[:requested]
        return [
            VectorHit(self.keys[index], score, rank)
            for rank, (score, index) in enumerate(rows, start=1)
        ]

    def _matrix(self, vectors: list[np.ndarray]) -> np.ndarray:
        matrix = np.ascontiguousarray(np.stack(vectors), dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != self.dimensions:
            raise ValueError(
                f"expected vectors with dimension {self.dimensions}, got {matrix.shape}"
            )
        norms = np.linalg.norm(matrix, axis=1)
        if np.any(~np.isfinite(matrix)) or np.any(~np.isfinite(norms)) or np.any(norms <= 0):
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
        # Materialize caller-owned values before mutating FAISS so conversion failures
        # cannot leave the parallel provenance arrays out of sync with the index.
        stored_media_keys = [str(value) for value in media_keys]
        stored_metadata = [dict(value) for value in metadata]
        if vectors:
            # Media search addresses FAISS rows directly, so retaining a second list of
            # decimal-string row keys would only duplicate per-occurrence memory.
            self.index._add_vectors(vectors)
        self.media_keys.extend(stored_media_keys)
        self.metadata.extend(stored_metadata)

    def search(self, queries: list[np.ndarray], *, limit: int) -> list[MediaVectorHit]:
        if limit <= 0 or not queries or not self.media_keys:
            return []
        matrix = self.index._matrix(queries)
        self.index.faiss.normalize_L2(matrix)
        best: dict[str, tuple[float, int, int]] = {}
        vector_count = len(self.media_keys)
        # FAISS returns float32 scores and int64 indices. Bound the temporary result
        # arrays while still amortizing calls across multiple query descriptors.
        rows_per_batch = max(1, (16 * 1024 * 1024) // (vector_count * 12))
        for start in range(0, len(queries), rows_per_batch):
            stop = min(start + rows_per_batch, len(queries))
            scores, indices = self.index.index.search(matrix[start:stop], vector_count)
            for offset, (query_scores, query_indices) in enumerate(
                zip(scores, indices, strict=True)
            ):
                query_index = start + offset
                for score, vector_index_raw in zip(
                    query_scores, query_indices, strict=True
                ):
                    vector_index = int(vector_index_raw)
                    if vector_index < 0:
                        continue
                    score = float(score)
                    media_key = self.media_keys[vector_index]
                    candidate = (score, -query_index, -vector_index)
                    current = best.get(media_key)
                    if current is None or candidate > (current[0], -current[2], -current[1]):
                        best[media_key] = (score, vector_index, query_index)
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
