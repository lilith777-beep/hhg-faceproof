from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .errors import FaceInputError


@dataclass(frozen=True, slots=True)
class VectorHit:
    key: str
    score: float
    rank: int


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
