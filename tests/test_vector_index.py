from __future__ import annotations

import numpy as np
import pytest

from faceproof.vector_index import ExactCosineIndex


def test_exact_faiss_cosine_ranking() -> None:
    index = ExactCosineIndex(2)
    index.add(
        ["same", "orthogonal", "opposite"],
        [
            np.array([2.0, 0.0], dtype=np.float32),
            np.array([0.0, 4.0], dtype=np.float32),
            np.array([-3.0, 0.0], dtype=np.float32),
        ],
    )
    hits = index.search(np.array([1.0, 0.0], dtype=np.float32), limit=3)
    assert [hit.key for hit in hits] == ["same", "orthogonal", "opposite"]
    assert hits[0].score == pytest.approx(1.0)
    assert hits[1].score == pytest.approx(0.0)
    assert hits[2].score == pytest.approx(-1.0)
    assert index.backend == "faiss.IndexFlatIP"


def test_exact_index_rejects_invalid_vectors() -> None:
    index = ExactCosineIndex(2)
    with pytest.raises(ValueError, match="dimension"):
        index.add(["bad"], [np.array([1.0, 2.0, 3.0], dtype=np.float32)])
    with pytest.raises(ValueError, match="non-zero"):
        index.add(["zero"], [np.zeros(2, dtype=np.float32)])
