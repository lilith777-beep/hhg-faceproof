from __future__ import annotations

import numpy as np
import pytest

from faceproof.vector_index import ExactCosineIndex, ExactMediaIndex


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


def test_media_index_matches_numpy_oracle_and_deduplicates_media() -> None:
    index = ExactMediaIndex(2)
    vectors = [
        np.array([0.8, 0.2], dtype=np.float32),
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0], dtype=np.float32),
    ]
    index.add(
        ["opaque-b", "opaque-a", "opaque-b"],
        vectors,
        [{"face": 0}, {"face": 0}, {"face": 1}],
    )
    query = np.array([3.0, 0.0], dtype=np.float32)
    hits = index.search([query], limit=20)
    oracle = [
        float(query @ value / (np.linalg.norm(query) * np.linalg.norm(value)))
        for value in vectors
    ]
    assert [hit.media_key for hit in hits] == ["opaque-a", "opaque-b"]
    assert hits[0].score == pytest.approx(max(oracle[1:2]), abs=1e-6)
    assert hits[1].score == pytest.approx(max(oracle[0], oracle[2]), abs=1e-6)
    assert hits[1].metadata == {"face": 0}


def test_media_index_empty_ties_multiple_queries_and_invalid_inputs() -> None:
    empty = ExactMediaIndex(2)
    assert empty.search([np.ones(2, dtype=np.float32)], limit=10) == []

    index = ExactMediaIndex(2)
    index.add(
        ["b", "a"],
        [np.array([1.0, 0.0]), np.array([1.0, 0.0])],
        [{"row": 0}, {"row": 1}],
    )
    hits = index.search(
        [np.array([0.0, 1.0]), np.array([1.0, 0.0])], limit=50
    )
    assert [hit.media_key for hit in hits] == ["a", "b"]
    assert all(hit.query_index == 1 for hit in hits)
    with pytest.raises(ValueError, match="equal length"):
        index.add(["x"], [np.ones(2)], [])
    with pytest.raises(ValueError, match="finite"):
        index.search([np.array([np.nan, 1.0])], limit=1)
