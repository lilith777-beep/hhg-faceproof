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
    with pytest.raises(ValueError, match="finite"):
        index.add(["infinite"], [np.array([np.inf, 1.0], dtype=np.float32)])


def test_exact_index_resolves_limit_boundary_ties_by_insertion_order() -> None:
    index = ExactCosineIndex(2)
    index.add(
        ["first", "second", "third"],
        [np.array([1.0, 0.0], dtype=np.float32) for _ in range(3)],
    )
    assert [hit.key for hit in index.search(np.array([1.0, 0.0]), limit=2)] == [
        "first",
        "second",
    ]


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


def test_media_index_multiquery_oracle_retains_responsible_occurrence() -> None:
    index = ExactMediaIndex(2)
    vectors = [
        np.array([1.0, 0.0]),
        np.array([1.0, 0.0]),
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([-1.0, 0.0]),
        np.array([0.0, -1.0]),
    ]
    media = ["z", "a", "z", "b", "a", "c"]
    queries = [np.array([1.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    index.add(media, vectors, [{"row": row} for row in range(len(vectors))])

    hits = index.search(queries, limit=10)

    query_matrix = np.stack(queries).astype(np.float64)
    query_matrix /= np.linalg.norm(query_matrix, axis=1, keepdims=True)
    vector_matrix = np.stack(vectors).astype(np.float64)
    vector_matrix /= np.linalg.norm(vector_matrix, axis=1, keepdims=True)
    scores = query_matrix @ vector_matrix.T
    oracle: dict[str, tuple[float, int, int]] = {}
    for query_index in range(len(queries)):
        for vector_index, media_key in enumerate(media):
            candidate = (float(scores[query_index, vector_index]), vector_index, query_index)
            current = oracle.get(media_key)
            if current is None or (candidate[0], -query_index, -vector_index) > (
                current[0],
                -current[2],
                -current[1],
            ):
                oracle[media_key] = candidate
    expected = sorted(oracle.items(), key=lambda item: (-item[1][0], item[0], item[1][1]))
    assert [hit.media_key for hit in hits] == [item[0] for item in expected]
    for hit, (_, (score, vector_index, query_index)) in zip(hits, expected, strict=True):
        assert hit.score == pytest.approx(score, abs=1e-6)
        assert (hit.vector_index, hit.query_index) == (vector_index, query_index)
        assert hit.metadata == {"row": vector_index}
    assert index.index.index.ntotal == len(vectors)
    assert index.index.keys == []


def test_media_index_batches_queries_in_one_faiss_call() -> None:
    index = ExactMediaIndex(2)
    index.add(
        [f"media-{row}" for row in range(4)],
        [np.array([row + 1.0, 1.0]) for row in range(4)],
        [{"row": row} for row in range(4)],
    )

    class CountingIndex:
        def __init__(self, delegate: object) -> None:
            self.delegate = delegate
            self.calls: list[tuple[int, int]] = []

        def search(self, queries: np.ndarray, limit: int) -> tuple[np.ndarray, np.ndarray]:
            self.calls.append((len(queries), limit))
            return self.delegate.search(queries, limit)

    counting = CountingIndex(index.index.index)
    index.index.index = counting
    hits = index.search([np.array([1.0, row + 1.0]) for row in range(8)], limit=2)
    assert len(hits) == 2
    assert counting.calls == [(8, 4)]


def test_media_index_add_is_atomic_when_metadata_is_invalid() -> None:
    index = ExactMediaIndex(2)
    with pytest.raises(TypeError):
        index.add(["media"], [np.ones(2)], [object()])  # type: ignore[list-item]
    assert index.index.index.ntotal == 0
    assert index.media_keys == []
    assert index.metadata == []
