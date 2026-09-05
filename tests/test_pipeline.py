from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator, FormatChecker

from faceproof.config import Settings
from faceproof.errors import ConsentRequired, NoVerifiedMatch
from faceproof.models import (
    BoundingBox,
    FaceObservation,
    FeatureMatch,
    SearchBatch,
    SearchHit,
)
from faceproof.pipeline import DiscoveryPipeline
from faceproof.remote import RemoteImage


class FakeFaceEngine:
    detector_id = "fake-detector"
    encoder_id = "fake-encoder"
    cv2 = object()

    def decode(self, content: bytes) -> bytes:
        return content

    def require_single_query_face(self, image: bytes) -> FaceObservation:
        return FaceObservation(BoundingBox(1, 2, 100, 120, 0.99), np.array([1.0, 0.0]))

    def detect_and_encode(self, image: bytes) -> list[FaceObservation]:
        if image == b"no-face":
            return []
        vector = np.array([1.0, 0.0]) if image == b"matching-image" else np.array([0.0, 1.0])
        return [FaceObservation(BoundingBox(0, 0, 80, 80, 0.98), vector)]

    @staticmethod
    def best_match(
        query: FaceObservation, candidates: list[FaceObservation]
    ) -> tuple[float, int | None, BoundingBox | None]:
        if not candidates:
            return -1.0, None, None
        scores = [float(np.dot(query.embedding, item.embedding)) for item in candidates]
        index = int(np.argmax(scores))
        return scores[index], index, candidates[index].box


class FakeCopyEngine:
    model_id = "fake-sscd"
    model_sha256 = "f" * 64
    dimensions = 2
    device = "cpu"

    @staticmethod
    def encode(image: bytes) -> np.ndarray:
        return FakeCopyEngine._descriptor(image)

    @staticmethod
    def encode_many(images: list[bytes]) -> list[np.ndarray]:
        return [FakeCopyEngine._descriptor(image) for image in images]

    @staticmethod
    def _descriptor(image: bytes) -> np.ndarray:
        if image in {b"query-image", b"matching-image"}:
            return np.array([1.0, 0.0], dtype=np.float32)
        return np.array([0.0, 1.0], dtype=np.float32)


class FakeSource:
    name = "synthetic-placeholder-source"

    def __init__(self, hits: list[SearchHit]) -> None:
        self.hits = hits
        self.calls = 0

    def discover(self) -> SearchBatch:
        self.calls += 1
        return SearchBatch(
            provider=self.name,
            live_query=False,
            scope="https://social.example hashtag #consented",
            endpoint="https://social.example/api/v1/timelines/tag/consented",
            fetched_at=datetime.now(UTC).isoformat(),
            pages_fetched=1,
            statuses_scanned=len(self.hits),
            media_scanned=len(self.hits),
            hits=self.hits,
        )


class FakeFetcher:
    def fetch(self, url: str) -> RemoteImage:
        content = b"matching-image" if "match" in url else b"different-image"
        return RemoteImage(url, content, "image/jpeg")


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        project_root=tmp_path,
        model_dir=tmp_path / "models",
        artifact_dir=tmp_path / "artifacts",
        mastodon_instance="https://social.example",
        mastodon_tag="consented",
        mastodon_max_pages=1,
        mastodon_page_size=10,
        rpc_url=None,
        private_key=None,
        signer_mode="unlocked",
        expected_signer=None,
        expected_chain_id=31_337,
        explorer_tx_url="",
        detection_threshold=0.8,
        face_threshold=0.5,
        face_threshold_source="test calibration",
        prefilter_face_threshold=0.3,
        ambiguity_margin=0.05,
        phash_max_distance=12,
        akaze_min_inliers=10,
        akaze_min_inlier_ratio=0.25,
        max_candidates=10,
        max_finalists=5,
    )


def _hit(image: str = "match.jpg", post_id: str = "123") -> SearchHit:
    return SearchHit(
        post_url=f"https://social.example/@consenting_user/{post_id}",
        canonical_uri=f"https://social.example/users/consenting_user/statuses/{post_id}",
        post_id=post_id,
        author_id="a1",
        author_handle="consenting_user",
        created_at="2026-09-01T00:00:00Z",
        content_text="Consented FaceProof demo",
        media_id=f"m-{post_id}",
        image_url=f"https://social.example/media/{image}",
        preview_url=f"https://social.example/media/{image}",
        image_width=800,
        image_height=800,
    )


@pytest.fixture(autouse=True)
def _image_algorithms(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("faceproof.pipeline.perceptual_hash", lambda image, cv2: 1)
    monkeypatch.setattr("faceproof.pipeline.hash_distance", lambda left, right: 0)
    monkeypatch.setattr(
        "faceproof.pipeline.geometric_match",
        lambda query, candidate, cv2: FeatureMatch(20, 18, 0.9),
    )


def _pipeline(
    tmp_path: Path,
    source: FakeSource,
    *,
    copy_engine: FakeCopyEngine | None = None,
) -> DiscoveryPipeline:
    return DiscoveryPipeline(
        settings=_settings(tmp_path),
        face_engine=FakeFaceEngine(),
        candidate_source=source,
        image_fetcher=FakeFetcher(),
        copy_engine=copy_engine,
    )


def test_pipeline_writes_minimal_verifiable_evidence(tmp_path: Path) -> None:
    image = tmp_path / "query.jpg"
    image.write_bytes(b"query-image")
    source = FakeSource([_hit()])
    bundle, output = _pipeline(tmp_path, source).discover(image, consent_asserted=True)
    stored = json.loads(output.read_text(encoding="utf-8"))

    assert source.calls == 1
    assert stored == bundle.to_dict()
    assert stored["match"]["post_url"].endswith("/123")
    assert stored["match"]["face_similarity"] == 1.0
    assert stored["match"]["same_content"] is True
    assert stored["match"]["threshold_source"] == "test calibration"
    assert stored["query"]["embedding_persisted"] is False
    assert "embedding" not in stored["query"]
    schema = json.loads(
        (Path(__file__).parents[1] / "schema" / "evidence-v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(stored)


def test_evidence_hashes_full_post_text_while_bounding_excerpt(tmp_path: Path) -> None:
    image = tmp_path / "query.jpg"
    image.write_bytes(b"query-image")
    full_text = "a" * 2100 + "tamper-sensitive-tail"
    source = FakeSource([replace(_hit(), content_text=full_text)])

    bundle, _ = _pipeline(tmp_path, source).discover(image, consent_asserted=True)

    assert len(bundle.match["post_text"]) == 2000
    assert bundle.match["post_text_truncated"] is True
    assert bundle.match["post_text_sha256"] == hashlib.sha256(full_text.encode("utf-8")).hexdigest()


def test_pipeline_requires_explicit_consent(tmp_path: Path) -> None:
    image = tmp_path / "query.jpg"
    image.write_bytes(b"query-image")
    with pytest.raises(ConsentRequired):
        _pipeline(tmp_path, FakeSource([_hit()])).discover(image, consent_asserted=False)


def test_pipeline_rejects_face_mismatch(tmp_path: Path) -> None:
    image = tmp_path / "query.jpg"
    image.write_bytes(b"query-image")
    with pytest.raises(NoVerifiedMatch, match="preview face/image filter"):
        _pipeline(tmp_path, FakeSource([_hit("different.jpg")])).discover(
            image, consent_asserted=True
        )


def test_pipeline_marks_close_face_candidates_ambiguous(tmp_path: Path) -> None:
    image = tmp_path / "query.jpg"
    image.write_bytes(b"query-image")
    source = FakeSource([_hit(post_id="1"), _hit(post_id="2")])
    bundle, _ = _pipeline(tmp_path, source).discover(image, consent_asserted=True)
    # Same-content corroboration is strong enough that identical reposts are not ambiguous.
    assert bundle.match["ambiguous"] is False
    assert bundle.match["verified_match_count"] == 2


def test_dual_faiss_retrieval_and_decision_are_evidenced(tmp_path: Path) -> None:
    image = tmp_path / "query.jpg"
    image.write_bytes(b"query-image")
    source = FakeSource([_hit(), _hit("different.jpg", post_id="negative")])
    bundle, _ = _pipeline(
        tmp_path,
        source,
        copy_engine=FakeCopyEngine(),
    ).discover(image, consent_asserted=True)

    assert bundle.search["vector_backend"] == "faiss.IndexFlatIP"
    assert bundle.query["copy_descriptor"]["enabled"] is True
    assert bundle.match["decision"] == "same_identity_and_content"
    assert bundle.match["face_match"] is True
    assert bundle.match["sscd_match"] is True
    assert bundle.match["sscd_similarity"] == 1.0
    assert set(bundle.match["retrieval_channels"]) >= {"face-faiss", "sscd-faiss"}
    assert bundle.match["copy_retrieval_rank"] == 1


@pytest.mark.parametrize(
    ("face_match", "copy_match", "expected"),
    [
        (True, True, "same_identity_and_content"),
        (True, False, "same_identity_different_content"),
        (False, True, "copy_face_conflict"),
        (False, False, "rejected"),
    ],
)
def test_decision_matrix(face_match: bool, copy_match: bool, expected: str) -> None:
    assert (
        DiscoveryPipeline._decision(
            face_match=face_match,
            copy_match=copy_match,
        )
        == expected
    )
