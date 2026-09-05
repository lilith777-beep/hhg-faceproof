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
from faceproof.errors import ConsentRequired
from faceproof.models import (
    BoundingBox,
    FaceObservation,
    FeatureMatch,
    SearchBatch,
    SearchHit,
)
from faceproof.pipeline import DiscoveryPipeline
from faceproof.policy import DecisionPolicy
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
        if image in {b"query-image", b"matching-image", b"no-face"}:
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
        if "no-face" in url:
            content = b"no-face"
        else:
            content = b"matching-image" if "match" in url else b"different-image"
        return RemoteImage(url, content, "image/jpeg")


class PreviewFailureFetcher(FakeFetcher):
    def fetch(self, url: str) -> RemoteImage:
        if "preview-broken" in url:
            raise OSError("preview unavailable")
        return RemoteImage(url, b"matching-image", "image/jpeg")


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
        policy=DecisionPolicy.synthetic(face_threshold=0.5, copy_threshold=0.5),
    )


def test_pipeline_writes_minimal_verifiable_evidence(tmp_path: Path) -> None:
    image = tmp_path / "query.jpg"
    image.write_bytes(b"query-image")
    source = FakeSource([_hit()])
    bundle, output = _pipeline(tmp_path, source).discover(
        image, copy_reference_paths=[image], consent_asserted=True
    )
    stored = json.loads(output.read_text(encoding="utf-8"))

    assert source.calls == 1
    assert stored == bundle.to_dict()
    assert stored["match"]["post_url"].endswith("/123")
    assert stored["match"]["face_similarity"] == 1.0
    assert stored["match"]["same_content"] is False
    assert stored["match"]["threshold_source"] == "synthetic-test-policy"
    assert stored["query"]["embedding_persisted"] is False
    assert "embedding" not in stored["query"]
    schema = json.loads(
        (Path(__file__).parents[1] / "schema" / "evidence-v2.schema.json").read_text(
            encoding="utf-8"
        )
    )
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(stored)


def test_evidence_hashes_full_post_text_while_bounding_excerpt(tmp_path: Path) -> None:
    image = tmp_path / "query.jpg"
    image.write_bytes(b"query-image")
    full_text = "a" * 2100 + "tamper-sensitive-tail"
    source = FakeSource([replace(_hit(), content_text=full_text)])

    bundle, _ = _pipeline(tmp_path, source).discover(
        image, copy_reference_paths=[image], consent_asserted=True
    )

    assert len(bundle.match["post_text"]) == 2000
    assert bundle.match["post_text_truncated"] is True
    assert bundle.match["post_text_sha256"] == hashlib.sha256(full_text.encode("utf-8")).hexdigest()


def test_pipeline_requires_explicit_consent(tmp_path: Path) -> None:
    image = tmp_path / "query.jpg"
    image.write_bytes(b"query-image")
    with pytest.raises(ConsentRequired):
        _pipeline(tmp_path, FakeSource([_hit()])).discover(
            image, copy_reference_paths=[image], consent_asserted=False
        )


def test_pipeline_rejects_face_mismatch(tmp_path: Path) -> None:
    image = tmp_path / "query.jpg"
    image.write_bytes(b"query-image")
    bundle, _ = _pipeline(tmp_path, FakeSource([_hit("different.jpg")])).discover(
        image, copy_reference_paths=[image], consent_asserted=True
    )
    assert bundle.match["decision"] == "no_accepted_match"


def test_pipeline_marks_close_face_candidates_ambiguous(tmp_path: Path) -> None:
    image = tmp_path / "query.jpg"
    image.write_bytes(b"query-image")
    source = FakeSource([_hit(post_id="1"), _hit(post_id="2")])
    bundle, _ = _pipeline(tmp_path, source).discover(
        image, copy_reference_paths=[image], consent_asserted=True
    )
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
    ).discover(image, copy_reference_paths=[image], consent_asserted=True)

    assert bundle.search["vector_backend"] == "faiss.IndexFlatIP"
    assert bundle.query["copy_descriptor"]["enabled"] is True
    assert bundle.match["decision"] == "both_supported_pending_human_confirmation"
    assert bundle.match["face_match"] is True
    assert bundle.match["sscd_match"] is True
    assert bundle.match["sscd_similarity"] == 1.0
    assert set(bundle.match["retrieval_channels"]) >= {"face-faiss", "sscd-faiss"}
    assert bundle.match["copy_retrieval_rank"] == 1


def test_copy_evidence_survives_unassessable_face(tmp_path: Path) -> None:
    image = tmp_path / "query.jpg"
    image.write_bytes(b"query-image")
    bundle, _ = _pipeline(
        tmp_path,
        FakeSource([_hit("no-face.jpg")]),
        copy_engine=FakeCopyEngine(),
    ).discover(image, copy_reference_paths=[image], consent_asserted=True)
    assert bundle.match["face_state"] == "UNASSESSABLE"
    assert bundle.match["copy_state"] == "SUPPORTED"
    assert bundle.match["decision"] == "copy_only_identity_unknown"


def test_exhaustive_mode_recovers_original_after_preview_failure(tmp_path: Path) -> None:
    image = tmp_path / "query.jpg"
    image.write_bytes(b"query-image")
    hit = replace(
        _hit(),
        image_url="https://social.example/media/match-full.jpg",
        preview_url="https://social.example/media/preview-broken.jpg",
    )
    pipeline = _pipeline(tmp_path, FakeSource([hit]), copy_engine=FakeCopyEngine())
    pipeline.image_fetcher = PreviewFailureFetcher()
    bundle, _ = pipeline.discover(
        image, copy_reference_paths=[image], consent_asserted=True
    )
    assert bundle.match["post_id"] == "123"
    assert "full-resolution-recovery" in bundle.match["retrieval_channels"]
    assert bundle.search["failures"][0]["media_id"] == hit.media_id
