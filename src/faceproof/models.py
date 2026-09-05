from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class BoundingBox:
    x: int
    y: int
    width: int
    height: int
    confidence: float


@dataclass(frozen=True, slots=True)
class FaceObservation:
    box: BoundingBox
    embedding: Any = field(repr=False, compare=False)
    landmarks: tuple[tuple[float, float], ...] = ()


@dataclass(frozen=True, slots=True)
class SearchHit:
    post_url: str
    canonical_uri: str
    post_id: str
    author_id: str
    author_handle: str
    created_at: str
    content_text: str
    media_id: str
    image_url: str
    preview_url: str
    image_width: int | None
    image_height: int | None
    wrapper_post_id: str | None = None
    wrapper_canonical_uri: str | None = None

    @property
    def occurrence_key(self) -> str:
        return hashlib.sha256(
            self.post_id.encode("utf-8") + b"\x00" + self.media_id.encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class SearchBatch:
    provider: str
    live_query: bool
    scope: str
    endpoint: str
    fetched_at: str
    pages_fetched: int
    statuses_scanned: int
    media_scanned: int
    hits: list[SearchHit]


@dataclass(frozen=True, slots=True)
class FeatureMatch:
    good_matches: int
    inliers: int
    inlier_ratio: float
    reprojection_error: float | None = None
    source_coverage: float = 0.0
    candidate_coverage: float = 0.0
    homography: tuple[float, ...] | None = None


@dataclass(frozen=True, slots=True)
class VerifiedMatch:
    hit: SearchHit
    face_similarity: float
    candidate_face_count: int
    matched_face_index: int | None
    matched_face_box: BoundingBox | None
    candidate_image_sha256: str
    candidate_media_type: str
    candidate_retrieved_url: str
    exact_image: bool
    perceptual_hash_distance: int
    feature_match: FeatureMatch
    same_content: bool
    face_match: bool
    sscd_similarity: float | None
    sscd_threshold: float | None
    sscd_match: bool
    decision: str
    retrieval_channels: tuple[str, ...]
    face_retrieval_rank: int | None
    copy_retrieval_rank: int | None
    phash_retrieval_rank: int | None
    face_state: str
    copy_state: str
    face_assessable: bool
    identity_reference_index: int | None
    copy_reference_index: int | None
    candidate_quality: dict[str, Any] | None
    candidate_consent_ref: str | None
    source_candidate_association: dict[str, Any]
    candidate_fetched_at: str | None


@dataclass(frozen=True, slots=True)
class EvidenceBundle:
    schema: str
    status: str
    created_at: str
    run_id: str
    consent: dict[str, Any]
    query: dict[str, Any]
    search: dict[str, Any]
    match: dict[str, Any]
    claims: dict[str, Any]
    provenance: dict[str, Any]
    human_review: dict[str, Any] | None
    nonce_hex: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AnchorReceipt:
    schema: str
    evidence_sha256: str
    payload_hex: str
    transaction_hash: str
    chain_id: int
    block_number: int
    block_hash: str
    sender: str
    recipient: str
    contract_code_sha256: str
    commitment_log_topic: str
    explorer_url: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class VerificationReport:
    verified: bool
    evidence_sha256: str
    transaction_hash: str
    chain_id: int
    block_number: int
    confirmations: int
    checks: dict[str, bool]
    reproduction: dict[str, str] = field(default_factory=dict)
