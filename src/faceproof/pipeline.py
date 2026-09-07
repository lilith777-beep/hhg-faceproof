from __future__ import annotations

import hashlib
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from . import __version__
from .config import Settings
from .copydetect import CopyDescriptorEngine
from .dataset import DatasetManifest
from .errors import BlockedState, ConsentRequired, FaceInputError, NoVerifiedMatch
from .face_analysis import (
    PARSER_MODEL_ID,
    PARSER_SHA256,
    POSE_TEMPLATE_SHA256,
    AnalysisState,
    FaceQualityAssessor,
)
from .faces import FaceEngine
from .image_match import geometric_match, hash_distance, perceptual_hash
from .integrity import SCHEMA, evidence_digest, write_json
from .models import EvidenceBundle, FeatureMatch, SearchBatch, SearchHit, VerifiedMatch
from .policy import AxisState, DecisionPolicy
from .remote import RemoteImage, SafeImageFetcher
from .search import CandidateSource
from .vector_index import ExactCosineIndex, ExactMediaIndex

ProgressCallback = Callable[[str], None]


@dataclass(slots=True)
class _PreviewCandidate:
    hit: SearchHit
    remote: RemoteImage
    image: Any
    faces: list[Any]
    face_similarity: float
    phash_distance: int
    matched_index: int | None
    matched_box: Any
    copy_descriptor: Any = None
    copy_similarity: float | None = None
    retrieval_channels: set[str] = field(default_factory=set)
    face_retrieval_rank: int | None = None
    copy_retrieval_rank: int | None = None
    phash_retrieval_rank: int | None = None
    identity_reference_index: int | None = None
    copy_reference_index: int | None = None
    face_consent_ref: str | None = None
    face_processing_allowed: bool = False


class DiscoveryPipeline:
    def __init__(
        self,
        *,
        settings: Settings,
        face_engine: FaceEngine,
        candidate_source: CandidateSource,
        image_fetcher: SafeImageFetcher,
        copy_engine: CopyDescriptorEngine | None = None,
        consent_manifest: DatasetManifest | None = None,
        policy: DecisionPolicy | None = None,
        quality_assessor: FaceQualityAssessor | None = None,
        progress: ProgressCallback | None = None,
    ) -> None:
        self.settings = settings
        self.face_engine = face_engine
        self.candidate_source = candidate_source
        self.image_fetcher = image_fetcher
        self.copy_engine = copy_engine
        self.consent_manifest = consent_manifest
        self.policy = policy or DecisionPolicy.review_only()
        self.quality_assessor = quality_assessor
        self.progress = progress or (lambda _: None)

    def discover(
        self,
        image_path: Path | list[Path] | tuple[Path, ...],
        *,
        copy_reference_paths: list[Path] | tuple[Path, ...] | None,
        enrollment_face_indices: list[int] | tuple[int, ...] | None = None,
        consent_asserted: bool,
        output_path: Path | None = None,
        exhaustive_full_resolution: bool = True,
    ) -> tuple[EvidenceBundle, Path]:
        pipeline_started = perf_counter()
        if not consent_asserted:
            raise ConsentRequired("explicit documented consent is required")
        if copy_reference_paths is None or not copy_reference_paths:
            raise FaceInputError(
                "copy_reference_images must be explicit; pass the same file explicitly "
                "if it has both roles"
            )
        enrollment_paths = [image_path] if isinstance(image_path, Path) else list(image_path)
        descriptor_cache: dict[str, Any] = {}
        face_cache: dict[str, list[Any]] = {}
        copy_paths = list(copy_reference_paths)
        selections = list(enrollment_face_indices or [])
        if selections and len(selections) != len(enrollment_paths):
            raise FaceInputError("provide exactly one --face-index per enrollment image")
        if not enrollment_paths:
            raise FaceInputError("at least one enrollment_face_image is required")
        if len(enrollment_paths) > 64 or len(copy_paths) > 64:
            raise FaceInputError("reference count exceeds the hard processing safety bound")
        if self.candidate_source.name == "mastodon-public-api" and self.consent_manifest is None:
            raise BlockedState(
                "BLOCKED_PERMISSION_MANIFEST",
                "live candidate faces require a consent manifest",
            )

        enrollment_rows = self._load_local_inputs(enrollment_paths, "enrollment_face")
        copy_rows = self._load_local_inputs(copy_paths, "copy_reference")
        self.progress("Encoding explicitly consented enrollment-face references")
        query_faces = [
            (
                self.face_engine.select_query_face(image, selections[index] if selections else None)
                if hasattr(self.face_engine, "select_query_face")
                else self.face_engine.require_single_query_face(image)
            )
            for index, (_, _, image, _) in enumerate(enrollment_rows)
        ]
        enrollment_qualities = (
            [
                self.quality_assessor.assess(row[2], query_faces[index])
                for index, row in enumerate(enrollment_rows)
            ]
            if self.quality_assessor is not None
            else []
        )
        quality_gate_active = (
            self.policy.pose_policy_validated and self.policy.visibility_policy_validated
        )
        enrollment_assessable = not quality_gate_active or not enrollment_qualities or all(
            quality.state == AnalysisState.ASSESSABLE for quality in enrollment_qualities
        )
        copy_images = [row[2] for row in copy_rows]
        copy_hashes = [row[3] for row in copy_rows]
        copy_phashes = [perceptual_hash(image, self.face_engine.cv2) for image in copy_images]
        query_copy_descriptors = (
            self._encode_copy_cached(copy_images, copy_hashes, descriptor_cache)
            if self.copy_engine is not None
            else []
        )

        self.progress("Enumerating the bounded public social-media scope")
        source_started = perf_counter()
        batch = self.candidate_source.discover()
        source_ms = (perf_counter() - source_started) * 1000
        if not batch.hits:
            raise NoVerifiedMatch(f"no public image posts were found in {batch.scope}")
        hits = batch.hits[: self.settings.max_candidates]
        policy_in_domain = (
            len(enrollment_paths) <= self.policy.max_enrollment_references
            and len(copy_paths) <= self.policy.max_copy_references
            and len(hits) <= self.policy.max_gallery_media
        )
        ledger = {
            hit.occurrence_key: {
                "post_id": hit.post_id,
                "media_id": hit.media_id,
                "disposition": "DISCOVERED",
                "detail": None,
            }
            for hit in hits
        }

        self.progress(f"Preview-processing {len(hits)} media candidates")
        preview_started = perf_counter()
        preview_rows, failures = self._download(hits, preview=True)
        for failure in failures:
            occurrence_key = failure.get("occurrence_key")
            if occurrence_key in ledger:
                ledger[occurrence_key].update(
                    disposition="PREVIEW_DOWNLOAD_ERROR", detail=failure["error"]
                )
        candidates: list[_PreviewCandidate] = []
        for hit, preview in preview_rows:
            try:
                preview_image = self.face_engine.decode(preview.content)
                consent_record = (
                    self.consent_manifest.face_record_for_hit(hit)
                    if self.consent_manifest is not None
                    else None
                )
                face_allowed = consent_record is not None or (
                    not batch.live_query
                    and self.policy.status == "SYNTHETIC_TEST_ONLY"
                    and self.consent_manifest is None
                )
                preview_faces = (
                    self._detect_faces_cached(
                        preview_image,
                        hashlib.sha256(preview.content).hexdigest(),
                        face_cache,
                    )
                    if face_allowed
                    else []
                )
                similarity, matched_index, matched_box, identity_ref = self._best_face_match(
                    query_faces, preview_faces
                )
                candidate_phash = perceptual_hash(preview_image, self.face_engine.cv2)
                distances = [hash_distance(value, candidate_phash) for value in copy_phashes]
                phash_distance = min(distances)
                copy_ref_index = distances.index(phash_distance)
                candidates.append(
                    _PreviewCandidate(
                        hit=hit,
                        remote=preview,
                        image=preview_image,
                        faces=preview_faces,
                        face_similarity=similarity,
                        phash_distance=phash_distance,
                        matched_index=matched_index,
                        matched_box=matched_box,
                        identity_reference_index=identity_ref,
                        copy_reference_index=copy_ref_index,
                        face_consent_ref=consent_record.consent_ref if consent_record else None,
                        face_processing_allowed=face_allowed,
                    )
                )
                if not face_allowed:
                    ledger[hit.occurrence_key]["disposition"] = (
                        "COPY_ONLY_FACE_CONSENT_INELIGIBLE"
                    )
            except FaceInputError as exc:
                ledger[hit.occurrence_key].update(
                    disposition="PREVIEW_ERROR", detail=type(exc).__name__
                )
                failures.append(self._failure(hit.preview_url, "preview_decode", exc, hit))

        descriptor_started = perf_counter()
        if self.copy_engine is not None and candidates:
            descriptors = self._encode_copy_cached(
                [item.image for item in candidates],
                [hashlib.sha256(item.remote.content).hexdigest() for item in candidates],
                descriptor_cache,
            )
            for item, descriptor in zip(candidates, descriptors, strict=True):
                item.copy_descriptor = descriptor
                scores = [float(query @ descriptor) for query in query_copy_descriptors]
                item.copy_similarity = max(scores)
                item.copy_reference_index = scores.index(item.copy_similarity)
        descriptor_ms = (perf_counter() - descriptor_started) * 1000

        retrieval_started = perf_counter()
        preview_shortlist = self._retrieve_candidates(
            query_faces=query_faces,
            query_copy_descriptors=query_copy_descriptors,
            candidates=candidates,
        )
        shortlist = candidates if exhaustive_full_resolution else preview_shortlist
        retrieval_ms = (perf_counter() - retrieval_started) * 1000
        preview_ms = (perf_counter() - preview_started) * 1000
        if not shortlist and not exhaustive_full_resolution:
            raise NoVerifiedMatch("no candidate survived bounded preview processing")
        selected_media = {item.hit.occurrence_key for item in shortlist}
        if not exhaustive_full_resolution:
            for item in candidates:
                if item.hit.occurrence_key not in selected_media:
                    ledger[item.hit.occurrence_key]["disposition"] = "PREVIEW_NOT_SELECTED"
        for item in shortlist:
            ledger[item.hit.occurrence_key]["disposition"] = "FULL_RESOLUTION_SELECTED"

        self.progress(f"Full-resolution verification of {len(shortlist)} finalists")
        final_started = perf_counter()
        preview_by_media = {item.hit.occurrence_key: item.remote for item in candidates}
        preview_cache = {item.hit.occurrence_key: item for item in candidates}
        finalist_hits = hits if exhaustive_full_resolution else [item.hit for item in shortlist]
        full_needed = [
            hit
            for hit in finalist_hits
            if hit.image_url != hit.preview_url or hit.occurrence_key not in preview_by_media
        ]
        full_rows, full_failures = self._download(full_needed, preview=False)
        failures.extend(full_failures)
        for failure in full_failures:
            occurrence_key = failure.get("occurrence_key")
            if occurrence_key in ledger:
                ledger[occurrence_key].update(
                    disposition="FULL_DOWNLOAD_ERROR", detail=failure["error"]
                )
        full_by_media = {hit.occurrence_key: remote for hit, remote in full_rows}

        decoded: list[dict[str, Any]] = []
        for hit in finalist_hits:
            remote = full_by_media.get(hit.occurrence_key) or preview_by_media.get(
                hit.occurrence_key
            )
            cached = preview_cache.get(hit.occurrence_key)
            if remote is None:
                continue
            try:
                if hit.image_url == hit.preview_url and cached is not None:
                    candidate_image = cached.image
                    candidate_faces = cached.faces
                    similarity = cached.face_similarity
                    matched_index = cached.matched_index
                    matched_box = cached.matched_box
                    identity_ref = cached.identity_reference_index
                    phash_distance = cached.phash_distance
                    copy_descriptor = cached.copy_descriptor
                else:
                    candidate_image = self.face_engine.decode(remote.content)
                    consent_record = (
                        self.consent_manifest.face_record_for_hit(hit)
                        if self.consent_manifest is not None
                        else None
                    )
                    face_allowed = consent_record is not None or (
                        not batch.live_query
                        and self.policy.status == "SYNTHETIC_TEST_ONLY"
                        and self.consent_manifest is None
                    )
                    candidate_faces = (
                        self._detect_faces_cached(
                            candidate_image,
                            hashlib.sha256(remote.content).hexdigest(),
                            face_cache,
                        )
                        if face_allowed
                        else []
                    )
                    similarity, matched_index, matched_box, identity_ref = self._best_face_match(
                        query_faces, candidate_faces
                    )
                    candidate_phash = perceptual_hash(candidate_image, self.face_engine.cv2)
                    phash_distance = min(
                        hash_distance(value, candidate_phash) for value in copy_phashes
                    )
                    copy_descriptor = None
                    copy_ref_index = min(
                        range(len(copy_phashes)),
                        key=lambda index: hash_distance(
                            copy_phashes[index], candidate_phash
                        ),
                    )
                    if cached is None:
                        cached = _PreviewCandidate(
                            hit=hit,
                            remote=remote,
                            image=candidate_image,
                            faces=candidate_faces,
                            face_similarity=similarity,
                            phash_distance=phash_distance,
                            matched_index=matched_index,
                            matched_box=matched_box,
                            identity_reference_index=identity_ref,
                            copy_reference_index=copy_ref_index,
                            face_consent_ref=(
                                consent_record.consent_ref if consent_record else None
                            ),
                            face_processing_allowed=face_allowed,
                            retrieval_channels={"full-resolution-recovery"},
                        )
                decoded.append(
                    {
                        "hit": hit,
                        "remote": remote,
                        "image": candidate_image,
                        "faces": candidate_faces,
                        "similarity": similarity,
                        "phash_distance": phash_distance,
                        "matched_index": matched_index,
                        "matched_box": matched_box,
                        "identity_reference_index": identity_ref,
                        "copy_descriptor": copy_descriptor,
                        "preview": cached,
                    }
                )
            except FaceInputError as exc:
                ledger[hit.occurrence_key].update(
                    disposition="FULL_DECODE_ERROR", detail=type(exc).__name__
                )
                failures.append(self._failure(remote.url, "final_decode", exc, hit))

        if self.copy_engine is not None:
            missing = [row for row in decoded if row["copy_descriptor"] is None]
            descriptors = self._encode_copy_cached(
                [row["image"] for row in missing],
                [hashlib.sha256(row["remote"].content).hexdigest() for row in missing],
                descriptor_cache,
            )
            for row, descriptor in zip(missing, descriptors, strict=True):
                row["copy_descriptor"] = descriptor

        assessed: list[VerifiedMatch] = []
        for row in decoded:
            hit = row["hit"]
            remote = row["remote"]
            candidate_image = row["image"]
            similarity = row["similarity"]
            preview = row["preview"]
            quality = None
            if row["matched_index"] is not None and self.quality_assessor is not None:
                quality = self.quality_assessor.assess(
                    candidate_image, row["faces"][row["matched_index"]]
                )
            face_assessable = bool(
                enrollment_assessable
                and preview.face_processing_allowed
                and row["matched_index"] is not None
                and (
                    not quality_gate_active
                    or quality is None
                    or quality.state == AnalysisState.ASSESSABLE
                )
            )
            face_state = self.policy.axis(
                similarity if row["matched_index"] is not None else None,
                threshold=self.policy.face_threshold if policy_in_domain else None,
                review_boundary=self.policy.face_review_boundary,
                assessable=face_assessable,
            )
            scores = (
                [float(query @ row["copy_descriptor"]) for query in query_copy_descriptors]
                if query_copy_descriptors and row["copy_descriptor"] is not None
                else []
            )
            copy_similarity = max(scores) if scores else None
            copy_ref_index = (
                scores.index(copy_similarity) if scores else preview.copy_reference_index
            )
            candidate_sha256 = hashlib.sha256(remote.content).hexdigest()
            exact_image = candidate_sha256 in copy_hashes
            diagnostic_sscd_match = bool(
                copy_similarity is not None and copy_similarity >= self.settings.sscd_threshold
            )
            copy_state = (
                AxisState.SUPPORTED
                if exact_image
                else self.policy.axis(
                    copy_similarity,
                    threshold=self.policy.copy_threshold if policy_in_domain else None,
                    review_boundary=self.policy.copy_review_boundary,
                    assessable=self.copy_engine is not None,
                    ran=self.copy_engine is not None,
                )
            )
            reference_index = copy_ref_index or 0
            phash_match = row["phash_distance"] <= self.settings.phash_max_distance
            should_run_geometry = exact_image or diagnostic_sscd_match or phash_match
            features = (
                geometric_match(
                    copy_images[reference_index], candidate_image, self.face_engine.cv2
                )
                if should_run_geometry
                else FeatureMatch(0, 0, 0.0)
            )
            geometric_copy = (
                features.inliers >= self.settings.akaze_min_inliers
                and features.inlier_ratio >= self.settings.akaze_min_inlier_ratio
                and features.source_coverage >= 0.05
                and features.candidate_coverage >= 0.05
                and features.reprojection_error is not None
                and features.reprojection_error <= 0.02
            )
            source_face = next(
                (
                    query_faces[index]
                    for index, enrollment in enumerate(enrollment_rows)
                    if enrollment[3] == copy_rows[reference_index][3]
                ),
                None,
            )
            association = self._source_candidate_association(
                source_face, row["matched_box"], features
            )
            decision = self.policy.decide(face_state, copy_state)
            same_content = copy_state == AxisState.SUPPORTED
            face_match = face_state == AxisState.SUPPORTED
            assessed.append(
                VerifiedMatch(
                    hit=hit,
                    face_similarity=similarity,
                    candidate_face_count=len(row["faces"]),
                    matched_face_index=row["matched_index"],
                    matched_face_box=row["matched_box"],
                    candidate_image_sha256=candidate_sha256,
                    candidate_media_type=remote.media_type,
                    candidate_retrieved_url=remote.url,
                    exact_image=exact_image,
                    perceptual_hash_distance=row["phash_distance"],
                    feature_match=features,
                    same_content=same_content,
                    face_match=face_match,
                    sscd_similarity=copy_similarity,
                    sscd_threshold=self.policy.copy_threshold,
                    sscd_match=copy_state == AxisState.SUPPORTED,
                    decision=decision,
                    retrieval_channels=tuple(sorted(preview.retrieval_channels)),
                    face_retrieval_rank=preview.face_retrieval_rank,
                    copy_retrieval_rank=preview.copy_retrieval_rank,
                    phash_retrieval_rank=preview.phash_retrieval_rank,
                    face_state=face_state,
                    copy_state=copy_state,
                    face_assessable=face_assessable,
                    identity_reference_index=row["identity_reference_index"],
                    copy_reference_index=copy_ref_index,
                    candidate_quality=quality.evidence() if quality else None,
                    candidate_consent_ref=preview.face_consent_ref,
                    source_candidate_association=association,
                    candidate_fetched_at=remote.fetched_at,
                )
            )
            ledger[hit.occurrence_key].update(
                disposition=decision,
                detail={
                    "face_state": face_state,
                    "copy_state": copy_state,
                    "face_similarity": round(similarity, 6),
                    "candidate_face_count": len(row["faces"]),
                    "copy_similarity": (
                        round(copy_similarity, 6) if copy_similarity is not None else None
                    ),
                    "identity_reference_index": row["identity_reference_index"],
                    "copy_reference_index": copy_ref_index,
                    "face_retrieval_rank": preview.face_retrieval_rank,
                    "copy_retrieval_rank": preview.copy_retrieval_rank,
                    "phash_diagnostic": phash_match,
                    "akaze_diagnostic": geometric_copy,
                },
            )
        if not assessed:
            raise NoVerifiedMatch("all selected originals failed terminal processing")
        assessed.sort(key=self._verified_rank, reverse=True)
        best = assessed[0]
        verified = [item for item in assessed if item.face_state == AxisState.SUPPORTED]
        ambiguous = best.decision in {
            "review",
            "copy_face_conflict_review",
            "execution_error",
        }
        final_ms = (perf_counter() - final_started) * 1000
        now = datetime.now(UTC)
        query_sha256 = enrollment_rows[0][3]
        run_id = f"fp-{now.strftime('%Y%m%dT%H%M%SZ')}-{query_sha256[:8]}"
        bundle = self._evidence_bundle(
            now=now,
            run_id=run_id,
            query_sha256=query_sha256,
            original_size=len(enrollment_rows[0][1]),
            query_faces=query_faces,
            enrollment_rows=enrollment_rows,
            enrollment_qualities=enrollment_qualities,
            copy_rows=copy_rows,
            batch=batch,
            hits_considered=len(hits),
            previews_downloaded=len(preview_rows),
            preview_bytes=sum(len(remote.content) for _, remote in preview_rows),
            finalists=len(finalist_hits),
            full_bytes=sum(len(remote.content) for _, remote in full_rows),
            failures=failures,
            dispositions=list(ledger.values()),
            assessed=assessed,
            verified=verified,
            best=best,
            selection_margin=None,
            ambiguous=ambiguous,
            exhaustive_full_resolution=exhaustive_full_resolution,
            policy_in_domain=policy_in_domain,
            timings={
                "source_ms": round(source_ms, 3),
                "preview_filter_ms": round(preview_ms, 3),
                "copy_descriptor_ms": round(descriptor_ms, 3),
                "vector_retrieval_ms": round(retrieval_ms, 3),
                "full_verification_ms": round(final_ms, 3),
                "total_ms": round((perf_counter() - pipeline_started) * 1000, 3),
            },
        )
        destination = output_path or self.settings.artifact_dir / run_id / "evidence.json"
        write_json(destination, bundle.to_dict())
        self.progress(f"Draft evidence fingerprint: {evidence_digest(bundle.to_dict())}")
        return bundle, destination

    def _retrieve_candidates(
        self,
        *,
        query_faces: list[Any],
        query_copy_descriptors: list[Any],
        candidates: list[_PreviewCandidate],
    ) -> list[_PreviewCandidate]:
        if not candidates:
            return []
        if self.copy_engine is None or not query_copy_descriptors:
            face_ranked = sorted(
                candidates,
                key=lambda row: (
                    -row.face_similarity,
                    row.hit.post_id,
                    row.hit.media_id,
                ),
            )
            for rank, item in enumerate(face_ranked, start=1):
                item.face_retrieval_rank = rank
                if rank <= self.settings.face_retrieval_k:
                    item.retrieval_channels.add("face-exact-sort")
            face_lane = face_ranked[: self.settings.face_retrieval_k]
            return face_lane[: self.settings.max_finalists]

        face_index = ExactMediaIndex(len(query_faces[0].embedding))
        face_keys: list[str] = []
        face_vectors: list[Any] = []
        face_metadata: list[dict[str, object]] = []
        for position, item in enumerate(candidates):
            for face_index_in_media, face in enumerate(item.faces):
                face_keys.append(item.hit.occurrence_key)
                face_vectors.append(face.embedding)
                face_metadata.append(
                    {
                        "candidate_position": position,
                        "media_key": item.hit.occurrence_key,
                        "face_index": face_index_in_media,
                        "source_box": asdict(face.box),
                        "consent_eligible": item.face_processing_allowed,
                    }
                )
        face_index.add(face_keys, face_vectors, face_metadata)
        face_results = face_index.search(
            [face.embedding for face in query_faces], limit=len(candidates)
        )
        face_positions: list[int] = []
        for result in face_results:
            position = int(result.metadata["candidate_position"])
            candidates[position].face_retrieval_rank = result.rank
            candidates[position].identity_reference_index = result.query_index
            if result.rank <= self.settings.face_retrieval_k:
                face_positions.append(position)
                candidates[position].retrieval_channels.add("face-faiss")

        copy_index = ExactMediaIndex(self.copy_engine.dimensions)
        copy_index.add(
            [item.hit.occurrence_key for item in candidates],
            [item.copy_descriptor for item in candidates],
            [
                {
                    "candidate_position": position,
                    "media_key": item.hit.occurrence_key,
                    "reference_association": "query_index_returned_with_search_hit",
                }
                for position, item in enumerate(candidates)
            ],
        )
        copy_results = copy_index.search(
            query_copy_descriptors,
            limit=len(candidates),
        )
        copy_positions: list[int] = []
        for result in copy_results:
            position = int(result.metadata["candidate_position"])
            candidates[position].copy_similarity = result.score
            candidates[position].copy_retrieval_rank = result.rank
            candidates[position].copy_reference_index = result.query_index
            if result.rank <= self.settings.copy_retrieval_k:
                copy_positions.append(position)
                candidates[position].retrieval_channels.add("sscd-faiss")

        phash_positions = [
            position
            for position, item in sorted(
                enumerate(candidates),
                key=lambda pair: (pair[1].phash_distance, pair[1].hit.post_id),
            )
            if item.phash_distance <= self.settings.phash_max_distance + 8
        ]
        for rank, position in enumerate(phash_positions, start=1):
            candidates[position].phash_retrieval_rank = rank
            candidates[position].retrieval_channels.add("phash-rescue")

        selected: list[_PreviewCandidate] = []
        selected_positions: set[int] = set()
        # Guarantee both learned retrieval lanes before using pHash as a rescue. Interleaving
        # all three lanes could let a broad pHash list evict valid top-K face/copy results.
        primary_channels = (face_positions, copy_positions)
        for rank in range(max((len(channel) for channel in primary_channels), default=0)):
            for channel in primary_channels:
                if rank >= len(channel):
                    continue
                position = channel[rank]
                if position in selected_positions:
                    continue
                selected_positions.add(position)
                selected.append(candidates[position])
                if len(selected) >= self.settings.max_finalists:
                    return selected
        for position in phash_positions:
            if position in selected_positions:
                continue
            selected_positions.add(position)
            selected.append(candidates[position])
            if len(selected) >= self.settings.max_finalists:
                break
        return selected

    def _load_local_inputs(
        self, paths: list[Path], role: str
    ) -> list[tuple[Any, bytes, Any, str]]:
        rows: list[tuple[Any, bytes, Any, str]] = []
        for path in paths:
            resolved = path.resolve()
            record = None
            if self.consent_manifest is not None:
                record = self.consent_manifest.record_for_path(resolved, role)
            elif self.policy.status != "SYNTHETIC_TEST_ONLY":
                raise BlockedState(
                    "BLOCKED_PERMISSION_MANIFEST",
                    f"{role} requires a documented dataset manifest",
                )
            raw = resolved.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            if record is not None and record.sha256 != digest:
                raise FaceInputError(f"{resolved.name} no longer matches its permission manifest")
            rows.append((record, raw, self.face_engine.decode(raw), digest))
        return rows

    def _encode_copy_cached(
        self, images: list[Any], keys: list[str], cache: dict[str, Any]
    ) -> list[Any]:
        if self.copy_engine is None:
            return []
        missing_keys: list[str] = []
        missing_images: list[Any] = []
        seen: set[str] = set()
        for key, image in zip(keys, images, strict=True):
            if key not in cache and key not in seen:
                seen.add(key)
                missing_keys.append(key)
                missing_images.append(image)
        descriptors = self.copy_engine.encode_many(missing_images)
        cache.update(zip(missing_keys, descriptors, strict=True))
        return [cache[key] for key in keys]

    def _detect_faces_cached(
        self, image: Any, content_sha256: str, cache: dict[str, list[Any]]
    ) -> list[Any]:
        if content_sha256 not in cache:
            cache[content_sha256] = self.face_engine.detect_and_encode(image)
        return cache[content_sha256]

    def _best_face_match(
        self, query_faces: list[Any], candidates: list[Any]
    ) -> tuple[float, int | None, Any, int | None]:
        best = (-1.0, None, None, None)
        for reference_index, query_face in enumerate(query_faces):
            score, candidate_index, box = self.face_engine.best_match(query_face, candidates)
            contender = (float(score), candidate_index, box, reference_index)
            if contender[0] > best[0]:
                best = contender
        return best

    def _download(
        self, hits: list[SearchHit], *, preview: bool
    ) -> tuple[list[tuple[SearchHit, RemoteImage]], list[dict[str, str]]]:
        if not hits:
            return [], []
        url_to_hits: dict[str, list[SearchHit]] = {}
        for hit in hits:
            url = hit.preview_url if preview else hit.image_url
            url_to_hits.setdefault(url, []).append(hit)

        remote_by_url: dict[str, RemoteImage] = {}
        error_by_url: dict[str, Exception] = {}
        workers = min(8, len(url_to_hits))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="media-fetch") as pool:
            future_to_url = {pool.submit(self.image_fetcher.fetch, url): url for url in url_to_hits}
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    remote_by_url[url] = future.result()
                except Exception as exc:
                    error_by_url[url] = exc

        # Network completion order must not influence vector insertion, tie handling, or
        # evidence bytes. Re-expand shared URLs in the provider's original media order.
        downloaded: list[tuple[SearchHit, RemoteImage]] = []
        failures: list[dict[str, str]] = []
        for hit in hits:
            url = hit.preview_url if preview else hit.image_url
            if url in remote_by_url:
                downloaded.append((hit, remote_by_url[url]))
            else:
                failures.append(self._failure(url, "download", error_by_url[url], hit))
        return downloaded, failures

    @staticmethod
    def _failure(
        url: str, stage: str, error: Exception, hit: SearchHit | None = None
    ) -> dict[str, str]:
        value = {
            "stage": stage,
            "url_sha256": hashlib.sha256(url.encode("utf-8")).hexdigest(),
            "error": type(error).__name__,
        }
        if hit is not None:
            value["media_id"] = hit.media_id
            value["post_id"] = hit.post_id
            value["occurrence_key"] = hit.occurrence_key
        return value

    def _evidence_bundle(
        self,
        *,
        now: datetime,
        run_id: str,
        query_sha256: str,
        original_size: int,
        query_faces: list[Any],
        enrollment_rows: list[tuple[Any, bytes, Any, str]],
        enrollment_qualities: list[Any],
        copy_rows: list[tuple[Any, bytes, Any, str]],
        batch: SearchBatch,
        hits_considered: int,
        previews_downloaded: int,
        preview_bytes: int,
        finalists: int,
        full_bytes: int,
        failures: list[dict[str, str]],
        dispositions: list[dict[str, Any]],
        assessed: list[VerifiedMatch],
        verified: list[VerifiedMatch],
        best: VerifiedMatch,
        selection_margin: float | None,
        ambiguous: bool,
        exhaustive_full_resolution: bool,
        policy_in_domain: bool,
        timings: dict[str, float],
    ) -> EvidenceBundle:
        hit = best.hit
        full_post_text = hit.content_text
        post_text = full_post_text[:2000]
        runner_summaries = [self._candidate_summary(item) for item in verified[:5]]
        return EvidenceBundle(
            schema=SCHEMA,
            status="DRAFT_REVIEW_REQUIRED",
            created_at=now.isoformat().replace("+00:00", "Z"),
            run_id=run_id,
            consent={
                "asserted_by_operator": True,
                "scope": "bounded identity and image-copy discovery for declared inputs",
                "biometric_comparison_performed": True,
                "identity_labels_inferred": False,
                "manifest_sha256": (
                    self.consent_manifest.sha256 if self.consent_manifest is not None else None
                ),
                "private_media_retention_policy": (
                    self.settings.private_media_retention_policy
                ),
            },
            query={
                "image_sha256": query_sha256,
                "image_bytes": original_size,
                "face_count": len(query_faces),
                "face_box": asdict(query_faces[0].box),
                "detector": self.face_engine.detector_id,
                "detector_model_sha256": getattr(self.face_engine, "detector_sha256", "unknown"),
                "detector_confidence_threshold": self.settings.detection_threshold,
                "encoder": self.face_engine.encoder_id,
                "encoder_model_sha256": getattr(self.face_engine, "encoder_sha256", "unknown"),
                "face_embedding_dimensions": len(query_faces[0].embedding),
                "embedding_persisted": False,
                "input_roles": {
                    "enrollment_face_images": [
                        {
                            "image_sha256": row[3],
                            "consent_ref": row[0].consent_ref if row[0] else "synthetic-fixture",
                            "selected_face_box": asdict(query_faces[index].box),
                            "quality": (
                                enrollment_qualities[index].evidence()
                                if enrollment_qualities
                                else None
                            ),
                        }
                        for index, row in enumerate(enrollment_rows)
                    ],
                    "copy_reference_images": [
                        {"image_sha256": row[3], "reference_index": index}
                        for index, row in enumerate(copy_rows)
                    ],
                },
                "copy_descriptor": (
                    {
                        "enabled": True,
                        "model": self.copy_engine.model_id,
                        "model_sha256": self.copy_engine.model_sha256,
                        "dimensions": self.copy_engine.dimensions,
                        "device": self.copy_engine.device,
                        "runtime_version": getattr(self.copy_engine, "runtime_version", "unknown"),
                        "preprocessing": "RGB square resize 320; ImageNet normalization",
                        "embedding_persisted": False,
                    }
                    if self.copy_engine is not None
                    else {"enabled": False, "embedding_persisted": False}
                ),
                "runtime": {
                    "faceproof_version": __version__,
                    "python_version": sys.version.split()[0],
                    "opencv_version": str(getattr(self.face_engine.cv2, "__version__", "unknown")),
                    "faiss_version": (
                        str(__import__("faiss").__version__)
                        if self.copy_engine is not None
                        else None
                    ),
                },
            },
            search={
                "provider": batch.provider,
                "live_query": batch.live_query,
                "scope": batch.scope,
                "endpoint": batch.endpoint,
                "provider_fetched_at": batch.fetched_at,
                "pages_fetched": batch.pages_fetched,
                "statuses_scanned": batch.statuses_scanned,
                "media_discovered": batch.media_scanned,
                "media_considered": hits_considered,
                "previews_downloaded": previews_downloaded,
                "preview_bytes_downloaded": preview_bytes,
                "full_resolution_finalists": finalists,
                "full_resolution_bytes_downloaded": full_bytes,
                "full_resolution_mode": (
                    "all" if exhaustive_full_resolution else "preview-top-k-union"
                ),
                "vector_backend": (
                    ExactCosineIndex.backend if self.copy_engine is not None else "exact-face-sort"
                ),
                "face_retrieval_k": self.settings.face_retrieval_k,
                "copy_retrieval_k": (
                    self.settings.copy_retrieval_k if self.copy_engine is not None else 0
                ),
                "failures": failures,
                "terminal_dispositions": dispositions,
                "timings": timings,
            },
            match={
                "post_url": hit.post_url,
                "canonical_uri": hit.canonical_uri,
                "post_id": hit.post_id,
                "author_id": hit.author_id,
                "author_handle": hit.author_handle,
                "post_created_at": hit.created_at,
                "post_text": post_text,
                "post_text_sha256": hashlib.sha256(full_post_text.encode("utf-8")).hexdigest(),
                "post_text_truncated": len(full_post_text) > len(post_text),
                "media_id": hit.media_id,
                "wrapper_post_id": hit.wrapper_post_id,
                "wrapper_canonical_uri": hit.wrapper_canonical_uri,
                "media_url": hit.image_url,
                "retrieved_media_url": best.candidate_retrieved_url,
                "media_fetch_observed_at": best.candidate_fetched_at,
                "candidate_image_sha256": best.candidate_image_sha256,
                "candidate_media_type": best.candidate_media_type,
                "candidate_face_count": best.candidate_face_count,
                "matched_face_index": best.matched_face_index,
                "matched_face_box": (
                    asdict(best.matched_face_box) if best.matched_face_box else None
                ),
                "face_similarity": round(best.face_similarity, 6),
                "face_threshold": self.policy.face_threshold,
                "face_state": best.face_state,
                "face_assessable": best.face_assessable,
                "face_match": best.face_match,
                "threshold_source": self.policy.policy_id,
                "exact_image": best.exact_image,
                "sscd_similarity": (
                    round(best.sscd_similarity, 6) if best.sscd_similarity is not None else None
                ),
                "sscd_threshold": best.sscd_threshold,
                "sscd_match": best.sscd_match,
                "copy_state": best.copy_state,
                "perceptual_hash_distance": best.perceptual_hash_distance,
                "akaze_good_matches": best.feature_match.good_matches,
                "akaze_inliers": best.feature_match.inliers,
                "akaze_inlier_ratio": round(best.feature_match.inlier_ratio, 6),
                "akaze_reprojection_error": best.feature_match.reprojection_error,
                "akaze_source_coverage": round(best.feature_match.source_coverage, 6),
                "akaze_candidate_coverage": round(
                    best.feature_match.candidate_coverage, 6
                ),
                "source_candidate_association": best.source_candidate_association,
                "content_signals": {
                    "exact_sha256": best.exact_image,
                    "sscd": best.sscd_match,
                    "perceptual_hash": (
                        best.perceptual_hash_distance <= self.settings.phash_max_distance
                    ),
                    "akaze_ransac": (
                        best.feature_match.inliers >= self.settings.akaze_min_inliers
                        and best.feature_match.inlier_ratio >= self.settings.akaze_min_inlier_ratio
                        and best.feature_match.source_coverage >= 0.05
                        and best.feature_match.candidate_coverage >= 0.05
                        and best.feature_match.reprojection_error is not None
                        and best.feature_match.reprojection_error <= 0.02
                    ),
                },
                "same_content": best.same_content,
                "decision": best.decision,
                "candidate_quality": best.candidate_quality,
                "candidate_consent_ref": best.candidate_consent_ref,
                "identity_reference_index": best.identity_reference_index,
                "copy_reference_index": best.copy_reference_index,
                "retrieval_channels": list(best.retrieval_channels),
                "face_retrieval_rank": best.face_retrieval_rank,
                "copy_retrieval_rank": best.copy_retrieval_rank,
                "phash_retrieval_rank": best.phash_retrieval_rank,
                "assessed_candidate_count": len(assessed),
                "copy_face_conflict_count": sum(
                    item.decision == "copy_face_conflict_review" for item in assessed
                ),
                "verified_match_count": len(verified),
                "selection_margin": (
                    round(selection_margin, 6) if selection_margin is not None else None
                ),
                "ambiguous": ambiguous,
                "human_confirmation_required": True,
                "policy": {
                    "policy_id": self.policy.policy_id,
                    "status": self.policy.status,
                    "policy_sha256": self.policy.source_sha256,
                    "model_lock_sha256": self.policy.model_lock_sha256,
                    "calibration_report_sha256": self.policy.calibration_report_sha256,
                    "test_report_sha256": self.policy.test_report_sha256,
                    "within_calibrated_bounds": policy_in_domain,
                },
                "top_candidates": runner_summaries,
            },
            claims={
                "face_claim_state": best.face_state,
                "copy_claim_state": best.copy_state,
                "submission_eligible_live_search": batch.live_query,
                "identity_match_supported": best.face_match,
                "same_content_supported": best.same_content,
                "decision": best.decision,
                "reviewable_claims": self._reviewable_claims(best),
                "not_verified": [
                    "the legal identity of any person",
                    "the truthfulness or authorship of the post",
                    "content outside the bounded social-search scope",
                    *(
                        ["candidate-face identity; no documented biometric permission"]
                        if not best.candidate_consent_ref and batch.live_query
                        else []
                    ),
                    *(
                        ["a calibrated operational claim"]
                        if not self.policy.calibrated
                        else []
                    ),
                    *(["a genuine social-search result"] if not batch.live_query else []),
                ],
                "blockchain_scope": (
                    "a later anchor proves integrity of this canonical evidence object "
                    "and transaction inclusion"
                ),
            },
            provenance={
                "actual_models": {
                    "detector": {
                        "id": self.face_engine.detector_id,
                        "sha256": getattr(self.face_engine, "detector_sha256", "unknown"),
                    },
                    "recognizer": {
                        "id": self.face_engine.encoder_id,
                        "sha256": getattr(self.face_engine, "encoder_sha256", "unknown"),
                    },
                    "copy": (
                        {
                            "id": self.copy_engine.model_id,
                            "sha256": self.copy_engine.model_sha256,
                        }
                        if self.copy_engine is not None
                        else None
                    ),
                    "parser": (
                        {"id": PARSER_MODEL_ID, "sha256": PARSER_SHA256}
                        if self.quality_assessor is not None
                        and self.quality_assessor.parser is not None
                        else None
                    ),
                    "pose_geometry_template_sha256": POSE_TEMPLATE_SHA256,
                },
                "model_lock_sha256": self.policy.model_lock_sha256,
                "policy_sha256": self.policy.source_sha256,
                "calibration_report_sha256": self.policy.calibration_report_sha256,
                "test_report_sha256": self.policy.test_report_sha256,
                "dependency_lock_sha256": self._optional_file_sha256(
                    self.settings.project_root / "requirements.lock"
                ),
                "source_code_fingerprint": self._source_fingerprint(),
                "serialization": (
                    "sorted UTF-8 JSON; fixed separators; allow_nan=False; not RFC8785"
                ),
            },
            human_review=None,
            nonce_hex=None,
        )

    @staticmethod
    def _candidate_summary(match: VerifiedMatch) -> dict[str, Any]:
        return {
            "post_url": match.hit.post_url,
            "media_id": match.hit.media_id,
            "face_similarity": round(match.face_similarity, 6),
            "face_match": match.face_match,
            "sscd_similarity": (
                round(match.sscd_similarity, 6) if match.sscd_similarity is not None else None
            ),
            "sscd_match": match.sscd_match,
            "same_content": match.same_content,
            "decision": match.decision,
            "retrieval_channels": list(match.retrieval_channels),
            "exact_image": match.exact_image,
            "perceptual_hash_distance": match.perceptual_hash_distance,
            "akaze_inlier_ratio": round(match.feature_match.inlier_ratio, 6),
            "source_candidate_association": match.source_candidate_association,
        }

    @staticmethod
    def _source_candidate_association(
        source_face: Any, candidate_box: Any, features: FeatureMatch
    ) -> dict[str, Any]:
        if source_face is None:
            return {"state": "UNKNOWN", "reason": "copy reference has no enrolled face"}
        if features.homography is None:
            return {"state": "UNKNOWN", "reason": "geometric transform unavailable"}
        if candidate_box is None:
            return {"state": "UNKNOWN", "reason": "candidate face is absent or unassessable"}
        try:
            matrix = np.asarray(features.homography, dtype="float64").reshape(3, 3)
        except (TypeError, ValueError):
            return {"state": "UNKNOWN", "reason": "geometric transform has invalid shape"}
        if not np.all(np.isfinite(matrix)):
            return {"state": "UNKNOWN", "reason": "geometric transform is non-finite"}
        box = source_face.box
        corners = np.array(
            [
                [box.x, box.y],
                [box.x + box.width, box.y],
                [box.x + box.width, box.y + box.height],
                [box.x, box.y + box.height],
            ],
            dtype="float64",
        )
        homogeneous = np.column_stack([corners, np.ones(4)])
        projected = homogeneous @ matrix.T
        if not np.all(np.isfinite(projected)) or np.any(abs(projected[:, 2]) < 1e-9):
            return {"state": "UNKNOWN", "reason": "geometric projection is degenerate"}
        projected = projected[:, :2] / projected[:, 2:]
        x1, y1 = projected.min(axis=0)
        x2, y2 = projected.max(axis=0)
        center_x = candidate_box.x + candidate_box.width / 2
        center_y = candidate_box.y + candidate_box.height / 2
        supported = bool(x1 <= center_x <= x2 and y1 <= center_y <= y2)
        return {
            "state": "CORROBORATED" if supported else "CONFLICT_REVIEW",
            "mapped_source_face_box": [
                round(float(x1), 3),
                round(float(y1), 3),
                round(float(x2 - x1), 3),
                round(float(y2 - y1), 3),
            ],
            "candidate_face_center_inside": supported,
        }

    @staticmethod
    def _optional_file_sha256(path: Path) -> str | None:
        return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None

    def _source_fingerprint(self) -> str:
        files = sorted((self.settings.project_root / "src" / "faceproof").glob("*.py"))
        digest = hashlib.sha256()
        for path in files:
            digest.update(path.name.encode("utf-8"))
            digest.update(b"\x00")
            digest.update(path.read_bytes())
            digest.update(b"\x00")
        return digest.hexdigest()

    @staticmethod
    def _reviewable_claims(match: VerifiedMatch) -> list[str]:
        claims: list[str] = []
        if match.face_state == AxisState.SUPPORTED:
            claims.append("identity comparison supported at the frozen operating point")
        if match.copy_state == AxisState.SUPPORTED:
            claims.append(
                "exact downloaded media bytes match an explicit copy-reference SHA-256"
                if match.exact_image
                else "image-copy relationship supported at the frozen operating point"
            )
        if not claims:
            claims.append("no automated face or copy claim accepted")
        if match.decision == "copy_face_conflict_review":
            claims.append("face/copy evidence conflict retained for review")
        return claims

    @staticmethod
    def _verified_rank(
        match: VerifiedMatch,
    ) -> tuple[int, int, float, float, int, float, int, str]:
        return (
            int(match.face_match and match.same_content),
            int(match.face_match),
            match.face_similarity,
            match.sscd_similarity if match.sscd_similarity is not None else -1.0,
            int(match.exact_image),
            match.feature_match.inlier_ratio,
            -match.perceptual_hash_distance,
            match.hit.post_id,
        )
