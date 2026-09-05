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

from . import __version__
from .config import Settings
from .copydetect import CopyDescriptorEngine
from .errors import ConsentRequired, FaceInputError, NoVerifiedMatch
from .faces import FaceEngine
from .image_match import geometric_match, hash_distance, perceptual_hash
from .integrity import SCHEMA, evidence_digest, write_json
from .models import EvidenceBundle, FeatureMatch, SearchBatch, SearchHit, VerifiedMatch
from .remote import RemoteImage, SafeImageFetcher
from .search import CandidateSource
from .vector_index import ExactCosineIndex

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


class DiscoveryPipeline:
    def __init__(
        self,
        *,
        settings: Settings,
        face_engine: FaceEngine,
        candidate_source: CandidateSource,
        image_fetcher: SafeImageFetcher,
        copy_engine: CopyDescriptorEngine | None = None,
        progress: ProgressCallback | None = None,
    ) -> None:
        self.settings = settings
        self.face_engine = face_engine
        self.candidate_source = candidate_source
        self.image_fetcher = image_fetcher
        self.copy_engine = copy_engine
        self.progress = progress or (lambda _: None)

    def discover(
        self,
        image_path: Path,
        *,
        consent_asserted: bool,
        output_path: Path | None = None,
    ) -> tuple[EvidenceBundle, Path]:
        pipeline_started = perf_counter()
        if not consent_asserted:
            raise ConsentRequired(
                "explicit consent is required; pass --i-have-consent only for your own face "
                "or a person who agreed to this search"
            )
        image_path = image_path.resolve()
        if not image_path.is_file():
            raise FaceInputError(f"input image does not exist: {image_path}")
        if image_path.stat().st_size > self.settings.max_image_bytes:
            raise FaceInputError("input image exceeds the configured size limit")

        original = image_path.read_bytes()
        query_sha256 = hashlib.sha256(original).hexdigest()
        self.progress("Validating, detecting and encoding the consenting query face locally")
        query_image = self.face_engine.decode(original)
        query_face = self.face_engine.require_single_query_face(query_image)
        query_phash = perceptual_hash(query_image, self.face_engine.cv2)
        query_copy_descriptor = (
            self.copy_engine.encode(query_image) if self.copy_engine is not None else None
        )

        self.progress("Enumerating current public social-media image posts")
        source_started = perf_counter()
        batch = self.candidate_source.discover()
        source_ms = (perf_counter() - source_started) * 1000
        if not batch.hits:
            raise NoVerifiedMatch(f"no public image posts were found in {batch.scope}")

        hits = batch.hits[: self.settings.max_candidates]
        self.progress(f"Preview-filtering {len(hits)} live media candidates")
        preview_started = perf_counter()
        preview_rows, failures = self._download(hits, preview=True)
        candidates: list[_PreviewCandidate] = []
        for hit, preview in preview_rows:
            try:
                preview_image = self.face_engine.decode(preview.content)
                preview_faces = self.face_engine.detect_and_encode(preview_image)
                similarity, matched_index, matched_box = self.face_engine.best_match(
                    query_face, preview_faces
                )
                phash_distance = hash_distance(
                    query_phash,
                    perceptual_hash(preview_image, self.face_engine.cv2),
                )
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
                    )
                )
            except FaceInputError as exc:
                failures.append(self._failure(hit.preview_url, "preview_decode", exc))

        descriptor_started = perf_counter()
        if self.copy_engine is not None and candidates:
            descriptors = self.copy_engine.encode_many([item.image for item in candidates])
            for item, descriptor in zip(candidates, descriptors, strict=True):
                item.copy_descriptor = descriptor
                item.copy_similarity = float(query_copy_descriptor @ descriptor)
        descriptor_ms = (perf_counter() - descriptor_started) * 1000

        retrieval_started = perf_counter()
        shortlist = self._retrieve_candidates(
            query_face=query_face,
            query_copy_descriptor=query_copy_descriptor,
            candidates=candidates,
        )
        retrieval_ms = (perf_counter() - retrieval_started) * 1000
        preview_ms = (perf_counter() - preview_started) * 1000
        if not shortlist:
            raise NoVerifiedMatch(
                "public posts were scanned, but none passed the local preview face/image filter"
            )

        self.progress(f"Full-resolution verification of {len(shortlist)} finalists")
        final_started = perf_counter()
        preview_by_media = {item.hit.media_id: item.remote for item in shortlist}
        preview_cache = {item.hit.media_id: item for item in shortlist}
        finalist_hits = [item.hit for item in shortlist]
        full_needed = [hit for hit in finalist_hits if hit.image_url != hit.preview_url]
        full_rows, full_failures = self._download(full_needed, preview=False)
        failures.extend(full_failures)
        full_by_media = {hit.media_id: remote for hit, remote in full_rows}

        decoded: list[dict[str, Any]] = []
        for hit in finalist_hits:
            remote = full_by_media.get(hit.media_id) or preview_by_media[hit.media_id]
            try:
                cached = preview_cache[hit.media_id]
                if hit.image_url == hit.preview_url and cached is not None:
                    candidate_image = cached.image
                    candidate_faces = cached.faces
                    similarity = cached.face_similarity
                    phash_distance = cached.phash_distance
                    matched_index = cached.matched_index
                    matched_box = cached.matched_box
                    copy_descriptor = cached.copy_descriptor
                else:
                    candidate_image = self.face_engine.decode(remote.content)
                    candidate_faces = self.face_engine.detect_and_encode(candidate_image)
                    similarity, matched_index, matched_box = self.face_engine.best_match(
                        query_face, candidate_faces
                    )
                    phash_distance = hash_distance(
                        query_phash,
                        perceptual_hash(candidate_image, self.face_engine.cv2),
                    )
                    copy_descriptor = None
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
                        "copy_descriptor": copy_descriptor,
                        "preview": cached,
                    }
                )
            except FaceInputError as exc:
                failures.append(self._failure(remote.url, "final_decode", exc))

        if self.copy_engine is not None:
            missing = [row for row in decoded if row["copy_descriptor"] is None]
            descriptors = self.copy_engine.encode_many([row["image"] for row in missing])
            for row, descriptor in zip(missing, descriptors, strict=True):
                row["copy_descriptor"] = descriptor

        assessed: list[VerifiedMatch] = []
        for row in decoded:
            hit = row["hit"]
            remote = row["remote"]
            candidate_image = row["image"]
            similarity = row["similarity"]
            phash_distance = row["phash_distance"]
            preview = row["preview"]
            face_match = similarity >= self.settings.face_threshold
            copy_similarity = (
                float(query_copy_descriptor @ row["copy_descriptor"])
                if query_copy_descriptor is not None and row["copy_descriptor"] is not None
                else None
            )
            sscd_match = bool(
                copy_similarity is not None and copy_similarity >= self.settings.sscd_threshold
            )
            try:
                candidate_sha256 = hashlib.sha256(remote.content).hexdigest()
                exact_image = candidate_sha256 == query_sha256
                phash_match = phash_distance <= self.settings.phash_max_distance
                should_run_geometry = face_match or exact_image or sscd_match or phash_match
                features = (
                    geometric_match(query_image, candidate_image, self.face_engine.cv2)
                    if should_run_geometry
                    else FeatureMatch(0, 0, 0.0)
                )
                geometric_copy = (
                    features.inliers >= self.settings.akaze_min_inliers
                    and features.inlier_ratio >= self.settings.akaze_min_inlier_ratio
                )
                same_content = exact_image or sscd_match or phash_match or geometric_copy
                if not face_match and not same_content:
                    continue
                decision = self._decision(face_match=face_match, copy_match=same_content)
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
                        perceptual_hash_distance=phash_distance,
                        feature_match=features,
                        same_content=same_content,
                        face_match=face_match,
                        sscd_similarity=copy_similarity,
                        sscd_threshold=(
                            self.settings.sscd_threshold if self.copy_engine is not None else None
                        ),
                        sscd_match=sscd_match,
                        decision=decision,
                        retrieval_channels=tuple(sorted(preview.retrieval_channels)),
                        face_retrieval_rank=preview.face_retrieval_rank,
                        copy_retrieval_rank=preview.copy_retrieval_rank,
                        phash_retrieval_rank=preview.phash_retrieval_rank,
                    )
                )
            except FaceInputError as exc:
                failures.append(self._failure(remote.url, "final_decode", exc))

        verified = [item for item in assessed if item.face_match]
        if not verified:
            conflict_count = sum(item.decision == "copy_face_conflict" for item in assessed)
            detail = (
                f"; {conflict_count} copy-like candidate(s) had no verified face"
                if conflict_count
                else ""
            )
            raise NoVerifiedMatch(
                "live candidates were found, but none passed full local face re-verification"
                + detail
            )
        verified.sort(key=self._verified_rank, reverse=True)
        best = verified[0]
        second = verified[1] if len(verified) > 1 else None
        margin = best.face_similarity - second.face_similarity if second else None
        same_remote_asset = bool(
            second and best.candidate_image_sha256 == second.candidate_image_sha256
        )
        content_uniquely_disambiguates = bool(
            best.same_content and second and not second.same_content
        )
        ambiguous = bool(
            second
            and margin is not None
            and margin < self.settings.ambiguity_margin
            and not same_remote_asset
            and not content_uniquely_disambiguates
        )
        final_ms = (perf_counter() - final_started) * 1000

        now = datetime.now(UTC)
        run_id = f"fp-{now.strftime('%Y%m%dT%H%M%SZ')}-{query_sha256[:8]}"
        bundle = self._evidence_bundle(
            now=now,
            run_id=run_id,
            query_sha256=query_sha256,
            original_size=len(original),
            query_face=query_face,
            batch=batch,
            hits_considered=len(hits),
            previews_downloaded=len(preview_rows),
            finalists=len(shortlist),
            failures=failures,
            assessed=assessed,
            verified=verified,
            best=best,
            selection_margin=margin,
            ambiguous=ambiguous,
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
        self.progress(f"Evidence fingerprint: {evidence_digest(bundle.to_dict())}")
        return bundle, destination

    def _retrieve_candidates(
        self,
        *,
        query_face: Any,
        query_copy_descriptor: Any,
        candidates: list[_PreviewCandidate],
    ) -> list[_PreviewCandidate]:
        if not candidates:
            return []
        if self.copy_engine is None or query_copy_descriptor is None:
            shortlist = [
                item
                for item in candidates
                if (
                    item.face_similarity >= self.settings.prefilter_face_threshold
                    or item.phash_distance <= self.settings.phash_max_distance + 8
                )
                and not (
                    item.hit.image_url == item.hit.preview_url
                    and item.face_similarity < self.settings.face_threshold
                )
            ]
            for rank, item in enumerate(
                sorted(shortlist, key=lambda row: (-row.face_similarity, row.phash_distance)),
                start=1,
            ):
                item.retrieval_channels.add("legacy-face-phash")
                item.face_retrieval_rank = rank
            return sorted(
                shortlist,
                key=lambda row: (-row.face_similarity, row.phash_distance, row.hit.post_id),
            )[: self.settings.max_finalists]

        face_index = ExactCosineIndex(len(query_face.embedding))
        face_keys: list[str] = []
        face_vectors: list[Any] = []
        face_key_to_position: dict[str, int] = {}
        for position, item in enumerate(candidates):
            for face_index_in_media, face in enumerate(item.faces):
                key = f"{position}:{face_index_in_media}"
                face_keys.append(key)
                face_vectors.append(face.embedding)
                face_key_to_position[key] = position
        face_index.add(face_keys, face_vectors)
        face_results = face_index.search(query_face.embedding, limit=len(face_keys))
        face_positions: list[int] = []
        seen_face_media: set[int] = set()
        for result in face_results:
            position = face_key_to_position[result.key]
            if position in seen_face_media or result.score < self.settings.prefilter_face_threshold:
                continue
            seen_face_media.add(position)
            face_positions.append(position)
            candidates[position].face_retrieval_rank = len(face_positions)
            candidates[position].retrieval_channels.add("face-faiss")
            if len(face_positions) >= self.settings.face_retrieval_k:
                break

        copy_index = ExactCosineIndex(self.copy_engine.dimensions)
        copy_index.add(
            [str(position) for position in range(len(candidates))],
            [item.copy_descriptor for item in candidates],
        )
        copy_results = copy_index.search(
            query_copy_descriptor,
            limit=min(self.settings.copy_retrieval_k, len(candidates)),
        )
        copy_positions = [int(result.key) for result in copy_results]
        for result, position in zip(copy_results, copy_positions, strict=True):
            candidates[position].copy_similarity = result.score
            candidates[position].copy_retrieval_rank = result.rank
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

    @staticmethod
    def _decision(*, face_match: bool, copy_match: bool) -> str:
        if face_match and copy_match:
            return "same_identity_and_content"
        if face_match:
            return "same_identity_different_content"
        if copy_match:
            return "copy_face_conflict"
        return "rejected"

    def _download(
        self, hits: list[SearchHit], *, preview: bool
    ) -> tuple[list[tuple[SearchHit, RemoteImage]], list[dict[str, str]]]:
        if not hits:
            return [], []
        downloaded: list[tuple[SearchHit, RemoteImage]] = []
        failures: list[dict[str, str]] = []
        url_to_hits: dict[str, list[SearchHit]] = {}
        for hit in hits:
            url = hit.preview_url if preview else hit.image_url
            url_to_hits.setdefault(url, []).append(hit)

        workers = min(8, len(url_to_hits))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="media-fetch") as pool:
            future_to_url = {pool.submit(self.image_fetcher.fetch, url): url for url in url_to_hits}
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    remote = future.result()
                    downloaded.extend((hit, remote) for hit in url_to_hits[url])
                except Exception as exc:
                    failures.append(self._failure(url, "download", exc))
        return downloaded, failures

    @staticmethod
    def _failure(url: str, stage: str, error: Exception) -> dict[str, str]:
        return {
            "stage": stage,
            "url_sha256": hashlib.sha256(url.encode("utf-8")).hexdigest(),
            "error": type(error).__name__,
        }

    def _evidence_bundle(
        self,
        *,
        now: datetime,
        run_id: str,
        query_sha256: str,
        original_size: int,
        query_face: Any,
        batch: SearchBatch,
        hits_considered: int,
        previews_downloaded: int,
        finalists: int,
        failures: list[dict[str, str]],
        assessed: list[VerifiedMatch],
        verified: list[VerifiedMatch],
        best: VerifiedMatch,
        selection_margin: float | None,
        ambiguous: bool,
        timings: dict[str, float],
    ) -> EvidenceBundle:
        hit = best.hit
        full_post_text = hit.content_text
        post_text = full_post_text[:2000]
        runner_summaries = [self._candidate_summary(item) for item in verified[:5]]
        return EvidenceBundle(
            schema=SCHEMA,
            created_at=now.isoformat().replace("+00:00", "Z"),
            run_id=run_id,
            consent={
                "asserted_by_operator": True,
                "scope": "matching-content discovery for the supplied face image",
                "identity_inference_performed": False,
            },
            query={
                "image_sha256": query_sha256,
                "image_bytes": original_size,
                "face_count": 1,
                "face_box": asdict(query_face.box),
                "detector": self.face_engine.detector_id,
                "detector_model_sha256": getattr(self.face_engine, "detector_sha256", "unknown"),
                "detector_confidence_threshold": self.settings.detection_threshold,
                "encoder": self.face_engine.encoder_id,
                "encoder_model_sha256": getattr(self.face_engine, "encoder_sha256", "unknown"),
                "face_embedding_dimensions": len(query_face.embedding),
                "embedding_persisted": False,
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
                "full_resolution_finalists": finalists,
                "vector_backend": (
                    ExactCosineIndex.backend if self.copy_engine is not None else "legacy-sort"
                ),
                "face_retrieval_k": self.settings.face_retrieval_k,
                "copy_retrieval_k": (
                    self.settings.copy_retrieval_k if self.copy_engine is not None else 0
                ),
                "failures": failures,
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
                "media_url": hit.image_url,
                "retrieved_media_url": best.candidate_retrieved_url,
                "candidate_image_sha256": best.candidate_image_sha256,
                "candidate_media_type": best.candidate_media_type,
                "candidate_face_count": best.candidate_face_count,
                "matched_face_index": best.matched_face_index,
                "matched_face_box": (
                    asdict(best.matched_face_box) if best.matched_face_box else None
                ),
                "face_similarity": round(best.face_similarity, 6),
                "face_threshold": self.settings.face_threshold,
                "face_match": best.face_match,
                "threshold_source": self.settings.face_threshold_source,
                "exact_image": best.exact_image,
                "sscd_similarity": (
                    round(best.sscd_similarity, 6) if best.sscd_similarity is not None else None
                ),
                "sscd_threshold": best.sscd_threshold,
                "sscd_match": best.sscd_match,
                "perceptual_hash_distance": best.perceptual_hash_distance,
                "akaze_good_matches": best.feature_match.good_matches,
                "akaze_inliers": best.feature_match.inliers,
                "akaze_inlier_ratio": round(best.feature_match.inlier_ratio, 6),
                "content_signals": {
                    "exact_sha256": best.exact_image,
                    "sscd": best.sscd_match,
                    "perceptual_hash": (
                        best.perceptual_hash_distance <= self.settings.phash_max_distance
                    ),
                    "akaze_ransac": (
                        best.feature_match.inliers >= self.settings.akaze_min_inliers
                        and best.feature_match.inlier_ratio >= self.settings.akaze_min_inlier_ratio
                    ),
                },
                "same_content": best.same_content,
                "decision": best.decision,
                "retrieval_channels": list(best.retrieval_channels),
                "face_retrieval_rank": best.face_retrieval_rank,
                "copy_retrieval_rank": best.copy_retrieval_rank,
                "phash_retrieval_rank": best.phash_retrieval_rank,
                "assessed_candidate_count": len(assessed),
                "copy_face_conflict_count": sum(
                    item.decision == "copy_face_conflict" for item in assessed
                ),
                "verified_match_count": len(verified),
                "selection_margin": (
                    round(selection_margin, 6) if selection_margin is not None else None
                ),
                "ambiguous": ambiguous,
                "human_confirmation_required": True,
                "top_candidates": runner_summaries,
            },
            claims={
                "verified": (
                    "the supplied face is visually consistent with the selected public-post image"
                    if batch.live_query
                    else "the synthetic query face is visually consistent with a synthetic fixture"
                ),
                "submission_eligible_live_search": batch.live_query,
                "identity_match_verified": best.face_match,
                "same_content_verified": best.same_content,
                "decision": best.decision,
                "not_verified": [
                    "the legal identity of any person",
                    "the truthfulness or authorship of the post",
                    "content outside the bounded social-search scope",
                    *(["a genuine social-search result"] if not batch.live_query else []),
                ],
                "blockchain_scope": (
                    "a later anchor proves integrity of this canonical evidence object "
                    "and transaction inclusion"
                ),
            },
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
        }

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
