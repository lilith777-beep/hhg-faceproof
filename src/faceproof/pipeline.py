from __future__ import annotations

import hashlib
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from .config import Settings
from .errors import ConsentRequired, FaceInputError, NoVerifiedMatch
from .faces import FaceEngine
from .image_match import geometric_match, hash_distance, perceptual_hash
from .integrity import SCHEMA, evidence_digest, write_json
from .models import EvidenceBundle, SearchBatch, SearchHit, VerifiedMatch
from .remote import RemoteImage, SafeImageFetcher
from .search import CandidateSource

ProgressCallback = Callable[[str], None]


class DiscoveryPipeline:
    def __init__(
        self,
        *,
        settings: Settings,
        face_engine: FaceEngine,
        candidate_source: CandidateSource,
        image_fetcher: SafeImageFetcher,
        progress: ProgressCallback | None = None,
    ) -> None:
        self.settings = settings
        self.face_engine = face_engine
        self.candidate_source = candidate_source
        self.image_fetcher = image_fetcher
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
        shortlist: list[tuple[SearchHit, RemoteImage, float, int]] = []
        for hit, preview in preview_rows:
            try:
                preview_image = self.face_engine.decode(preview.content)
                preview_faces = self.face_engine.detect_and_encode(preview_image)
                similarity, _, _ = self.face_engine.best_match(query_face, preview_faces)
                phash_distance = hash_distance(
                    query_phash,
                    perceptual_hash(preview_image, self.face_engine.cv2),
                )
                if (
                    similarity >= self.settings.prefilter_face_threshold
                    or phash_distance <= self.settings.phash_max_distance + 8
                ):
                    shortlist.append((hit, preview, similarity, phash_distance))
            except FaceInputError as exc:
                failures.append(self._failure(hit.preview_url, "preview_decode", exc))

        shortlist.sort(key=lambda row: (-row[2], row[3], row[0].post_id))
        shortlist = shortlist[: self.settings.max_finalists]
        preview_ms = (perf_counter() - preview_started) * 1000
        if not shortlist:
            raise NoVerifiedMatch(
                "public posts were scanned, but none passed the local preview face/image filter"
            )

        self.progress(f"Full-resolution verification of {len(shortlist)} finalists")
        final_started = perf_counter()
        preview_by_media = {hit.media_id: remote for hit, remote, _, _ in shortlist}
        finalist_hits = [row[0] for row in shortlist]
        full_needed = [hit for hit in finalist_hits if hit.image_url != hit.preview_url]
        full_rows, full_failures = self._download(full_needed, preview=False)
        failures.extend(full_failures)
        full_by_media = {hit.media_id: remote for hit, remote in full_rows}

        verified: list[VerifiedMatch] = []
        for hit in finalist_hits:
            remote = full_by_media.get(hit.media_id) or preview_by_media[hit.media_id]
            try:
                candidate_image = self.face_engine.decode(remote.content)
                candidate_faces = self.face_engine.detect_and_encode(candidate_image)
                similarity, matched_index, matched_box = self.face_engine.best_match(
                    query_face, candidate_faces
                )
                if similarity < self.settings.face_threshold:
                    continue
                candidate_sha256 = hashlib.sha256(remote.content).hexdigest()
                phash_distance = hash_distance(
                    query_phash,
                    perceptual_hash(candidate_image, self.face_engine.cv2),
                )
                features = geometric_match(query_image, candidate_image, self.face_engine.cv2)
                exact_image = candidate_sha256 == query_sha256
                same_content = (
                    exact_image
                    or phash_distance <= self.settings.phash_max_distance
                    or (
                        features.inliers >= self.settings.akaze_min_inliers
                        and features.inlier_ratio >= self.settings.akaze_min_inlier_ratio
                    )
                )
                verified.append(
                    VerifiedMatch(
                        hit=hit,
                        face_similarity=similarity,
                        candidate_face_count=len(candidate_faces),
                        matched_face_index=matched_index,
                        matched_face_box=matched_box,
                        candidate_image_sha256=candidate_sha256,
                        candidate_media_type=remote.media_type,
                        candidate_retrieved_url=remote.url,
                        exact_image=exact_image,
                        perceptual_hash_distance=phash_distance,
                        feature_match=features,
                        same_content=same_content,
                    )
                )
            except FaceInputError as exc:
                failures.append(self._failure(remote.url, "final_decode", exc))

        if not verified:
            raise NoVerifiedMatch(
                "live candidates were found, but none passed full local face re-verification"
            )
        verified.sort(key=self._verified_rank, reverse=True)
        best = verified[0]
        second = verified[1] if len(verified) > 1 else None
        margin = best.face_similarity - second.face_similarity if second else None
        ambiguous = bool(
            second
            and not best.same_content
            and margin is not None
            and margin < self.settings.ambiguity_margin
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
            verified=verified,
            best=best,
            selection_margin=margin,
            ambiguous=ambiguous,
            timings={
                "source_ms": round(source_ms, 3),
                "preview_filter_ms": round(preview_ms, 3),
                "full_verification_ms": round(final_ms, 3),
                "total_ms": round((perf_counter() - pipeline_started) * 1000, 3),
            },
        )
        destination = output_path or self.settings.artifact_dir / run_id / "evidence.json"
        write_json(destination, bundle.to_dict())
        self.progress(f"Evidence fingerprint: {evidence_digest(bundle.to_dict())}")
        return bundle, destination

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
        verified: list[VerifiedMatch],
        best: VerifiedMatch,
        selection_margin: float | None,
        ambiguous: bool,
        timings: dict[str, float],
    ) -> EvidenceBundle:
        hit = best.hit
        post_text = hit.content_text[:2000]
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
                "encoder": self.face_engine.encoder_id,
                "embedding_persisted": False,
            },
            search={
                "provider": batch.provider,
                "live_query": True,
                "scope": batch.scope,
                "endpoint": batch.endpoint,
                "provider_fetched_at": batch.fetched_at,
                "pages_fetched": batch.pages_fetched,
                "statuses_scanned": batch.statuses_scanned,
                "media_discovered": batch.media_scanned,
                "media_considered": hits_considered,
                "previews_downloaded": previews_downloaded,
                "full_resolution_finalists": finalists,
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
                "post_text_sha256": hashlib.sha256(post_text.encode("utf-8")).hexdigest(),
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
                "threshold_source": self.settings.face_threshold_source,
                "exact_image": best.exact_image,
                "perceptual_hash_distance": best.perceptual_hash_distance,
                "akaze_good_matches": best.feature_match.good_matches,
                "akaze_inliers": best.feature_match.inliers,
                "akaze_inlier_ratio": round(best.feature_match.inlier_ratio, 6),
                "same_content": best.same_content,
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
                ),
                "same_content_verified": best.same_content,
                "not_verified": [
                    "the legal identity of any person",
                    "the truthfulness or authorship of the post",
                    "content outside the bounded social-search scope",
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
            "same_content": match.same_content,
            "exact_image": match.exact_image,
            "perceptual_hash_distance": match.perceptual_hash_distance,
            "akaze_inlier_ratio": round(match.feature_match.inlier_ratio, 6),
        }

    @staticmethod
    def _verified_rank(match: VerifiedMatch) -> tuple[int, int, float, float, int, str]:
        return (
            int(match.exact_image),
            int(match.same_content),
            match.face_similarity,
            match.feature_match.inlier_ratio,
            -match.perceptual_hash_distance,
            match.hit.post_id,
        )
