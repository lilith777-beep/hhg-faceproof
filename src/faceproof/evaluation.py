from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .dataset import DatasetManifest, ManifestRecord
from .errors import BlockedState, FaceInputError
from .integrity import write_json
from .models import SearchBatch, SearchHit
from .pipeline import DiscoveryPipeline
from .remote import RemoteImage

EVALUABLE_AXIS_STATES = {"SUPPORTED", "NOT_SUPPORTED", "BORDERLINE"}


class ManifestMediaSource:
    """Offline evaluator source; candidates come only from the validated manifest."""

    name = "manifest-fixture-source"

    def __init__(self, manifest: DatasetManifest, split: str) -> None:
        self.records = tuple(
            record
            for record in manifest.records
            if record.split == split
            and record.path is not None
            and record.roles & {"candidate_face", "candidate_copy"}
        )
        self.split = split

    def discover(self) -> SearchBatch:
        hits = [_record_hit(record) for record in self.records]
        return SearchBatch(
            provider=self.name,
            live_query=False,
            scope=f"validated manifest split={self.split}",
            endpoint=f"manifest://{self.split}",
            fetched_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            pages_fetched=1,
            statuses_scanned=len(hits),
            media_scanned=len(hits),
            hits=hits,
        )


class ManifestImageFetcher:
    def __init__(self, manifest: DatasetManifest) -> None:
        self.by_id = {record.image_id: record for record in manifest.records}

    def fetch(self, url: str) -> RemoteImage:
        image_id = url.removeprefix("manifest://")
        record = self.by_id.get(image_id)
        if record is None or record.path is None:
            raise FaceInputError(f"manifest media unavailable: {image_id}")
        content = record.path.read_bytes()
        if hashlib.sha256(content).hexdigest() != record.sha256:
            raise FaceInputError(f"manifest media hash changed: {image_id}")
        return RemoteImage(url, content, _media_type(record.path))


@dataclass(frozen=True, slots=True)
class EvaluationCounts:
    searches: int
    evaluable_searches: int
    known_searches: int
    unknown_searches: int
    unevaluable_known_searches: int
    unevaluable_unknown_searches: int
    searches_with_failures: int
    false_positive_searches: int
    false_negative_searches: int
    true_positive_searches: int
    incorrect_known_returns: int


def run_manifest_evaluation(
    manifest: DatasetManifest,
    pipeline: DiscoveryPipeline,
    *,
    split: str,
    output: Path,
) -> dict[str, Any]:
    if split not in {"development", "calibration", "test"}:
        raise FaceInputError(f"unsupported evaluation split: {split}")
    queries = [
        record
        for record in manifest.records
        if record.split == split and "enrollment_face" in record.roles and record.path is not None
    ]
    gallery = [
        record
        for record in manifest.records
        if record.split == split
        and record.path is not None
        and record.roles & {"candidate_face", "candidate_copy"}
    ]
    unavailable_gallery = [
        record.image_id
        for record in manifest.records
        if record.split == split
        and record.path is None
        and record.roles & {"candidate_face", "candidate_copy"}
    ]
    if unavailable_gallery:
        raise BlockedState(
            "BLOCKED_REAL_CALIBRATION_DATA",
            f"split {split} candidate media need local files: {sorted(unavailable_gallery)}",
        )
    if not queries or not gallery:
        raise BlockedState(
            "BLOCKED_REAL_CALIBRATION_DATA",
            f"split {split} needs enrollment queries and candidate media",
        )
    _validate_evaluation_partition(queries, gallery)
    by_id = {record.image_id: record for record in manifest.records}
    face_gallery_media = {
        record.media_id or record.image_id
        for record in gallery
        if "candidate_face" in record.roles
    }
    copy_gallery_media = {
        record.media_id or record.image_id
        for record in gallery
        if "candidate_copy" in record.roles
    }
    search_rows: list[dict[str, Any]] = []
    for query in queries:
        copy_refs = _copy_references(query, by_id)
        bundle, _ = pipeline.discover(
            query.path,
            copy_reference_paths=[record.path for record in copy_refs if record.path],
            consent_asserted=True,
            exhaustive_full_resolution=True,
        )
        dispositions = {
            str(row["media_id"]): row for row in bundle.search["terminal_dispositions"]
        }
        known_media = {
            record.media_id or record.image_id
            for record in gallery
            if set(record.participant_ids) & set(query.participant_ids)
            and "candidate_face" in record.roles
        }
        copy_media = {
            record.media_id or record.image_id
            for record in gallery
            if "candidate_copy" in record.roles
            and (
                record.copy_parent_image_id in {item.image_id for item in copy_refs}
                or record.image_id in {item.image_id for item in copy_refs}
            )
        }
        accepted = {
            media_id
            for media_id, row in dispositions.items()
            if (row.get("detail") or {}).get("face_state") == "SUPPORTED"
        }
        accepted_copy = {
            media_id
            for media_id, row in dispositions.items()
            if (row.get("detail") or {}).get("copy_state") == "SUPPORTED"
        }
        assessed_face = {
            media_id
            for media_id, row in dispositions.items()
            if media_id in face_gallery_media
            and isinstance(row.get("detail"), dict)
            and row["detail"].get("face_state") in EVALUABLE_AXIS_STATES
        }
        assessed_copy = {
            media_id
            for media_id, row in dispositions.items()
            if media_id in copy_gallery_media
            and isinstance(row.get("detail"), dict)
            and row["detail"].get("copy_state") in EVALUABLE_AXIS_STATES
        }
        wrong = accepted - known_media
        correct_return = bool(accepted & known_media)
        copy_correct_return = bool(accepted_copy & copy_media)
        face_evaluable = (
            bool(face_gallery_media)
            and (
                correct_return
                or (
                    known_media <= assessed_face
                    if known_media
                    else face_gallery_media <= assessed_face
                )
            )
        )
        copy_evaluable = (
            bool(copy_gallery_media)
            and (
                copy_correct_return
                or (
                    copy_media <= assessed_copy
                    if copy_media
                    else copy_gallery_media <= assessed_copy
                )
            )
        )
        search_rows.append(
            {
                "query_image_id": query.image_id,
                "participant_ids": list(query.participant_ids),
                "source_image_family_id": query.source_image_family_id,
                "copy_reference_family_ids": sorted(
                    {record.source_image_family_id for record in copy_refs}
                ),
                "known_subject": bool(known_media),
                "correct_media": sorted(known_media),
                "copy_correct_media": sorted(copy_media),
                "accepted_face_media": sorted(accepted),
                "accepted_copy_media": sorted(accepted_copy),
                "incorrect_media": sorted(wrong),
                "correct_return": correct_return,
                "copy_correct_return": copy_correct_return,
                "face_search_evaluable": face_evaluable,
                "copy_search_evaluable": copy_evaluable,
                "preview": _preview_metrics(query, copy_refs, gallery, dispositions),
                "bytes_downloaded": {
                    "preview": bundle.search["preview_bytes_downloaded"],
                    "full_resolution": bundle.search["full_resolution_bytes_downloaded"],
                },
                "latency_ms": bundle.search["timings"],
                "failure_count": len(bundle.search["failures"]),
                "terminal_dispositions": list(dispositions.values()),
            }
        )
    known = [row for row in search_rows if row["known_subject"]]
    unknown = [row for row in search_rows if not row["known_subject"]]
    known_evaluable = [row for row in known if row["face_search_evaluable"]]
    unknown_evaluable = [row for row in unknown if row["face_search_evaluable"]]
    fp = sum(bool(row["accepted_face_media"]) for row in unknown_evaluable)
    tp = sum(bool(row["correct_return"]) for row in known_evaluable)
    fn = len(known_evaluable) - tp
    incorrect = sum(bool(row["incorrect_media"]) for row in known)
    counts = EvaluationCounts(
        len(search_rows),
        len(known_evaluable) + len(unknown_evaluable),
        len(known_evaluable),
        len(unknown_evaluable),
        len(known) - len(known_evaluable),
        len(unknown) - len(unknown_evaluable),
        sum(bool(row["failure_count"]) for row in search_rows),
        fp,
        fn,
        tp,
        incorrect,
    )
    face_fpir_clustered = _clustered_search_interval(
        unknown_evaluable,
        event_name="accepted_face_media",
        cluster_name="participant_ids",
        require_all=False,
    )
    face_tpir_clustered = _clustered_search_interval(
        known_evaluable,
        event_name="correct_return",
        cluster_name="participant_ids",
        require_all=True,
    )
    copy_metrics = _copy_search_metrics(search_rows)
    report = {
        "schema": "faceproof.open-set-evaluation.v1",
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "manifest_sha256": manifest.sha256,
        "split": split,
        "pipeline_mode": "actual DiscoveryPipeline; exhaustive full-resolution all",
        "policy": {
            "policy_id": pipeline.policy.policy_id,
            "status": pipeline.policy.status,
            "policy_sha256": pipeline.policy.source_sha256,
            "calibration_report_sha256": pipeline.policy.calibration_report_sha256,
            "test_report_sha256": pipeline.policy.test_report_sha256,
        },
        "gallery": {
            "media_items": len(gallery),
            "face_vectors": sum(
                int((row.get("detail") or {}).get("candidate_face_count") or 0)
                for row in search_rows[0]["terminal_dispositions"]
            ),
            "distinct_known_identities": len(
                {participant for row in gallery for participant in row.participant_ids}
            ),
            "query_reference_images": len(queries),
        },
        "counts": asdict(counts),
        "metrics": {
            "search_fpir": _ratio(fp, len(unknown_evaluable)),
            "search_fpir_interval_95": _wilson(fp, len(unknown_evaluable)),
            "search_fpir_interval_95_clustered": face_fpir_clustered["interval"],
            "search_zero_event_upper_95": (
                1 - 0.05 ** (1 / len(unknown_evaluable))
                if fp == 0 and unknown_evaluable
                else None
            ),
            "known_tpir": _ratio(tp, len(known_evaluable)),
            "known_tpir_interval_95": _wilson(tp, len(known_evaluable)),
            "known_tpir_interval_95_clustered": face_tpir_clustered["interval"],
            "known_fnir": _ratio(fn, len(known_evaluable)),
            "incorrect_known_return_rate": _ratio(incorrect, len(known)),
            "face_confidence_unit": "participant cluster",
            "face_cluster_counts": {
                "known": face_tpir_clustered["clusters"],
                "unknown": face_fpir_clustered["clusters"],
            },
            "copy_search": copy_metrics,
            "preview_recall": _aggregate_preview(search_rows),
            "quality_visibility_pose": _quality_evaluation(manifest, pipeline, split),
        },
        "searches": search_rows,
        "limitations": [
            "clustered intervals treat participant/source-family clusters as independent",
            "results are conditional on this frozen gallery",
            "pretrained-model identity overlap is unknown",
        ],
    }
    write_json(output, report)
    return report


def calibrate_from_evaluation(
    evaluation_path: Path,
    output: Path,
    *,
    policy_id: str,
    fpir_upper_target: float = 0.01,
    tpir_lower_target: float = 0.90,
) -> dict[str, Any]:
    """Select thresholds on calibration searches only, under declared confidence targets."""
    raw_bytes = evaluation_path.read_bytes()
    import json

    report = json.loads(raw_bytes)
    if report.get("schema") != "faceproof.open-set-evaluation.v1":
        raise FaceInputError("unsupported evaluation report")
    if report.get("split") != "calibration":
        raise FaceInputError("thresholds may only be selected from the calibration split")
    searches = report.get("searches") or []
    _validate_calibration_report(report, searches)
    face = _select_search_threshold(
        searches,
        score_name="face_similarity",
        truth_name="correct_media",
        fpir_upper_target=fpir_upper_target,
        tpir_lower_target=tpir_lower_target,
        eligibility_name="face_search_evaluable",
        cluster_name="participant_ids",
    )
    copy_axis = _select_search_threshold(
        searches,
        score_name="copy_similarity",
        truth_name="copy_correct_media",
        fpir_upper_target=fpir_upper_target,
        tpir_lower_target=tpir_lower_target,
        eligibility_name="copy_search_evaluable",
        cluster_name="copy_reference_family_ids",
    )
    if face is None or copy_axis is None:
        raise BlockedState(
            "BLOCKED_REAL_CALIBRATION_DATA",
            "no operating point meets the declared false-alarm and useful-recall targets",
        )
    decision = {
        "schema": "faceproof.calibration-decision.v1",
        "status": "FROZEN_BEFORE_TEST",
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "policy_id": policy_id,
        "calibration_evaluation_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "thresholds": {
            "face_accept": face["threshold"],
            "face_review": None,
            "copy_accept": copy_axis["threshold"],
            "copy_review": None,
        },
        "targets": {
            "search_fpir_upper_95_max": fpir_upper_target,
            "known_tpir_lower_95_min": tpir_lower_target,
            "copy_search_false_alarm_upper_95_max": fpir_upper_target,
            "copy_search_recall_lower_95_min": tpir_lower_target,
        },
        "selected_metrics": {"face": face, "copy": copy_axis},
        "quality_validation": {"pose": False, "visibility": False},
    }
    write_json(output, decision)
    return decision


def _validate_calibration_report(
    report: dict[str, Any], searches: list[dict[str, Any]]
) -> None:
    policy = report.get("policy")
    if not isinstance(policy, dict) or policy.get("status") != "REVIEW_ONLY":
        raise FaceInputError(
            "calibration evaluation must be produced under an uncalibrated review-only policy"
        )
    if policy.get("calibration_report_sha256") or policy.get("test_report_sha256"):
        raise FaceInputError("calibration evaluation is contaminated by post-calibration artifacts")
    if not searches:
        raise BlockedState(
            "BLOCKED_REAL_CALIBRATION_DATA", "calibration evaluation has no searches"
        )
    query_ids: set[str] = set()
    for search in searches:
        query_id = search.get("query_image_id")
        if not isinstance(query_id, str) or not query_id:
            raise FaceInputError("calibration search lacks query_image_id")
        if query_id in query_ids:
            raise FaceInputError(f"duplicate calibration query_image_id: {query_id}")
        query_ids.add(query_id)
        for name in ("participant_ids", "copy_reference_family_ids"):
            values = search.get(name)
            if not isinstance(values, list) or not values:
                raise FaceInputError(f"calibration search lacks {name}")
        for name in ("face_search_evaluable", "copy_search_evaluable"):
            if not isinstance(search.get(name), bool):
                raise FaceInputError(f"calibration search lacks boolean {name}")


def _record_hit(record: ManifestRecord) -> SearchHit:
    media_id = record.media_id or record.image_id
    labels = record.labels
    return SearchHit(
        post_url=str(labels.get("post_url") or f"https://offline.invalid/post/{media_id}"),
        canonical_uri=str(labels.get("canonical_uri") or f"urn:faceproof:{media_id}"),
        post_id=str(labels.get("post_id") or media_id),
        author_id=str(labels.get("author_id") or "authorized-fixture"),
        author_handle=str(labels.get("author_handle") or "authorized-fixture"),
        created_at=str(labels.get("created_at") or "1970-01-01T00:00:00Z"),
        content_text=str(labels.get("content_text") or "authorized evaluation media"),
        media_id=media_id,
        image_url=f"manifest://{record.image_id}",
        preview_url=f"manifest://{record.image_id}",
        image_width=None,
        image_height=None,
    )


def _validate_evaluation_partition(
    queries: list[ManifestRecord], gallery: list[ManifestRecord]
) -> None:
    """Reject within-split query/gallery leakage that inflates identity results."""
    face_gallery = [record for record in gallery if "candidate_face" in record.roles]
    gallery_hashes = {record.sha256 for record in face_gallery}
    gallery_families = {record.source_image_family_id for record in face_gallery}
    for query in queries:
        if query.sha256 in gallery_hashes:
            raise FaceInputError(
                f"identity query/gallery exact-image leakage: {query.image_id}"
            )
        if query.source_image_family_id in gallery_families:
            raise FaceInputError(
                f"identity query/gallery source-family leakage: {query.image_id}"
            )


def _copy_references(
    query: ManifestRecord, by_id: dict[str, ManifestRecord]
) -> list[ManifestRecord]:
    declared = query.labels.get("copy_reference_image_ids")
    if declared is None and "copy_reference" in query.roles:
        declared = [query.image_id]
    if not isinstance(declared, list) or not declared:
        raise BlockedState(
            "BLOCKED_REAL_CALIBRATION_DATA",
            f"query {query.image_id} needs explicit copy_reference_image_ids",
        )
    result = [by_id.get(str(image_id)) for image_id in declared]
    if any(
        record is None
        or "copy_reference" not in record.roles
        or record.split != query.split
        or record.path is None
        for record in result
    ):
        raise FaceInputError(f"query {query.image_id} refers to an invalid copy reference")
    return [record for record in result if record is not None]


def _preview_metrics(
    query: ManifestRecord,
    refs: list[ManifestRecord],
    gallery: list[ManifestRecord],
    dispositions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    ref_ids = {record.image_id for record in refs}
    identity_positive = {
        record.media_id or record.image_id
        for record in gallery
        if "candidate_face" in record.roles
        and set(record.participant_ids) & set(query.participant_ids)
    }
    copy_positive = {
        record.media_id or record.image_id
        for record in gallery
        if "candidate_copy" in record.roles
        and (record.copy_parent_image_id in ref_ids or record.image_id in ref_ids)
    }
    verifier_positive = {
        media_id
        for media_id, row in dispositions.items()
        if "SUPPORTED"
        in {
            str((row.get("detail") or {}).get("face_state")),
            str((row.get("detail") or {}).get("copy_state")),
        }
    }
    result: dict[str, Any] = {}
    copy_positive_by_family = {
        reference.source_image_family_id: {
            record.media_id or record.image_id
            for record in gallery
            if "candidate_copy" in record.roles
            and (
                record.copy_parent_image_id == reference.image_id
                or record.image_id == reference.image_id
            )
        }
        for reference in refs
    }
    for k in (5, 10, 20, 50):
        face_retained = {
            media_id
            for media_id, row in dispositions.items()
            if ((row.get("detail") or {}).get("face_retrieval_rank") or math.inf) <= k
        }
        copy_retained = {
            media_id
            for media_id, row in dispositions.items()
            if ((row.get("detail") or {}).get("copy_retrieval_rank") or math.inf) <= k
        }
        union = face_retained | copy_retained
        result[str(k)] = {
            "per_axis_k": k,
            "union_size": len(union),
            "identity_positive_recall": _set_recall(identity_positive, union),
            "copy_positive_recall": _set_recall(copy_positive, union),
            "all_positive_recovery": _set_recall(identity_positive | copy_positive, union),
            "exhaustive_verifier_positive_retention": _set_recall(
                verifier_positive, union
            ),
            "exhaustive_positive_recovered_outside_preview": len(
                verifier_positive - union
            ),
            "truth_positive_count": len(identity_positive | copy_positive),
            "truth_positive_retained": len((identity_positive | copy_positive) & union),
            "truth_any_hit": bool((identity_positive | copy_positive) & union),
            "identity_all_retained": identity_positive <= union,
            "copy_all_retained": copy_positive <= union,
            "copy_family_all_retained": {
                family: positives <= union
                for family, positives in copy_positive_by_family.items()
                if positives
            },
            "identity_positive_count": len(identity_positive),
            "copy_positive_count": len(copy_positive),
        }
    return result


def _aggregate_preview(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for k in (5, 10, 20, 50):
        values = [row["preview"][str(k)] for row in rows]
        total_positive = sum(int(value["truth_positive_count"]) for value in values)
        total_retained = sum(int(value["truth_positive_retained"]) for value in values)
        result[str(k)] = {
            key: _mean_defined([value[key] for value in values])
            for key in (
                "identity_positive_recall",
                "copy_positive_recall",
                "all_positive_recovery",
                "exhaustive_verifier_positive_retention",
            )
        }
        result[str(k)]["macro_all_positive_recall"] = result[str(k)][
            "all_positive_recovery"
        ]
        result[str(k)]["micro_all_positive_recall"] = (
            round(total_retained / total_positive, 6) if total_positive else None
        )
        truth_queries = [value for value in values if value["truth_positive_count"]]
        result[str(k)]["any_hit_query_recall"] = round(
            sum(bool(value["truth_any_hit"]) for value in truth_queries)
            / len(truth_queries),
            6,
        ) if truth_queries else None
        identity_clusters: dict[tuple[str, ...], bool] = {}
        copy_clusters: dict[str, bool] = {}
        for row, value in zip(rows, values, strict=True):
            participant_key = tuple(sorted(str(item) for item in row["participant_ids"]))
            if int(value["identity_positive_count"]) > 0:
                identity_clusters[participant_key] = identity_clusters.get(
                    participant_key, True
                ) and bool(value["identity_all_retained"])
            for family, retained in value["copy_family_all_retained"].items():
                copy_clusters[family] = copy_clusters.get(family, True) and bool(retained)
        identity_success = sum(identity_clusters.values())
        copy_success = sum(copy_clusters.values())
        result[str(k)]["identity_candidate_recall_interval_95_clustered"] = _wilson(
            identity_success, len(identity_clusters)
        )
        result[str(k)]["copy_candidate_recall_interval_95_clustered"] = _wilson(
            copy_success, len(copy_clusters)
        )
        result[str(k)]["confidence_unit"] = "participant / source-image-family"
        result[str(k)]["identity_cluster_count"] = len(identity_clusters)
        result[str(k)]["copy_source_family_cluster_count"] = len(copy_clusters)
    return result


def _clustered_search_interval(
    rows: list[dict[str, Any]],
    *,
    event_name: str,
    cluster_name: str,
    require_all: bool,
) -> dict[str, Any]:
    clusters: dict[tuple[str, ...], list[bool]] = {}
    for row in rows:
        raw_cluster = row.get(cluster_name) or []
        key = tuple(sorted(str(value) for value in raw_cluster))
        if not key:
            raise FaceInputError(f"evaluation search lacks {cluster_name}")
        clusters.setdefault(key, []).append(bool(row.get(event_name)))
    outcomes = [all(events) if require_all else any(events) for events in clusters.values()]
    return {"clusters": len(outcomes), "interval": _wilson(sum(outcomes), len(outcomes))}


def _copy_search_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    known = [
        row
        for row in rows
        if row["copy_correct_media"] and row["copy_search_evaluable"]
    ]
    unknown = [
        row
        for row in rows
        if not row["copy_correct_media"] and row["copy_search_evaluable"]
    ]
    tp = sum(bool(row["copy_correct_return"]) for row in known)
    fp = sum(bool(row["accepted_copy_media"]) for row in unknown)
    known_clustered = _clustered_search_interval(
        known,
        event_name="copy_correct_return",
        cluster_name="copy_reference_family_ids",
        require_all=True,
    )
    unknown_clustered = _clustered_search_interval(
        unknown,
        event_name="accepted_copy_media",
        cluster_name="copy_reference_family_ids",
        require_all=False,
    )
    return {
        "known_searches": len(known),
        "unknown_searches": len(unknown),
        "unevaluable_searches": sum(not row["copy_search_evaluable"] for row in rows),
        "true_positive_searches": tp,
        "false_negative_searches": len(known) - tp,
        "false_positive_searches": fp,
        "recall": _ratio(tp, len(known)),
        "recall_interval_95": _wilson(tp, len(known)),
        "recall_interval_95_clustered": known_clustered["interval"],
        "false_alarm_rate": _ratio(fp, len(unknown)),
        "false_alarm_interval_95": _wilson(fp, len(unknown)),
        "false_alarm_interval_95_clustered": unknown_clustered["interval"],
        "known_source_family_clusters": known_clustered["clusters"],
        "unknown_source_family_clusters": unknown_clustered["clusters"],
        "confidence_unit": "copy-reference source-image-family cluster",
    }


def _ratio(events: int, total: int) -> float | None:
    return round(events / total, 6) if total else None


def _set_recall(positive: set[str], retained: set[str]) -> float | None:
    return round(len(positive & retained) / len(positive), 6) if positive else None


def _mean_defined(values: list[Any]) -> float | None:
    defined = [float(value) for value in values if value is not None]
    return round(sum(defined) / len(defined), 6) if defined else None


def _wilson(events: int, total: int) -> list[float] | None:
    if not total:
        return None
    z = 1.959963984540054
    p = events / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return [round(max(0.0, center - margin), 6), round(min(1.0, center + margin), 6)]


def _media_type(path: Path) -> str:
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }.get(path.suffix.lower(), "application/octet-stream")


def _candidate_rows(search: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row
        for row in search.get("terminal_dispositions") or []
        if isinstance(row.get("detail"), dict)
    ]


def _select_search_threshold(
    searches: list[dict[str, Any]],
    *,
    score_name: str,
    truth_name: str,
    fpir_upper_target: float,
    tpir_lower_target: float,
    eligibility_name: str,
    cluster_name: str,
) -> dict[str, Any] | None:
    scores = sorted(
        {
            float(row["detail"][score_name])
            for search in searches
            if search.get(eligibility_name) is True
            for row in _candidate_rows(search)
            if row["detail"].get(score_name) is not None
        }
    )
    candidates = scores + ([scores[-1] + 1e-6] if scores else [])
    eligible: list[dict[str, Any]] = []
    for threshold in candidates:
        evaluated = [search for search in searches if search.get(eligibility_name) is True]
        known = [search for search in evaluated if search.get(truth_name)]
        unknown = [search for search in evaluated if not search.get(truth_name)]
        known_outcomes: list[tuple[dict[str, Any], bool]] = []
        for search in known:
            truth = set(search.get(truth_name) or [])
            accepted = {
                str(row["media_id"])
                for row in _candidate_rows(search)
                if float(row["detail"].get(score_name, -math.inf)) >= threshold
            }
            known_outcomes.append((search, bool(accepted & truth)))
        unknown_outcomes = [
            (
                search,
                any(
                float(row["detail"].get(score_name, -math.inf)) >= threshold
                for row in _candidate_rows(search)
                ),
            )
            for search in unknown
        ]
        tp = sum(outcome for _, outcome in known_outcomes)
        fp = sum(outcome for _, outcome in unknown_outcomes)
        tp_clustered = _clustered_outcomes(
            known_outcomes, cluster_name=cluster_name, require_all=True
        )
        fp_clustered = _clustered_outcomes(
            unknown_outcomes, cluster_name=cluster_name, require_all=False
        )
        fp_interval = _wilson(sum(fp_clustered), len(fp_clustered))
        tp_interval = _wilson(sum(tp_clustered), len(tp_clustered))
        if (
            fp_interval
            and tp_interval
            and fp_interval[1] <= fpir_upper_target
            and tp_interval[0] >= tpir_lower_target
        ):
            eligible.append(
                {
                    "threshold": round(threshold, 6),
                    "fpir": _ratio(fp, len(unknown)),
                    "fpir_interval_95": fp_interval,
                    "tpir": _ratio(tp, len(known)),
                    "tpir_interval_95": tp_interval,
                    "known_searches": len(known),
                    "unknown_searches": len(unknown),
                    "known_clusters": len(tp_clustered),
                    "unknown_clusters": len(fp_clustered),
                    "confidence_unit": cluster_name,
                    "search_fpir": _ratio(fp, len(unknown)),
                    "search_tpir": _ratio(tp, len(known)),
                }
            )
    return min(eligible, key=lambda row: row["threshold"], default=None)


def _clustered_outcomes(
    outcomes: list[tuple[dict[str, Any], bool]],
    *,
    cluster_name: str,
    require_all: bool,
) -> list[bool]:
    clusters: dict[tuple[str, ...], list[bool]] = {}
    for search, outcome in outcomes:
        raw_cluster = search.get(cluster_name) or []
        key = tuple(sorted(str(value) for value in raw_cluster))
        if not key:
            raise FaceInputError(f"calibration search lacks {cluster_name}")
        clusters.setdefault(key, []).append(outcome)
    return [all(values) if require_all else any(values) for values in clusters.values()]


def _quality_evaluation(
    manifest: DatasetManifest, pipeline: DiscoveryPipeline, split: str
) -> dict[str, Any]:
    annotated = [
        record
        for record in manifest.records
        if record.split == split
        and record.path is not None
        and "candidate_face" in record.roles
        and record.face_annotations
    ]
    if not annotated or pipeline.quality_assessor is None:
        return {
            "status": "BLOCKED_QUALITY_ANNOTATIONS",
            "reason": "manual visibility and real pose-bin annotations are unavailable",
        }
    confusion: dict[str, dict[str, int]] = {}
    false_clear = 0
    occluded_total = 0
    unknown = 0
    region_total = 0
    pose_correct = 0
    pose_total = 0
    pose_unknown = 0
    quality_review = 0
    faces_total = 0
    for record in annotated:
        image = pipeline.face_engine.decode(record.path.read_bytes())
        faces = pipeline.face_engine.detect_and_encode(image)
        for annotation in record.face_annotations:
            face_index = annotation.get("face_index")
            if not isinstance(face_index, int) or not 0 <= face_index < len(faces):
                unknown += 3
                region_total += 3
                continue
            assessment = pipeline.quality_assessor.assess(image, faces[face_index])
            faces_total += 1
            quality_review += assessment.state != "ASSESSABLE"
            truth_visibility = annotation.get("visibility") or {}
            evidence = assessment.parsing.regions if assessment.parsing else {}
            for region in ("eyes", "nose", "mouth"):
                truth = str(truth_visibility.get(region) or "UNKNOWN").upper()
                predicted = str(evidence[region].visibility) if region in evidence else "UNKNOWN"
                confusion.setdefault(truth, {}).setdefault(predicted, 0)
                confusion[truth][predicted] += 1
                region_total += 1
                unknown += predicted == "UNKNOWN"
                if truth == "OCCLUDED":
                    occluded_total += 1
                    false_clear += predicted == "VISIBLE"
            truth_pose = annotation.get("pose_bin")
            if truth_pose:
                pose_total += 1
                if assessment.pose.coarse_bin == "UNKNOWN":
                    pose_unknown += 1
                pose_correct += assessment.pose.coarse_bin == str(truth_pose).upper()
    result = {
        "status": "MEASURED",
        "annotated_faces": faces_total,
        "region_confusion": confusion,
        "region_false_clear_rate": _ratio(false_clear, occluded_total),
        "region_unknown_coverage": _ratio(unknown, region_total),
        "pose_bin_accuracy": _ratio(pose_correct, pose_total),
        "pose_unknown_rate": _ratio(pose_unknown, pose_total),
        "quality_needs_review_rate": _ratio(quality_review, faces_total),
        "note": "parser evidence is not treated as universal occlusion ground truth",
    }
    targets = pipeline.policy.targets
    required = {
        "visibility_false_clear_max",
        "visibility_unknown_max",
        "pose_bin_accuracy_min",
    }
    if not required <= targets.keys():
        result["validation_decision"] = "BLOCKED_PREDECLARED_TARGETS"
    elif None in {
        result["region_false_clear_rate"],
        result["region_unknown_coverage"],
        result["pose_bin_accuracy"],
    }:
        result["validation_decision"] = "BLOCKED_INSUFFICIENT_ANNOTATIONS"
    else:
        passed = (
            float(result["region_false_clear_rate"])
            <= targets["visibility_false_clear_max"]
            and float(result["region_unknown_coverage"])
            <= targets["visibility_unknown_max"]
            and float(result["pose_bin_accuracy"])
            >= targets["pose_bin_accuracy_min"]
        )
        result["validation_decision"] = "PASS" if passed else "FAIL"
        result["predeclared_targets"] = {
            key: targets[key] for key in sorted(required)
        }
    return result
