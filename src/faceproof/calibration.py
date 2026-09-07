from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from .copydetect import CopyDescriptorEngine
from .errors import BlockedState, FaceInputError
from .faces import FaceEngine
from .integrity import write_json

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(frozen=True, slots=True)
class ThresholdMetrics:
    threshold: float
    true_accept_rate: float
    true_reject_rate: float
    false_accept_rate: float
    false_reject_rate: float
    balanced_accuracy: float


def choose_threshold(
    positive_scores: list[float], negative_scores: list[float]
) -> ThresholdMetrics:
    """Choose a deterministic balanced-accuracy threshold, preferring lower FAR."""
    if len(positive_scores) < 2 or len(negative_scores) < 5:
        raise BlockedState(
            "BLOCKED_REAL_CALIBRATION_DATA",
            "calibration needs at least 2 positive and 5 negative comparisons",
        )
    values = sorted(set(positive_scores + negative_scores))
    candidates = [values[0] - 1e-6, values[-1] + 1e-6]
    candidates.extend((left + right) / 2 for left, right in pairwise(values))
    metrics: list[ThresholdMetrics] = []
    for threshold in candidates:
        true_accepts = sum(score >= threshold for score in positive_scores)
        false_accepts = sum(score >= threshold for score in negative_scores)
        tar = true_accepts / len(positive_scores)
        far = false_accepts / len(negative_scores)
        trr = 1.0 - far
        metrics.append(
            ThresholdMetrics(
                threshold=threshold,
                true_accept_rate=tar,
                true_reject_rate=trr,
                false_accept_rate=far,
                false_reject_rate=1.0 - tar,
                balanced_accuracy=(tar + trr) / 2,
            )
        )
    return max(
        metrics,
        key=lambda item: (item.balanced_accuracy, -item.false_accept_rate, item.threshold),
    )


def _paths(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FaceInputError(f"calibration directory does not exist: {directory}")
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _score(engine: FaceEngine, query: Any, path: Path) -> tuple[float, str]:
    content = path.read_bytes()
    image = engine.decode(content)
    faces = engine.detect_and_encode(image)
    score, _, _ = engine.best_match(query, faces)
    if not faces:
        raise FaceInputError(f"no face detected in calibration image: {path.name}")
    return score, hashlib.sha256(content).hexdigest()


def calibrate(
    engine: FaceEngine,
    reference: Path,
    positives: Path,
    negatives: Path,
    output: Path,
) -> dict[str, Any]:
    reference_bytes = reference.read_bytes()
    query = engine.require_single_query_face(engine.decode(reference_bytes))
    positive_rows = [(_path.name, *_score(engine, query, _path)) for _path in _paths(positives)]
    negative_rows = [(_path.name, *_score(engine, query, _path)) for _path in _paths(negatives)]
    result = choose_threshold([row[1] for row in positive_rows], [row[1] for row in negative_rows])
    report: dict[str, Any] = {
        "schema": "faceproof.calibration.v1",
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "detector": engine.detector_id,
        "encoder": engine.encoder_id,
        "reference_sha256": hashlib.sha256(reference_bytes).hexdigest(),
        "positive_count": len(positive_rows),
        "negative_count": len(negative_rows),
        "recommended_threshold": round(result.threshold, 6),
        "metrics": {
            "true_accept_rate": round(result.true_accept_rate, 6),
            "true_reject_rate": round(result.true_reject_rate, 6),
            "false_accept_rate": round(result.false_accept_rate, 6),
            "false_reject_rate": round(result.false_reject_rate, 6),
            "balanced_accuracy": round(result.balanced_accuracy, 6),
        },
        "samples": {
            "positive": [
                {"filename": name, "score": round(score, 6), "sha256": digest}
                for name, score, digest in positive_rows
            ],
            "negative": [
                {"filename": name, "score": round(score, 6), "sha256": digest}
                for name, score, digest in negative_rows
            ],
        },
        "warning": (
            "This threshold is specific to this validation set; human review remains required."
        ),
    }
    write_json(output, report)
    return report


def calibrate_copy(
    engine: CopyDescriptorEngine,
    decode: Callable[[bytes], Any],
    reference: Path,
    positives: Path,
    negatives: Path,
    output: Path,
) -> dict[str, Any]:
    """Calibrate copy similarity from derived copies and unrelated-image negatives."""
    positive_paths = _paths(positives)
    negative_paths = _paths(negatives)
    reference_bytes = reference.read_bytes()
    all_paths = positive_paths + negative_paths
    contents = [path.read_bytes() for path in all_paths]
    reference_descriptor = engine.encode(decode(reference_bytes))
    descriptors = engine.encode_many([decode(content) for content in contents])
    scores = [float(np.dot(reference_descriptor, descriptor)) for descriptor in descriptors]
    positive_scores = scores[: len(positive_paths)]
    negative_scores = scores[len(positive_paths) :]
    result = choose_threshold(positive_scores, negative_scores)

    def rows(paths: list[Path], row_scores: list[float], offset: int) -> list[dict[str, Any]]:
        return [
            {
                "filename": path.name,
                "score": round(score, 6),
                "sha256": hashlib.sha256(contents[offset + index]).hexdigest(),
            }
            for index, (path, score) in enumerate(zip(paths, row_scores, strict=True))
        ]

    report: dict[str, Any] = {
        "schema": "faceproof.copy-calibration.v1",
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "model": engine.model_id,
        "model_sha256": engine.model_sha256,
        "dimensions": engine.dimensions,
        "reference_sha256": hashlib.sha256(reference_bytes).hexdigest(),
        "positive_definition": "transformed or recompressed copies of the reference image",
        "negative_definition": "unrelated images, including visually similar hard negatives",
        "positive_count": len(positive_paths),
        "negative_count": len(negative_paths),
        "recommended_threshold": round(result.threshold, 6),
        "metrics": {
            "true_accept_rate": round(result.true_accept_rate, 6),
            "true_reject_rate": round(result.true_reject_rate, 6),
            "false_accept_rate": round(result.false_accept_rate, 6),
            "false_reject_rate": round(result.false_reject_rate, 6),
            "balanced_accuracy": round(result.balanced_accuracy, 6),
        },
        "samples": {
            "positive": rows(positive_paths, positive_scores, 0),
            "negative": rows(negative_paths, negative_scores, len(positive_paths)),
        },
        "warning": (
            "Use held-out social-media transformations before changing the production threshold; "
            "same-person photos are not copy-positive samples."
        ),
    }
    write_json(output, report)
    return report
