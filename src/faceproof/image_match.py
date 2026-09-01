from __future__ import annotations

from typing import Any

import numpy as np

from .models import FeatureMatch


def perceptual_hash(image: Any, cv2: Any) -> int:
    """Return a compact DCT perceptual hash using only OpenCV core operations."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(resized)
    low = dct[:8, :8].reshape(-1)
    median = float(np.median(low[1:]))
    value = 0
    for bit in low > median:
        value = (value << 1) | int(bit)
    return value


def hash_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def _bounded_gray(image: Any, cv2: Any, max_dimension: int = 512) -> Any:
    height, width = image.shape[:2]
    scale = min(1.0, max_dimension / max(height, width))
    if scale < 1.0:
        image = cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


def geometric_match(query: Any, candidate: Any, cv2: Any) -> FeatureMatch:
    query_gray = _bounded_gray(query, cv2)
    candidate_gray = _bounded_gray(candidate, cv2)
    detector = cv2.AKAZE_create()
    query_keypoints, query_descriptors = detector.detectAndCompute(query_gray, None)
    candidate_keypoints, candidate_descriptors = detector.detectAndCompute(candidate_gray, None)
    if query_descriptors is None or candidate_descriptors is None:
        return FeatureMatch(0, 0, 0.0)
    if len(query_descriptors) < 2 or len(candidate_descriptors) < 2:
        return FeatureMatch(0, 0, 0.0)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(query_descriptors, candidate_descriptors, k=2)
    good = [
        pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance
    ]
    if len(good) < 4:
        return FeatureMatch(len(good), 0, 0.0)

    query_points = np.float32([query_keypoints[item.queryIdx].pt for item in good])
    candidate_points = np.float32([candidate_keypoints[item.trainIdx].pt for item in good])
    _, mask = cv2.findHomography(query_points, candidate_points, cv2.RANSAC, 3.0)
    if mask is None:
        return FeatureMatch(len(good), 0, 0.0)
    inliers = int(mask.ravel().sum())
    return FeatureMatch(len(good), inliers, inliers / len(good))
