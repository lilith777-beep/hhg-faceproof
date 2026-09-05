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


def _bounded_gray(image: Any, cv2: Any, max_dimension: int = 512) -> tuple[Any, float]:
    height, width = image.shape[:2]
    scale = min(1.0, max_dimension / max(height, width))
    if scale < 1.0:
        image = cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return gray, scale


def geometric_match(query: Any, candidate: Any, cv2: Any) -> FeatureMatch:
    query_gray, query_scale = _bounded_gray(query, cv2)
    candidate_gray, candidate_scale = _bounded_gray(candidate, cv2)
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
    homography, mask = cv2.findHomography(query_points, candidate_points, cv2.RANSAC, 3.0)
    if mask is None or homography is None or not np.all(np.isfinite(homography)):
        return FeatureMatch(len(good), 0, 0.0)
    inliers = int(mask.ravel().sum())
    if inliers < 4 or abs(float(np.linalg.det(homography[:2, :2]))) < 1e-8:
        return FeatureMatch(len(good), inliers, inliers / len(good))
    keep = mask.ravel().astype(bool)
    source_inliers = query_points[keep]
    candidate_inliers = candidate_points[keep]
    projected = cv2.perspectiveTransform(
        source_inliers.reshape(-1, 1, 2), homography
    ).reshape(-1, 2)
    diagonal = float(np.hypot(*candidate_gray.shape[:2]))
    reprojection = float(np.linalg.norm(projected - candidate_inliers, axis=1).mean() / diagonal)

    def coverage(points: Any, shape: tuple[int, ...]) -> float:
        hull = cv2.convexHull(points.reshape(-1, 1, 2))
        return float(cv2.contourArea(hull) / (shape[0] * shape[1]))

    source_coverage = coverage(source_inliers, query_gray.shape)
    candidate_coverage = coverage(candidate_inliers, candidate_gray.shape)
    source_transform = np.array(
        [[query_scale, 0, 0], [0, query_scale, 0], [0, 0, 1]], dtype=np.float64
    )
    candidate_inverse = np.array(
        [[1 / candidate_scale, 0, 0], [0, 1 / candidate_scale, 0], [0, 0, 1]],
        dtype=np.float64,
    )
    full_homography = candidate_inverse @ homography @ source_transform
    full_homography /= full_homography[2, 2]
    return FeatureMatch(
        len(good),
        inliers,
        inliers / len(good),
        reprojection,
        source_coverage,
        candidate_coverage,
        tuple(float(value) for value in full_homography.reshape(-1)),
    )
