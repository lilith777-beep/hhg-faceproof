from __future__ import annotations

import cv2
import numpy as np

from faceproof.image_match import geometric_match, hash_distance, perceptual_hash


def _pattern() -> np.ndarray:
    image = np.full((420, 420, 3), 240, dtype=np.uint8)
    cv2.rectangle(image, (40, 40), (380, 380), (20, 70, 160), 5)
    cv2.circle(image, (210, 210), 95, (30, 160, 30), 8)
    cv2.putText(image, "FACEPROOF", (80, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
    for index in range(20):
        point = (30 + index * 18, 60 + (index * 47) % 300)
        cv2.circle(image, point, 3, (index * 7, 20, 255 - index * 7), -1)
    return image


def test_perceptual_hash_is_stable_under_jpeg_round_trip() -> None:
    original = _pattern()
    ok, encoded = cv2.imencode(".jpg", original, [cv2.IMWRITE_JPEG_QUALITY, 80])
    assert ok
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    distance = hash_distance(perceptual_hash(original, cv2), perceptual_hash(decoded, cv2))
    assert distance <= 5


def test_akaze_confirms_a_geometrically_transformed_copy() -> None:
    original = _pattern()
    matrix = cv2.getRotationMatrix2D((210, 210), 4, 0.96)
    transformed = cv2.warpAffine(original, matrix, (420, 420))
    match = geometric_match(original, transformed, cv2)
    assert match.good_matches >= 10
    assert match.inliers >= 8
    assert match.inlier_ratio >= 0.5


def test_akaze_handles_textureless_images() -> None:
    blank = np.zeros((200, 200, 3), dtype=np.uint8)
    assert geometric_match(blank, blank, cv2).good_matches == 0
