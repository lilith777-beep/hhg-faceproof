import numpy as np
import pytest

from faceproof.errors import FaceInputError
from faceproof.faces import FaceEngine
from faceproof.models import BoundingBox, FaceObservation


def test_multiple_enrollment_faces_require_explicit_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    faces = [
        FaceObservation(BoundingBox(0, 0, 200, 200, 0.99), np.array([1.0, 0.0])),
        FaceObservation(BoundingBox(20, 20, 80, 80, 0.98), np.array([0.0, 1.0])),
    ]
    monkeypatch.setattr(FaceEngine, "detect_and_encode", lambda self, image: faces)
    monkeypatch.setattr(FaceEngine, "_require_query_quality", lambda self, image, face: None)
    engine = FaceEngine.__new__(FaceEngine)
    with pytest.raises(FaceInputError, match="explicitly select"):
        engine.select_query_face(object())
    assert engine.select_query_face(object(), 1) is faces[1]
    with pytest.raises(FaceInputError, match="out of range"):
        engine.select_query_face(object(), 2)
