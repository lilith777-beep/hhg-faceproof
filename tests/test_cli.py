from faceproof.cli import _show_match, console


def test_show_match_renders_review_only_candidate_without_thresholds() -> None:
    bundle = {
        "match": {
            "post_url": "https://social.example/@owner/1",
            "author_handle": "owner",
            "canonical_uri": "https://social.example/users/owner/statuses/1",
            "face_similarity": 0.81,
            "face_threshold": None,
            "face_match": False,
            "sscd_similarity": 0.73,
            "sscd_threshold": None,
            "decision": "review",
            "same_content": False,
            "ambiguous": True,
            "candidate_image_sha256": "a" * 64,
        }
    }

    with console.capture() as capture:
        _show_match(bundle)

    rendered = capture.get()
    assert rendered.count("not calibrated") == 2
    assert "review" in rendered
