from __future__ import annotations

import httpx
import pytest

from faceproof.errors import SearchError
from faceproof.search import MastodonMediaSource, _plain_text


@pytest.fixture(autouse=True)
def _skip_dns_for_mock_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("faceproof.search.validate_public_https_url", lambda url: None)


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_mastodon_discovers_real_post_metadata_and_prefers_cached_media() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/timelines/tag/HHGoaFaceProof"
        return httpx.Response(
            200,
            json=[
                {
                    "id": "123",
                    "uri": "https://origin.example/users/alice/statuses/123",
                    "url": "https://social.example/@alice/123",
                    "visibility": "public",
                    "created_at": "2026-09-01T00:00:00.000Z",
                    "content": "<p>Hello &amp; <b>Goa</b></p>",
                    "account": {"id": "a1", "acct": "alice@origin.example"},
                    "media_attachments": [
                        {
                            "id": "m1",
                            "type": "image",
                            "url": "https://social.example/cache/full.jpg",
                            "remote_url": "https://origin.example/full.jpg",
                            "preview_url": "https://social.example/cache/preview.jpg",
                            "meta": {"original": {"width": 1200, "height": 900}},
                        }
                    ],
                }
            ],
        )

    with _client(handler) as client:
        batch = MastodonMediaSource(
            instance="https://social.example",
            tag="#HHGoaFaceProof",
            max_pages=2,
            page_size=40,
            max_media=10,
            timeout_s=1,
            client=client,
        ).discover()

    assert batch.provider == "mastodon-public-api"
    assert batch.statuses_scanned == 1
    assert batch.media_scanned == 1
    hit = batch.hits[0]
    assert hit.post_url == "https://social.example/@alice/123"
    assert hit.image_url == "https://social.example/cache/full.jpg"
    assert hit.canonical_uri.endswith("/statuses/123")
    assert hit.content_text == "Hello & Goa"
    assert (hit.image_width, hit.image_height) == (1200, 900)


def test_mastodon_ignores_private_and_non_image_media() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[
                {
                    "id": "1",
                    "visibility": "private",
                    "media_attachments": [{"id": "m", "type": "image", "url": "https://x/a"}],
                },
                {
                    "id": "2",
                    "visibility": "public",
                    "media_attachments": [{"id": "m", "type": "video", "url": "https://x/v"}],
                },
            ],
        )

    with _client(handler) as client:
        batch = MastodonMediaSource(
            instance="https://social.example",
            tag=None,
            max_pages=1,
            page_size=40,
            max_media=10,
            timeout_s=1,
            client=client,
        ).discover()
    assert batch.hits == []
    assert batch.statuses_scanned == 2


def test_rejects_non_origin_instance() -> None:
    with pytest.raises((SearchError, ValueError)):
        MastodonMediaSource(
            instance="https://social.example/path",
            tag=None,
            max_pages=1,
            page_size=1,
            max_media=1,
            timeout_s=1,
        )


def test_plain_text_removes_markup() -> None:
    assert _plain_text("<p>A<br>B &amp; C</p>") == "A B & C"


def test_public_timeline_policy_failure_is_actionable_and_not_retried() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(422, json={"error": "public timeline disabled"})

    with _client(handler) as client:
        source = MastodonMediaSource(
            instance="https://social.example",
            tag=None,
            max_pages=1,
            page_size=40,
            max_media=10,
            timeout_s=1,
            client=client,
        )
        with pytest.raises(SearchError, match="configure FACEPROOF_MASTODON_TAG"):
            source.discover()
    assert calls == 1


def test_authorized_multi_post_pagination_boosts_and_multi_image() -> None:
    calls: list[str | None] = []

    def status(identifier: str, account: str, media_count: int, *, reblog: bool = False) -> dict:
        actual = {
            "id": identifier,
            "uri": f"https://social.example/users/{account}/statuses/{identifier}",
            "url": f"https://social.example/@{account}/{identifier}",
            "visibility": "public",
            "created_at": "2026-09-05T00:00:00Z",
            "content": "<p>authorized demo</p>",
            "account": {"id": f"id-{account}", "acct": account},
            "media_attachments": [
                {
                    "id": f"{identifier}-m{index}",
                    "type": "image",
                    "url": f"https://social.example/media/{identifier}-{index}.jpg",
                    "preview_url": f"https://social.example/media/{identifier}-{index}-p.jpg",
                }
                for index in range(media_count)
            ],
        }
        if not reblog:
            return actual
        return {
            "id": f"boost-{identifier}",
            "uri": f"https://social.example/boost/{identifier}",
            "reblog": actual,
        }

    def handler(request: httpx.Request) -> httpx.Response:
        max_id = request.url.params.get("max_id")
        calls.append(max_id)
        if max_id is None:
            return httpx.Response(200, json=[status("3", "alice", 2), status("2", "mallory", 1)])
        if max_id == "2":
            return httpx.Response(200, json=[status("1", "alice", 1, reblog=True)])
        return httpx.Response(200, json=[])

    with _client(handler) as client:
        batch = MastodonMediaSource(
            instance="https://social.example",
            tag="demo",
            max_pages=3,
            page_size=2,
            max_media=10,
            timeout_s=1,
            allowed_accounts=frozenset({"alice"}),
            client=client,
        ).discover()

    assert calls == [None, "2"]
    assert [hit.media_id for hit in batch.hits] == ["3-m0", "3-m1", "1-m0"]
    assert batch.hits[-1].wrapper_post_id == "boost-1"
    assert batch.hits[-1].post_id == "1"
