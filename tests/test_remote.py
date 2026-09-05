import socket

import httpx
import pytest

from faceproof.errors import UnsafeRemoteResource
from faceproof.remote import SafeImageFetcher, image_media_type, validate_public_https_url


def test_rejects_non_https_before_dns() -> None:
    with pytest.raises(UnsafeRemoteResource, match="HTTPS"):
        validate_public_https_url("http://example.com/image.jpg")


def test_rejects_embedded_credentials() -> None:
    with pytest.raises(UnsafeRemoteResource, match="credentials"):
        validate_public_https_url("https://user:pass@example.com/image.jpg")


def test_rejects_private_dns_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args, **kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))],
    )
    with pytest.raises(UnsafeRemoteResource, match="non-public"):
        validate_public_https_url("https://images.example/image.jpg")


def test_accepts_public_dns_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args, **kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
        ],
    )
    validate_public_https_url("https://images.example/image.jpg")


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (b"\xff\xd8\xffrest", "image/jpeg"),
        (b"\x89PNG\r\n\x1a\nrest", "image/png"),
        (b"GIF89arest", "image/gif"),
        (b"BMrest", "image/bmp"),
        (b"RIFF\x00\x00\x00\x00WEBPrest", "image/webp"),
        (b"<html>not an image</html>", None),
    ],
)
def test_image_signature_detection(content: bytes, expected: str | None) -> None:
    assert image_media_type(content) == expected


class _PeerStream:
    def __init__(self, address: str) -> None:
        self.address = address

    def get_extra_info(self, name: str):
        return (self.address, 443) if name == "server_addr" else None


def test_connected_peer_blocks_dns_rebinding_and_test_loopback_is_explicit() -> None:
    response = httpx.Response(
        200, extensions={"network_stream": _PeerStream("127.0.0.1")}
    )
    fetcher = SafeImageFetcher(timeout_s=1, max_bytes=10, max_redirects=0)
    with pytest.raises(UnsafeRemoteResource, match="not a public"):
        fetcher._validate_connected_peer(response)
    test_fetcher = SafeImageFetcher(
        timeout_s=1, max_bytes=10, max_redirects=0, allow_loopback_for_tests=True
    )
    test_fetcher._validate_connected_peer(response)
    fetcher.close()
    test_fetcher.close()
