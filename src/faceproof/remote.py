from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit

import httpx

from .errors import SearchError, UnsafeRemoteResource

ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/gif",
    "image/bmp",
}


def image_media_type(content: bytes) -> str | None:
    if content.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if content.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if content.startswith(b"BM"):
        return "image/bmp"
    if len(content) >= 12 and content[:4] == b"RIFF" and content[8:12] == b"WEBP":
        return "image/webp"
    return None


@dataclass(frozen=True, slots=True)
class RemoteImage:
    url: str
    content: bytes
    media_type: str


def _is_public_ip(address: str) -> bool:
    ip = ipaddress.ip_address(address)
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def validate_public_https_url(url: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme.lower() != "https":
        raise UnsafeRemoteResource("candidate image URL must use HTTPS")
    if parsed.username or parsed.password:
        raise UnsafeRemoteResource("candidate URL must not contain credentials")
    if not parsed.hostname:
        raise UnsafeRemoteResource("candidate URL has no hostname")

    try:
        addresses = {
            item[4][0]
            for item in socket.getaddrinfo(
                parsed.hostname, parsed.port or 443, type=socket.SOCK_STREAM
            )
        }
    except socket.gaierror as exc:
        raise UnsafeRemoteResource(
            f"candidate hostname does not resolve: {parsed.hostname}"
        ) from exc

    if not addresses or any(not _is_public_ip(address) for address in addresses):
        raise UnsafeRemoteResource("candidate URL resolves to a non-public address")


class SafeImageFetcher:
    def __init__(
        self,
        *,
        timeout_s: float,
        max_bytes: int,
        max_redirects: int,
        user_agent: str = "FaceProof/0.3 (+https://github.com/BlueBlaze6335/hhg-faceproof)",
    ) -> None:
        self.timeout_s = timeout_s
        self.max_bytes = max_bytes
        self.max_redirects = max_redirects
        self.user_agent = user_agent
        self._client = httpx.Client(
            timeout=self.timeout_s,
            follow_redirects=False,
            trust_env=False,
            limits=httpx.Limits(max_connections=12, max_keepalive_connections=8),
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> SafeImageFetcher:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def fetch(self, url: str) -> RemoteImage:
        current = url
        headers = {
            "Accept": "image/avif,image/webp,image/png,image/jpeg,image/*;q=0.8",
            "User-Agent": self.user_agent,
        }
        for redirect_count in range(self.max_redirects + 1):
            validate_public_https_url(current)
            try:
                with self._client.stream("GET", current, headers=headers) as response:
                    if response.status_code in {301, 302, 303, 307, 308}:
                        location = response.headers.get("location")
                        if not location:
                            raise SearchError("image redirect omitted Location header")
                        if redirect_count == self.max_redirects:
                            raise SearchError("image exceeded redirect limit")
                        current = urljoin(current, location)
                        continue

                    response.raise_for_status()
                    content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
                    if content_type not in ALLOWED_IMAGE_TYPES:
                        displayed_type = content_type or "unknown"
                        raise SearchError(
                            f"candidate returned unsupported content type {displayed_type}"
                        )

                    declared = int(response.headers.get("content-length", "0") or 0)
                    if declared > self.max_bytes:
                        raise SearchError("candidate image exceeds the configured size limit")

                    chunks: list[bytes] = []
                    received = 0
                    for chunk in response.iter_bytes():
                        received += len(chunk)
                        if received > self.max_bytes:
                            raise SearchError("candidate image exceeds the configured size limit")
                        chunks.append(chunk)
                    if not chunks:
                        raise SearchError("candidate image was empty")
                    content = b"".join(chunks)
                    detected_type = image_media_type(content)
                    if detected_type is None:
                        raise SearchError("candidate body has no supported image signature")
                    if detected_type != content_type:
                        raise SearchError(
                            "candidate image signature does not match its declared content type"
                        )
                    return RemoteImage(current, content, detected_type)
            except httpx.HTTPError as exc:
                raise SearchError(f"candidate download failed: {exc}") from exc

        raise SearchError("candidate download did not complete")
