from __future__ import annotations

import html
import time
from datetime import UTC, datetime
from html.parser import HTMLParser
from typing import Any, Protocol
from urllib.parse import quote, urlsplit

import httpx

from .errors import SearchError
from .models import SearchBatch, SearchHit
from .remote import validate_public_https_url


class CandidateSource(Protocol):
    name: str

    def discover(self) -> SearchBatch: ...


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def _plain_text(markup: str) -> str:
    parser = _TextExtractor()
    parser.feed(markup)
    return " ".join(html.unescape(" ".join(parser.parts)).split())


class MastodonMediaSource:
    """Enumerate current public image posts from an open Mastodon timeline."""

    name = "mastodon-public-api"

    def __init__(
        self,
        *,
        instance: str,
        tag: str | None,
        max_pages: int,
        page_size: int,
        max_media: int,
        timeout_s: float,
        allowed_accounts: frozenset[str] | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        instance = instance.rstrip("/")
        validate_public_https_url(instance)
        parsed = urlsplit(instance)
        if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
            raise SearchError("Mastodon instance must be an HTTPS origin without a path")
        self.instance = instance
        self.tag = tag.lstrip("#") if tag else None
        self.max_pages = max_pages
        self.page_size = page_size
        self.max_media = max_media
        self.allowed_accounts = allowed_accounts
        self.client = client or httpx.Client(
            timeout=timeout_s,
            follow_redirects=False,
            trust_env=False,
            limits=httpx.Limits(max_connections=6, max_keepalive_connections=4),
        )
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def _endpoint(self) -> str:
        if self.tag:
            return f"{self.instance}/api/v1/timelines/tag/{quote(self.tag, safe='')}"
        return f"{self.instance}/api/v1/timelines/public"

    def _get_page(self, endpoint: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        for attempt in range(3):
            try:
                response = self.client.get(
                    endpoint,
                    params=params,
                    headers={"Accept": "application/json", "User-Agent": "FaceProof/0.3"},
                )
                if response.status_code == 429:
                    delay = min(float(response.headers.get("retry-after", "1") or 1), 3.0)
                    time.sleep(delay)
                    continue
                if response.status_code >= 500 and attempt < 2:
                    time.sleep(0.25 * (2**attempt))
                    continue
                if response.status_code in {401, 403, 422} and not self.tag:
                    raise SearchError(
                        "this instance does not expose its anonymous public timeline; "
                        "configure FACEPROOF_MASTODON_TAG or pass --tag"
                    )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, list):
                    raise SearchError("Mastodon timeline returned a non-list response")
                return [item for item in payload if isinstance(item, dict)]
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code < 500 and exc.response.status_code != 429:
                    raise SearchError(
                        f"Mastodon rejected the timeline request with HTTP "
                        f"{exc.response.status_code}"
                    ) from exc
                if attempt == 2:
                    raise SearchError(f"Mastodon timeline request failed: {exc}") from exc
                time.sleep(0.25 * (2**attempt))
            except (httpx.HTTPError, ValueError) as exc:
                if attempt == 2:
                    raise SearchError(f"Mastodon timeline request failed: {exc}") from exc
                time.sleep(0.25 * (2**attempt))
        raise SearchError("Mastodon timeline request exhausted retries")

    def discover(self) -> SearchBatch:
        endpoint = self._endpoint()
        hits: list[SearchHit] = []
        seen_media: set[tuple[str, str]] = set()
        pages_fetched = 0
        statuses_scanned = 0
        max_id: str | None = None

        try:
            for _ in range(self.max_pages):
                params: dict[str, Any] = {"limit": self.page_size, "only_media": "true"}
                if max_id:
                    params["max_id"] = max_id
                statuses = self._get_page(endpoint, params)
                pages_fetched += 1
                if not statuses:
                    break
                statuses_scanned += len(statuses)

                for wrapper in statuses:
                    status = wrapper.get("reblog") or wrapper
                    if not isinstance(status, dict) or status.get("visibility") != "public":
                        continue
                    account = status.get("account") or {}
                    account_keys = {
                        str(account.get("id") or ""),
                        str(account.get("acct") or "").lower(),
                        str(account.get("username") or "").lower(),
                    }
                    if self.allowed_accounts and not (
                        account_keys & {value.lower() for value in self.allowed_accounts}
                    ):
                        continue
                    for media in status.get("media_attachments") or []:
                        if not isinstance(media, dict) or media.get("type") != "image":
                            continue
                        # Prefer the instance-cached copy. It is within the operator-selected
                        # origin and remains available when a federated origin is slow or gone.
                        image_url = str(media.get("url") or media.get("remote_url") or "")
                        preview_url = str(media.get("preview_url") or image_url)
                        post_url = str(status.get("url") or status.get("uri") or "")
                        media_id = str(media.get("id") or image_url)
                        key = (str(status.get("id", "")), media_id)
                        if not image_url or not post_url or key in seen_media:
                            continue
                        seen_media.add(key)
                        original_meta = (media.get("meta") or {}).get("original") or {}
                        hits.append(
                            SearchHit(
                                post_url=post_url,
                                canonical_uri=str(status.get("uri") or post_url),
                                post_id=str(status.get("id") or ""),
                                author_id=str(account.get("id") or ""),
                                author_handle=str(
                                    account.get("acct") or account.get("username") or ""
                                ),
                                created_at=str(status.get("created_at") or ""),
                                content_text=_plain_text(str(status.get("content") or "")),
                                media_id=media_id,
                                image_url=image_url,
                                preview_url=preview_url,
                                image_width=_optional_int(original_meta.get("width")),
                                image_height=_optional_int(original_meta.get("height")),
                                wrapper_post_id=(
                                    str(wrapper.get("id") or "")
                                    if wrapper is not status
                                    else None
                                ),
                                wrapper_canonical_uri=(
                                    str(wrapper.get("uri") or wrapper.get("url") or "")
                                    if wrapper is not status
                                    else None
                                ),
                            )
                        )
                        if len(hits) >= self.max_media:
                            break
                    if len(hits) >= self.max_media:
                        break
                if len(hits) >= self.max_media or len(statuses) < self.page_size:
                    break
                max_id = str(statuses[-1].get("id") or "") or None
                if max_id is None:
                    break
        finally:
            self.close()

        scope = f"hashtag #{self.tag}" if self.tag else "public media timeline"
        return SearchBatch(
            provider=self.name,
            live_query=True,
            scope=f"{self.instance} {scope}",
            endpoint=endpoint,
            fetched_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            pages_fetched=pages_fetched,
            statuses_scanned=statuses_scanned,
            media_scanned=len(hits),
            hits=hits,
        )


def _optional_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
