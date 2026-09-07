"""
api.py — FastAPI surface for the harness.
`build_local_harness()` wires the in-memory/mock stack so the API runs + demos with ZERO keys
(great for local + the Vercel-hosted frontend during dev). Modal swaps in the real backends.
"""
from __future__ import annotations

import threading
import time as _t
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# the funnel/tunnel endpoints are PUBLIC: a 3GB "audio" part would spool to disk, load into
# RAM (x2: Starlette part + httpx re-encode for Sarvam) and OOM the single serving box
MAX_BODY_BYTES = 12 * 1024 * 1024
MIN_AUDIO_BYTES = 512          # below any real container+frame: reject before spending Sarvam
# per-IP token buckets (requests, window seconds). /ask carries Sarvam spend -> stricter.
RATE_LIMITS = {"voice": (20, 60.0), "text": (60, 60.0)}
# browsers only enforce CORS; curl is unaffected. Allow the prod frontend, its Vercel
# previews, and localhost dev — everything the demo uses, nothing else.
ALLOWED_ORIGIN_RE = (r"^https://hhgvrag(-[a-z0-9]+-blueblaze6335s-projects)?\.vercel\.app$"
                     r"|^https?://(localhost|127\.0\.0\.1)(:\d+)?$")


class _TooLarge(Exception):
    pass


class BodyLimitMiddleware:
    """ASGI-level request-body cap: fast-reject on declared Content-Length, then count the
    streamed bytes (chunked/lying clients) and abort with 413 before the app spools them."""

    def __init__(self, app, max_bytes: int = MAX_BODY_BYTES):
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        for k, v in scope.get("headers", []):
            if k == b"content-length":
                try:
                    if int(v) > self.max_bytes:
                        await self._send_413(send)
                        return
                except ValueError:
                    pass
        seen = 0
        started = False

        async def recv():
            nonlocal seen
            msg = await receive()
            if msg["type"] == "http.request":
                seen += len(msg.get("body", b""))
                if seen > self.max_bytes:
                    raise _TooLarge()
            return msg

        async def snd(msg):
            nonlocal started
            if msg["type"] == "http.response.start":
                started = True
            await send(msg)

        try:
            await self.app(scope, recv, snd)
        except _TooLarge:
            if not started:
                await self._send_413(send)
            # response already started -> nothing clean left to send; connection ends

    @staticmethod
    async def _send_413(send):
        body = b'{"detail":"request body too large"}'
        await send({"type": "http.response.start", "status": 413,
                    "headers": [(b"content-type", b"application/json"),
                                (b"content-length", str(len(body)).encode()),
                                (b"access-control-allow-origin", b"*")]})
        await send({"type": "http.response.body", "body": body})


class _RateLimiter:
    """Tiny per-IP token bucket. Funnel/cloudflared put the real client in X-Forwarded-For;
    bare loopback with no XFF is local dev / tests -> exempt. In-memory by design: limits
    reset on restart, which is fine for an abuse brake (not billing)."""

    def __init__(self):
        self._buckets: dict = {}
        self._lock = threading.Lock()

    @staticmethod
    def client_ip(request) -> Optional[str]:
        xff = request.headers.get("x-forwarded-for", "")
        if xff:
            return xff.split(",")[0].strip() or None
        host = request.client.host if request.client else ""
        if host in ("127.0.0.1", "::1", "localhost", "testclient", ""):
            return None                                    # local / test: exempt
        return host

    def allow(self, ip: str, bucket: str) -> bool:
        limit, window = RATE_LIMITS[bucket]
        now = _t.monotonic()
        with self._lock:
            if len(self._buckets) > 4096:                  # abuse-scale churn: drop stale
                cutoff = now - 2 * window
                for k in [k for k, (_, ts) in self._buckets.items() if ts < cutoff]:
                    del self._buckets[k]
            tokens, ts = self._buckets.get((ip, bucket), (float(limit), now))
            tokens = min(float(limit), tokens + (now - ts) * (limit / window))
            if tokens < 1.0:
                self._buckets[(ip, bucket)] = (tokens, now)
                return False
            self._buckets[(ip, bucket)] = (tokens - 1.0, now)
            return True

import chunking as ck
from cache import SemanticResponseCache, SessionContext
from embeddings import HashEmbedder
from generation import ExtractiveGenerator
from guardrails import KeywordSafety
from harness import RAGHarness
from index_build import synthetic_docs
from normalize import QueryNormalizer
from reranker import LexicalReranker
from retrieval import Retriever, memory_client
from router import HeuristicRouter
from schemas import Query, RAGResponse
from stt import MockSTT


class AskText(BaseModel):
    # max_length: an unbounded string costs seconds of CPU across lower()/regex/tokenize/
    # normalize/embed — free to repeat for an abuser, and no real question needs more
    text: str = Field(max_length=4000)
    top_k: int = Field(default=12, gt=0, le=100)   # Stage-A winner k=12 on 73ca3e90
    session_id: Optional[str] = Field(default=None, max_length=64, pattern=r"^[\w-]{1,64}$")
    quality_mode: bool = False
    retrieval_mode: Optional[str] = None   # "hybrid" (default) | "pageindex" (vectorless)


def build_local_harness() -> RAGHarness:
    """No-keys, in-memory harness for local dev / tests / the demo frontend."""
    from pageindex import PageIndexTree
    from raptor import ExtractSummarizer, RaptorTreeBuilder

    emb = HashEmbedder(dim=512)
    client = memory_client()
    retr = Retriever(client, emb, use_sparse=True)
    chunker = ck.PassageAwareChunker(min_tokens=1, max_tokens=200)
    docs = synthetic_docs(60)
    chunks = chunker.chunk_corpus(docs)
    embeds = emb.embed_docs([c.text for c in chunks])
    retr.index("demo__passage", chunks, embeds)
    summaries, _ = RaptorTreeBuilder(emb, ExtractSummarizer(),
                                     cluster_size=5, max_levels=3).build(chunks, embeds)
    pageindex = PageIndexTree.from_nodes(chunks, summaries)
    return RAGHarness(
        embedder=emb, retriever=retr, generator=ExtractiveGenerator(),
        safety=KeywordSafety(), collection="demo__passage",
        ood_threshold=0.15, grounding_min_support=0.25,
        stt=MockSTT("what is qdrant", "en"),
        normalizer=QueryNormalizer(), router=HeuristicRouter(),
        # rerank_strong collapses to rerank_min here: the bars are per-reranker
        # calibrations, and LexicalReranker's "confident match" tops out ~0.67 where
        # BGE's sits >=0.9 — the two-bar policy itself is covered by feature tests.
        reranker=LexicalReranker(), rerank_min=0.2, rerank_strong=0.2, rerank_candidates=20,
        response_cache=SemanticResponseCache(threshold=0.92, max_size=256),
        session_ctx=SessionContext(max_turns=5), pageindex=pageindex,
        stage_timeouts={"retrieve": 60, "rerank": 40, "generate": 150, "ground_check": 40})


def create_app(harness: Optional[RAGHarness] = None) -> FastAPI:
    # lazy: never build a harness at import time (Modal's serve() does `from api import create_app`,
    # so an eager build would spin a throwaway in-memory index on every GPU cold start).
    state = {"h": harness}

    def get_h() -> RAGHarness:
        if state["h"] is None:
            state["h"] = build_local_harness()
        return state["h"]

    app = FastAPI(title="hhgvrag — Voice RAG", version="0.1")
    app.add_middleware(CORSMiddleware, allow_origins=[],
                       allow_origin_regex=ALLOWED_ORIGIN_RE,
                       allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["*"])
    app.add_middleware(BodyLimitMiddleware, max_bytes=MAX_BODY_BYTES)

    limiter = _RateLimiter()

    @app.middleware("http")
    async def open_liveness_cors(request, call_next):
        # /health and /status are public, read-only, no-credential liveness/metadata — CORS
        # on them protects nothing (they expose only up-ness + already-public corpus counts)
        # but a strict allowlist breaks legitimate cross-origin health probes (the failover
        # health check from any frontend, preview, or test origin). Keep /ask* locked to the
        # allowlist; make these two always-answerable. Single ACAO header, always.
        resp = await call_next(request)
        if request.url.path in ("/health", "/status"):
            resp.headers["Access-Control-Allow-Origin"] = "*"
        return resp

    @app.middleware("http")
    async def rate_limit(request, call_next):
        path = request.url.path
        if path.startswith("/ask"):
            ip = _RateLimiter.client_ip(request)
            if ip is not None:
                bucket = "voice" if path == "/ask" else "text"
                if not limiter.allow(ip, bucket):
                    from fastapi.responses import JSONResponse
                    return JSONResponse({"detail": "rate limited; slow down"},
                                        status_code=429,
                                        headers={"Retry-After": "30"})
        return await call_next(request)

    @app.middleware("http")
    async def allow_private_network(request, call_next):
        # Chrome Private Network Access: a public page (vercel.app) fetching a host that
        # resolves to a private address (Tailscale-client machines resolve ts.net to the
        # tailnet IP) must see this header on the preflight, or fetch() dies with
        # "Failed to fetch". Public-DNS visitors (judges) never trigger it; harmless extra.
        resp = await call_next(request)
        if request.method == "OPTIONS":
            resp.headers["Access-Control-Allow-Private-Network"] = "true"
        return resp

    @app.on_event("startup")
    async def _bind_quality_loop():
        # AsyncVLLMQuality's sync generate() adapter (harness /ask_text quality path) needs
        # the running server loop; /ask_stream's astream runs in-loop and doesn't. Best-effort.
        import asyncio
        h = state.get("h")
        qg = getattr(h, "quality_generator", None) if h else None
        if qg is not None and hasattr(qg, "bind_loop"):
            try:
                qg.bind_loop(asyncio.get_running_loop())
                print("quality generator bound to server loop (streaming enabled)")
            except Exception as e:  # noqa: BLE001
                print(f"bind_loop skipped: {e}")

    # /health and /status are async so liveness survives threadpool exhaustion: sync
    # endpoints share anyio's ~40-token pool with /ask*, and a stall there must not make
    # the box look dead to the frontend's health cycle (they do no blocking work).
    @app.get("/health")
    async def health():
        h = get_h()
        return {"status": "ok", "collection": h.collection, "budget_ms": h.budget_ms}

    @app.get("/status")
    async def status():
        """api_version 2 contract (amendment 8): the frontend renders corpus truth from HERE,
        never inferred from collection names. Values update only at promotion time."""
        from config import settings
        h = get_h()
        return {
            "api_version": "2",
            "build_manifest": settings.build_manifest,
            "indexed_docs": settings.status_indexed_docs,
            "heldout_docs": settings.status_heldout_docs,
            "live_strategy": settings.live_strategy,
            "collection_suffix": h.collection,
            "status": "ready",
        }

    @app.post("/ask_text", response_model=RAGResponse)
    def ask_text(body: AskText):
        return get_h().answer(Query(text=body.text, top_k=body.top_k,
                                    session_id=body.session_id,
                                    quality_mode=body.quality_mode,
                                    retrieval_mode=body.retrieval_mode))

    @app.post("/ask_stream")
    async def ask_stream(body: AskText):
        """Streaming quality: the grounded extractive answer is emitted FIRST (event
        `result`, the <200ms answer of record), then the 7B elaboration streams as
        `elab_delta` events, closing with `elab_done` {ms, grounded}. When the quality
        backend can't token-stream, the elaboration arrives as one delta — same contract."""
        import json as _json
        import time as _time
        from fastapi.responses import StreamingResponse
        from fastapi.concurrency import run_in_threadpool
        from guardrails import grounding_check
        from schemas import RetrievedChunk

        h = get_h()

        async def gen():
            def sse(event, obj):
                return f"event: {event}\ndata: {_json.dumps(obj, ensure_ascii=False)}\n\n"
            grounded_resp = await run_in_threadpool(
                h.answer, Query(text=body.text, top_k=body.top_k,
                                session_id=body.session_id, quality_mode=False,
                                retrieval_mode=body.retrieval_mode))
            yield sse("result", _json.loads(grounded_resp.model_dump_json()))
            qg = h.quality_generator
            if not body.quality_mode or qg is None or grounded_resp.decision.value != "answer":
                yield sse("elab_done", {"ms": 0, "grounded": True, "skipped": True})
                return
            evidence = [RetrievedChunk(chunk_id=c.chunk_id, text=c.text, score=1.0)
                        for c in grounded_resp.citations]
            t0 = _time.perf_counter()
            full = ""
            try:
                if hasattr(qg, "astream"):
                    import asyncio as _aio
                    # inter-delta timeout: a hung engine must close the stream honestly,
                    # not hold the SSE connection (and its threadpool token) open forever
                    it = qg.astream(body.text, evidence).__aiter__()
                    while True:
                        try:
                            delta = await _aio.wait_for(it.__anext__(), timeout=30.0)
                        except StopAsyncIteration:
                            break
                        full += delta
                        yield sse("elab_delta", {"t": delta})
                else:
                    out = await run_in_threadpool(qg.generate, body.text, evidence)
                    full = out.text
                    yield sse("elab_delta", {"t": full})
                g = await run_in_threadpool(
                    grounding_check, full, evidence, h.grounding_min_support)
                yield sse("elab_done", {"ms": round((_time.perf_counter() - t0) * 1000, 1),
                                        "grounded": bool(g.grounded)})
            except Exception as e:  # noqa: BLE001 — stream must close honestly, never hang
                yield sse("elab_done", {"ms": round((_time.perf_counter() - t0) * 1000, 1),
                                        "grounded": False, "error": str(e)[:80]})

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"cache-control": "no-cache"})

    @app.post("/ask", response_model=RAGResponse)
    def ask(file: UploadFile = File(...), language: Optional[str] = Form(None),
            session_id: Optional[str] = Form(None),
            quality_mode: bool = Form(False),
            retrieval_mode: Optional[str] = Form(None)):
        # sync def -> FastAPI threadpool. An async def here would run the blocking Sarvam
        # HTTP call on the event loop and freeze every other request.
        audio = file.file.read(MAX_BODY_BYTES + 1)
        if len(audio) > MAX_BODY_BYTES:      # belt: BodyLimitMiddleware already rejects
            from schemas import Decision, RAGResponse as _R
            return _R(answer="Audio too large.", decision=Decision.ERROR,
                      abstained=True, reason="audio: too large")
        if len(audio) < MIN_AUDIO_BYTES:     # no real recording is this small: save the
            from schemas import Decision, RAGResponse as _R   # Sarvam spend + round trip
            return _R(answer="I couldn't hear that — please hold and speak again.",
                      decision=Decision.ERROR, abstained=True, reason="audio: too short")
        lang = language if (language and len(language) <= 12
                            and language.replace("-", "").isalnum()) else None
        return get_h().answer_voice(audio, lang, session_id=session_id,
                                    quality_mode=quality_mode,
                                    retrieval_mode=retrieval_mode)

    return app


# module-level ASGI app for `uvicorn api:app` — builds NO harness until the first request
app = create_app()
