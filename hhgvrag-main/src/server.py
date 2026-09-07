"""
server.py — the real-backend server for a GPU box (Forge / on-prem). Zero Modal.
Identical wiring to modal_app.RAGService.load: BGE-M3 + reranker + Qdrant (local path) +
extractive default + 3B quality mode + Sarvam STT + all guardrails, served by uvicorn.

Run (from hhgvrag/, inside the GPU venv, after src/build_real_index.py):

    set SARVAM_API_KEY=...   (Windows)  |  export SARVAM_API_KEY=...  (Linux)
    python src/server.py --host 0.0.0.0 --port 8000

Then expose it publicly (Cloudflare Tunnel recommended) and point web/ at the URL.
Heavy imports live inside build_real_harness() — importing this module stays cheap.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
# vLLM forks its engine core; a fork after CUDA init dies with "Cannot re-initialize CUDA
# in forked subprocess". Fix = ORDERING: build_real_harness constructs vLLM FIRST, before
# BGE-M3 (or anything else) initializes CUDA in this process. (Do NOT force spawn mode —
# vLLM 0.27's spawn path breaks on Python 3.11 with "array.array is not subscriptable".)


def build_real_harness(qdrant_path: str = "./qdrant_data", qdrant_url: str = None):
    from qdrant_client import QdrantClient

    from config import settings
    from embeddings import BGEM3Embedder
    from generation import ExtractiveGenerator, LocalLLMGenerator
    from guardrails import KeywordSafety
    from harness import RAGHarness
    from retrieval import Retriever
    from stt import SarvamSTT, SarvamTransliterator

    api_key = os.environ.get("SARVAM_API_KEY", "")
    if not api_key or api_key.startswith("PLACEHOLDER"):
        print("WARNING: SARVAM_API_KEY not set — voice endpoint will fail; text works.")

    # ORDER MATTERS: vLLM first — its engine fork must happen before ANY CUDA init in this
    # process (BGE-M3 would initialize CUDA and poison the fork).
    print(f"loading quality-mode model ({settings.gen_model}, {settings.gen_backend})...")
    if settings.gen_backend == "vllm":
        try:
            # ASYNC engine first: real token streaming for /ask_stream (no 3s dead wait —
            # the answer builds progressively). bind_loop() is called at app startup.
            from generation import AsyncVLLMQuality
            quality_gen = AsyncVLLMQuality(settings.gen_model, settings.gen_max_tokens,
                                           settings.gen_temperature,
                                           gpu_memory_utilization=settings.gen_gpu_mem_util)
        except BaseException as e:  # noqa: BLE001 — engine death raises SystemExit-class
            print(f"async vLLM unavailable ({type(e).__name__}: {e}); trying sync VLLMGenerator")
            try:
                from generation import VLLMGenerator
                quality_gen = VLLMGenerator(settings.gen_model, settings.gen_max_tokens,
                                            settings.gen_temperature,
                                            gpu_memory_utilization=settings.gen_gpu_mem_util)
            except BaseException as e2:  # noqa: BLE001
                print(f"vLLM unavailable ({type(e2).__name__}: {e2}); falling back to HF")
                quality_gen = LocalLLMGenerator(settings.gen_model, settings.gen_max_tokens,
                                                settings.gen_temperature)
    else:
        quality_gen = LocalLLMGenerator(settings.gen_model, settings.gen_max_tokens,
                                        settings.gen_temperature)

    print("loading BGE-M3...")
    emb = BGEM3Embedder()
    client = QdrantClient(url=qdrant_url) if qdrant_url else QdrantClient(path=qdrant_path)
    retr = Retriever(client, emb, use_sparse=settings.use_sparse)
    gen = ExtractiveGenerator()                       # H8: <200ms default
    stt = SarvamSTT(api_key, settings.sarvam_stt_model, settings.sarvam_stt_mode)
    transliterator = (SarvamTransliterator(api_key)
                      if (settings.use_transliteration and api_key) else None)

    reranker = router = normalizer = None
    if settings.use_reranker:
        print("loading reranker...")
        from reranker import BGEReranker
        reranker = BGEReranker(settings.rerank_model)
    if settings.use_router:
        from router import HeuristicRouter
        router = HeuristicRouter(language_filter=settings.router_language_filter)
    if settings.use_normalizer:
        from normalize import QueryNormalizer
        normalizer = QueryNormalizer()

    response_cache = session_ctx = None
    if settings.use_response_cache:
        from cache import SemanticResponseCache
        response_cache = SemanticResponseCache(
            threshold=settings.cache_similarity_threshold,
            max_size=settings.cache_max_size)
    if settings.use_session_context:
        from cache import SessionContext
        session_ctx = SessionContext(max_turns=settings.session_max_turns)

    raptor_coll = f"{settings.collection_prefix}__raptor"
    collection = raptor_coll
    if not (settings.use_raptor and client.collection_exists(collection)):
        collection = f"{settings.collection_prefix}__{settings.live_strategy}"
    print(f"live collection: {collection}")

    # vectorless PageIndex tree — built from the SERVED collection (passage when raptor is
    # disabled), NOT unconditionally from raptor: the corrupted raptor summaries would make a
    # partially-bad tree, and a flat leaf tree over passage is exactly the honest vectorless
    # story (same as the edge tier). No vectors are read; payload text only.
    pageindex = None
    pi_coll = raptor_coll if (settings.use_raptor and client.collection_exists(raptor_coll)) else collection
    if settings.use_pageindex and client.collection_exists(pi_coll):
        from pageindex import PageIndexTree
        pageindex = PageIndexTree.from_qdrant(client, pi_coll,
                                              beam=settings.pageindex_beam)
        print(f"pageindex tree ({pi_coll}): {pageindex.size} nodes, {len(pageindex._roots)} roots")

    harness = RAGHarness(
        embedder=emb, retriever=retr, generator=gen, safety=KeywordSafety(),
        collection=collection,
        ood_threshold=settings.ood_score_threshold,
        grounding_min_support=settings.grounding_min_support,
        budget_ms=settings.budget_ms, stage_timeouts=settings.stage_timeout_ms, stt=stt,
        normalizer=normalizer, router=router, reranker=reranker,
        rerank_min=settings.rerank_min, rerank_candidates=settings.rerank_candidates,
        rerank_veto=settings.rerank_veto,
        response_cache=response_cache, session_ctx=session_ctx,
        quality_generator=quality_gen, pageindex=pageindex)
    harness.verify_ready()

    # warmup: pay CUDA/lazy-init now, not inside the first user's timed stages
    try:
        from schemas import RetrievedChunk
        wc = RetrievedChunk(chunk_id="warmup", text="warmup passage about vectors.", score=0.0)
        emb.embed_query("warmup query")
        if reranker is not None:
            reranker.rerank("warmup query", [wc], top_n=1)
        quality_gen.generate("warmup query", [wc])
        print("warmup complete")
    except Exception as e:  # noqa: BLE001 — warmup is best-effort
        print(f"warmup skipped: {type(e).__name__}: {e}")
    return harness


def main():
    ap = argparse.ArgumentParser(description="hhgvrag GPU server (Forge/on-prem)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--qdrant-path", default="./qdrant_data")
    ap.add_argument("--qdrant-url", default=None,
                    help="Qdrant SERVER url (HNSW, concurrent) — overrides --qdrant-path")
    args = ap.parse_args()

    import uvicorn
    from api import create_app

    app = create_app(build_real_harness(args.qdrant_path, args.qdrant_url))
    print(f"serving on http://{args.host}:{args.port}  (docs at /docs)")
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
