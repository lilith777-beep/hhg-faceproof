"""
modal_app.py — the GPU deployment. Co-locates BGE-M3 + on-disk Qdrant + the local 3B + FastAPI
on ONE box so the retrieval->output path stays under 200 ms (no cross-service network hops).

Setup (human):
    pip install modal && modal token new
    modal secret create sarvam SARVAM_API_KEY=...        # Sarvam STT
    modal secret create hf HF_TOKEN=...                  # dataset + model pulls
    modal run   src/modal_app.py::build_index            # offline: build the Qdrant collections
    modal deploy src/modal_app.py                        # serves the FastAPI app (prints URL)
Point web/ at the printed URL and deploy web/ on Vercel.
"""
from __future__ import annotations

import modal

app = modal.App("hhgvrag")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "transformers>=4.44", "accelerate>=0.30", "FlagEmbedding>=1.2",
        "qdrant-client>=1.9", "fastapi>=0.110", "pydantic>=2.5", "httpx>=0.27",
        "datasets>=2.16", "python-multipart>=0.0.9",
    )
    .env({"HF_HOME": "/data/hf"})   # model weights cached on the Volume — no re-download
    .add_local_python_source(
        "chunking", "embeddings", "retrieval", "guardrails", "generation",
        "stt", "harness", "schemas", "config", "index_build", "api",
        "reranker", "router", "normalize", "raptor", "cache", "topics",
        "textnorm", "pageindex",   # generation/guardrails/pageindex import textnorm at
    )                              # module top — omitting it was ModuleNotFoundError on boot
)

vol = modal.Volume.from_name("hhgvrag-data", create_if_missing=True)
DATA = "/data"
QDRANT_PATH = f"{DATA}/qdrant"


@app.function(image=image, gpu="A10G", volumes={DATA: vol},
              secrets=[modal.Secret.from_name("hf")], timeout=60 * 60 * 4)
def build_index(max_docs: int = 20_000, languages: str = "hi",
                llm_raptor: bool = True, include_semantic: bool = False):
    """Offline: load MSMARCO-XI slice -> every chunking strategy -> Qdrant collections on the Volume.

    Defaults are the SAFE first-deploy scale: 20k docs keeps the in-process Qdrant exact
    scan inside the 60ms retrieve budget and the whole build well under the timeout.
    `include_semantic=False` because SemanticChunker embeds per-doc (50k tiny GPU calls)
    and on short MS MARCO passages degenerates to whole-passage chunks anyway — the
    semantic strategy still runs in the local A/B eval. Scale up deliberately, measured."""
    from qdrant_client import QdrantClient
    from embeddings import BGEM3Embedder
    from index_build import build_all_strategies, load_msmarco_slice

    from topics import EmbeddingTopicClassifier

    langs = tuple(x.strip() for x in languages.split(",") if x.strip())
    docs = load_msmarco_slice(languages=langs, max_docs=max_docs)
    emb = BGEM3Embedder()
    client = QdrantClient(path=QDRANT_PATH)
    topic_clf = EmbeddingTopicClassifier(n_topics=12)
    from config import settings as _settings
    # MUST match what RAGService.load looks up, or every cold start fails verify_ready()
    report = build_all_strategies(docs, emb, client, prefix=_settings.collection_prefix,
                                  include_semantic=include_semantic,
                                  topic_classifier=topic_clf)

    from raptor import ExtractSummarizer, load_hf_summarizer
    from index_build import build_raptor_index

    cleanup = None
    if llm_raptor:
        print("loading 3B model for RAPTOR abstractive summaries...")
        summarizer, cleanup = load_hf_summarizer()
    else:
        summarizer = ExtractSummarizer()

    raptor_report = build_raptor_index(
        docs, emb, summarizer, client, prefix="msmarco_xi",
        strategy_name="passage", cluster_size=5, max_levels=3)
    report["raptor"] = raptor_report
    print("raptor:", raptor_report)
    if cleanup is not None:
        cleanup()

    vol.commit()
    print("built:", {k: v["n_chunks"] for k, v in report.items()})
    return report


@app.cls(image=image, gpu="A10G", volumes={DATA: vol},
         secrets=[modal.Secret.from_name("sarvam"), modal.Secret.from_name("hf")],
         max_containers=1, scaledown_window=15 * 60, timeout=60 * 5)
@modal.concurrent(max_inputs=4)
class RAGService:
    # SCALE-TO-ZERO: no min_containers — an always-on A10G is ~$26/day and would blow the
    # $30 free credits in ~27h. Idle 15 min -> container stops; next request pays a ~60-90s
    # cold start (warm it before demos/judging). max_containers=1: sessions + semantic cache
    # live in container memory — a second container would silently lose both. concurrent(4):
    # overlapping requests share the container (cache/session are lock-protected).
    @modal.enter()
    def load(self):
        import os
        from qdrant_client import QdrantClient
        from embeddings import BGEM3Embedder
        from generation import ExtractiveGenerator, LocalLLMGenerator
        from guardrails import KeywordSafety   # swap to LlamaGuardSafety(gen) for production
        from harness import RAGHarness
        from retrieval import Retriever
        from stt import SarvamSTT
        from config import settings

        emb = BGEM3Embedder()
        client = QdrantClient(path=QDRANT_PATH)
        retr = Retriever(client, emb, use_sparse=settings.use_sparse)
        gen = ExtractiveGenerator()   # H8: grounded extractive = default (<200ms)
        quality_gen = LocalLLMGenerator(settings.gen_model, settings.gen_max_tokens,
                                        settings.gen_temperature)  # H8: LLM = quality mode
        stt = SarvamSTT(os.environ["SARVAM_API_KEY"], settings.sarvam_stt_model,
                        settings.sarvam_stt_mode)

        # classifier stack: reranker (R1) + router (R2) + noisy-ASR normalizer (R3)
        reranker = router = normalizer = None
        if settings.use_reranker:
            from reranker import BGEReranker
            reranker = BGEReranker(settings.rerank_model)
        if settings.use_router:
            from router import HeuristicRouter
            router = HeuristicRouter(language_filter=settings.router_language_filter)
        if settings.use_normalizer:
            from normalize import QueryNormalizer
            normalizer = QueryNormalizer()

        # CAG: semantic response cache (R5a) + session context (R5c)
        response_cache = session_ctx = None
        if settings.use_response_cache:
            from cache import SemanticResponseCache
            response_cache = SemanticResponseCache(
                threshold=settings.cache_similarity_threshold,
                max_size=settings.cache_max_size)
        if settings.use_session_context:
            from cache import SessionContext
            session_ctx = SessionContext(max_turns=settings.session_max_turns)

        # use RAPTOR collection (leaves + summaries) if available, else leaf-only
        collection = f"{settings.collection_prefix}__raptor"
        if not (settings.use_raptor and client.collection_exists(collection)):
            collection = f"{settings.collection_prefix}__{settings.live_strategy}"

        self.harness = RAGHarness(
            embedder=emb, retriever=retr, generator=gen, safety=KeywordSafety(),
            collection=collection,
            ood_threshold=settings.ood_score_threshold,
            grounding_min_support=settings.grounding_min_support,
            budget_ms=settings.budget_ms, stage_timeouts=settings.stage_timeout_ms, stt=stt,
            normalizer=normalizer, router=router, reranker=reranker,
            rerank_min=settings.rerank_min, rerank_candidates=settings.rerank_candidates,
            rerank_veto=settings.rerank_veto,
            response_cache=response_cache, session_ctx=session_ctx,
            quality_generator=quality_gen)
        self.harness.verify_ready()   # fail the cold start (clear msg) if build_index never ran

        # warmup: pay CUDA/lazy-init NOW, not inside the first user's timed stages
        try:
            from schemas import RetrievedChunk
            _wc = RetrievedChunk(chunk_id="warmup", text="warmup passage about vectors.", score=0.0)
            emb.embed_query("warmup query")
            if reranker is not None:
                reranker.rerank("warmup query", [_wc], top_n=1)
            quality_gen.generate("warmup query", [_wc])
            print("warmup complete")
        except Exception as e:  # noqa: BLE001 — warmup is best-effort
            print("warmup skipped:", e)

    @modal.asgi_app()
    def serve(self):
        from api import create_app
        return create_app(self.harness)


@app.local_entrypoint()
def main():
    print("Run `modal run src/modal_app.py::build_index` first, then `modal deploy src/modal_app.py`.")
