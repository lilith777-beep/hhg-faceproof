"""Central config — model ids, thresholds, latency budget, and secrets (from env)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    # --- STT (Sarvam) ---
    sarvam_api_key: str = field(default_factory=lambda: os.getenv("SARVAM_API_KEY", ""))
    sarvam_stt_model: str = "saaras:v3"            # /speech-to-text default (verified in docs)
    sarvam_stt_mode: str = "transcribe"           # transcribe | translate (saaras:v3 only)
    sarvam_stt_lang: str = "unknown"               # BCP-47 (e.g. hi-IN) or 'unknown' = auto-detect

    # --- embeddings (BGE-M3: dense + sparse + colbert) ---
    embed_model: str = "BAAI/bge-m3"
    embed_dim: int = 1024
    use_sparse: bool = True                        # hybrid dense+sparse retrieval

    # --- vector DB (Qdrant) ---
    qdrant_url: str = field(default_factory=lambda: os.getenv("QDRANT_URL", ":memory:"))
    qdrant_api_key: str = field(default_factory=lambda: os.getenv("QDRANT_API_KEY", ""))
    collection_prefix: str = "msmarco_xi_val14_73ca3e90"   # <prefix>__<strategy>
    live_strategy: str = "passage"                 # promoted after the offline A/B (eval/)

    # --- /status contract (api_version 2) — updated ONLY at promotion time ---
    build_manifest: str = "73ca3e90"               # signed manifest, exhaustively verified GREEN
    status_indexed_docs: int = 875_859
    status_heldout_docs: int = 10_100
    hnsw_ef: int = 96          # applies to a real Qdrant server; in-process local mode
                               # (QdrantClient(path=…)) is an EXACT scan — fine at ≤20k docs

    # --- generation (quality mode; extractive stays the <200ms default) ---
    gen_model: str = "Qwen/Qwen2.5-7B-Instruct"    # Forge A6000 48GB: 7B fp16 ≈ 15GB, easy
    gen_backend: str = "vllm"                      # vllm (prefix cache, fast) | hf (fallback)
    gen_gpu_mem_util: float = 0.5                  # vLLM's share of VRAM (rest: BGE-M3+reranker)
    gen_max_tokens: int = 420        # quality mode is out-of-budget + STREAMED, so room for a
                                     # comprehensive grounded answer matters more than token count
                                     # (was 160 → 7B stopped after 2 sentences, terser than extractive)
    gen_temperature: float = 0.2

    # --- guardrails / thresholds ---
    ood_score_threshold: float = 0.53              # 73ca3e90-CALIBRATED (19,458 cal queries):
                                                   # dense qrel-hit p02=0.537 keeps 98% of
                                                   # provably-correct answers; old 20k τ=0.58
                                                   # would abstain 13.6% of answerable queries
    grounding_min_support: float = 0.5             # answer must be >=50% supported by context
    safety_model: str = "meta-llama/Llama-Guard-3-1B"

    # --- classifier: reranker (R1) + router (R2) + noisy-ASR normalizer (R3) ---
    use_reranker: bool = True
    use_router: bool = True
    use_normalizer: bool = True
    use_transliteration: bool = True   # typed romanized-Indic -> native script (Sarvam
                                       # /transliterate, cached) so it uses the fast native
                                       # retrieval path instead of the slow romanized dual-arm
    rerank_model: str = "BAAI/bge-reranker-v2-m3"  # cross-encoder; score = answerability signal
    rerank_min: float = 0.30                        # 73ca3e90-calibrated: qrel-hit p02=0.303
                                                    # (98% of correct answers admitted)
    rerank_veto: float = 0.03                       # dense passed BUT cross-encoder certain
                                                    # nothing is relevant -> abstain anyway
                                                    # (live: junk pairs score ≤0.026, real ≥0.35)
    rerank_candidates: int = 16                     # Stage-A winner at 876k: depth 16 matches
                                                    # depth-32 recall at 32ms p50 (vs 58ms);
                                                    # (40 pairs blew the stage cap on live A6000)
    router_language_filter: bool = False            # keep cross-lingual retrieval open (no hard filter)

    # --- RAPTOR (R4) ---
    # DISABLED on 73ca3e90: the live __raptor collection is corrupted — the summary
    # upsert-id collision (fixed in code for future builds, task #23) kept only 6,833 of
    # ~60k summaries AND cross-linked their children across languages, so summary-expansion
    # intermittently retrieves unresolvable leaf ids -> decision:error under load. Until a
    # clean rebuild, serve the complete, verified __passage collection (958,999 pts). The
    # promoted live strategy was "passage" anyway (won the offline A/B).
    use_raptor: bool = False
    raptor_cluster_size: int = 5
    raptor_max_levels: int = 3

    # --- PageIndex (vectorless tree retrieval; per-query toggle, hybrid stays default) ---
    use_pageindex: bool = True
    pageindex_beam: int = 5     # lexical descent needs a wider beam on a 169-root live tree

    # --- CAG / cache (R5) ---
    use_response_cache: bool = True
    cache_similarity_threshold: float = 0.92       # cosine threshold for semantic cache hit
    cache_max_size: int = 256
    use_session_context: bool = True
    session_max_turns: int = 5

    # --- latency budget (ms), retrieval -> output (excludes STT) ---
    # budget_ms is enforced END-TO-END by the harness: every budget-scoped stage is capped at
    # the REMAINING request budget (with a small floor), so per-stage values below are upper
    # guards, not a partition. "generate_quality" and "stt" run OUTSIDE the 200ms budget:
    # quality mode is the explicitly over-budget path (seconds-scale, separately timed) and
    # STT is upstream of the measured pipeline.
    budget_ms: int = 200
    stage_timeout_ms: dict = field(default_factory=lambda: {
        "safety": 80, "retrieve": 60, "rerank": 110, "generate": 150, "ground_check": 40,
        "generate_quality": 10_000, "stt": 8_000, "transliterate": 2_000,
    })

    # --- dataset slice (documented + reproducible) ---
    dataset: str = "ai4bharat/MSMARCO-XI"
    dataset_languages: tuple = ("hi",)             # Indic CONFIG codes (no 'en' config exists)
    dataset_include_english: bool = True           # index passages.English_passages (English corpus)
    dataset_include_translated: bool = True        # + passages.Translated_passages (Indic corpus)
    dataset_max_docs: int = 50_000                 # curated slice so hosting + latency hold


settings = Settings()
