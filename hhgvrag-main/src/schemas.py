"""
schemas.py — the typed contract for the harness (requirement #5).

Every harness stage consumes and returns one of these Pydantic models, so the pipeline is
structured input/output end-to-end (never a raw string in / string out), each stage is
independently testable, and the QueryTrace drives the latency analytics (requirement #4) and
the guardrail decisions (requirement #6).
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class Decision(str, Enum):
    ANSWER = "answer"           # grounded answer returned
    ABSTAIN_OOD = "abstain_ood"          # off-topic / not covered by the corpus
    ABSTAIN_UNGROUNDED = "abstain_ungrounded"  # generated text not supported by context
    REFUSE_UNSAFE = "refuse_unsafe"      # unsafe / inappropriate input
    SMALL_TALK = "small_talk"            # greeting / meta ("who are you") -> intentional non-RAG reply
    ERROR = "error"             # pipeline failure -> safe fallback


class Query(BaseModel):
    # TRUNCATED, not rejected: the API boundary (AskText) 422s oversized input honestly;
    # this internal model must never raise ("every entrypoint returns a typed RAGResponse")
    # — unbounded text is CPU abuse (regex/tokenize/embed are all O(len)), so cap it here
    # as defense-in-depth for internal callers (voice transcripts, evals).
    text: str
    language: Optional[str] = Field(default=None, max_length=12)  # ISO code; None -> auto

    @field_validator("text", mode="before")
    @classmethod
    def _bound_text(cls, v):
        return v[:4000] if isinstance(v, str) else v
    top_k: int = Field(default=12, gt=0, le=100)  # Stage-A winner k=12 (876k); gt=0: top_k<=0
                                                  # would drop/misslice hits
    strategy: Optional[str] = None       # override the promoted chunking collection
    asr_confidence: float = 1.0          # 0..1 from STT (Sarvam language_probability) -> noise-aware
    session_id: Optional[str] = Field(default=None, max_length=64)  # R5c: conversation history
    quality_mode: bool = False           # H8: use LLM generator instead of extractive (may exceed 200ms)
    retrieval_mode: Optional[str] = None  # "hybrid" (vector, default) | "pageindex" (vectorless tree)


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    score: float
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    rerank_score: Optional[float] = None         # cross-encoder relevance (R1) — the answerability signal
    payload: dict = Field(default_factory=dict)  # doc_id, passage_id, language, spans, strategy…


class RouteResult(BaseModel):
    """Query router (R2) output — the 'strong classifier' front-half."""
    language: Optional[str] = None       # detected script/language (telemetry; not a hard filter)
    intent: str = "qa"                   # qa | chitchat | meta | unsafe
    filters: dict = Field(default_factory=dict)  # metadata filters to apply at retrieval
    collection: Optional[str] = None     # optional collection/level override
    expand: bool = False                 # low ASR confidence -> widen retrieval
    detail: Optional[str] = None


class SafetyResult(BaseModel):
    safe: bool
    category: Optional[str] = None       # e.g. violence, self-harm, sexual, hate…
    detail: Optional[str] = None


class OODResult(BaseModel):
    in_domain: bool
    top_score: float                     # true max DENSE cosine (not an RRF rank score)
    threshold: float
    lexical_top: float = 0.0             # best query<->chunk lexical overlap (noisy-ASR rescue)
    signal: str = "dense"                # which signal admitted it: dense | lexical | none


class GroundingResult(BaseModel):
    grounded: bool
    support: float                       # 0..1 fraction of the answer supported by context
    cited_chunk_ids: list = Field(default_factory=list)
    detail: Optional[str] = None


class StageTiming(BaseModel):
    stage: str
    ms: float
    ok: bool = True
    retries: int = 0
    note: Optional[str] = None


class QueryTrace(BaseModel):
    query: str
    decision: Decision
    total_ms: float = 0.0
    stages: list = Field(default_factory=list)     # list[StageTiming]
    safety: Optional[SafetyResult] = None
    ood: Optional[OODResult] = None
    grounding: Optional[GroundingResult] = None
    route: Optional[RouteResult] = None            # R2
    normalized: Optional[str] = None               # R3 — cleaned/repaired query actually retrieved on
    rerank_top: Optional[float] = None             # R1 — best cross-encoder score
    cache_hit: bool = False                        # R5a — response served from semantic cache
    quality_mode: bool = False                     # H8 — LLM generator used (may exceed 200ms budget)
    retrieval_mode: str = "hybrid"                 # which retriever ran: hybrid | pageindex
    n_retrieved: int = 0

    def add(self, t: StageTiming) -> None:
        self.stages.append(t)

    @property
    def retrieval_to_output_ms(self) -> float:
        """The number the brief's 200 ms budget is measured against. Excludes UPSTREAM
        normalization that isn't part of retrieval->output: STT (voice) and transliterate
        (typed romanized-Indic -> native script, a ~370ms Sarvam call before the pipeline)."""
        return sum(s.ms for s in self.stages if s.stage not in ("stt", "transliterate"))


class Citation(BaseModel):
    chunk_id: str
    doc_id: Optional[str] = None
    passage_id: Optional[int] = None
    text: str


class RAGResponse(BaseModel):
    answer: str
    decision: Decision
    citations: list = Field(default_factory=list)  # list[Citation]
    abstained: bool = False
    reason: Optional[str] = None
    trace: Optional[QueryTrace] = None
