"""
retrieval.py — Qdrant hybrid retrieval (dense + learned-sparse) with metadata filters.

We fuse dense + sparse with Reciprocal Rank Fusion in Python (portable to Qdrant's in-memory
mode used by the tests) and keep the dense cosine score around because the OOD guardrail needs a
real similarity value, not a rank-fusion score.
"""
from __future__ import annotations

import uuid
from typing import Optional, Sequence, Union

from qdrant_client import QdrantClient, models

from embeddings import EmbedResult, Embedder
from schemas import RetrievedChunk

_NS = uuid.UUID("00000000-0000-0000-0000-00000000c0de")


def upsert_with_retry(client, name: str, points, tries: int = 5) -> None:
    """Bulk-ingest upsert with exponential backoff. A background HNSW/optimizer stall can
    push a single upsert past the client timeout (killed the 2026-08-16 build at 594k
    points); point ids are deterministic so re-sending a possibly-applied batch is
    idempotent and safe."""
    import time as _time
    delay = 1.0
    for attempt in range(tries):
        try:
            client.upsert(name, points=points)
            return
        except Exception as e:
            if attempt == tries - 1:
                raise
            print(f"  upsert retry {attempt + 1}/{tries - 1} ({type(e).__name__}: "
                  f"{str(e)[:80]}) in {delay:.0f}s")
            _time.sleep(delay)
            delay = min(delay * 2, 30.0)


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_NS, chunk_id))


class Retriever:
    def __init__(self, client: QdrantClient, embedder: Embedder,
                 use_sparse: bool = True, rrf_k: int = 60):
        self.client = client
        self.embedder = embedder
        self.use_sparse = use_sparse
        self.rrf_k = rrf_k

    # --- indexing -----------------------------------------------------------------------
    def ensure_collection(self, name: str) -> None:
        if self.client.collection_exists(name):
            return
        self.client.create_collection(
            collection_name=name,
            vectors_config={"dense": models.VectorParams(
                size=self.embedder.dim, distance=models.Distance.COSINE)},
            sparse_vectors_config={"sparse": models.SparseVectorParams()},
        )

    def index(self, name: str, chunks: Sequence, embeds: Sequence[EmbedResult],
              batch: int = 256) -> int:
        # streams PointStructs per batch — materializing all of them first costs multiple GB
        # of pydantic overhead at the 1M+-point versioned-build scale
        self.ensure_collection(name)
        if len(chunks) != len(embeds):   # zip would silently drop the tail — corrupt index
            raise ValueError(f"index({name}): {len(chunks)} chunks vs {len(embeds)} embeds")
        n, buf = 0, []
        for ch, er in zip(chunks, embeds):
            vec = {"dense": er.dense}
            if self.use_sparse and er.sparse:
                vec["sparse"] = models.SparseVector(
                    indices=list(er.sparse.keys()), values=list(er.sparse.values()))
            buf.append(models.PointStruct(
                id=point_id(ch.chunk_id), vector=vec, payload=ch.to_payload()))
            if len(buf) >= batch:
                upsert_with_retry(self.client, name, buf)
                n += len(buf)
                buf = []
        if buf:
            upsert_with_retry(self.client, name, buf)
            n += len(buf)
        return n

    # --- search -------------------------------------------------------------------------
    def _filter(self, filters: Optional[dict]) -> Optional[models.Filter]:
        if not filters:
            return None
        must = [models.FieldCondition(key=k, match=models.MatchValue(value=v))
                for k, v in filters.items()]
        return models.Filter(must=must)

    def search(self, name: str, query: Union[str, EmbedResult], top_k: int = 8,
               filters: Optional[dict] = None, prefetch: int = 40) -> list:
        er = self.embedder.embed_query(query) if isinstance(query, str) else query
        qfilter = self._filter(filters)
        prefetch = max(prefetch, top_k)   # wide-net requests (noisy-ASR top_k widening) must
                                          # not be silently capped by the per-arm prefetch

        dense_hits = self.client.query_points(
            name, query=er.dense, using="dense", limit=prefetch,
            query_filter=qfilter, with_payload=True).points
        dense_score = {h.id: h.score for h in dense_hits}
        payloads = {h.id: h.payload for h in dense_hits}
        ranks = [[h.id for h in dense_hits]]

        sparse_score = {}
        if self.use_sparse and er.sparse:
            sparse_hits = self.client.query_points(
                name, query=models.SparseVector(
                    indices=list(er.sparse.keys()), values=list(er.sparse.values())),
                using="sparse", limit=prefetch, query_filter=qfilter, with_payload=True).points
            sparse_score = {h.id: h.score for h in sparse_hits}
            for h in sparse_hits:
                payloads.setdefault(h.id, h.payload)
            ranks.append([h.id for h in sparse_hits])

        fused = self._rrf(ranks)
        out = []
        for pid, fscore in sorted(fused.items(), key=lambda kv: -kv[1])[:top_k]:
            pl = payloads.get(pid, {})
            out.append(RetrievedChunk(
                chunk_id=pl.get("chunk_id", str(pid)),
                text=pl.get("text", ""),
                score=fscore,
                dense_score=dense_score.get(pid),
                sparse_score=sparse_score.get(pid),
                payload=pl,
            ))
        return out

    def _rrf(self, rank_lists: list) -> dict:
        fused: dict = {}
        for lst in rank_lists:
            for rank, pid in enumerate(lst):
                fused[pid] = fused.get(pid, 0.0) + 1.0 / (self.rrf_k + rank + 1)
        return fused


    # --- leaf-evidence support (P0.3) ---------------------------------------------------
    def fetch_by_chunk_ids(self, name: str, chunk_ids: Sequence[str]) -> list:
        """Fetch specific points by chunk_id (payload-only; no vector search)."""
        if not chunk_ids:
            return []
        pts = self.client.retrieve(name, ids=[point_id(c) for c in chunk_ids],
                                   with_payload=True)
        out = []
        for p in pts:
            pl = p.payload or {}
            out.append(RetrievedChunk(chunk_id=pl.get("chunk_id", str(p.id)),
                                      text=pl.get("text", ""), score=0.0, payload=pl))
        return out


def is_summary_payload(payload: dict) -> bool:
    """Summary detector: new contract (top-level is_summary) OR legacy raptor marker."""
    return bool(payload.get("is_summary") or payload.get("strategy") == "raptor")


def expand_to_leaves(retriever: "Retriever", collection: str, hits: Sequence,
                     top_k: int, max_depth: int = 4) -> list:
    """P0.3 leaf-evidence expansion: summaries are NAVIGATION, never evidence.

    Summary hits are expanded to their leaf descendants (contract: extra.leaf_descendants;
    legacy fallback: recursive resolution via extra.children). The returned list contains
    ONLY leaves: directly-retrieved leaves keep their rank/scores; fetched descendants
    inherit their parent summary's fused score (the reranker re-scores everything after).
    If nothing resolves to a leaf, the result is EMPTY — the caller must abstain, never
    fall back to summaries."""
    leaves = [h for h in hits if not is_summary_payload(h.payload)]
    summaries = [h for h in hits if is_summary_payload(h.payload)]
    if not summaries:
        return list(hits)[:top_k] if top_k else list(hits)

    have = {h.chunk_id for h in leaves}
    ordered_leaf_ids: list = []
    parent_score: dict = {}
    MAX_RESOLVE = 400   # hard cap on legacy-resolution work per request

    def _extra(payload: dict) -> dict:
        return payload.get("extra") or {}

    # contract path first: leaf_descendants is a single lookup, no fetches
    pending: list = []          # (chunk_ids_to_resolve, originating_summary_score)
    for s in summaries:
        desc = _extra(s.payload).get("leaf_descendants")
        if desc:
            for lid in desc:
                if lid not in have and lid not in parent_score:
                    parent_score[lid] = s.score
                    ordered_leaf_ids.append(lid)
        else:
            kids = list(_extra(s.payload).get("children") or [])
            if kids:
                pending.append((kids, s.score))

    # legacy path: level-wise BATCHED BFS — ONE fetch per level for ALL summaries together
    # (per-summary recursion did dozens of round-trips and blew the request deadline live)
    depth = max_depth
    while pending and depth > 0:
        batch_ids, score_of = [], {}
        for kids, sc in pending:
            for k in kids:
                if k not in score_of and len(batch_ids) < MAX_RESOLVE:
                    score_of[k] = sc
                    batch_ids.append(k)
        if not batch_ids:
            break
        fetched = retriever.fetch_by_chunk_ids(collection, batch_ids)
        pending = []
        for f in fetched:
            sc = score_of.get(f.chunk_id, 0.0)
            if is_summary_payload(f.payload):
                desc = _extra(f.payload).get("leaf_descendants")
                if desc:
                    for lid in desc:
                        if lid not in have and lid not in parent_score:
                            parent_score[lid] = sc
                            ordered_leaf_ids.append(lid)
                else:
                    kids = list(_extra(f.payload).get("children") or [])
                    if kids:
                        pending.append((kids, sc))
            else:
                if f.chunk_id not in have and f.chunk_id not in parent_score:
                    parent_score[f.chunk_id] = sc
                    ordered_leaf_ids.append(f.chunk_id)
        depth -= 1

    budget = max(0, (top_k or len(hits)) - len(leaves))
    to_fetch = ordered_leaf_ids[:budget]
    fetched = retriever.fetch_by_chunk_ids(collection, to_fetch)
    by_id = {f.chunk_id: f for f in fetched}
    expanded = []
    for lid in to_fetch:
        f = by_id.get(lid)
        if f is not None and not is_summary_payload(f.payload):
            f.score = parent_score.get(lid, 0.0)
            expanded.append(f)
    return (leaves + expanded)[:top_k] if top_k else leaves + expanded


def memory_client() -> QdrantClient:
    """In-memory Qdrant — used by tests and small local runs."""
    return QdrantClient(":memory:")
