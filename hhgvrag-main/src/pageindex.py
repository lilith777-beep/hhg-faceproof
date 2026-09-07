"""
pageindex.py — vectorless, reasoning-style retrieval over the hierarchical summary tree.

PageIndex posture: retrieval WITHOUT embeddings at query time. The RAPTOR build already
gives us the tree (leaves + level-N summaries with `children` links); this module navigates
it by scored beam descent — root summaries → best branches → leaves — using pure lexical
scoring (CPU, <10ms), so it fits the 200ms budget. An optional LLM-guided descent exists
for quality mode (each hop is an LLM call — honestly over-budget).

This is an ADDITIONAL retrieval mode next to Qdrant hybrid (the brief names a vector DB;
hybrid stays the default). Toggle per-query via Query.retrieval_mode="pageindex".
"""
from __future__ import annotations

import re
from typing import Optional, Sequence

from schemas import RetrievedChunk

from textnorm import WORD_RE as _TOK   # Indic-correct: keeps matras/virama attached
_STOP = set(
    "a an the of to in on for and or is are was were be been being this that these those "
    "it its as at by with from into about over under how what when where who whom which why "
    "do does did can could should would will may might i you he she they we me my your".split()
)


def _stem(t: str) -> str:
    """Cheap plural fold (matches generation._stem): corporation ↔ corporations."""
    if len(t) > 4 and t.endswith("es"):
        return t[:-2]
    if len(t) > 3 and t.endswith("s"):
        return t[:-1]
    return t


def _tokens(text: str) -> set:
    return {_stem(t) for t in _TOK.findall(text.lower()) if t not in _STOP and len(t) > 1}


class _Node:
    __slots__ = ("node_id", "text", "level", "children", "payload", "toks")

    def __init__(self, node_id, text, level, children, payload):
        self.node_id = node_id
        self.text = text
        self.level = level
        self.children = children or []
        self.payload = payload or {}
        self.toks = _tokens(text)


class PageIndexTree:
    """Vectorless tree index: beam descent by lexical overlap, leaves out."""

    def __init__(self, beam: int = 3, max_hops: int = 6):
        self.beam = beam
        self.max_hops = max_hops
        self._nodes: dict = {}
        self._roots: list = []

    # ---- construction ------------------------------------------------------------------
    @classmethod
    def from_nodes(cls, leaf_chunks: Sequence, summary_chunks: Sequence,
                   beam: int = 3) -> "PageIndexTree":
        """Build from RaptorTreeBuilder outputs (leaves + summaries with extra.children)."""
        tree = cls(beam=beam)
        for c in leaf_chunks:
            tree._nodes[c.chunk_id] = _Node(c.chunk_id, c.text, 0, [], c.to_payload())
        for s in summary_chunks:
            lvl = s.extra.get("raptor_level", 1)
            tree._nodes[s.chunk_id] = _Node(s.chunk_id, s.text, lvl,
                                            list(s.extra.get("children", [])), s.to_payload())
        tree._finalize()
        return tree

    @classmethod
    def from_qdrant(cls, client, collection: str, beam: int = 3) -> "PageIndexTree":
        """Reconstruct the tree from an indexed __raptor collection (payloads only —
        no vectors are read; this mode never touches them)."""
        tree = cls(beam=beam)
        offset = None
        while True:
            pts, offset = client.scroll(collection, limit=1024, offset=offset,
                                        with_payload=True, with_vectors=False)
            for p in pts:
                pl = p.payload or {}
                cid = pl.get("chunk_id", str(p.id))
                extra = pl.get("extra") or {}
                lvl = extra.get("raptor_level", 0)
                tree._nodes[cid] = _Node(cid, pl.get("text", ""), lvl,
                                         list(extra.get("children", [])), pl)
            if offset is None:
                break
        tree._finalize()
        return tree

    def _finalize(self):
        referenced = set()
        for n in self._nodes.values():
            referenced.update(n.children)
        self._roots = [n for n in self._nodes.values() if n.node_id not in referenced]
        # inverted token index over LEAVES: the vectorless EVIDENCE lookup. Beam descent
        # through abstract LLM summaries lost the token trail on the live tree ("what is a
        # corporation" surfaced junk → veto → wrong abstain). Postings are exact, stem-folded,
        # zero embeddings, O(query_terms) at query time. The tree remains for llm_navigate.
        self._postings: dict = {}
        self._n_leaves = 0
        for n in self._nodes.values():
            if not n.children:
                self._n_leaves += 1
                for t in n.toks:
                    self._postings.setdefault(t, set()).add(n.node_id)

    @property
    def size(self) -> int:
        return len(self._nodes)

    # ---- retrieval ---------------------------------------------------------------------
    def _score(self, qtok: set, node: _Node) -> float:
        if not qtok or not node.toks:
            return 0.0
        return len(qtok & node.toks) / len(qtok)

    def search(self, query: str, top_k: int = 8, beam: Optional[int] = None) -> list:
        """Vectorless evidence lookup: idf-weighted postings over leaves (exact, stem-folded,
        no embeddings), length-normalized. Returns LEAF evidence only."""
        import math
        qtok = _tokens(query)
        if not qtok or not self._nodes or not self._n_leaves:
            return []
        scores: dict = {}
        for t in qtok:
            ids = self._postings.get(t)
            if not ids:
                continue
            idf = math.log(1.0 + self._n_leaves / len(ids))
            for nid in ids:
                scores[nid] = scores.get(nid, 0.0) + idf
        if not scores:
            return []
        # mild length normalization so long leaves don't win on volume alone
        ranked = sorted(scores.items(),
                        key=lambda kv: -(kv[1] / (1.0 + 0.08 * math.log(
                            1 + len(self._nodes[kv[0]].toks)))))[:top_k]
        top = ranked[0][1] or 1.0
        return [RetrievedChunk(chunk_id=nid, text=self._nodes[nid].text,
                               score=round(s / top, 4), dense_score=None,
                               sparse_score=None, payload=self._nodes[nid].payload)
                for nid, s in ranked]

    def llm_navigate(self, query: str, generate_fn, top_k: int = 8) -> list:
        """Quality-mode variant: the LLM picks the branch at each hop (over-budget by design).
        generate_fn(prompt) -> str containing the chosen option number(s)."""
        qtok = _tokens(query)
        frontier = sorted(self._roots, key=lambda n: -self._score(qtok, n))[:6]
        for _ in range(self.max_hops):
            expandable = [n for n in frontier if n.children]
            if not expandable:
                break
            options = "\n".join(f"[{i + 1}] {n.text[:160]}" for i, n in enumerate(expandable))
            prompt = (f"Question: {query}\n\nWhich sections most likely contain the answer? "
                      f"Reply with up to 2 numbers.\n{options}\nNumbers:")
            reply = generate_fn(prompt) or ""
            picks = [int(m) - 1 for m in re.findall(r"\d+", reply)[:2]
                     if 0 < int(m) <= len(expandable)]
            if not picks:
                picks = [0]
            nxt = []
            for i in picks:
                nxt.extend(self._nodes[c] for c in expandable[i].children if c in self._nodes)
            keep_leaves = [n for n in frontier if not n.children]
            frontier = keep_leaves + nxt
        scored = [(self._score(qtok, n), n) for n in frontier if not n.children]
        scored.sort(key=lambda t: -t[0])
        return [RetrievedChunk(chunk_id=n.node_id, text=n.text, score=s,
                               payload=n.payload) for s, n in scored[:top_k]]
