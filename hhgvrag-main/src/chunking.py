"""
chunking.py — the "vast chunking" layer for hhgvrag.

The brief explicitly rejects a single naive fixed-size split. This module implements SIX
strategies behind one interface, each emitting metadata-rich chunks, so the offline index
build can produce one Qdrant collection per strategy and we can A/B them on retrieval metrics
(recall@k / MRR / nDCG) and promote the winner to the live path.

Strategies
----------
1. FixedSizeChunker      — token fixed-size + configurable overlap (the baseline we must beat).
2. RecursiveChunker      — recursively split on a separator hierarchy (¶ → sentence → word),
                           packing to a token budget with overlap (structure-preserving).
3. SentenceWindowChunker — sentence-boundary sliding window (N sentences, stride S); indexes a
                           small "focus" window, can return a wider "context" window at retrieval.
4. PassageAwareChunker   — respects MS-MARCO passage boundaries; merges tiny passages up to a
                           floor and splits oversized ones (metadata-aware: keeps passage_id).
5. SemanticChunker       — embeds sentences, cuts at semantic troughs (adjacent-similarity local
                           minima below a percentile), so a chunk = one coherent idea.
6. HierarchicalChunker   — parent (coarse) + child (fine) chunks linked by parent_id; retrieve on
                           children for precision, hand parents to the LLM for context.

Design notes
------------
- Indian-language aware: the sentence splitter treats the Devanagari danda "।" / "॥" and the
  Urdu full stop "۔" as terminators (MSMARCO-XI is cross-lingual Indian).
- Tokenizer is pluggable: default is a fast whitespace/script approximation for local dev/tests;
  production injects the BGE-M3 HF tokenizer for exact token budgets.
- Pure-stdlib except the optional embedder (SemanticChunker) — so the whole module is unit-
  testable on CPU with no models.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Callable, Iterable, Optional, Protocol, Sequence

# ---- tokenizer abstraction -------------------------------------------------------------
# A Tokenizer maps text -> list of token strings. Default approximates; prod uses HF BGE-M3.
Tokenizer = Callable[[str], list]

_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def approx_tokenizer(text: str) -> list:
    """Cheap, dependency-free token approximation (words + standalone punctuation)."""
    return _WORD_RE.findall(text)


# ---- offset-mapping tokenizer (P1.1 frozen transform) ----------------------------------
# The production chunker must NOT slice text by naive token-string search: a SentencePiece
# token like "▁foo" is not a substring of the source, repeats resolve to the wrong occurrence,
# and combining marks/emoji desync byte vs char offsets. An OffsetTokenizer returns CHAR spans
# per token (from HF `return_offsets_mapping`), and chunks are sliced by those char boundaries
# so `chunk.text == source[char_start:char_end]` holds by construction. `embeddings.py` exposes
# the real BGE-M3 offset tokenizer via `get_offset_tokenizer()`; the approximation below keeps
# the whole path CPU-testable with no model.
class OffsetTokenizer(Protocol):
    def encode_offsets(self, text: str) -> list: ...   # -> list[(char_start, char_end)]


class ApproxOffsetTokenizer:
    """Regex offset tokenizer for local dev/tests. Spans come from `re.finditer` (real match
    offsets), so they are monotonic, within bounds, and slice back to the exact token text —
    never a `.find()` guess. Matches `approx_tokenizer`'s token boundaries."""

    def encode_offsets(self, text: str) -> list:
        return [(m.start(), m.end()) for m in _WORD_RE.finditer(text or "")]


approx_offset_tokenizer = ApproxOffsetTokenizer()


def chunk_by_offsets(text: str, offsets: Sequence, size: int, overlap: int) -> list:
    """Fixed-size token windows sliced by CHAR offsets. Returns
    [(chunk_text, char_start, char_end, tok_start, tok_end), ...] with the invariants the
    P1.1 test matrix asserts:
      * chunk_text == text[char_start:char_end]           (offset slicing, never token strings)
      * char spans monotonically non-decreasing across chunks
      * each window holds <= size tokens (exact ceiling)
      * consecutive windows overlap by exactly `overlap` tokens (until the final short window)
      * every span lies within [0, len(text)]
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if not (0 <= overlap < size):
        raise ValueError("overlap must satisfy 0 <= overlap < size")
    n = len(offsets)
    if n == 0:
        return []
    step = size - overlap
    out, i = [], 0
    while i < n:
        j = min(i + size, n)
        cstart = offsets[i][0]
        cend = offsets[j - 1][1]
        out.append((text[cstart:cend], cstart, cend, i, j))
        if j >= n:
            break
        i += step
    return out


# ---- sentence splitting (multilingual) -------------------------------------------------
# Terminators: Latin . ! ?  · Devanagari danda । and double-danda ॥ · Urdu ۔
_SENT_END = re.compile(r"(?<=[\.\!\?।॥۔])\s+")


def split_sentences(text: str) -> list:
    """Split into sentences across Latin + Indic scripts. Never returns empty strings."""
    text = text.strip()
    if not text:
        return []
    parts = _SENT_END.split(text)
    return [p.strip() for p in parts if p and p.strip()]


# ---- data model ------------------------------------------------------------------------
@dataclass
class Document:
    """One source document. `passages` (optional) preserves MS-MARCO passage structure."""
    doc_id: str
    text: str
    language: str = "unknown"
    source: str = "msmarco-xi"
    passages: Optional[Sequence[str]] = None  # if set, PassageAwareChunker uses these
    stable_doc_id: Optional[str] = None       # corpus_spec.stable_doc_id — leakage-safe identity


@dataclass
class Chunk:
    text: str
    doc_id: str
    strategy: str
    chunk_index: int
    char_start: int
    char_end: int
    token_count: int
    language: str = "unknown"
    source: str = "msmarco-xi"
    passage_id: Optional[int] = None
    parent_id: Optional[str] = None  # HierarchicalChunker: child -> parent chunk id
    is_summary: bool = False         # P0.3 contract: leaves False, RAPTOR summaries True
    stable_doc_id: Optional[str] = None  # leaf's leakage-safe corpus identity (from its Document)
    extra: dict = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        return f"{self.doc_id}::{self.strategy}::{self.chunk_index}"

    def to_payload(self) -> dict:
        """Flat dict for the Qdrant point payload (metadata-aware retrieval + filtering)."""
        d = asdict(self)
        d["chunk_id"] = self.chunk_id
        # lift topic to the TOP level: Retriever._filter builds FieldCondition(key="topic"),
        # which can never match a key nested under extra.*
        topic = self.extra.get("topic") if isinstance(self.extra, dict) else None
        if topic:
            d["topic"] = topic
        # P0.3 leaf-evidence contract: is_summary is a first-class retrieval signal (the harness
        # expands summary hits to leaf descendants and cites LEAVES only). Guarantee it at the
        # TOP level so retrieval.is_summary_payload / any Qdrant filter can read it — a value
        # nested under extra.* is never matched. Honor extra["is_summary"] as a fallback source.
        d["is_summary"] = bool(self.is_summary or (isinstance(self.extra, dict)
                                                   and self.extra.get("is_summary")))
        return d


# ---- base ------------------------------------------------------------------------------
class Chunker:
    name = "base"

    def __init__(self, tokenizer: Tokenizer = approx_tokenizer):
        self.tok = tokenizer

    def _ntok(self, text: str) -> int:
        return len(self.tok(text))

    def chunk(self, doc: Document) -> list:  # -> list[Chunk]
        raise NotImplementedError

    def chunk_corpus(self, docs: Iterable[Document]) -> list:
        out = []
        for d in docs:
            out.extend(self.chunk(d))
        return out

    # helper: emit a Chunk with consistent metadata. `token_count` may be supplied by callers
    # that already know the exact token count (the offset-tokenizer path) so we don't re-tokenize
    # with the approximation; stable_doc_id is inherited from the source Document.
    def _mk(self, doc, text, idx, cstart, cend, token_count=None, **kw) -> Chunk:
        return Chunk(
            text=text, doc_id=doc.doc_id, strategy=self.name, chunk_index=idx,
            char_start=cstart, char_end=cend,
            token_count=self._ntok(text) if token_count is None else token_count,
            language=doc.language, source=doc.source,
            stable_doc_id=doc.stable_doc_id, **kw,
        )


# ---- 1. fixed-size + overlap -----------------------------------------------------------
class FixedSizeChunker(Chunker):
    name = "fixed"

    def __init__(self, size: int = 256, overlap: int = 40, tokenizer: Tokenizer = approx_tokenizer,
                 offset_tokenizer: Optional[OffsetTokenizer] = None):
        super().__init__(tokenizer)
        assert 0 <= overlap < size
        self.size, self.overlap = size, overlap
        # when injected (production BGE-M3 path), spans come from real char offsets and chunk
        # text is sliced by those boundaries — the P1.1 offset-correct transform.
        self.offset_tok = offset_tokenizer

    def chunk(self, doc: Document) -> list:
        if self.offset_tok is not None:
            offsets = self.offset_tok.encode_offsets(doc.text)
            chunks = []
            for idx, (ctext, cstart, cend, ti, tj) in enumerate(
                    chunk_by_offsets(doc.text, offsets, self.size, self.overlap)):
                chunks.append(self._mk(doc, ctext, idx, cstart, cend, token_count=tj - ti))
            return chunks
        toks = self.tok(doc.text)
        if not toks:
            return []
        # rebuild char spans by walking the source text token-by-token
        spans = _token_spans(doc.text, toks)
        chunks, i, idx = [], 0, 0
        step = self.size - self.overlap
        while i < len(toks):
            j = min(i + self.size, len(toks))
            cstart = spans[i][0]
            cend = spans[j - 1][1]
            chunks.append(self._mk(doc, doc.text[cstart:cend], idx, cstart, cend))
            idx += 1
            if j >= len(toks):
                break
            i += step
        return chunks


# ---- 2. recursive (separator hierarchy) ------------------------------------------------
class RecursiveChunker(Chunker):
    name = "recursive"
    # separator hierarchy; the trailing "" is the terminal char-split fallback (ONLY it splits
    # to characters). Gating on `sep.strip()` was the bug: "\n\n"/"\n"/" " all strip to "" and
    # were char-splitting every document into `\n\n`-joined character soup.
    SEPARATORS = ["\n\n", "\n", ". ", "। ", " ", ""]

    def __init__(self, size: int = 256, overlap: int = 40, tokenizer: Tokenizer = approx_tokenizer):
        super().__init__(tokenizer)
        self.size, self.overlap = size, overlap

    def _split(self, text: str, sep_i: int) -> list:
        if self._ntok(text) <= self.size or sep_i >= len(self.SEPARATORS):
            return [text]
        sep = self.SEPARATORS[sep_i]
        parts = text.split(sep) if sep else list(text)   # only the "" sentinel -> characters
        # re-attach separator so char offsets stay findable; pack greedily to token budget
        out, buf = [], ""
        for p in parts:
            piece = (buf + sep + p) if buf else p
            if self._ntok(piece) > self.size and buf:
                out.append(buf)
                buf = p
            else:
                buf = piece
        if buf:
            out.append(buf)
        # any still-too-big piece recurses to the next finer separator
        final = []
        for o in out:
            final.extend(self._split(o, sep_i + 1) if self._ntok(o) > self.size else [o])
        return final

    def chunk(self, doc: Document) -> list:
        pieces = [p for p in self._split(doc.text, 0) if p.strip()]
        pieces = _apply_overlap(pieces, self.overlap, self.tok)
        chunks, cursor = [], 0
        for idx, p in enumerate(pieces):
            cstart = doc.text.find(p.strip()[:24], cursor) if p.strip() else cursor
            if cstart < 0:
                cstart = cursor
            cend = cstart + len(p)
            cursor = max(cursor, cstart + 1)
            chunks.append(self._mk(doc, p.strip(), idx, cstart, min(cend, len(doc.text))))
        return chunks


# ---- 3. sentence window ----------------------------------------------------------------
class SentenceWindowChunker(Chunker):
    name = "sentwin"

    def __init__(self, window: int = 3, stride: int = 2, tokenizer: Tokenizer = approx_tokenizer):
        super().__init__(tokenizer)
        assert 1 <= stride <= window
        self.window, self.stride = window, stride

    def chunk(self, doc: Document) -> list:
        sents = split_sentences(doc.text)
        if not sents:
            return []
        chunks, idx, i, cursor = [], 0, 0, 0
        while i < len(sents):
            win = sents[i:i + self.window]
            text = " ".join(win)
            hit = doc.text.find(win[0][:24], cursor)   # monotonic cursor -> right span for repeats
            cstart = hit if hit >= 0 else cursor
            cursor = max(cursor, cstart + 1)
            cend = min(cstart + len(text), len(doc.text))
            chunks.append(self._mk(doc, text, idx, cstart, cend,
                                   extra={"n_sentences": len(win)}))
            idx += 1
            if i + self.window >= len(sents):
                break
            i += self.stride
        return chunks


# ---- 4. passage-aware (MS-MARCO structure) ---------------------------------------------
class PassageAwareChunker(Chunker):
    name = "passage"

    def __init__(self, min_tokens: int = 48, max_tokens: int = 320,
                 tokenizer: Tokenizer = approx_tokenizer):
        super().__init__(tokenizer)
        self.min_tokens, self.max_tokens = min_tokens, max_tokens

    def chunk(self, doc: Document) -> list:
        passages = list(doc.passages) if doc.passages else _paragraphs(doc.text)
        # merge tiny adjacent passages up to the floor; split oversized ones by sentences
        merged, buf, buf_ids = [], "", []
        for pid, p in enumerate(passages):
            p = p.strip()
            if not p:
                continue
            cand = (buf + " " + p).strip() if buf else p
            if self._ntok(cand) < self.min_tokens:
                buf, buf_ids = cand, buf_ids + [pid]
            else:
                merged.append((cand, buf_ids + [pid]))
                buf, buf_ids = "", []
        if buf:
            merged.append((buf, buf_ids))

        chunks, idx, cursor = [], 0, 0
        for text, ids in merged:
            for piece in self._split_oversized(text):
                cstart = doc.text.find(piece[:24], cursor)
                cstart = cstart if cstart >= 0 else cursor
                cend = min(cstart + len(piece), len(doc.text))
                cursor = max(cursor, cstart + 1)
                chunks.append(self._mk(doc, piece, idx, cstart, cend,
                                       passage_id=ids[0], extra={"passage_ids": ids}))
                idx += 1
        return chunks

    def _split_oversized(self, text: str) -> list:
        if self._ntok(text) <= self.max_tokens:
            return [text]
        sents, out, buf = split_sentences(text), [], ""
        for s in sents:
            cand = (buf + " " + s).strip() if buf else s
            if self._ntok(cand) > self.max_tokens and buf:
                out.append(buf)
                buf = s
            else:
                buf = cand
        if buf:
            out.append(buf)
        return out or [text]


# ---- 5. semantic (embedding-boundary) --------------------------------------------------
Embedder = Callable[[Sequence[str]], "list"]  # sentences -> list of vectors (list[float])


class SemanticChunker(Chunker):
    name = "semantic"

    def __init__(self, embedder: Embedder, breakpoint_percentile: int = 25,
                 max_tokens: int = 320, tokenizer: Tokenizer = approx_tokenizer):
        super().__init__(tokenizer)
        self.embedder = embedder
        self.pct = breakpoint_percentile
        self.max_tokens = max_tokens

    def chunk(self, doc: Document) -> list:
        sents = split_sentences(doc.text)
        if len(sents) <= 1:
            return [self._mk(doc, doc.text.strip(), 0, 0, len(doc.text))] if doc.text.strip() else []
        vecs = self.embedder(sents)
        sims = [_cosine(vecs[i], vecs[i + 1]) for i in range(len(sents) - 1)]
        # cut where adjacent similarity is a local trough below the percentile threshold
        thresh = _percentile(sims, self.pct)
        groups, cur = [], [sents[0]]
        for i, s in enumerate(sents[1:]):
            if sims[i] < thresh or self._ntok(" ".join(cur + [s])) > self.max_tokens:
                groups.append(cur)
                cur = [s]
            else:
                cur.append(s)
        groups.append(cur)

        chunks, idx, cursor = [], 0, 0
        for g in groups:
            text = " ".join(g)
            hit = doc.text.find(g[0][:24], cursor)     # monotonic cursor -> right span for repeats
            cstart = hit if hit >= 0 else cursor
            cursor = max(cursor, cstart + 1)
            cend = min(cstart + len(text), len(doc.text))
            chunks.append(self._mk(doc, text, idx, cstart, cend,
                                   extra={"n_sentences": len(g), "sim_threshold": round(thresh, 4)}))
            idx += 1
        return chunks


# ---- 6. hierarchical (parent-child) ----------------------------------------------------
class HierarchicalChunker(Chunker):
    name = "hierarchical"

    def __init__(self, parent_size: int = 512, child_size: int = 160, overlap: int = 24,
                 tokenizer: Tokenizer = approx_tokenizer,
                 offset_tokenizer: Optional[OffsetTokenizer] = None):
        super().__init__(tokenizer)
        self.parent = FixedSizeChunker(parent_size, 0, tokenizer, offset_tokenizer)
        self.child = FixedSizeChunker(child_size, overlap, tokenizer, offset_tokenizer)

    def chunk(self, doc: Document) -> list:
        parents = self.parent.chunk(doc)
        out, idx = [], 0
        for p in parents:
            p.strategy = self.name
            p.chunk_index = idx
            p.extra = {**p.extra, "role": "parent"}
            parent_id = p.chunk_id
            out.append(p)
            idx += 1
            sub = Document(doc_id=doc.doc_id, text=p.text, language=doc.language,
                           source=doc.source, stable_doc_id=doc.stable_doc_id)
            for c in self.child.chunk(sub):
                c.strategy = self.name
                c.chunk_index = idx
                c.parent_id = parent_id
                c.char_start += p.char_start
                c.char_end += p.char_start
                c.extra = {"role": "child"}
                out.append(c)
                idx += 1
        return out


# ---- registry --------------------------------------------------------------------------
def all_strategies(tokenizer: Tokenizer = approx_tokenizer,
                   embedder: Optional[Embedder] = None) -> dict:
    """Every chunker keyed by name. SemanticChunker only if an embedder is supplied."""
    reg = {
        "fixed": FixedSizeChunker(tokenizer=tokenizer),
        "recursive": RecursiveChunker(tokenizer=tokenizer),
        "sentwin": SentenceWindowChunker(tokenizer=tokenizer),
        "passage": PassageAwareChunker(tokenizer=tokenizer),
        "hierarchical": HierarchicalChunker(tokenizer=tokenizer),
    }
    if embedder is not None:
        reg["semantic"] = SemanticChunker(embedder, tokenizer=tokenizer)
    return reg


# ---- small helpers ---------------------------------------------------------------------
def _token_spans(text: str, toks: list) -> list:
    """Char (start, end) for each token by scanning forward — robust to repeats."""
    spans, cursor = [], 0
    for t in toks:
        i = text.find(t, cursor)
        if i < 0:
            i = cursor
        spans.append((i, i + len(t)))
        cursor = i + len(t)
    return spans


def _apply_overlap(pieces: list, overlap: int, tok: Tokenizer) -> list:
    if overlap <= 0 or len(pieces) <= 1:
        return pieces
    out = [pieces[0]]
    for k in range(1, len(pieces)):
        prev_toks = tok(pieces[k - 1])
        tail = prev_toks[-overlap:] if len(prev_toks) > overlap else prev_toks
        out.append((" ".join(tail) + " " + pieces[k]).strip())
    return out


def _paragraphs(text: str) -> list:
    parts = re.split(r"\n\s*\n", text)
    return [p.strip() for p in parts if p.strip()] or [text]


def _cosine(a, b) -> float:
    num = sum(x * y for x, y in zip(a, b))
    da = sum(x * x for x in a) ** 0.5
    db = sum(y * y for y in b) ** 0.5
    return num / (da * db) if da and db else 0.0


def _percentile(xs: list, pct: int) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round((pct / 100.0) * (len(s) - 1)))))
    return s[k]
