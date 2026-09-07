"""
embeddings.py — BGE-M3 hybrid embeddings behind a small Protocol.

Two backends share one interface so the whole pipeline is testable on CPU:
- `HashEmbedder`  — deterministic, dependency-free (hashing vectorizer for dense + token TF for
                    sparse). Similar texts → similar dense vectors, so retrieval is meaningful
                    in local tests without any model.
- `BGEM3Embedder` — the real thing (FlagEmbedding BGE-M3: dense + learned sparse). Imported
                    lazily so importing this module never requires torch on the laptop.
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

_TOK = re.compile(r"\w+", re.UNICODE)


@dataclass
class EmbedResult:
    dense: list = field(default_factory=list)           # list[float], unit-normalized
    sparse: dict = field(default_factory=dict)          # {token_index(int): weight(float)}


@runtime_checkable
class Embedder(Protocol):
    dim: int
    def embed_docs(self, texts: Sequence[str]) -> list: ...     # -> list[EmbedResult]
    def embed_query(self, text: str) -> EmbedResult: ...
    def get_offset_tokenizer(self): ...   # -> OffsetTokenizer: .encode_offsets(text)->[(cs,ce)]


# --- offset-mapping tokenizer adapter (P1.1) --------------------------------------------
# The production chunker slices text by CHAR offsets, never by naive token-string search. The
# embedder owns the model, so it exposes the tokenizer here (chunking.py never reaches into
# private model fields). `encode_offsets(text) -> [(char_start, char_end), ...]`.
class HFOffsetTokenizer:
    """Wraps a HuggingFace *fast* tokenizer to yield per-token CHAR spans via
    `return_offsets_mapping`. Zero-width spans (special tokens, SentencePiece meta markers that
    map to s==e) are dropped so every span slices back to real source text — a '▁foo' piece is
    never treated as a substring of the source, and combining marks / emoji stay byte-aligned."""

    def __init__(self, hf_tokenizer):
        self._t = hf_tokenizer

    def encode_offsets(self, text: str) -> list:
        if not text:
            return []
        enc = self._t(text, return_offsets_mapping=True, add_special_tokens=False)
        return [(int(s), int(e)) for (s, e) in enc["offset_mapping"] if e > s]


# --- deterministic CPU backend (local dev + tests) --------------------------------------
class HashEmbedder:
    """Hashing vectorizer: dense = L2-normalized hashed bag-of-tokens; sparse = token TF."""

    def __init__(self, dim: int = 256, sparse_dim: int = 2 ** 20):
        self.dim = dim
        self.sparse_dim = sparse_dim

    def _tokens(self, text: str) -> list:
        return _TOK.findall(text.lower())

    def _h(self, tok: str, mod: int) -> int:
        return int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16) % mod

    def _one(self, text: str) -> EmbedResult:
        toks = self._tokens(text)
        dense = [0.0] * self.dim
        sparse: dict = {}
        for t in toks:
            di = self._h(t, self.dim)
            sign = 1.0 if self._h(t + "#s", 2) else -1.0    # signed hashing reduces collisions
            dense[di] += sign
            si = self._h(t, self.sparse_dim)
            sparse[si] = sparse.get(si, 0.0) + 1.0
        n = math.sqrt(sum(x * x for x in dense)) or 1.0
        dense = [x / n for x in dense]
        # sublinear TF for sparse weights
        sparse = {k: 1.0 + math.log(v) for k, v in sparse.items()}
        return EmbedResult(dense=dense, sparse=sparse)

    def embed_docs(self, texts: Sequence[str]) -> list:
        return [self._one(t) for t in texts]

    def embed_query(self, text: str) -> EmbedResult:
        return self._one(text)

    def get_offset_tokenizer(self):
        """Regex offset tokenizer (real `re` match offsets) so the offset-correct chunk path is
        fully exercisable on CPU with no model — same token boundaries as `approx_tokenizer`."""
        from chunking import approx_offset_tokenizer
        return approx_offset_tokenizer


# --- real backend (Modal GPU) -----------------------------------------------------------
class BGEM3Embedder:
    """BAAI/bge-m3 via FlagEmbedding — dense (1024) + learned sparse (lexical weights)."""

    def __init__(self, model_name: str = "BAAI/bge-m3", use_fp16: bool = True,
                 batch_size: int = 64):
        from FlagEmbedding import BGEM3FlagModel  # lazy: only on Modal
        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16)
        self.model_name = model_name
        self.dim = 1024
        self.batch_size = batch_size

    def _encode_batch(self, texts: Sequence[str]) -> list:
        out = self.model.encode(list(texts), return_dense=True, return_sparse=True,
                                return_colbert_vecs=False)
        results = []
        for dense, sparse in zip(out["dense_vecs"], out["lexical_weights"]):
            results.append(EmbedResult(
                dense=[float(x) for x in dense],
                sparse={int(k): float(v) for k, v in sparse.items()},
            ))
        return results

    def embed_docs(self, texts: Sequence[str]) -> list:
        texts = list(texts)
        if len(texts) <= self.batch_size:
            return self._encode_batch(texts)
        results = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            results.extend(self._encode_batch(batch))
            if i > 0 and i % (self.batch_size * 10) == 0:
                print(f"  embedded {i}/{len(texts)} chunks")
        return results

    def embed_query(self, text: str) -> EmbedResult:
        return self._encode_batch([text])[0]

    def get_offset_tokenizer(self) -> HFOffsetTokenizer:
        """The BGE-M3 fast tokenizer wrapped for offset-correct chunking. Locates the tokenizer
        on the FlagEmbedding model (layout varies across versions); falls back to loading it by
        name. Reaching into the model happens HERE so chunking.py stays model-agnostic."""
        tok = (getattr(self.model, "tokenizer", None)
               or getattr(getattr(self.model, "model", None), "tokenizer", None))
        if tok is None:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(self.model_name)
        return HFOffsetTokenizer(tok)
