"""
reranker.py — cross-encoder reranking (R1), the back-half of the "strong classifier".

First-stage hybrid retrieval is recall-oriented and noisy (especially on garbled ASR queries). A
cross-encoder jointly reads (query, passage) and scores true relevance — it rescues the right
passage from a wide candidate set and its calibrated score doubles as the ANSWERABILITY signal
(a better off-topic gate than a bi-encoder cosine τ).

- `LexicalReranker`  — deterministic, CPU: token-overlap + phonetic overlap (so it still helps on
  noisy queries). Used in tests and the no-GPU local demo.
- `BGEReranker`      — BAAI/bge-reranker-v2-m3 via FlagEmbedding, normalized scores. Modal GPU.
Both attach `rerank_score` to each chunk, sort descending, and return the top-n.
"""
from __future__ import annotations

import re
from typing import Optional, Protocol, Sequence, runtime_checkable

_TOK = re.compile(r"\w+", re.UNICODE)


def _toks(text: str) -> set:
    return {t for t in _TOK.findall(text.lower()) if len(t) > 1}


@runtime_checkable
class Reranker(Protocol):
    def rerank(self, query: str, chunks: Sequence, top_n: int = 8) -> list: ...


class LexicalReranker:
    """CPU stand-in: relevance = overlap coefficient of query/passage tokens, with a phonetic
    back-off so near-miss ASR spellings still score. Deterministic -> testable, and a real signal
    (not a stub) for the local demo."""

    def __init__(self, phonetic: bool = True):
        self.phonetic = phonetic

    def _score(self, qtok: set, qphon: set, text: str) -> float:
        ctok = _toks(text)
        if not qtok or not ctok:
            return 0.0
        overlap = len(qtok & ctok) / len(qtok)          # how much of the query the passage covers
        if overlap < 1.0 and self.phonetic and qphon:
            from normalize import _metaphone_lite
            cphon = {_metaphone_lite(t) for t in ctok}
            phon = len(qphon & cphon) / len(qphon)
            overlap = max(overlap, 0.85 * phon)          # phonetic hits count, slightly discounted
        return round(overlap, 4)

    def rerank(self, query: str, chunks: Sequence, top_n: int = 8) -> list:
        from normalize import _metaphone_lite
        qtok = _toks(query)
        qphon = {_metaphone_lite(t) for t in qtok} if self.phonetic else set()
        for c in chunks:
            c.rerank_score = self._score(qtok, qphon, c.text)
        ranked = sorted(chunks, key=lambda c: (c.rerank_score or 0.0), reverse=True)
        return ranked[:top_n]


class BGEReranker:
    """BAAI/bge-reranker-v2-m3 cross-encoder, transformers-NATIVE (GPU). No FlagEmbedding:
    its compute_score path calls tokenizer.prepare_for_model, REMOVED in transformers 5.x —
    it died at runtime on the real box. Plain seq-classification forward works everywhere.
    Normalized sigmoid scores 0..1."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", use_fp16: bool = True,
                 max_length: int = 512, batch_size: int = 32):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self._torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.float16 if (use_fp16 and torch.cuda.is_available()) else None
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, torch_dtype=dtype)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()
        self.max_length = max_length
        self.batch_size = batch_size

    def rerank(self, query: str, chunks: Sequence, top_n: int = 8) -> list:
        chunks = list(chunks)
        if not chunks:
            return []
        torch = self._torch
        scores: list = []
        with torch.no_grad():
            for i in range(0, len(chunks), self.batch_size):
                batch = chunks[i:i + self.batch_size]
                enc = self.tok([query] * len(batch), [c.text for c in batch],
                               padding=True, truncation=True,
                               max_length=self.max_length, return_tensors="pt")
                enc = {k: v.to(self.model.device) for k, v in enc.items()}
                logits = self.model(**enc).logits.squeeze(-1)
                scores.extend(torch.sigmoid(logits).float().cpu().tolist())
        for c, s in zip(chunks, scores):
            c.rerank_score = float(s)
        ranked = sorted(chunks, key=lambda c: (c.rerank_score or 0.0), reverse=True)
        return ranked[:top_n]


class ONNXReranker:
    """BAAI/bge-reranker-v2-m3 via ONNX Runtime (fp16 CUDA) — SAME model, SAME scores as
    BGEReranker, ~1.5-1.8x faster inference (a pure serving optimization, zero accuracy
    trade). Export the model first with `eval/export_reranker_onnx.py`, then point
    `onnx_dir` at the output. Identical `rerank()` contract; falls to CPUExecutionProvider
    if CUDA ORT isn't present. max_length is a real latency knob (batch pads to the longest
    pair — capping it bounds per-pair compute); calibration verifies recall holds."""

    def __init__(self, onnx_dir: str, model_name: str = "BAAI/bge-reranker-v2-m3",
                 max_length: int = 512, batch_size: int = 32):
        import os
        import numpy as np
        import onnxruntime as ort
        from transformers import AutoTokenizer
        self._np = np
        self.tok = AutoTokenizer.from_pretrained(model_name)
        avail = ort.get_available_providers()
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if "CUDAExecutionProvider" in avail else ["CPUExecutionProvider"])
        onnx_path = onnx_dir if onnx_dir.endswith(".onnx") else os.path.join(onnx_dir, "model.onnx")
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
        self._inputs = {i.name for i in self.sess.get_inputs()}
        self.max_length = max_length
        self.batch_size = batch_size

    def rerank(self, query: str, chunks: Sequence, top_n: int = 8) -> list:
        chunks = list(chunks)
        if not chunks:
            return []
        np = self._np
        scores: list = []
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            enc = self.tok([query] * len(batch), [c.text for c in batch],
                           padding=True, truncation=True, max_length=self.max_length,
                           return_tensors="np")
            feed = {k: v for k, v in enc.items() if k in self._inputs}
            logits = self.sess.run(None, feed)[0].squeeze(-1)
            scores.extend((1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))).tolist())
        for c, s in zip(chunks, scores):
            c.rerank_score = float(s)
        return sorted(chunks, key=lambda c: (c.rerank_score or 0.0), reverse=True)[:top_n]
