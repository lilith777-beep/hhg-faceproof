"""
raptor.py — R4: Recursive Abstractive Processing for Tree-Organized Retrieval.

Offline: leaf chunks → cluster by embedding cosine → summarize clusters → embed summaries →
repeat. All levels (leaves + summaries) index into one collection — broad queries match
high-level summaries, specific queries match leaf chunks, the reranker sorts everything.

CPU-testable: ExtractSummarizer needs no model. Real: LLMSummarizer uses the 3B on Modal.
"""
from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from chunking import Chunk, split_sentences


@runtime_checkable
class Summarizer(Protocol):
    def summarize(self, texts: Sequence[str]) -> str: ...


class ExtractSummarizer:
    """Pick the longest (most informative) sentences across a cluster — CPU-testable."""
    def __init__(self, max_sentences: int = 3):
        self.max_sents = max_sentences

    def summarize(self, texts: Sequence[str]) -> str:
        sents = []
        for t in texts:
            sents.extend(split_sentences(t))
        if not sents:
            return " ".join(texts)[:500]
        # tie-break on the text, not set iteration order — otherwise summary content (and
        # everything downstream of it: clustering, ids) varies with PYTHONHASHSEED
        by_len = sorted(set(sents), key=lambda s: (-len(s), s))
        return " ".join(by_len[:self.max_sents])


class LLMSummarizer:
    """Abstractive summaries via the co-located 3B model on Modal.

    `batch_fn` (prompts -> list[str]) is REQUIRED for real corpora: a 50k-doc build produces
    ~10k+ cluster summaries, and one unbatched 3B generate per cluster (~1-2s each) is hours of
    wall-clock. Batched HF generation brings it to tens of minutes."""
    def __init__(self, generate_fn, batch_fn=None):
        self._gen = generate_fn
        self._batch = batch_fn

    def _prompt(self, texts: Sequence[str]) -> str:
        # cap the prompt: a skewed k-means cluster can hold hundreds of members, and an
        # unbounded join would exceed the 3B's context window and crash hours into a build
        combined = "\n---\n".join(list(texts)[:12])[:6000]
        return (f"Summarize these passages in 2-3 sentences, capturing key facts:\n\n"
                f"{combined}\n\nSummary:")

    def summarize(self, texts: Sequence[str]) -> str:
        return (self._gen(self._prompt(texts)) or "").strip() or " ".join(texts)[:200]

    def summarize_batch(self, text_groups: Sequence[Sequence[str]]) -> list:
        prompts = [self._prompt(g) for g in text_groups]
        if self._batch is not None:
            outs = self._batch(prompts)
        else:
            outs = [self._gen(p) for p in prompts]
        return [(o or "").strip() or " ".join(g)[:200]
                for o, g in zip(outs, text_groups)]


def load_hf_summarizer(model_name: str = "Qwen/Qwen2.5-3B-Instruct", batch: int = 16):
    """Load a batched 3B summarizer for RAPTOR builds on ANY GPU box (Modal or on-prem).
    Returns (LLMSummarizer, cleanup_fn). Lazy imports — never touches torch on the laptop."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"            # decoder-only: MUST left-pad for batch generate
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto")

    def generate_batch(prompts, _batch=batch):
        outs = []
        for i in range(0, len(prompts), _batch):
            chunk = prompts[i:i + _batch]
            texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                             add_generation_prompt=True, tokenize=False)
                     for p in chunk]
            enc = tok(texts, return_tensors="pt", padding=True,
                      truncation=True, max_length=2048).to(model.device)
            gen = model.generate(**enc, max_new_tokens=128, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
            for j in range(len(chunk)):
                outs.append(tok.decode(gen[j][enc["input_ids"].shape[1]:],
                                       skip_special_tokens=True).strip())
            if i and i % (_batch * 10) == 0:
                print(f"  raptor summaries: {i}/{len(prompts)}")
        return outs

    def generate_one(prompt):
        return generate_batch([prompt])[0]

    def cleanup():
        nonlocal model, tok
        del model, tok
        torch.cuda.empty_cache()

    return LLMSummarizer(generate_one, batch_fn=generate_batch), cleanup


def _cluster(vecs, target_size=5, max_iter=8):
    """K-means on L2-normalized vectors. Returns list[list[int]] (cluster → member indices)."""
    n = len(vecs)
    if n <= target_size:
        return [list(range(n))]
    k = max(2, n // target_size)
    mat = np.array(vecs, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)
    mat = mat / norms

    # farthest-first seeding, INCREMENTAL: track each point's similarity to its nearest seed
    # and update with one matvec per new seed — O(n·d·k). The naive version (full matmul vs
    # all seeds every iteration) is O(n·d·k²) ≈ hours of numpy at 50k leaves.
    seeds = [0]
    closest = (mat @ mat[0]).astype(np.float32)
    for _ in range(min(k, n) - 1):
        nxt = int(np.argmin(closest))
        if nxt in seeds or closest[nxt] >= 0.9999:   # duplicate-heavy corpus: stop early
            break
        seeds.append(nxt)
        np.maximum(closest, mat @ mat[nxt], out=closest)
    centers = mat[seeds].copy()

    for _ in range(max_iter):
        sims = mat @ centers.T
        labels = sims.argmax(axis=1)
        for j in range(len(centers)):
            members = mat[labels == j]
            if len(members):
                c = members.mean(axis=0)
                norm = np.linalg.norm(c)
                centers[j] = c / norm if norm > 1e-9 else c

    labels = (mat @ centers.T).argmax(axis=1)
    clusters = {}
    for i, la in enumerate(labels):
        clusters.setdefault(int(la), []).append(i)
    return [v for v in clusters.values() if v]


def _leaf_closure(children: Sequence, leaf_desc: dict) -> list:
    """Full transitive closure of LEAF chunk_ids under a summary's immediate children.
    `leaf_desc` maps every already-built node id (leaf or lower summary) to its leaf-descendant
    list; a leaf maps to itself. Order is first-seen + deduped -> deterministic given the
    (deterministic) clustering. Unknown children degrade to themselves (defensive; shouldn't
    happen because children always come from the current working set)."""
    seen, out = set(), []
    for ch in children:
        for lid in leaf_desc.get(ch, (ch,)):
            if lid not in seen:
                seen.add(lid)
                out.append(lid)
    return out


class RaptorTreeBuilder:
    """Build a RAPTOR tree on top of existing leaf chunks.

    Emits summary chunks that satisfy the P0.3 leaf-evidence payload contract: `is_summary=True`
    plus `extra.{raptor_level, children, leaf_descendants, build_manifest}`. `leaf_descendants`
    is the FULL transitive closure of stable leaf chunk_ids, so the harness expands a summary hit
    straight to its leaves (retrieval.expand_to_leaves reads leaf_descendants first) without a
    recursive per-level fetch."""
    def __init__(self, embedder, summarizer: Summarizer,
                 cluster_size: int = 5, max_levels: int = 3, min_cluster: int = 3,
                 build_manifest: str = ""):
        self.embedder = embedder
        self.summarizer = summarizer
        self.cluster_size = cluster_size
        self.max_levels = max_levels
        self.min_cluster = min_cluster
        self.build_manifest = build_manifest   # 8-char CorpusBuildSpec.manifest8 (lineage ref)

    def build(self, chunks: Sequence[Chunk], embeddings: Sequence) -> tuple:
        """Build the RAPTOR tree. Returns (summary_chunks, summary_embeddings)."""
        vecs = [e.dense if hasattr(e, "dense") else list(e) for e in embeddings]
        cur_texts = [c.text for c in chunks]
        cur_ids = [c.chunk_id for c in chunks]
        cur_vecs = list(vecs)

        # every leaf is its own sole leaf-descendant; summaries accumulate the closure as levels
        # build up (a level-2 summary's leaves = union of its child summaries' leaves)
        leaf_desc: dict = {c.chunk_id: [c.chunk_id] for c in chunks}

        summaries, summary_idx = [], 0

        for level in range(1, self.max_levels + 1):
            if len(cur_texts) < self.min_cluster:
                break
            clusters = _cluster(cur_vecs, self.cluster_size)
            multi = [c for c in clusters if len(c) > 1]
            if not multi:
                break

            next_texts, next_ids, next_vecs = [], [], []
            # batch the whole level: one summarize_batch call + one embed_docs call, instead of
            # 2 model calls per cluster (the difference between minutes and hours at 50k docs)
            groups = [[cur_texts[i] for i in ci] for ci in multi]
            children_per = [[cur_ids[i] for i in ci] for ci in multi]
            if hasattr(self.summarizer, "summarize_batch"):
                summary_texts = self.summarizer.summarize_batch(groups)
            else:
                summary_texts = [self.summarizer.summarize(g) for g in groups]

            keep = [(t, ch) for t, ch in zip(summary_texts, children_per) if t.strip()]
            if keep:
                embs = self.embedder.embed_docs([t for t, _ in keep])
                for (summary_text, children), emb in zip(keep, embs):
                    vec = emb.dense if hasattr(emb, "dense") else list(emb)
                    descendants = _leaf_closure(children, leaf_desc)
                    # identity = content (sorted child ids), NEVER a per-build counter:
                    # a counter resets on every build() call, and the versioned pipeline
                    # calls build() once per (language x slice) group into ONE collection —
                    # colliding `raptor::raptor::N` ids last-writer-overwrote ~89% of the
                    # summary tier (965,832-point collection carries 6,833 summaries, not
                    # ~60k). Content-derived ids are also stable across rebuild runs.
                    import hashlib as _hl
                    sig = _hl.sha256("\x1f".join(sorted(children)).encode("utf-8")).digest()
                    chunk = Chunk(
                        text=summary_text, doc_id="raptor", strategy="raptor",
                        chunk_index=int.from_bytes(sig[:8], "big"),
                        char_start=0, char_end=len(summary_text),
                        token_count=len(summary_text.split()), is_summary=True,
                        extra={"raptor_level": level, "children": children,
                               "leaf_descendants": descendants,
                               "build_manifest": self.build_manifest})
                    leaf_desc[chunk.chunk_id] = descendants
                    summaries.append((chunk, emb))
                    next_texts.append(summary_text)
                    next_ids.append(chunk.chunk_id)
                    next_vecs.append(vec)
                    summary_idx += 1

            for c in clusters:
                if len(c) == 1:
                    idx = c[0]
                    next_texts.append(cur_texts[idx])
                    next_ids.append(cur_ids[idx])
                    next_vecs.append(cur_vecs[idx])

            cur_texts, cur_ids, cur_vecs = next_texts, next_ids, next_vecs

        return [s[0] for s in summaries], [s[1] for s in summaries]
