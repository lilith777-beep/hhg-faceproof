"""
eval_retrieval_modes.py — hybrid (vector) vs pageindex (vectorless) A/B on the same corpus.

Local demo: synthetic corpus + HashEmbedder, topical hit-rate + latency per mode.
On Forge: run after build_real_index; swap in real qrels via build_eval_set for recall@k.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import chunking as ck
from embeddings import HashEmbedder
from index_build import synthetic_docs
from pageindex import PageIndexTree
from raptor import ExtractSummarizer, RaptorTreeBuilder
from retrieval import Retriever, memory_client

# query -> the doc_id topic prefix that counts as a hit (synthetic_docs naming)
QUERIES = [
    ("what is qdrant vector database", "qdrant"),
    ("hybrid dense sparse search", "qdrant"),
    ("goa beaches portuguese forts", "goa"),
    ("western india coastal state", "goa"),
    ("python for data science", "python"),
    ("programming language web backends", "python"),
    ("sarvam speech transcription hindi", "sarvam"),
    ("retrieval augmented generation grounded", "rag"),
]


def main():
    emb = HashEmbedder(dim=512)
    client = memory_client()
    retr = Retriever(client, emb, use_sparse=True)
    chunks = ck.PassageAwareChunker(min_tokens=1, max_tokens=200).chunk_corpus(
        synthetic_docs(60))
    embeds = emb.embed_docs([c.text for c in chunks])
    retr.index("ab__passage", chunks, embeds)
    summaries, _ = RaptorTreeBuilder(emb, ExtractSummarizer(),
                                     cluster_size=5, max_levels=3).build(chunks, embeds)
    tree = PageIndexTree.from_nodes(chunks, summaries)

    def hit(results, topic):
        return any(r.payload.get("doc_id", "").startswith(topic) for r in results[:3])

    rows = []
    for mode in ("hybrid", "pageindex"):
        hits, total_ms = 0, 0.0
        for q, topic in QUERIES:
            t0 = time.perf_counter()
            if mode == "hybrid":
                res = retr.search("ab__passage", q, top_k=8)
            else:
                res = tree.search(q, top_k=8)
            total_ms += (time.perf_counter() - t0) * 1000
            hits += hit(res, topic)
        rows.append((mode, hits, len(QUERIES), total_ms / len(QUERIES)))

    print(f"\n  {'mode':<10} {'hit@3':>7} {'avg ms':>8}")
    print(f"  {'-'*10} {'-'*7} {'-'*8}")
    for mode, h, n, ms in rows:
        print(f"  {mode:<10} {h:>4}/{n:<2} {ms:>8.2f}")
    print(f"\n  tree: {tree.size} nodes ({len(summaries)} summaries), "
          f"embeddings touched at query time: hybrid=yes, pageindex=NO")


if __name__ == "__main__":
    main()
