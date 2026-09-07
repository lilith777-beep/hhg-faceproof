# -*- coding: utf-8 -*-
"""
calibrate_ood_live.py — M7 on the REAL stack: BGE-M3 + the live Qdrant server + the real
corpus. In-domain = actual MSMARCO-XI queries whose passages are IN the index (sampled from
the same shard rows the build ingested). OOD = queries a static web-QA corpus cannot answer:
personal, real-time, fictional-lore, device-local. Prints the score distribution and the
F1-optimal dense-cosine τ.

Run on the GPU box:  python eval/calibrate_ood_live.py --qdrant-url http://localhost:6333
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

OOD = [
    "what time is my dentist appointment tomorrow",
    "summarize the last email i received",
    "who is the king of westeros right now",
    "what did elon musk tweet this morning",
    "turn off my bedroom lights",
    "what is my wifi password",
    "how do i beat level 47 of candy crush",
    "what will the stock market do next week",
    "read me my grocery list",
    "when is my mother's birthday",
    "what is the score of today's ipl match",
    "translate my last whatsapp message",
    "which of my friends is online now",
    "what is the weather in goa right now",
    "book me a cab to the airport",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qdrant-url", default="http://localhost:6333")
    ap.add_argument("--collection", default="msmarco_xi__raptor")
    ap.add_argument("--n-indomain", type=int, default=40)
    args = ap.parse_args()

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from qdrant_client import QdrantClient
    from embeddings import BGEM3Embedder
    from retrieval import Retriever

    path = hf_hub_download("ai4bharat/MSMARCO-XI", "validation/hinval.parquet",
                           repo_type="dataset")
    rows = next(pq.ParquetFile(path).iter_batches(batch_size=400)).to_pylist()
    random.seed(7)
    sample = random.sample(rows, args.n_indomain)
    half = args.n_indomain // 2
    in_domain = ([r["Eng_Query"].strip().lstrip(". ") for r in sample[:half]] +
                 [r["query"].strip() for r in sample[half:]])

    print("loading BGE-M3...")
    emb = BGEM3Embedder()
    retr = Retriever(QdrantClient(url=args.qdrant_url), emb, use_sparse=True)

    def top_dense(q):
        res = retr.search(args.collection, q, top_k=8)
        ds = [r.dense_score for r in res if r.dense_score is not None]
        return max(ds) if ds else 0.0

    scores = ([(top_dense(q), True, q) for q in in_domain] +
              [(top_dense(q), False, q) for q in OOD])

    best_f1, best_t = 0.0, 0.5
    for t in sorted(set(s for s, _, _ in scores)):
        tp = sum(1 for s, l, _ in scores if s >= t and l)
        fp = sum(1 for s, l, _ in scores if s >= t and not l)
        fn = sum(1 for s, l, _ in scores if s < t and l)
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        if f1 >= best_f1:
            best_f1, best_t = f1, t

    print(f"\nOPTIMAL_TAU {best_t:.4f} F1 {best_f1:.3f}  "
          f"({len(in_domain)} in-domain, {len(OOD)} ood)")
    print(f"{'lbl':>4} {'score':>7}  query")
    for s, l, q in sorted(scores, key=lambda x: -x[0]):
        print(f"{'IN' if l else 'OOD':>4} {s:>7.4f}  {q[:64]}")


if __name__ == "__main__":
    main()
