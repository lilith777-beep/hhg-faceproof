"""
calibrate_ood.py — M7: derive the OOD gate threshold from labeled score distributions.

Runs locally with HashEmbedder for a demo; run on Modal with BGE-M3 for the real threshold.
Outputs the optimal τ (dense cosine) and the score distribution for in-domain vs OOD queries.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import chunking as ck
from embeddings import HashEmbedder
from index_build import synthetic_docs
from retrieval import Retriever, memory_client

IN_DOMAIN = [
    "what is qdrant", "qdrant vector database hybrid search",
    "goa beaches and portuguese forts", "goa western india",
    "python programming language data science", "python web development",
    "sarvam speech recognition hindi english", "sarvam saarika transcription",
    "retrieval augmented generation passages", "rag grounded answer",
]

OUT_OF_DOMAIN = [
    "quantum chromodynamics quark gluon plasma",
    "recipe for chocolate cake with buttercream frosting",
    "the history of the roman empire and senate",
    "how to install a car engine and transmission",
    "deep sea bioluminescent organisms jellyfish",
    "instructions for knitting a winter sweater pattern",
    "cryptocurrency blockchain mining staking rewards",
    "orbital mechanics hohmann transfer delta v",
    "medieval castle construction drawbridge moat",
    "sourdough bread fermentation gluten development",
]


def calibrate(retriever, collection, in_domain=IN_DOMAIN, ood=OUT_OF_DOMAIN):
    scores = []
    for q in in_domain:
        results = retriever.search(collection, q, top_k=8)
        dense = [r.dense_score for r in results if r.dense_score is not None]
        scores.append((max(dense) if dense else 0.0, True, q))
    for q in ood:
        results = retriever.search(collection, q, top_k=8)
        dense = [r.dense_score for r in results if r.dense_score is not None]
        scores.append((max(dense) if dense else 0.0, False, q))

    thresholds = sorted(set(s[0] for s in scores))
    best_f1, best_t = 0.0, 0.3
    for t in thresholds:
        tp = sum(1 for s, label, _ in scores if s >= t and label)
        fp = sum(1 for s, label, _ in scores if s >= t and not label)
        fn = sum(1 for s, label, _ in scores if s < t and label)
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        if f1 >= best_f1:
            best_f1, best_t = f1, t

    return best_t, best_f1, scores


def main():
    emb = HashEmbedder(dim=512)
    client = memory_client()
    retr = Retriever(client, emb, use_sparse=True)

    docs = synthetic_docs(60)
    chunker = ck.PassageAwareChunker(min_tokens=1, max_tokens=200)
    chunks = chunker.chunk_corpus(docs)
    embeds = emb.embed_docs([c.text for c in chunks])
    retr.index("cal__passage", chunks, embeds)

    threshold, f1, scores = calibrate(retr, "cal__passage")
    print(f"\n  Optimal threshold: {threshold:.4f}  (F1 = {f1:.3f})")
    print(f"  In-domain queries:  {len(IN_DOMAIN)}")
    print(f"  Out-of-domain:      {len(OUT_OF_DOMAIN)}")
    print(f"\n  {'Label':>5}  {'Score':>7}  Query")
    print(f"  {'-----':>5}  {'-----':>7}  {'-----'}")
    for score, is_in, q in sorted(scores, key=lambda x: -x[0]):
        label = "IN" if is_in else "OOD"
        bar = "#" * int(score * 30)
        print(f"  {label:>5}  {score:>7.4f}  {bar}  {q[:50]}")

    return threshold


if __name__ == "__main__":
    main()
