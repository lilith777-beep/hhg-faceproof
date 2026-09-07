"""Tests for offline index build, latency analytics, chunking A/B eval, and the FastAPI surface."""
import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "eval"))

from embeddings import HashEmbedder          # noqa: E402
from index_build import build_all_strategies, synthetic_docs  # noqa: E402
from retrieval import Retriever, memory_client  # noqa: E402
import latency as lat                         # noqa: E402
import eval_chunking as ec                     # noqa: E402


def test_index_build_all_strategies():
    emb = HashEmbedder(dim=256)
    client = memory_client()
    docs = synthetic_docs(30)
    report = build_all_strategies(docs, emb, client, prefix="t", include_semantic=True)
    # every strategy (incl. semantic) built a non-empty collection
    assert {"fixed", "recursive", "sentwin", "passage", "hierarchical", "semantic"} <= set(report)
    for s, info in report.items():
        assert info["n_chunks"] > 0
        assert client.collection_exists(info["collection"])


def test_chunking_ab_eval():
    emb = HashEmbedder(dim=256)
    client = memory_client()
    docs = synthetic_docs(40)
    report = build_all_strategies(docs, emb, client, prefix="ab", include_semantic=False)
    retr = Retriever(client, emb, use_sparse=True)
    # qrels: a topic query is relevant to every doc whose id starts with that topic
    topics = ["goa", "rag", "python", "sarvam", "qdrant"]
    queries = [f"{t} information" for t in topics]
    qrels = {f"{t} information": {d.doc_id for d in docs if d.doc_id.startswith(t)}
             for t in topics}
    results = ec.evaluate_strategies(report, retr, queries, qrels, k=10)
    assert results and all("ndcg@10" in v for v in results.values())
    winner = ec.pick_winner(results, "ndcg@10")
    assert winner in results
    # sanity: retrieval finds relevant docs (recall > 0 for at least the winner)
    assert results[winner]["recall@10"] > 0


def test_latency_percentiles():
    from api import build_local_harness
    h = build_local_harness()
    queries = ["what is qdrant", "tell me about goa", "what is retrieval augmented generation",
               "python language", "sarvam speech", "zzzq unknown topic xyz"] * 4
    report = lat.measure_latency(h, queries, warmup=2)
    assert report["n_queries"] == len(queries)
    for p in ("p50", "p70", "p100"):
        assert report["total_ms"][p] >= 0
    assert report["total_ms"]["p50"] <= report["total_ms"]["p100"]
    assert sum(report["decisions"].values()) == len(queries)
    # write the markdown report (artifact)
    out = os.path.join(ROOT, "eval", "latency_report.md")
    lat.write_report(report, out)
    assert os.path.exists(out)
    print("    local (mock) P50/P70/P100 ms:", report["total_ms"])


def test_api_endpoints():
    from fastapi.testclient import TestClient
    from api import create_app
    client = TestClient(create_app())
    assert client.get("/health").json()["status"] == "ok"
    r = client.post("/ask_text", json={"text": "what is qdrant"})
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] == "answer" and body["citations"]
    assert body["trace"]["stages"]  # trace serialized
    off = client.post("/ask_text", json={"text": "zzzq quarkgluon lagrangian xyztopic"}).json()
    assert off["decision"] == "abstain_ood" and off["abstained"]


def test_ndcg_never_exceeds_one():
    """M9: many chunks of ONE relevant doc must not inflate nDCG past 1.0 (was ~2.95)."""
    import eval_chunking as ec

    class Hit:
        def __init__(self, did):
            self.payload = {"doc_id": did}

    class Retr:
        def search(self, coll, q, top_k=10):
            return [Hit("d1") for _ in range(5)] + [Hit("d2") for _ in range(5)]

    report = {"x": {"collection": "c", "n_chunks": 10}}
    res = ec.evaluate_strategies(report, Retr(), ["q"], {"q": {"d1"}}, k=10)
    assert 0.0 <= res["x"]["ndcg@10"] <= 1.0
    assert res["x"]["recall@10"] == 1.0 and res["x"]["mrr"] == 1.0


def test_msmarco_row_parsing():
    """Parse a row shaped like the real MSMARCO-XI schema (English/Translated passages + is_selected)."""
    from index_build import _row_passages, _valid_configs, _norm_key
    row = {
        "target_lang": "hi",
        "query": "गोवा में समुद्र तट",
        "Eng_Query": "beaches in goa",
        "passages": {
            "is_selected": [0, 1],
            "English_passages": ["Goa has many beaches.", "Panaji is the capital of Goa."],
            "Translated_passages": ["गोवा में कई समुद्र तट हैं।", "पणजी गोवा की राजधानी है।"],
        },
    }
    en = list(_row_passages(row, include_english=True, include_translated=False))
    assert [t[1] for t in en] == ["en", "en"] and [t[2] for t in en] == [0, 1]
    assert en[0][0] == "Goa has many beaches."
    both = list(_row_passages(row, include_english=True, include_translated=True))
    assert len(both) == 4 and any(lang == "hi" for _, lang, _ in both)
    # 'en' is not a real config -> falls back to a valid Indic config; dedup preserves order
    assert _valid_configs(("en", "hi")) == ["hi"]
    assert _norm_key("  Goa   HAS  Beaches ") == "goa has beaches"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        passed += 1
        print(f"  PASS  {fn.__name__}")
    print(f"\n{passed}/{len(fns)} build/eval/api tests passed")
