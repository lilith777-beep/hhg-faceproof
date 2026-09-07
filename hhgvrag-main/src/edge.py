"""
edge.py — TIER E: hhgvrag on a 2GB-GPU / CPU-only device (user-approved architecture,
2026-08-17). Same harness, same guardrails, same honesty — zero heavyweight models in the
official path:

    retrieval  = PageIndex vectorless (inverted stem-folded postings, ~0.3ms, no embedder)
    gate       = calibrated lexical arm of ood_gate (dense arm absent -> lexical governs)
    answer     = extractive MMR composer, verbatim + [n] citations   (<50ms total, CPU)
    LID        = script + romanized-vernacular detection, localized refusals (13+roman)
    polish     = OPTIONAL Qwen2.5 0.5B/1.5B Q4 GGUF via llama-cpp (streams at reading
                 speed, separately timed — the polish is never the grounded claim)
    stt        = OPTIONAL faster-whisper tiny int8 on-device; else the Sarvam API online

Corpus ships as a JSONL of leaf chunks (one {"text", "chunk_id", "doc_id", "language",
"stable_doc_id", ...payload} per line), exported from any server-tier collection:

    # on the server box: export the newest N leaves for the edge bundle
    python src/edge.py --export-corpus edge_corpus.jsonl --limit 50000 \
        --qdrant-url http://localhost:6333 --collection msmarco_xi__passage
    # on the edge device (no GPU, no Qdrant, no torch needed):
    python src/edge.py --corpus edge_corpus.jsonl --port 8000
    # with the streamed polisher (needs pip install llama-cpp-python + a GGUF file):
    python src/edge.py --corpus edge_corpus.jsonl --polisher qwen2.5-0.5b-instruct-q4.gguf
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(__file__))


# ---- edge corpus ------------------------------------------------------------------------
def load_edge_corpus(path: str) -> list:
    """JSONL -> chunk-like objects (text + chunk_id + payload) for PageIndex + composer."""
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            chunks.append(_EdgeChunk(
                text=d.get("text", ""), chunk_id=d.get("chunk_id", str(len(chunks))),
                doc_id=d.get("doc_id", ""), language=d.get("language", "en"),
                is_summary=False, payload=d))
    return chunks


class _EdgeChunk(SimpleNamespace):
    """PageIndexTree.from_nodes expects the Chunk contract (`.to_payload()`), not a bare
    attribute bag — the JSONL payload IS the Qdrant payload, so return it directly."""

    def to_payload(self) -> dict:
        return dict(self.payload)


def export_corpus(qdrant_url: str, collection: str, out_path: str, limit: int,
                  languages: tuple = ()) -> int:
    """Server-side helper: scroll leaf points into the edge JSONL bundle.

    Plain mode scrolls the first `limit` leaves — but the collection's scroll order is
    NOT language-balanced, so that yields a Hindi-heavy subset (the thin languages get a
    handful of docs and the vectorless edge tier can't retrieve for them). Pass
    `languages` to instead take an EVEN per-language quota (limit / n_langs each, via a
    server-side payload filter), so every declared language is genuinely searchable on
    the edge. Languages with fewer than the quota contribute all they have."""
    from qdrant_client import QdrantClient
    from qdrant_client import models as qm
    client = QdrantClient(url=qdrant_url, timeout=120)

    def _scroll(flt, cap, f):
        n, offset = 0, None
        while n < cap:
            pts, offset = client.scroll(collection, scroll_filter=flt,
                                        limit=min(1024, cap - n), offset=offset,
                                        with_payload=True)
            for p in pts:
                pay = p.payload or {}
                if pay.get("is_summary"):
                    continue
                f.write(json.dumps(pay, ensure_ascii=False) + "\n")
                n += 1
            if offset is None:
                break
        return n

    total = 0
    with open(out_path, "w", encoding="utf-8") as f:
        if languages:
            quota = max(1, limit // len(languages))
            for lang in languages:
                flt = qm.Filter(must=[qm.FieldCondition(
                    key="language", match=qm.MatchValue(value=lang))])
                got = _scroll(flt, quota, f)
                print(f"  export[{lang}] {got} leaves (quota {quota})")
                total += got
        else:
            total = _scroll(None, limit, f)
    return total


# ---- optional adapters (import only when configured; degrade loudly, never crash) -------
class EdgePolisher:
    """Constrained-rewrite polisher on llama-cpp (Q4 GGUF). Same contract as the server's
    quality generator: generate(query, contexts) -> text. The extractive answer is always
    produced FIRST by the harness; this only rephrases, and the grounding gate re-checks."""

    def __init__(self, gguf_path: str, n_gpu_layers: int = -1, max_tokens: int = 96):
        from llama_cpp import Llama       # optional dep: pip install llama-cpp-python
        self.llm = Llama(model_path=gguf_path, n_ctx=2048, n_gpu_layers=n_gpu_layers,
                         verbose=False)
        self.max_tokens = max_tokens

    def generate(self, query: str, contexts) -> "SimpleNamespace":
        evidence = " ".join((c.text or "")[:300] for c in list(contexts)[:3])
        prompt = (f"<|im_start|>system\nRewrite the EVIDENCE into a fluent, factual answer "
                  f"to the QUESTION. Use ONLY facts from the EVIDENCE. Reply in the "
                  f"question's language and script.<|im_end|>\n<|im_start|>user\n"
                  f"QUESTION: {query}\nEVIDENCE: {evidence}<|im_end|>\n"
                  f"<|im_start|>assistant\n")
        out = self.llm(prompt, max_tokens=self.max_tokens, temperature=0.0,
                       stop=["<|im_end|>"])
        text = out["choices"][0]["text"].strip()
        return SimpleNamespace(text=text,
                               cited_chunk_ids=[c.chunk_id for c in list(contexts)[:3]])


def load_edge_stt(size: str = "tiny"):
    """faster-whisper int8 on-device STT -> Transcript-compatible shim (optional dep)."""
    from faster_whisper import WhisperModel
    import tempfile
    model = WhisperModel(size, device="auto", compute_type="int8")

    class _EdgeSTT:
        def transcribe(self, audio: bytes, language=None):
            from stt import Transcript
            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
                f.write(audio)
                p = f.name
            try:
                segs, info = model.transcribe(p, language=language)
                text = " ".join(s.text for s in segs).strip()
                return Transcript(text=text, language=info.language or "unknown",
                                  confidence=float(info.language_probability or 1.0))
            finally:
                os.unlink(p)
    return _EdgeSTT()


# ---- the Tier-E harness -----------------------------------------------------------------
def build_edge_harness(corpus_path: str, polisher_gguf: str = "",
                       stt_size: str = "") -> "RAGHarness":
    from generation import ExtractiveGenerator
    from guardrails import KeywordSafety
    from harness import RAGHarness
    from pageindex import PageIndexTree
    from router import HeuristicRouter

    t0 = time.time()
    chunks = load_edge_corpus(corpus_path)
    pageindex = PageIndexTree.from_nodes(chunks, [])
    print(f"[edge] {len(chunks)} leaves indexed vectorless in {time.time() - t0:.1f}s")

    # "Grounded, or nothing" holds on the lexical tier too: common-token overlap ("king" in
    # "king of westeros") must not admit evidence when the query's discriminative token is
    # absent. df from the tree's own postings — language-agnostic, no stopword lists.
    from pageindex import _tokens as _pi_tokens
    n_docs = max(1, len(chunks))
    _pi_search = pageindex.search

    def _grounded_search(query, top_k=8, beam=None):
        qtok = {t for t in _pi_tokens(query) if len(t) > 1}
        dfs = {t: len(pageindex._postings.get(t, ())) for t in qtok}
        content = {t for t, df in dfs.items() if df < 0.2 * n_docs}
        if not content:
            return _pi_search(query, top_k=top_k, beam=beam)
        rarest = min(content, key=lambda t: dfs[t])
        if dfs[rarest] == 0:
            return []          # discriminative token provably absent from corpus -> abstain
        hits = _pi_search(query, top_k=top_k, beam=beam)
        kept = []
        for h in hits:
            toks = getattr(h, "toks", None) or _pi_tokens(getattr(h, "text", ""))
            if rarest in toks:
                kept.append(h)
        return kept

    pageindex.search = _grounded_search

    quality = None
    if polisher_gguf:
        try:
            quality = EdgePolisher(polisher_gguf)
            print(f"[edge] polisher loaded: {os.path.basename(polisher_gguf)}")
        except Exception as e:
            print(f"[edge] polisher unavailable ({e}) — extractive-only tier")
    # STT resolution: explicit --stt (operator intent) > Sarvam via SARVAM_API_KEY (the
    # laptop has internet — voice must work on this tier too) > none (typed error, not 500)
    stt = None
    if stt_size:
        try:
            stt = load_edge_stt(stt_size)
            print(f"[edge] on-device STT: whisper-{stt_size} int8")
        except Exception as e:
            print(f"[edge] on-device STT unavailable ({e}) — trying Sarvam")
    if stt is None:
        key = os.environ.get("SARVAM_API_KEY", "").strip()
        if key:
            from stt import SarvamSTT
            stt = SarvamSTT(api_key=key)
            print("[edge] STT: Sarvam saaras:v3 (API)")
        else:
            print("[edge] NO STT — set SARVAM_API_KEY or pass --stt tiny; "
                  "voice requests return a typed error")

    class _NoRetriever:
        """Hybrid arm absent on Tier E: pageindex mode is forced by the edge Query default;
        anything reaching the dense path is a bug we want loud."""
        def search(self, *a, **k):
            raise RuntimeError("edge tier is vectorless: hybrid retrieval not available")

    class _NoEmbed:
        dim = 0
        def embed_query(self, text):
            return SimpleNamespace(dense=[], sparse={})
        def embed_docs(self, texts):
            return [self.embed_query(t) for t in texts]

    xlit = None
    _xkey = os.environ.get("SARVAM_API_KEY", "").strip()
    if _xkey:
        from stt import SarvamTransliterator
        xlit = SarvamTransliterator(_xkey)   # typed romanized-Indic -> native (edge is lexical)
    h = RAGHarness(
        embedder=_NoEmbed(), retriever=_NoRetriever(), generator=ExtractiveGenerator(),
        safety=KeywordSafety(), collection="edge_vectorless",
        router=HeuristicRouter(), stt=stt, quality_generator=quality,
        pageindex=pageindex, reranker=None, response_cache=None, session_ctx=None,
        transliterator=xlit)

    # Tier E is vectorless-ONLY: api.py's shared endpoints default retrieval_mode to hybrid,
    # which on this tier is _NoRetriever (loud failure). Force pageindex on every query.
    _answer = h.answer

    def _edge_answer(query, *a, **k):
        query.retrieval_mode = "pageindex"
        return _answer(query, *a, **k)

    h.answer = _edge_answer
    return h


def _load_env_file() -> None:
    """Load SARVAM_API_KEY (and any KEY=VALUE lines) from a local env file if the process
    env doesn't already carry it — so the edge server can use Sarvam STT without the key
    being set as a shell env var (which is fragile to capture on Windows). Checked paths,
    first hit wins; existing os.environ values are never overwritten. Never prints values."""
    if os.environ.get("SARVAM_API_KEY"):
        return
    import pathlib
    home = pathlib.Path.home()
    for p in (home / ".hhgvrag.env", home / ".config" / "hhgvrag.env",
              pathlib.Path("hhgvrag.env")):
        try:
            if not p.is_file():
                continue
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k and v and not os.environ.get(k):
                    os.environ[k] = v
            print(f"[edge] loaded env from {p} "
                  f"(SARVAM_API_KEY {'present' if os.environ.get('SARVAM_API_KEY') else 'absent'})")
            return
        except Exception as e:  # noqa: BLE001
            print(f"[edge] env file {p} unreadable: {e}")


def _fetch_key_from_host(host: str) -> None:
    """Service-side secret load: pull SARVAM_API_KEY from the AUTHORITATIVE env file on the
    given ssh host (Forge's ~/.config/hhgvrag.env) into this process's environment at
    startup. This is standard 12-factor secret handling — the key is read straight into the
    service that needs it, never printed, never written to disk here. No-op if the key is
    already present or the ssh fails (falls back to whisper). Requires key-based ssh."""
    if os.environ.get("SARVAM_API_KEY") or not host:
        return
    import subprocess
    try:
        out = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host,
             "grep -oP '^SARVAM_API_KEY=\\K.*' ~/.config/hhgvrag.env"],
            capture_output=True, text=True, timeout=20)
        val = (out.stdout or "").strip()
        if len(val) > 20:
            os.environ["SARVAM_API_KEY"] = val
            print(f"[edge] SARVAM_API_KEY loaded from {host} (Sarvam STT enabled)")
        else:
            print(f"[edge] could not fetch key from {host} (rc={out.returncode}) — whisper fallback")
    except Exception as e:  # noqa: BLE001
        print(f"[edge] key fetch from {host} failed ({type(e).__name__}) — whisper fallback")


def main() -> int:
    ap = argparse.ArgumentParser(description="hhgvrag Tier-E edge server (2GB GPU / CPU)")
    ap.add_argument("--corpus", default="edge_corpus.jsonl")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--polisher", default="", help="path to a Q4 GGUF (llama-cpp)")
    ap.add_argument("--stt", default="", help="faster-whisper size (tiny/base) for offline")
    ap.add_argument("--export-corpus", default="", help="server-side: write edge JSONL")
    ap.add_argument("--limit", type=int, default=50_000)
    ap.add_argument("--qdrant-url", default="http://localhost:6333")
    ap.add_argument("--collection", default="msmarco_xi__passage")
    ap.add_argument("--languages", default="",
                    help="comma-separated lang codes for an EVEN per-language export "
                         "(e.g. hi,bn,ta,te,kn,ml,mr,gu,pa,or,ur,en,sa,ne); empty = first-N scroll")
    ap.add_argument("--sarvam-forge-host", default="",
                    help="ssh host (user@ip) to pull SARVAM_API_KEY from at startup "
                         "(the authoritative env file); enables Sarvam STT without a local key")
    args = ap.parse_args()

    if args.export_corpus:
        langs = tuple(x.strip() for x in args.languages.split(",") if x.strip())
        n = export_corpus(args.qdrant_url, args.collection, args.export_corpus,
                          args.limit, languages=langs)
        print(f"exported {n} leaves -> {args.export_corpus}")
        return 0

    _load_env_file()   # pick up SARVAM_API_KEY from a local env file (before harness build)
    _fetch_key_from_host(args.sarvam_forge_host)   # else pull it from the authoritative host
    import uvicorn
    from api import create_app
    h = build_edge_harness(args.corpus, args.polisher, args.stt)
    uvicorn.run(create_app(h), host="0.0.0.0", port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
