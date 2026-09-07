"""
index_build.py — offline: build one Qdrant collection PER chunking strategy so we can A/B them.

`build_all_strategies` is CPU/in-memory testable (synthetic docs + HashEmbedder). The real run
(`build_from_msmarco`, Modal GPU) loads a documented ai4bharat/MSMARCO-XI slice + BGE-M3.
"""
from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import chunking as ck
import corpus_spec as cspec
from embeddings import Embedder


def _semantic_adapter(embedder: Embedder):
    """SemanticChunker wants sentences -> list[vector]; adapt the doc embedder's dense vecs."""
    def fn(sents: Sequence[str]) -> list:
        return [er.dense for er in embedder.embed_docs(list(sents))]
    return fn


def build_all_strategies(docs: Sequence, embedder: Embedder, client, prefix: str,
                         include_semantic: bool = True,
                         topic_classifier=None) -> dict:
    """
    For every chunking strategy: chunk all docs -> embed -> classify topics -> index.
    Returns {strategy: {"collection": name, "n_chunks": int}}.
    """
    from retrieval import Retriever
    from topics import apply_topic_labels, KeywordTopicClassifier
    retriever = Retriever(client, embedder, use_sparse=True)
    strategies = ck.all_strategies(
        embedder=_semantic_adapter(embedder) if include_semantic else None)
    clf = topic_classifier or KeywordTopicClassifier()

    report = {}
    for sname, chunker in strategies.items():
        chunks = chunker.chunk_corpus(docs)
        if not chunks:
            continue
        print(f"  embedding {len(chunks)} chunks for strategy '{sname}'...")
        embeds = embedder.embed_docs([c.text for c in chunks])
        apply_topic_labels(chunks, embeds, clf)
        coll = f"{prefix}__{sname}"
        n = retriever.index(coll, chunks, embeds)
        report[sname] = {"collection": coll, "n_chunks": n}
    return report


# ---- MSMARCO-XI loading (schema verified LIVE on the real repo, 2026-08-15) -------------
# The hub repo has ONE config ('default') and per-language parquet SHARDS:
#   train/hintrain.parquet (~3.7GB) · validation/hinval.parquet (~460MB) · etc.
# datasets' streaming mode CRASHES on the nested `passages` column
# (pyarrow ArrowNotImplementedError: nested chunked conversions) — so we hf_hub_download the
# shard (bounded) and iterate it with pyarrow directly. Row schema (verified):
#   query (target-lang) · Eng_Query · Answer · Eng_Answer · query_id · query_type ·
#   target_lang ('hin_Deva') · passages = {English_passages:[...], Translated_passages:[...],
#   is_selected:[...]} — is_selected==1 marks the answering passage -> free qrels.
MSMARCO_CONFIGS = ("as", "bn", "gu", "hi", "kn", "ml", "mr", "ne", "or",
                   "pa", "sa", "ta", "te", "ur")
_LANG_FILE = {"as": "asm", "bn": "ben", "gu": "guj", "hi": "hin", "kn": "kan",
              "ml": "mal", "mr": "mar", "ne": "nep", "or": "ori", "pa": "pan",
              "sa": "san", "ta": "tam", "te": "tel", "ur": "urd"}


def _iter_msmarco_rows(cfg: str, split: str = "validation", revision: Optional[str] = None):
    """Yield rows from one language's parquet shard. Validation (~460MB/lang) is the default:
    plenty for a ≤50k corpus, 8x smaller download than train, and it carries is_selected.
    `revision` pins the hub commit SHA (CorpusBuildSpec.dataset_revision) — None keeps the
    legacy un-pinned behavior for the pre-protocol callers."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    suffix = "train" if split == "train" else "val"
    path = hf_hub_download("ai4bharat/MSMARCO-XI",
                           f"{split}/{_LANG_FILE[cfg]}{suffix}.parquet",
                           repo_type="dataset", revision=revision)
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=64):
        yield from batch.to_pylist()


_SCRIPT_PATTERNS = [   # compiled once — this runs per passage over 100k+ passages
    ("hi", re.compile(r"[ऀ-ॿ]")),   # Devanagari
    ("bn", re.compile(r"[ঀ-৿]")),   # Bengali
    ("pa", re.compile(r"[਀-੿]")),   # Gurmukhi
    ("gu", re.compile(r"[઀-૿]")),   # Gujarati
    ("or", re.compile(r"[଀-୿]")),   # Odia
    ("ta", re.compile(r"[஀-௿]")),   # Tamil
    ("te", re.compile(r"[ఀ-౿]")),   # Telugu
    ("kn", re.compile(r"[ಀ-೿]")),   # Kannada
    ("ml", re.compile(r"[ഀ-ൿ]")),   # Malayalam
    ("ur", re.compile(r"[؀-ۿ]")),   # Arabic/Urdu
]


def _detect_lang(text: str, fallback: str = "en") -> str:
    """Detect language from actual content via Unicode script blocks (same logic as the router)."""
    for lang, rx in _SCRIPT_PATTERNS:
        if rx.search(text):
            return lang
    return fallback


def _row_passages(row: dict, include_english: bool, include_translated: bool,
                  fallback_lang: str = "hi"):
    """Yield (text, language, is_selected) for each passage in one MSMARCO-XI row.
    Language comes from the ACTUAL text (script detection); a romanized translated passage
    falls back to the config language — never an out-of-vocabulary tag like 'xx' that no
    filter could ever match."""
    p = row.get("passages") or {}
    sel = p.get("is_selected") or []
    en = p.get("English_passages") or []
    tr = p.get("Translated_passages") or []
    for i in range(max(len(en), len(tr), len(sel))):
        s = int(sel[i]) if i < len(sel) and sel[i] is not None else 0
        if include_english and i < len(en) and en[i] and en[i].strip():
            text = en[i].strip()
            yield text, _detect_lang(text, "en"), s
        if include_translated and i < len(tr) and tr[i] and tr[i].strip():
            text = tr[i].strip()
            yield text, _detect_lang(text, fallback_lang), s


def _norm_key(text: str) -> str:
    """Dedup key — English_passages repeat across queries, so collapse by normalized text."""
    return " ".join(text.lower().split())[:400]


def _valid_configs(languages) -> list:
    out = []
    for c in languages:
        c = c if c in MSMARCO_CONFIGS else "hi"   # 'en' isn't a config; English lives in every row
        if c not in out:
            out.append(c)
    return out


def _load_msmarco(languages, max_docs, include_english, include_translated,
                  collect_eval=False, max_queries=500):
    """Stream MSMARCO-XI -> quality-filter + dedup passages into Documents (+ optional
    cross-lingual qrels). The quality filter runs HERE, before doc-id assignment, so the
    eval corpus and the live corpus are the SAME corpus — filtering only one of them would
    make the A/B verdict describe an index we don't ship."""
    docs, by_text, queries, qrels, full = [], {}, [], {}, False
    dropped = 0
    for cfg in _valid_configs(languages):
        for row in _iter_msmarco_rows(cfg):
            rel = set()
            for text, lang, sel in _row_passages(row, include_english, include_translated,
                                                 fallback_lang=cfg):
                if not _passage_quality(text):
                    dropped += 1
                    continue
                key = _norm_key(text)
                did = by_text.get(key)
                if did is None:
                    if len(docs) >= max_docs:
                        full = True
                        continue
                    did = f"{lang}-{len(docs)}"
                    by_text[key] = did
                    docs.append(ck.Document(doc_id=did, text=text, language=lang))
                if sel:
                    rel.add(did)
            if collect_eval and rel and len(qrels) < max_queries:
                q = (row.get("query") or row.get("Eng_Query") or "").strip()
                if q:
                    if q not in qrels:
                        queries.append(q)
                    qrels.setdefault(q, set()).update(rel)
            if full and (not collect_eval or len(qrels) >= max_queries):
                if dropped:
                    print(f"  quality filter: dropped {dropped} passages pre-dedup")
                return docs, queries, qrels
    if dropped:
        print(f"  quality filter: dropped {dropped} passages pre-dedup")
    return docs, queries, qrels


def _passage_quality(text: str) -> bool:
    """Return False for passages too short, too repetitive, or likely boilerplate."""
    words = text.split()
    if len(words) < 5:
        return False
    unique = set(w.lower() for w in words)
    if len(unique) / len(words) < 0.3:
        return False
    low = text.lower()
    # boilerplate markers only — NOT bare URLs ("what is javascript" is a canonical MS MARCO
    # topic, and web passages legitimately cite links; URL-only fragments die on <5 words)
    if any(bp in low for bp in ("click here", "cookie policy",
                                 "terms of service", "privacy policy",
                                 "subscribe to our", "sign up for our", "lorem ipsum")):
        return False
    return True


def load_msmarco_slice(languages=("hi",), max_docs: int = 50_000,
                       include_english: bool = True, include_translated: bool = True) -> list:
    """A documented, dedup'd, quality-filtered MSMARCO-XI corpus slice. -> list[Document].
    (Filtering happens inside _load_msmarco so eval + live share one corpus definition.)"""
    docs, _, _ = _load_msmarco(languages, max_docs, include_english, include_translated)
    return docs


def build_eval_set(languages=("hi",), max_docs: int = 20_000, max_queries: int = 500,
                   include_english: bool = True, include_translated: bool = False):
    """Corpus + CROSS-LINGUAL eval (Indic query -> English passage), qrels from is_selected.
    Returns (docs, queries, qrels) so eval_chunking can score recall@k/MRR/nDCG on real labels."""
    return _load_msmarco(languages, max_docs, include_english, include_translated,
                         collect_eval=True, max_queries=max_queries)


def build_raptor_index(docs, embedder, summarizer, client, prefix,
                       strategy_name="passage", cluster_size=5, max_levels=3,
                       build_manifest: str = ""):
    """Build a RAPTOR tree collection on top of the winning chunking strategy.
    Indexes leaves + summaries in '{prefix}__raptor'. `build_manifest` (8-char CorpusBuildSpec
    manifest) is stamped into every summary's lineage payload."""
    from raptor import RaptorTreeBuilder
    from retrieval import Retriever

    strategies = {
        "passage": lambda: ck.PassageAwareChunker(),
        "fixed": lambda: ck.FixedSizeChunker(),
        "recursive": lambda: ck.RecursiveChunker(),
        "sentwin": lambda: ck.SentenceWindowChunker(),
        "hierarchical": lambda: ck.HierarchicalChunker(),
    }
    chunker = strategies.get(strategy_name, strategies["passage"])()
    chunks = chunker.chunk_corpus(docs)
    embeds = embedder.embed_docs([c.text for c in chunks])

    builder = RaptorTreeBuilder(embedder, summarizer, cluster_size, max_levels,
                                build_manifest=build_manifest)
    summary_chunks, summary_embeds = builder.build(chunks, embeds)

    retriever = Retriever(client, embedder, use_sparse=True)
    all_chunks = list(chunks) + summary_chunks
    all_embeds = list(embeds) + summary_embeds
    collection = f"{prefix}__raptor"
    n = retriever.index(collection, all_chunks, all_embeds)

    max_level = 0
    for c in summary_chunks:
        lvl = c.extra.get("raptor_level", 0)
        if lvl > max_level:
            max_level = lvl

    return {"collection": collection, "n_chunks": n,
            "n_leaves": len(chunks), "n_summaries": len(summary_chunks),
            "max_level": max_level}


def synthetic_docs(n: int = 40) -> list:
    """Deterministic corpus for local index-build tests (no dataset needed)."""
    topics = [
        ("goa", "Goa is a coastal state in western India known for beaches and Portuguese forts."),
        ("rag", "Retrieval augmented generation retrieves passages then generates a grounded answer."),
        ("python", "Python is a programming language used for data science and web backends."),
        ("sarvam", "Sarvam Saarika transcribes Indian-language speech including Hindi and English."),
        ("qdrant", "Qdrant is a vector database supporting dense and sparse hybrid search."),
    ]
    out = []
    for i in range(n):
        key, base = topics[i % len(topics)]
        out.append(ck.Document(doc_id=f"{key}-{i}", language="en",
                               text=f"{base} This is document number {i} about {key}."))
    return out


# =========================================================================================
# LEAKAGE-SAFE PASSAGE-FAMILY SPLIT (P0.4)
# -----------------------------------------------------------------------------------------
# Replaces the row-level eval split. The pipeline:
#   rows -> passage OCCURRENCES (quality-filtered, canonicalized, stable_doc_id'd)
#        -> exact-dedup to canonical passages (one stable_doc_id each)
#        -> FAMILIES (union of: identical-canonical-text, conservative SimHash near-dup,
#                     translated-variant-of-same-row+index)
#        -> whole-family allocation to indexed(40k) / heldout(10k), seeded + deterministic
#        -> query classification: a query is HELDOUT iff its COMPLETE positive family set is
#           held out; INDEXED iff all positives are indexed; else EXCLUDED (unrealizable, counted)
#        -> cal/dev/sealed QUERY partitions for BOTH in-corpus and absent-evidence pools
# Whole-family allocation makes the leakage assertions (exact-hash intersection == 0 AND
# near-dup-family intersection == 0) true BY CONSTRUCTION; assert_split_integrity proves it.
# Every function here is CPU/synthetic-testable — the real run just feeds _iter_msmarco_rows.
# =========================================================================================


class _UnionFind:
    """Tiny union-find over hashable ids (path-compression + union-by-size). Deterministic:
    the representative of a set is always its lexicographically smallest member."""

    def __init__(self):
        self.parent: dict = {}

    def add(self, x) -> None:
        self.parent.setdefault(x, x)

    def find(self, x):
        self.add(x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:      # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        lo, hi = (ra, rb) if ra < rb else (rb, ra)  # keep the smaller id as the root
        self.parent[hi] = lo

    def groups(self) -> dict:
        out: dict = {}
        for x in list(self.parent):
            out.setdefault(self.find(x), []).append(x)
        for r in out:
            out[r].sort()
        return out


@dataclass
class PassageOccurrence:
    """One (passage, variant) instance in one row. Provenance (row_id, passage_index, variant)
    drives the translated-variant family link; stable_doc_id collapses exact duplicates.
    `query_group` is the UN-namespaced source query id: the same underlying MS MARCO query
    appears once per language shard (each with its own translated query text), and the group
    key ties those per-language variants together for partition co-location."""
    row_id: str
    passage_index: int
    variant: str          # "en" | "tr:<lang>"
    lang: str
    text: str             # raw (stripped) passage text
    stable_doc_id: str
    is_selected: int
    query_id: str
    query_text: str
    query_group: str = ""


def _row_occurrences(row: dict, row_id: str, include_english: bool, include_translated: bool,
                     cfg: str, qid_prefix: str = "") -> list:
    """Extract quality-filtered PassageOccurrences from one MSMARCO-XI row, preserving the
    passage index + variant so English_passages[i] and Translated_passages[i] can be linked.
    `qid_prefix` namespaces query_ids per shard (the raw id becomes the cross-shard group)."""
    p = row.get("passages") or {}
    sel = p.get("is_selected") or []
    en = p.get("English_passages") or []
    tr = p.get("Translated_passages") or []
    raw_qid = row.get("query_id")
    has_qid = raw_qid is not None and str(raw_qid).strip() != ""
    # with a real query_id: namespace per shard, group by the raw id (cross-shard identity).
    # without one: fall back to the (already-prefixed) row_id for BOTH -> no cross-shard link.
    qid = f"{qid_prefix}{raw_qid}" if has_qid else row_id
    qgroup = str(raw_qid) if has_qid else row_id
    qtext = (row.get("query") or row.get("Eng_Query") or "").strip()
    out = []
    for i in range(max(len(en), len(tr), len(sel))):
        s = int(sel[i]) if i < len(sel) and sel[i] is not None else 0
        if include_english and i < len(en) and en[i] and en[i].strip():
            text = en[i].strip()
            if _passage_quality(text):
                out.append(PassageOccurrence(
                    row_id=row_id, passage_index=i, variant=cspec.variant_tag(True, "en"),
                    lang=_detect_lang(text, "en"),
                    text=text, stable_doc_id=cspec.stable_doc_id(text, cspec.variant_tag(True, "en")),
                    is_selected=s, query_id=qid, query_text=qtext, query_group=qgroup))
        if include_translated and i < len(tr) and tr[i] and tr[i].strip():
            text = tr[i].strip()
            if _passage_quality(text):
                lang = _detect_lang(text, cfg)
                var = cspec.variant_tag(False, lang)
                out.append(PassageOccurrence(
                    row_id=row_id, passage_index=i, variant=var, lang=lang,
                    text=text, stable_doc_id=cspec.stable_doc_id(text, var),
                    is_selected=s, query_id=qid, query_text=qtext, query_group=qgroup))
    return out


def iter_passage_occurrences(rows: Iterable, include_english: bool = True,
                             include_translated: bool = True, cfg: str = "hi",
                             row_prefix: str = "") -> list:
    """Flatten rows -> list[PassageOccurrence]. `row_prefix` (e.g. 'hi:') keeps row_ids AND
    query_ids disjoint when concatenating multiple language shards — without it, shard-local
    row indices collide and the (row_id, passage_index) translation link would join a Tamil
    translation to a Hindi row's English passage."""
    out = []
    for idx, row in enumerate(rows):
        rid = f"{row_prefix}{idx}" if row_prefix else str(idx)
        out.extend(_row_occurrences(row, rid, include_english, include_translated, cfg,
                                    qid_prefix=row_prefix))
    return out


def occurrences_from_shards(shards: Sequence, spec: cspec.CorpusBuildSpec) -> list:
    """Multi-shard occurrence collection with DISJOINT per-shard namespaces and ONE unified
    family graph downstream. `shards` = sequence of (cfg, rows_iterable); each shard's row_ids
    and query_ids get the '<cfg>:' prefix so provenance links never collide across shards,
    while exact-hash dedup still collapses the English passages that repeat across shards into
    a single cross-shard family (rule 1), and each shard's translations join that family via
    their own (prefixed) row provenance (rule 3)."""
    occ = []
    for cfg, rows in shards:
        occ.extend(iter_passage_occurrences(rows, spec.include_english, spec.include_translated,
                                            cfg=cfg, row_prefix=f"{cfg}:"))
    return occ


@dataclass
class FamilyIndex:
    """Canonical passages grouped into leakage-safe families."""
    family_of: dict          # stable_doc_id -> family_id (rep = smallest sdid in the family)
    members: dict            # family_id -> sorted list[stable_doc_id]  (unique canonical passages)
    size: dict               # family_id -> int (docs = # unique canonical passages)
    sdid_text: dict          # stable_doc_id -> representative raw text
    sdid_lang: dict          # stable_doc_id -> language
    sdid_variant: dict       # stable_doc_id -> variant tag
    exact_groups: dict       # exact_hash(canonical text) -> sorted list[stable_doc_id]

    @property
    def total_docs(self) -> int:
        return sum(self.size.values())

    def family_id(self, sdid: str) -> str:
        return self.family_of[sdid]


def build_family_index(occurrences: Sequence[PassageOccurrence], spec: cspec.CorpusBuildSpec
                       ) -> FamilyIndex:
    """Group canonical passages into families under THREE union rules (all conservative):
      1. identical canonical text (variant-agnostic) -> same family  [makes exact-hash
         intersection provably 0 after whole-family allocation]
      2. SimHash Hamming <= spec.simhash_hamming_max -> same family   [conservative near-dup]
      3. same (row_id, passage_index) across variants  -> same family [translated variant joins
         its source English passage's family]
    """
    uf = _UnionFind()
    sdid_text: dict = {}
    sdid_lang: dict = {}
    sdid_variant: dict = {}
    exact_groups: dict = {}
    simhash_of: dict = {}
    # provenance -> {variant: sdid} for the translation link
    prov: dict = {}

    for occ in occurrences:
        sid = occ.stable_doc_id
        uf.add(sid)
        if sid not in sdid_text:                      # first occurrence is the representative
            sdid_text[sid] = occ.text
            sdid_lang[sid] = occ.lang
            sdid_variant[sid] = occ.variant
            simhash_of[sid] = cspec.simhash64(occ.text)
            eh = cspec.exact_hash(occ.text)
            exact_groups.setdefault(eh, [])
            if sid not in exact_groups[eh]:
                exact_groups[eh].append(sid)
        prov.setdefault((occ.row_id, occ.passage_index), {})[occ.variant] = sid

    # rule 1: exact canonical-text duplicates across variants
    for eh, sids in exact_groups.items():
        for other in sids[1:]:
            uf.union(sids[0], other)

    # rule 2: conservative SimHash near-dup via LSH banding (candidate buckets, then exact
    # Hamming check <= threshold). Banding guarantees candidate recall for d < n_bands.
    buckets: dict = {}
    for sid, h in simhash_of.items():
        for band in cspec.simhash_bands(h, spec.simhash_bands, spec.simhash_band_bits):
            buckets.setdefault(band, []).append(sid)
    checked: set = set()
    for band, sids in buckets.items():
        if len(sids) < 2:
            continue
        for a in range(len(sids)):
            for b in range(a + 1, len(sids)):
                pair = (sids[a], sids[b]) if sids[a] < sids[b] else (sids[b], sids[a])
                if pair in checked:
                    continue
                checked.add(pair)
                if cspec.hamming(simhash_of[pair[0]], simhash_of[pair[1]]) <= spec.simhash_hamming_max:
                    uf.union(pair[0], pair[1])

    # rule 3: translated-variant family link (same row + passage index, different variant)
    for variants in prov.values():
        sids = list(variants.values())
        for other in sids[1:]:
            uf.union(sids[0], other)

    groups = uf.groups()
    family_of, members, size = {}, {}, {}
    for rep, sids in groups.items():
        members[rep] = sids
        size[rep] = len(sids)
        for s in sids:
            family_of[s] = rep
    return FamilyIndex(family_of=family_of, members=members, size=size,
                       sdid_text=sdid_text, sdid_lang=sdid_lang, sdid_variant=sdid_variant,
                       exact_groups={k: sorted(v) for k, v in exact_groups.items()})


@dataclass
class QrelIndex:
    """Query -> its selected (positive) canonical passages and their families."""
    query_text: dict         # query_id -> text
    positives: dict          # query_id -> set[stable_doc_id]  (is_selected == 1)
    families: dict           # query_id -> set[family_id]
    group_of: dict = field(default_factory=dict)   # query_id -> translation-group key

    @property
    def query_ids(self) -> list:
        return sorted(self.positives)


def build_qrels(occurrences: Sequence[PassageOccurrence], family_index: FamilyIndex) -> QrelIndex:
    """Collect per-query positive passages (is_selected==1) and map them to families."""
    query_text, positives, families, group_of = {}, {}, {}, {}
    for occ in occurrences:
        if occ.query_text:
            query_text.setdefault(occ.query_id, occ.query_text)
        group_of.setdefault(occ.query_id, occ.query_group or occ.query_id)
        if occ.is_selected:
            positives.setdefault(occ.query_id, set()).add(occ.stable_doc_id)
    for qid, sids in positives.items():
        families[qid] = {family_index.family_of[s] for s in sids if s in family_index.family_of}
    return QrelIndex(query_text=query_text, positives=positives, families=families,
                     group_of=group_of)


class BuildConstraintError(Exception):
    """Raised when the leakage-safe split cannot be realized within the spec's constraints
    (amendment 5). The message names the BINDING constraint — never silently weakened."""


@dataclass
class Allocation:
    indexed_families: set
    heldout_families: set
    unused_families: set
    indexed_docs: int
    heldout_docs: int
    indexed_queries: list        # queries whose COMPLETE positive family set is indexed
    heldout_queries: list        # queries whose COMPLETE positive family set is held out
    excluded_queries: list       # list[(query_id, reason)] — unrealizable, counted (never dropped silently)
    algorithm: str

    def indexed_stable_doc_ids(self, family_index: FamilyIndex) -> list:
        out = []
        for fam in self.indexed_families:
            out.extend(family_index.members[fam])
        return out

    def heldout_stable_doc_ids(self, family_index: FamilyIndex) -> list:
        out = []
        for fam in self.heldout_families:
            out.extend(family_index.members[fam])
        return out


def _within_band(n: int, target: int, tol: float) -> bool:
    return target * (1 - tol) <= n <= target * (1 + tol)


def allocate_families(family_index: FamilyIndex, qrels: QrelIndex,
                      spec: cspec.CorpusBuildSpec) -> Allocation:
    """Seeded, deterministic whole-family allocation.

    Held-out first (query-first greedy so absent-evidence queries are fully covered), then
    indexed from what remains, then classify queries. Raises BuildConstraintError naming the
    binding constraint if the targets/query floors cannot be met (amendment 5)."""
    spec.validate()
    all_fams = set(family_index.members)
    size = family_index.size
    rng = random.Random(spec.allocation_seed)

    hi_hb = spec.heldout_docs_target * (1 + spec.size_tolerance)
    lo_hb = spec.heldout_docs_target * (1 - spec.size_tolerance)
    hi_ib = spec.indexed_docs_target * (1 + spec.size_tolerance)
    lo_ib = spec.indexed_docs_target * (1 - spec.size_tolerance)

    fams_touched_by_query = set()
    for fams in qrels.families.values():
        fams_touched_by_query |= fams

    # ---- phase 1: held-out (query-first) ----
    q_order = sorted(qrels.families)
    rng.shuffle(q_order)
    H: set = set()
    H_docs = 0
    for qid in q_order:
        fams = qrels.families[qid]
        if not fams:
            continue
        new = fams - H
        add = sum(size[f] for f in new)
        if H_docs + add <= hi_hb:
            H |= new
            H_docs += add
    # pad toward the target with families NO query touches (keeps query realizability intact)
    if H_docs < spec.heldout_docs_target:
        pad = sorted(all_fams - H - fams_touched_by_query)
        rng.shuffle(pad)
        for f in pad:
            if H_docs >= spec.heldout_docs_target:
                break
            if H_docs + size[f] <= hi_hb:
                H.add(f)
                H_docs += size[f]
    if not _within_band(H_docs, spec.heldout_docs_target, spec.size_tolerance):
        raise BuildConstraintError(
            f"held-out docs {H_docs} outside {int(lo_hb)}..{int(hi_hb)} band "
            f"(target {spec.heldout_docs_target} +/-{spec.size_tolerance:.0%}); binding constraint: "
            f"insufficient non-leaking families to fill the held-out partition")

    # ---- phase 2: indexed ----
    avail = sorted(all_fams - H)
    avail_set = set(avail)
    if spec.indexed_docs_target == 0:
        # WHOLE-CORPUS-MINUS-HOLDOUT (sentinel 0): index EVERY family not held out. No indexed
        # band — the tolerance governs the holdout only. Maximizes qrel realizability: the only
        # unrealizable queries left are those whose positives straddle indexed and held-out.
        I = set(avail)
        I_docs = sum(size[f] for f in avail)
    else:
        # fixed target: query-first greedy over families not held out, then pad to the target
        I = set()
        I_docs = 0
        idx_q_order = [q for q in sorted(qrels.families)
                       if qrels.families[q] and qrels.families[q] <= avail_set]
        rng.shuffle(idx_q_order)
        for qid in idx_q_order:
            new = qrels.families[qid] - I
            add = sum(size[f] for f in new)
            if I_docs + add <= hi_ib:
                I |= new
                I_docs += add
        if I_docs < spec.indexed_docs_target:
            pad = [f for f in avail if f not in I]
            rng.shuffle(pad)
            for f in pad:
                if I_docs >= spec.indexed_docs_target:
                    break
                if I_docs + size[f] <= hi_ib:
                    I.add(f)
                    I_docs += size[f]
        if not _within_band(I_docs, spec.indexed_docs_target, spec.size_tolerance):
            raise BuildConstraintError(
                f"indexed docs {I_docs} outside {int(lo_ib)}..{int(hi_ib)} band "
                f"(target {spec.indexed_docs_target} +/-{spec.size_tolerance:.0%}); binding constraint: "
                f"insufficient non-held-out families to fill the indexed partition")

    unused = all_fams - H - I

    # ---- phase 3: classify queries against the realized partition ----
    indexed_q, heldout_q, excluded_q = [], [], []
    for qid in sorted(qrels.families):
        fams = qrels.families[qid]
        if not fams:
            excluded_q.append((qid, "no positive families (no is_selected passage survived filtering)"))
            continue
        if fams <= H:
            heldout_q.append(qid)
        elif fams <= I:
            indexed_q.append(qid)
        else:
            in_h, in_i, in_u = bool(fams & H), bool(fams & I), bool(fams & unused)
            if in_h and in_i:
                reason = "positive families split across indexed and held-out"
            elif in_u and not (in_h or in_i):
                reason = "positive families entirely unused (corpus larger than indexed+heldout targets)"
            else:
                reason = "positive families partially unused (some placed, some unused)"
            excluded_q.append((qid, reason))

    if len(heldout_q) < spec.min_heldout_queries:
        raise BuildConstraintError(
            f"only {len(heldout_q)} fully-held-out queries (need >= {spec.min_heldout_queries}); "
            f"binding constraint: not enough queries whose complete positive family set is held out")

    return Allocation(indexed_families=I, heldout_families=H, unused_families=unused,
                      indexed_docs=I_docs, heldout_docs=H_docs,
                      indexed_queries=indexed_q, heldout_queries=heldout_q,
                      excluded_queries=excluded_q, algorithm=spec.allocation_algorithm)


@dataclass
class QueryPartition:
    calibration: list
    dev: list
    sealed: list

    def counts(self) -> dict:
        return {"calibration": len(self.calibration), "dev": len(self.dev),
                "sealed": len(self.sealed)}


def partition_queries(query_ids: Sequence[str], spec: cspec.CorpusBuildSpec,
                      salt: str = "", groups: Optional[dict] = None) -> QueryPartition:
    """Split query ids into disjoint calibration/dev/sealed (amendment: sealed ids are written
    but their CONTENT is never used in any dev path). Seeded + deterministic; `salt`
    differentiates the in-corpus vs absent-evidence pools while sharing partition_seed.

    `groups` (optional) maps qid -> translation-group key (the un-namespaced source query id).
    Fractions are then applied over GROUPS and every member of a group lands in the SAME
    partition — the Hindi and Tamil variants of one source query can never straddle calibration
    and sealed (that would leak tuning knowledge of a sealed query's content through its
    sibling language). With no groups (or identity groups) this reduces exactly to the original
    per-query behavior. Published sizes are realized QUERY counts (`counts()`)."""
    ids = sorted(set(query_ids))
    gmap: dict = {}
    for q in ids:
        gmap.setdefault((groups or {}).get(q, q), []).append(q)
    gkeys = sorted(gmap)
    # deterministic salt hash — Python's built-in hash() is per-process randomized (PYTHONHASHSEED)
    # and would break cross-run reproducibility (a P0 exit gate).
    salt_h = int.from_bytes(hashlib.blake2b(salt.encode("utf-8"), digest_size=4).digest(), "big")
    seed = (spec.partition_seed ^ salt_h) if salt else spec.partition_seed
    random.Random(seed).shuffle(gkeys)
    n = len(gkeys)
    n_cal = min(int(round(spec.calibration_frac * n)), n)
    n_dev = min(int(round(spec.dev_frac * n)), n - n_cal)
    cal = [q for g in gkeys[:n_cal] for q in gmap[g]]
    dev = [q for g in gkeys[n_cal:n_cal + n_dev] for q in gmap[g]]
    sealed = [q for g in gkeys[n_cal + n_dev:] for q in gmap[g]]
    return QueryPartition(calibration=sorted(cal), dev=sorted(dev), sealed=sorted(sealed))


def assert_split_integrity(allocation: Allocation, family_index: FamilyIndex) -> dict:
    """Prove the leakage assertions the P0 exit gate demands and RETURN the report (which the
    manifest publishes). Raises AssertionError on any leak — never downgraded to a warning."""
    fam_intersection = allocation.indexed_families & allocation.heldout_families
    assert not fam_intersection, f"near-dup-family intersection != 0: {sorted(fam_intersection)[:5]}"

    idx_sdids = set(allocation.indexed_stable_doc_ids(family_index))
    held_sdids = set(allocation.heldout_stable_doc_ids(family_index))
    sdid_intersection = idx_sdids & held_sdids
    assert not sdid_intersection, f"stable_doc_id intersection != 0: {sorted(sdid_intersection)[:5]}"

    idx_exact = {cspec.exact_hash(family_index.sdid_text[s]) for s in idx_sdids}
    held_exact = {cspec.exact_hash(family_index.sdid_text[s]) for s in held_sdids}
    exact_intersection = idx_exact & held_exact
    assert not exact_intersection, f"exact-hash intersection != 0: {len(exact_intersection)} texts"

    return {
        "family_intersection": 0,
        "stable_doc_id_intersection": 0,
        "exact_hash_intersection": 0,
        "indexed_docs": len(idx_sdids),
        "heldout_docs": len(held_sdids),
        "indexed_families": len(allocation.indexed_families),
        "heldout_families": len(allocation.heldout_families),
        "indexed_exact_hashes": len(idx_exact),
        "heldout_exact_hashes": len(held_exact),
    }


def remap_qrels(qrels: QrelIndex, allocation: Allocation, family_index: FamilyIndex,
                query_ids: Sequence[str]) -> dict:
    """Remap a set of (in-corpus) queries' positives to the stable_doc_ids that actually resolve
    in the INDEXED corpus. Returns {query_id: sorted[stable_doc_id]}. For a correctly-classified
    in-corpus query every positive resolves; use assert_positives_resolve to enforce it."""
    indexed = set(allocation.indexed_stable_doc_ids(family_index))
    out = {}
    for qid in query_ids:
        pos = qrels.positives.get(qid, set())
        out[qid] = sorted(p for p in pos if p in indexed)
    return out


def assert_positives_resolve(remapped: dict, corpus_stable_doc_ids: Sequence[str],
                             qrels: QrelIndex) -> dict:
    """Assert every reported qrel positive resolves in the given leaf collection (P0 exit gate:
    'every scored positive resolves in every compared leaf collection'). Returns a realizability
    report; raises AssertionError listing the first unresolved positives."""
    corpus = set(corpus_stable_doc_ids)
    unresolved = []
    total = 0
    for qid, sids in remapped.items():
        # a realizable in-corpus query must keep ALL of its original positives
        original = qrels.positives.get(qid, set())
        for p in original:
            total += 1
            if p not in corpus:
                unresolved.append((qid, p))
    assert not unresolved, (f"{len(unresolved)} qrel positives do not resolve in the corpus "
                            f"(first: {unresolved[:3]})")
    return {"queries": len(remapped), "positives_checked": total, "unresolved": 0}


@dataclass
class SplitPlan:
    """The full realized plan — the object the dry-run prints and the real build consumes."""
    spec: cspec.CorpusBuildSpec
    family_index: FamilyIndex
    qrels: QrelIndex
    allocation: Allocation
    incorpus_partition: QueryPartition      # cal/dev/sealed over indexed (answerable) queries
    absent_partition: QueryPartition        # cal/dev/sealed over held-out (absent-evidence) queries
    integrity: dict
    realization: cspec.BuildRealization

    def collection_names(self, strategies: Sequence[str] = ("passage", "raptor")) -> list:
        return [self.spec.collection_name(s) for s in strategies]


def _plan_from_occurrences(occ: list, spec: cspec.CorpusBuildSpec) -> SplitPlan:
    """Shared planning core: occurrences -> UNIFIED family graph -> qrels -> allocation ->
    group-aware partitions -> integrity assertions -> realization manifest."""
    fam = build_family_index(occ, spec)
    qr = build_qrels(occ, fam)
    alloc = allocate_families(fam, qr, spec)
    integrity = assert_split_integrity(alloc, fam)

    incorpus = partition_queries(alloc.indexed_queries, spec, salt="in_corpus",
                                 groups=qr.group_of)
    absent = partition_queries(alloc.heldout_queries, spec, salt="absent_evidence",
                               groups=qr.group_of)
    for part_name, part in (("in_corpus", incorpus), ("absent_evidence", absent)):
        for pname in ("calibration", "dev", "sealed"):
            if len(getattr(part, pname)) < spec.min_partition_queries:
                raise BuildConstraintError(
                    f"{part_name}/{pname} has {len(getattr(part, pname))} queries "
                    f"(need >= {spec.min_partition_queries}); binding constraint: too few "
                    f"realizable queries to form a {pname} partition")

    idx_sdids = alloc.indexed_stable_doc_ids(fam)
    realization = cspec.BuildRealization(
        manifest8=spec.manifest8(), spec_json=spec.canonical_json(),
        source_rows=len({o.row_id for o in occ}), source_passages=len(occ),
        unique_canonical=len(fam.family_of), n_families=len(fam.members),
        indexed_docs=alloc.indexed_docs, heldout_docs=alloc.heldout_docs,
        indexed_families=len(alloc.indexed_families), heldout_families=len(alloc.heldout_families),
        heldout_queries=len(alloc.heldout_queries), indexed_queries=len(alloc.indexed_queries),
        excluded_queries=len(alloc.excluded_queries),
        calibration_queries=len(incorpus.calibration) + len(absent.calibration),
        dev_queries=len(incorpus.dev) + len(absent.dev),
        sealed_queries=len(incorpus.sealed) + len(absent.sealed),
        corpus_hash=cspec.corpus_hash(idx_sdids),
        notes={"in_corpus_partition": incorpus.counts(),
               "absent_evidence_partition": absent.counts(),
               "excluded_reasons": _summarize_exclusions(alloc.excluded_queries),
               "query_groups": {
                   "indexed": len({qr.group_of.get(q, q) for q in alloc.indexed_queries}),
                   "heldout": len({qr.group_of.get(q, q) for q in alloc.heldout_queries})}})
    return SplitPlan(spec=spec, family_index=fam, qrels=qr, allocation=alloc,
                     incorpus_partition=incorpus, absent_partition=absent,
                     integrity=integrity, realization=realization)


def plan_split(rows: Iterable, spec: cspec.CorpusBuildSpec, cfg: str = "hi",
               row_prefix: str = "") -> SplitPlan:
    """Single-shard orchestrator: rows -> occurrences -> plan. Pure planning (no Qdrant, no
    models). Kept for single-language runs and tests; multi-language runs MUST use
    plan_split_multi (disjoint per-shard namespaces)."""
    occ = iter_passage_occurrences(rows, spec.include_english, spec.include_translated,
                                   cfg=cfg, row_prefix=row_prefix)
    return _plan_from_occurrences(occ, spec)


def plan_split_multi(shards: Sequence, spec: cspec.CorpusBuildSpec) -> SplitPlan:
    """Multi-shard orchestrator for the ALL-14-languages posture. `shards` = sequence of
    (cfg, rows_iterable); each shard gets the '<cfg>:' row/query namespace (no provenance
    collisions) while the family graph is built ONCE over all shards, so English passages that
    repeat across shards collapse into a single cross-shard family and every shard's
    translations join it. Both the --dry-run command and the real Forge build call this."""
    return _plan_from_occurrences(occurrences_from_shards(shards, spec), spec)


def _summarize_exclusions(excluded: Sequence) -> dict:
    out: dict = {}
    for _, reason in excluded:
        out[reason] = out.get(reason, 0) + 1
    return out


def documents_from_allocation(allocation: Allocation, family_index: FamilyIndex,
                              spec: cspec.CorpusBuildSpec) -> list:
    """Materialize the INDEXED partition as ck.Document objects (stable_doc_id set), ready for
    chunking + indexing on Forge. Deterministic doc_ids derived from the stable_doc_id so two
    loader runs produce identical ids. Never returns held-out or unused passages."""
    docs = []
    for sdid in sorted(allocation.indexed_stable_doc_ids(family_index)):
        lang = family_index.sdid_lang.get(sdid, "en")
        # full stable_doc_id in the doc_id -> globally unique by construction (it's a SHA256),
        # so no chunk_id/point_id collision can silently overwrite a Qdrant point.
        docs.append(ck.Document(doc_id=f"{lang}-{sdid}", text=family_index.sdid_text[sdid],
                                language=lang, stable_doc_id=sdid))
    return docs


_SYN_VOCAB = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike "
              "november oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee "
              "zulu apple river mountain planet signal harbor lantern compass anchor meadow "
              "cipher photon quartz basalt").split()


def synthetic_msmarco_rows(n_rows: int = 40, seed: int = 0, shared_answer_rate: float = 0.15,
                           dup_rate: float = 0.2, neardup_rate: float = 0.1,
                           multi_positive_rate: float = 0.0,
                           with_translations: bool = True) -> list:
    """Deterministic MSMARCO-XI-shaped rows for testing the split machinery with NO dataset.

    Passages are DIVERSE (random word samples + a unique marker) so distinct passages sit far
    apart in SimHash space (no accidental chaining). Controlled structure is injected on purpose:
      * `shared_answer_rate` — a row's SELECTED passage is an exact copy of an earlier row's
        selected passage -> two queries share a positive family (the case leakage-safety must
        force onto one side).
      * `dup_rate`          — a distractor is an exact duplicate of an earlier passage.
      * `neardup_rate`      — a distractor is a 1-char typo of this row's answer (near-dup family).
      * `with_translations` — a parallel Hindi variant of the answer (unique per row) that links
        to its English source only via (row, passage_index) provenance.
    """
    rng = random.Random(seed)
    rows: list = []
    answers: list = []      # each row's selected English passage text
    for r in range(n_rows):
        words = " ".join(rng.sample(_SYN_VOCAB, 8))
        if answers and rng.random() < shared_answer_rate:
            answer = rng.choice(answers)                 # share a POSITIVE with an earlier query
        else:
            answer = f"Passage {r}: {words}. Reference marker number {r} for retrieval evaluation."
        answers.append(answer)

        en = [answer]
        tr = [f"अनुच्छेद {r}: {words} संदर्भ चिह्न संख्या {r}." if with_translations else ""]
        sel = [1]

        # one distractor passage
        roll = rng.random()
        if answers[:-1] and roll < dup_rate:
            distractor = rng.choice(answers[:-1])        # exact duplicate of an earlier passage
        elif roll < dup_rate + neardup_rate:
            distractor = answer[:-1] + "X."              # 1-char near-dup of this row's answer
        else:
            w2 = " ".join(rng.sample(_SYN_VOCAB, 8))
            distractor = f"Distractor {r}: {w2}. Unrelated note number {r} kept separate here."
        en.append(distractor)
        tr.append("")
        # occasionally the distractor is ALSO a positive -> a multi-family query (exercises the
        # 'positives split across partitions -> excluded' path when its families are allocated apart)
        sel.append(1 if (answers[:-1] and roll < dup_rate and rng.random() < multi_positive_rate)
                   else 0)

        rows.append({
            "query": f"query {r} about {words.split()[0]} topic",
            "Eng_Query": f"query {r} about {words.split()[0]} topic",
            "query_id": f"q{r}",
            "passages": {"English_passages": en, "Translated_passages": tr, "is_selected": sel},
        })
    return rows
