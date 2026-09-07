"""
corpus_spec.py — immutable corpus identity for the leakage-safe MSMARCO-XI build (P0.4).

This module is the single source of truth for THREE reproducible identities the evaluation
protocol depends on, and nothing here ever touches the network, a model, or Qdrant — so it is
100% unit-testable on the laptop and produces byte-identical results on Forge:

1. `canonicalize(text)`            — the ONE text normalization every hash agrees on.
2. `stable_doc_id(text, variant)`  — SHA256(canonical text + variant metadata); collapses exact
                                     duplicates (English passages repeat across queries) and
                                     separates variants (an English passage vs a coincidentally
                                     identical translated string never collide).
3. `CorpusBuildSpec` + `manifest8` — the frozen manifest hash (first 8 hex of SHA256 of the
                                     canonical spec JSON) that names versioned collections
                                     `msmarco_xi_40k_<manifest8>__<strategy>` and that the sealed
                                     test command demands before it will run.

Plus the hand-rolled 64-bit SimHash used for conservative near-duplicate FAMILY grouping
(no new dependency — stdlib hashlib only).

Design rule (verifier): the manifest hash is computed from the BUILD INPUTS only (dataset
revision, shards, seeds, algorithm + threshold versions, tokenizer/model revisions, chunk +
RAPTOR params, targets). Realized OUTPUTS (counts, hashes, query manifests) live in a separate
`BuildRealization` artifact tagged with the same manifest8 — so freezing the spec fixes the
collection name and the sealed-test key BEFORE the build runs, and the outputs never mutate it.
"""
from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Optional

# ---- canonical text identity -----------------------------------------------------------
# One normalization, used by stable_doc_id AND by exact-duplicate grouping AND by SimHash, so
# the three can never disagree about what "the same passage" means. NFC (compose combining
# marks so "e"+U+0301 == "é"), collapse ALL whitespace runs to one space, strip ends. Case is
# PRESERVED: two passages differing only in case are treated as distinct documents (safer than
# silently merging — case can be semantic, and near-dup SimHash still links true variants).
_HASH_SCHEME = "v1"   # bump if canonicalize() ever changes — old stable_doc_ids stay attributable


def canonicalize(text: str) -> str:
    """The canonical form of a passage. Deterministic, dependency-free, script-agnostic."""
    if not text:
        return ""
    t = unicodedata.normalize("NFC", text)
    return " ".join(t.split()).strip()


def variant_tag(is_english: bool, lang: str) -> str:
    """Stable variant metadata folded into the doc id: 'en' for the shared English passage,
    else 'tr:<lang>' for a translated variant. Keeps English dedup global while separating a
    translated string that happens to canonicalize to the same bytes as an English one."""
    return "en" if is_english else f"tr:{lang or 'xx'}"


def stable_doc_id(text: str, variant: str, *, hex_len: int = 64) -> str:
    """SHA256(canonical passage text + stable variant metadata) -> hex.

    Reproducible across independent loader runs (P0 exit gate). `hex_len` lets callers trade
    payload size for collision margin; default is the full 64-hex digest (collision-free)."""
    canon = canonicalize(text)
    material = f"{_HASH_SCHEME}\x1f{variant}\x1f{canon}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()[:hex_len]


def exact_hash(text: str) -> str:
    """Exact-duplicate key: SHA256 of the canonical text only (variant-agnostic). Used for the
    leakage assertion `exact-hash intersection == 0` across partitions."""
    return hashlib.sha256((_HASH_SCHEME + "\x1f" + canonicalize(text)).encode("utf-8")).hexdigest()


# ---- hand-rolled 64-bit SimHash (no deps) ----------------------------------------------
_SIMHASH_BITS = 64


def _feature_hash64(token: str) -> int:
    """64-bit stable hash of one token (blake2b truncated — deterministic across runs/OS)."""
    return int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "big")


def simhash64(text: str) -> int:
    """Charikar SimHash over whitespace tokens of the canonical text -> 64-bit int.

    Near-identical passages (small edits) produce hashes a few bits apart; unrelated passages
    are ~32 bits apart. Tokenization is script-agnostic (splits on whitespace after canonical
    normalization), so it works for Devanagari/Urdu as well as Latin."""
    toks = canonicalize(text).lower().split()
    if not toks:
        return 0
    acc = [0] * _SIMHASH_BITS
    counts: dict = {}
    for t in toks:
        counts[t] = counts.get(t, 0) + 1
    for tok, w in counts.items():
        h = _feature_hash64(tok)
        for b in range(_SIMHASH_BITS):
            if (h >> b) & 1:
                acc[b] += w
            else:
                acc[b] -= w
    out = 0
    for b in range(_SIMHASH_BITS):
        if acc[b] > 0:
            out |= (1 << b)
    return out


def hamming(a: int, b: int) -> int:
    """Population count of the XOR — number of differing bits between two 64-bit SimHashes."""
    return (a ^ b).bit_count()


def simhash_bands(h: int, n_bands: int, band_bits: int) -> list:
    """Split a 64-bit SimHash into `n_bands` sub-keys of `band_bits` bits each (LSH banding).

    Two hashes within Hamming distance d < n_bands are GUARANTEED to share at least one band
    (pigeonhole) — so banding candidate-generation never misses a near-dup below that bound.
    Returns [(band_index, band_value), ...] as the bucket keys."""
    mask = (1 << band_bits) - 1
    return [(i, (h >> (i * band_bits)) & mask) for i in range(n_bands)]


# ---- the immutable build spec ----------------------------------------------------------
@dataclass
class CorpusBuildSpec:
    """The frozen inputs that define ONE reproducible corpus build. `manifest8()` hashes the
    canonical JSON of exactly these fields — change any of them and you get a new manifest, a
    new collection name, and a new sealed-test key (that is the point)."""

    # --- dataset identity ---
    dataset: str = "ai4bharat/MSMARCO-XI"
    dataset_revision: str = "UNPINNED"          # HF commit SHA — MUST be pinned before freeze
    shards: tuple = ("validation/hinval.parquet",)  # ordered shard paths actually iterated
    splits: tuple = ("validation",)
    languages: tuple = ("hi",)                  # MSMARCO-XI config codes (no 'en' config exists)
    include_english: bool = True
    include_translated: bool = True
    # rows consumed PER SHARD (0 = the whole shard). HASHED: this bounds the realized corpus,
    # so two builds with different caps must never share a manifest — the sizing probe proved
    # the gap (tiers 3000 and 6000 initially collapsed to one manifest8).
    max_rows_per_shard: int = 0

    # --- allocation (leakage-safe family split) ---
    allocation_seed: int = 20260817
    allocation_algorithm: str = "whole-family-query-first-greedy-v1"
    # indexed_docs_target semantics (hashed — changing the mode changes the manifest):
    #   N > 0  -> fixed target: indexed partition fills to N within +/- size_tolerance.
    #   0      -> SENTINEL "whole-corpus-minus-holdout": indexed := EVERY family not held out
    #             (realized indexed = total unique canonical docs - realized holdout). The
    #             tolerance band applies to the HOLDOUT only. This maximizes qrel
    #             realizability — the only unrealizable queries left are those whose positive
    #             families straddle indexed and held-out.
    indexed_docs_target: int = 40_000
    heldout_docs_target: int = 10_000
    size_tolerance: float = 0.01                # +/-1% band (amendment 1; holdout-only when
                                                # indexed_docs_target == 0)

    # --- transform / dedup / gate versions + thresholds ---
    filter_version: str = "passage_quality-v1"  # _passage_quality() contract
    dedup_version: str = "canonical-nfc-v1"     # canonicalize() contract
    simhash_bits: int = 64
    simhash_bands: int = 4                       # 4 bands x 16 bits: catches Hamming <= 3
    simhash_band_bits: int = 16
    simhash_hamming_max: int = 3                 # conservative near-dup ceiling
    translation_gate_version: str = "audit-triggered-v1"  # amendment 4 (no unconditional drop)

    # --- tokenizer + embedding model revisions ---
    tokenizer: str = "BAAI/bge-m3"
    tokenizer_revision: str = "UNPINNED"
    embed_model: str = "BAAI/bge-m3"
    embed_revision: str = "UNPINNED"

    # --- chunking (promoted strategy + its params) ---
    chunk_strategy: str = "passage"
    chunk_params: dict = field(default_factory=lambda: {"min_tokens": 48, "max_tokens": 320})

    # --- RAPTOR ---
    raptor_cluster_size: int = 5
    raptor_max_levels: int = 3
    raptor_summarizer: str = "Qwen/Qwen2.5-3B-Instruct"

    # --- query partitioning (QUERY counts, seeded, disjoint) ---
    partition_seed: int = 20260818
    calibration_frac: float = 0.4
    dev_frac: float = 0.3
    sealed_frac: float = 0.3                     # cal+dev+sealed must sum to 1.0
    min_heldout_queries: int = 30               # build-failure floor (amendment 5)
    min_partition_queries: int = 5              # each cal/dev/sealed partition floor

    # --- collection naming ---
    collection_prefix: str = "msmarco_xi_40k"

    # ---- canonicalization + hashing ----------------------------------------------------
    def to_canonical_dict(self) -> dict:
        """A plain, JSON-safe dict with deterministic ordering for hashing. Tuples -> lists,
        nested dicts key-sorted by json.dumps(sort_keys=True)."""
        d = asdict(self)
        # normalize tuple/list containers to lists so JSON is stable regardless of caller type
        for k, v in list(d.items()):
            if isinstance(v, tuple):
                d[k] = list(v)
        return d

    def canonical_json(self) -> str:
        return json.dumps(self.to_canonical_dict(), sort_keys=True,
                          separators=(",", ":"), ensure_ascii=True)

    def manifest_full(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def manifest8(self) -> str:
        """First 8 hex of SHA256(canonical spec JSON) — the versioned build identity."""
        return self.manifest_full()[:8]

    def collection_name(self, strategy: str) -> str:
        """Versioned collection name: never a live prefix, always manifest-scoped.
        e.g. msmarco_xi_40k_<manifest8>__passage"""
        return f"{self.collection_prefix}_{self.manifest8()}__{strategy}"

    def validate(self) -> None:
        """Cheap structural sanity checks (NOT a build gate — that is the allocator's job)."""
        s = self.calibration_frac + self.dev_frac + self.sealed_frac
        if abs(s - 1.0) > 1e-9:
            raise ValueError(f"partition fractions must sum to 1.0, got {s}")
        if not (0.0 <= self.size_tolerance < 0.5):
            raise ValueError("size_tolerance must be in [0, 0.5)")
        if self.indexed_docs_target < 0:
            raise ValueError("indexed_docs_target must be >= 0 (0 = whole-corpus-minus-holdout)")
        if self.heldout_docs_target <= 0:
            raise ValueError("heldout_docs_target must be positive")
        if self.simhash_bands * self.simhash_band_bits != self.simhash_bits:
            raise ValueError("simhash_bands * simhash_band_bits must equal simhash_bits")
        if self.simhash_hamming_max >= self.simhash_bands:
            # pigeonhole completeness: banding only guarantees candidate recall for d < n_bands
            raise ValueError("simhash_hamming_max must be < simhash_bands for LSH completeness")


@dataclass
class BuildRealization:
    """The realized OUTPUTS of a build, tagged with the spec's manifest8. Separate from the
    spec so it never perturbs the manifest hash. This is the object serialized into the
    immutable build manifest / split-integrity report alongside the spec."""
    manifest8: str
    spec_json: str                              # the exact canonical_json() that was frozen
    source_rows: int = 0
    source_passages: int = 0                    # raw passage occurrences seen
    unique_canonical: int = 0                   # distinct stable_doc_ids
    n_families: int = 0
    indexed_docs: int = 0                       # realized (within target +/- tolerance)
    heldout_docs: int = 0
    indexed_families: int = 0
    heldout_families: int = 0
    heldout_queries: int = 0
    indexed_queries: int = 0
    excluded_queries: int = 0                   # split-family (unrealizable) queries, counted
    calibration_queries: int = 0
    dev_queries: int = 0
    sealed_queries: int = 0
    corpus_hash: str = ""                       # hash over the sorted indexed stable_doc_ids
    notes: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def corpus_hash(stable_doc_ids) -> str:
    """Order-independent hash over a set of stable_doc_ids — the fingerprint of a realized
    corpus partition. Two builds with the same members hash identically regardless of order."""
    h = hashlib.sha256()
    for sid in sorted(set(stable_doc_ids)):
        h.update(sid.encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()
