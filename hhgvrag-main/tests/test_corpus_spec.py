"""
test_corpus_spec.py — immutable corpus identity (P0.4): canonicalization, stable_doc_id,
exact-hash, hand-rolled 64-bit SimHash + LSH banding, and the CorpusBuildSpec manifest hash.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from corpus_spec import (  # noqa: E402
    canonicalize, variant_tag, stable_doc_id, exact_hash, simhash64, hamming, simhash_bands,
    CorpusBuildSpec, BuildRealization, corpus_hash,
)


# ---- canonicalization ------------------------------------------------------------------
def test_canonicalize_collapses_whitespace():
    assert canonicalize("  Hello   world\n\ttest ") == "Hello world test"

def test_canonicalize_nfc_unifies_combining_marks():
    # "e" + combining acute  ==  precomposed "é"
    assert canonicalize("café") == canonicalize("café")

def test_canonicalize_empty():
    assert canonicalize("") == "" and canonicalize(None) == ""


# ---- stable_doc_id ---------------------------------------------------------------------
def test_stable_doc_id_deterministic_and_whitespace_invariant():
    a = stable_doc_id("The quick brown fox", variant_tag(True, "en"))
    b = stable_doc_id("The   quick  brown   fox", variant_tag(True, "en"))
    assert a == b, "whitespace-variant text must collapse to the same stable id"
    assert len(a) == 64

def test_stable_doc_id_variant_separates_en_from_translated():
    en = stable_doc_id("bank", variant_tag(True, "en"))
    tr = stable_doc_id("bank", variant_tag(False, "hi"))
    assert en != tr, "same text, different variant metadata -> different stable id"

def test_stable_doc_id_reproduces_across_calls():
    # P0 exit gate: stable IDs reproduce across independent loader runs
    ids = {stable_doc_id("reproducible passage text", variant_tag(True, "en")) for _ in range(5)}
    assert len(ids) == 1

def test_stable_doc_id_hex_len():
    assert len(stable_doc_id("x", "en", hex_len=16)) == 16


# ---- exact hash ------------------------------------------------------------------------
def test_exact_hash_variant_agnostic():
    # exact_hash ignores variant (canonical TEXT only) -> catches cross-variant identical text
    assert exact_hash("same text here") == exact_hash("same   text here")
    assert exact_hash("a") != exact_hash("b")


# ---- SimHash + banding -----------------------------------------------------------------
def test_simhash_near_beats_far():
    base = "the vector database supports dense and sparse hybrid retrieval with metadata filters"
    near = base.replace("filters", "filtering")          # tiny edit
    far = "goa is a coastal indian state famous for its beaches and portuguese colonial forts"
    assert hamming(simhash64(base), simhash64(near)) < hamming(simhash64(base), simhash64(far))

def test_simhash_identical_zero_distance():
    assert hamming(simhash64("identical text"), simhash64("identical   text")) == 0

def test_simhash_empty():
    assert simhash64("") == 0

def test_simhash_banding_pigeonhole():
    """Two hashes within Hamming < n_bands MUST share at least one band (LSH completeness)."""
    a = simhash64("alpha bravo charlie delta echo foxtrot golf hotel india")
    # flip 3 bits of a -> guaranteed to share a band with 4 bands
    b = a ^ 0b111
    ba, bb = dict(simhash_bands(a, 4, 16)), dict(simhash_bands(b, 4, 16))
    assert any(ba[i] == bb[i] for i in range(4)), "d=3 < 4 bands must collide in >=1 band"

def test_simhash_bands_shape():
    bands = simhash_bands(simhash64("hello world"), 4, 16)
    assert len(bands) == 4 and all(0 <= v < (1 << 16) for _, v in bands)


# ---- CorpusBuildSpec / manifest --------------------------------------------------------
def test_manifest8_deterministic_and_sensitive():
    s1 = CorpusBuildSpec(dataset_revision="abc123")
    s2 = CorpusBuildSpec(dataset_revision="abc123")
    assert s1.manifest8() == s2.manifest8() and len(s1.manifest8()) == 8
    assert CorpusBuildSpec(dataset_revision="other").manifest8() != s1.manifest8()

def test_manifest8_sensitive_to_every_axis():
    base = CorpusBuildSpec(dataset_revision="rev")
    m = base.manifest8()
    variants = [
        CorpusBuildSpec(dataset_revision="rev", allocation_seed=999),
        CorpusBuildSpec(dataset_revision="rev", chunk_strategy="fixed"),
        CorpusBuildSpec(dataset_revision="rev", simhash_hamming_max=2),
        CorpusBuildSpec(dataset_revision="rev", tokenizer_revision="pinned"),
        CorpusBuildSpec(dataset_revision="rev", indexed_docs_target=30000),
    ]
    assert all(v.manifest8() != m for v in variants), "each build axis must move the manifest"

def test_manifest_sensitive_to_row_cap_and_shards():
    """The sizing-probe gap: corpus-shaping inputs (per-shard row cap, realized shard list)
    MUST move the manifest — two different corpora may never share a manifest8."""
    base = CorpusBuildSpec(dataset_revision="r")
    a = CorpusBuildSpec(dataset_revision="r", max_rows_per_shard=3000)
    b = CorpusBuildSpec(dataset_revision="r", max_rows_per_shard=6000)
    assert len({base.manifest8(), a.manifest8(), b.manifest8()}) == 3
    c = CorpusBuildSpec(dataset_revision="r",
                        shards=("validation/hinval.parquet", "validation/tamval.parquet"))
    assert c.manifest8() != base.manifest8()

def test_collection_name_versioned():
    s = CorpusBuildSpec(dataset_revision="rev")
    name = s.collection_name("passage")
    assert name == f"msmarco_xi_40k_{s.manifest8()}__passage"
    assert "40k" in name and s.manifest8() in name

def test_canonical_json_stable_ordering():
    s = CorpusBuildSpec(dataset_revision="rev")
    # tuples serialize as lists, keys sorted -> identical string every call
    assert s.canonical_json() == s.canonical_json()
    assert '"dataset_revision":"rev"' in s.canonical_json()

def test_spec_validate_rejects_bad_fractions():
    for kw in ({"calibration_frac": 0.5, "dev_frac": 0.3, "sealed_frac": 0.3},
               {"size_tolerance": 0.9},
               {"simhash_bands": 5, "simhash_band_bits": 16},   # 5*16 != 64
               {"simhash_hamming_max": 4, "simhash_bands": 4}):  # hamming_max !< bands
        try:
            CorpusBuildSpec(dataset_revision="r", **kw).validate()
            assert False, f"validate should reject {kw}"
        except ValueError:
            pass

def test_spec_validate_accepts_defaults():
    CorpusBuildSpec(dataset_revision="r").validate()   # must not raise

def test_spec_indexed_sentinel_zero_is_whole_corpus_mode():
    # 0 = whole-corpus-minus-holdout: valid, hashed (differs from any fixed target)
    s0 = CorpusBuildSpec(dataset_revision="r", indexed_docs_target=0)
    s0.validate()
    assert s0.manifest8() != CorpusBuildSpec(dataset_revision="r").manifest8()
    for bad in ({"indexed_docs_target": -1}, {"heldout_docs_target": 0}):
        try:
            CorpusBuildSpec(dataset_revision="r", **bad).validate()
            assert False, f"validate should reject {bad}"
        except ValueError:
            pass


# ---- BuildRealization / corpus_hash ----------------------------------------------------
def test_corpus_hash_order_independent():
    assert corpus_hash(["a", "b", "c"]) == corpus_hash(["c", "a", "b"])
    assert corpus_hash(["a", "b"]) != corpus_hash(["a", "b", "c"])

def test_build_realization_roundtrip():
    r = BuildRealization(manifest8="deadbeef", spec_json="{}", indexed_docs=40000)
    d = r.to_dict()
    assert d["manifest8"] == "deadbeef" and d["indexed_docs"] == 40000


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        passed += 1
        print(f"  PASS  {fn.__name__}")
    print(f"\n{passed}/{len(fns)} corpus-spec tests passed")
