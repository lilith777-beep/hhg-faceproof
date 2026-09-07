"""
test_split.py — leakage-safe passage-family split (P0.4 + amendments 1,4,5).

Covers: family grouping (exact-dup / conservative SimHash near-dup / translated-variant link),
whole-family allocation with target bands + query-first held-out, the build-failure rule
(amendment 5, named constraint), the query partitioner (seeded, disjoint, sealed separated),
the leakage assertions (family + exact-hash intersection == 0, and that they FIRE on a leak),
qrel remapping + realizability, and end-to-end plan_split reproducibility.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import corpus_spec as cspec                              # noqa: E402
import index_build as ib                                 # noqa: E402
from index_build import (PassageOccurrence, build_family_index, build_qrels,  # noqa: E402
                         allocate_families, partition_queries, assert_split_integrity,
                         remap_qrels, assert_positives_resolve, plan_split, Allocation,
                         BuildConstraintError, iter_passage_occurrences, synthetic_msmarco_rows,
                         documents_from_allocation)


def _spec(**kw):
    base = dict(dataset_revision="test", indexed_docs_target=350, heldout_docs_target=150,
                size_tolerance=0.3, min_heldout_queries=5, min_partition_queries=1)
    base.update(kw)
    return cspec.CorpusBuildSpec(**base)


def _occ(row_id, idx, variant, lang, text, sel=0, qid="q", qtext="q text"):
    return PassageOccurrence(row_id=row_id, passage_index=idx, variant=variant, lang=lang,
                             text=text, stable_doc_id=cspec.stable_doc_id(text, variant),
                             is_selected=sel, query_id=qid, query_text=qtext)


# ---- family rule 1: exact-dup --------------------------------------------------------
def test_family_exact_dup_merges():
    spec = _spec()
    occ = [
        _occ("r0", 0, "en", "en", "The census is published every ten years by the bureau office."),
        _occ("r1", 0, "en", "en", "The census is published every  ten years by the bureau office."),  # ws-variant
        _occ("r2", 0, "en", "en", "A totally different passage about coastal geography and beaches."),
    ]
    fam = build_family_index(occ, spec)
    f0 = fam.family_of[occ[0].stable_doc_id]
    f1 = fam.family_of[occ[1].stable_doc_id]
    f2 = fam.family_of[occ[2].stable_doc_id]
    assert f0 == f1, "whitespace-variant exact dup must share a family"
    assert f2 != f0, "unrelated passage must be its own family"


def test_family_exact_dup_cross_variant_same_text():
    """Identical canonical TEXT in different variants -> same family (guarantees exact-hash
    intersection == 0 after whole-family allocation)."""
    spec = _spec()
    txt = "romanized identical string appearing as both english and translated variant here now"
    occ = [_occ("r0", 0, "en", "en", txt), _occ("r1", 3, "tr:hi", "hi", txt)]
    fam = build_family_index(occ, spec)
    assert fam.family_of[occ[0].stable_doc_id] == fam.family_of[occ[1].stable_doc_id]


# ---- family rule 2: conservative SimHash near-dup ------------------------------------
def test_family_near_dup_merges_within_threshold():
    spec = _spec()   # default simhash_hamming_max = 3
    long = ("the vector database supports dense and sparse hybrid retrieval with metadata "
            "filters over a large multilingual corpus of passages")
    near = long + " today"                    # Hamming 3 -> within threshold
    far = "goa is a coastal indian state famous for beaches and portuguese forts and cuisine"
    occ = [_occ("r0", 0, "en", "en", long), _occ("r1", 0, "en", "en", near),
           _occ("r2", 0, "en", "en", far)]
    fam = build_family_index(occ, spec)
    assert fam.family_of[occ[0].stable_doc_id] == fam.family_of[occ[1].stable_doc_id], \
        "near-dup within Hamming threshold must merge"
    assert fam.family_of[occ[2].stable_doc_id] != fam.family_of[occ[0].stable_doc_id], \
        "far passage must not merge"


def test_family_near_dup_conservative_excludes_far():
    """A tighter threshold does NOT merge a moderately-similar pair (conservative posture)."""
    spec = _spec(simhash_hamming_max=1)   # bands=4 still valid (1 < 4)
    long = ("the vector database supports dense and sparse hybrid retrieval with metadata "
            "filters over a large multilingual corpus of passages")
    near = long + " today"                # Hamming 3 -> now ABOVE threshold 1
    occ = [_occ("r0", 0, "en", "en", long), _occ("r1", 0, "en", "en", near)]
    fam = build_family_index(occ, spec)
    assert fam.family_of[occ[0].stable_doc_id] != fam.family_of[occ[1].stable_doc_id]


# ---- family rule 3: translated-variant link ------------------------------------------
def test_family_translation_link_same_row_index():
    """English_passages[i] and Translated_passages[i] (same row+index) join one family, even
    though their SimHash/exact-hash differ (different scripts)."""
    spec = _spec()
    en = _occ("r5", 2, "en", "en", "The monsoon season brings heavy rainfall to the region.")
    hi = _occ("r5", 2, "tr:hi", "hi", "मानसून का मौसम इस क्षेत्र में भारी वर्षा लाता है।")
    other = _occ("r6", 0, "en", "en", "Unrelated passage about semiconductor manufacturing.")
    fam = build_family_index([en, hi, other], spec)
    assert fam.family_of[en.stable_doc_id] == fam.family_of[hi.stable_doc_id], \
        "translated variant must join its source English family"
    assert fam.family_of[other.stable_doc_id] != fam.family_of[en.stable_doc_id]


def test_family_no_link_when_different_index():
    """Different passage index -> NOT translation-linked (no accidental cross-passage merge)."""
    spec = _spec()
    en = _occ("r7", 0, "en", "en", "Passage A about coastal erosion patterns over decades.")
    hi = _occ("r7", 1, "tr:hi", "hi", "यह अनुच्छेद बी पूरी तरह से अलग विषय पर है।")
    fam = build_family_index([en, hi], spec)
    assert fam.family_of[en.stable_doc_id] != fam.family_of[hi.stable_doc_id]


# ---- allocator -----------------------------------------------------------------------
def test_allocation_whole_families_and_bands():
    spec = _spec()
    rows = synthetic_msmarco_rows(n_rows=300, seed=3)
    occ = iter_passage_occurrences(rows)
    fam = build_family_index(occ, spec)
    qr = build_qrels(occ, fam)
    alloc = allocate_families(fam, qr, spec)
    # whole families: no family on both sides
    assert not (alloc.indexed_families & alloc.heldout_families)
    # within bands
    assert 350 * 0.7 <= alloc.indexed_docs <= 350 * 1.3
    assert 150 * 0.7 <= alloc.heldout_docs <= 150 * 1.3
    # every held-out query's COMPLETE positive family set is held out
    for qid in alloc.heldout_queries:
        assert qr.families[qid] <= alloc.heldout_families
    for qid in alloc.indexed_queries:
        assert qr.families[qid] <= alloc.indexed_families


def test_allocation_deterministic():
    spec = _spec()
    rows = synthetic_msmarco_rows(n_rows=250, seed=4)
    occ = iter_passage_occurrences(rows)
    fam = build_family_index(occ, spec)
    qr = build_qrels(occ, fam)
    a1 = allocate_families(fam, qr, spec)
    a2 = allocate_families(fam, qr, spec)
    assert a1.heldout_families == a2.heldout_families
    assert a1.indexed_queries == a2.indexed_queries


def test_allocation_seed_changes_split():
    rows = synthetic_msmarco_rows(n_rows=250, seed=4)
    occ = iter_passage_occurrences(rows)
    fam = build_family_index(occ, _spec())
    qr = build_qrels(occ, fam)
    a1 = allocate_families(fam, qr, _spec(allocation_seed=1))
    a2 = allocate_families(fam, qr, _spec(allocation_seed=2))
    assert a1.heldout_families != a2.heldout_families, "different seed -> different allocation"


def test_build_failure_named_constraint():
    rows = synthetic_msmarco_rows(n_rows=200, seed=5)
    occ = iter_passage_occurrences(rows)
    fam = build_family_index(occ, _spec())
    qr = build_qrels(occ, fam)
    # impossible held-out target
    try:
        allocate_families(fam, qr, _spec(heldout_docs_target=10_000_000, size_tolerance=0.01))
        assert False, "should have failed"
    except BuildConstraintError as e:
        assert "held-out" in str(e) and "binding constraint" in str(e)
    # impossible held-out query floor
    try:
        allocate_families(fam, qr, _spec(min_heldout_queries=10_000_000))
        assert False, "should have failed"
    except BuildConstraintError as e:
        assert "fully-held-out queries" in str(e)


def test_excluded_queries_counted_not_dropped():
    """A query whose positives split across partitions is EXCLUDED with a reason (never silently
    dropped) — amendment: no silent qrel exclusion."""
    spec = _spec()
    rows = synthetic_msmarco_rows(n_rows=300, seed=6, multi_positive_rate=0.6)
    occ = iter_passage_occurrences(rows)
    fam = build_family_index(occ, spec)
    qr = build_qrels(occ, fam)
    alloc = allocate_families(fam, qr, spec)
    total = len(alloc.indexed_queries) + len(alloc.heldout_queries) + len(alloc.excluded_queries)
    assert total == len(qr.families), "every query is classified (indexed|heldout|excluded)"
    for qid, reason in alloc.excluded_queries:
        assert reason, "every exclusion carries a reason"


# ---- partitioner ---------------------------------------------------------------------
def test_partition_disjoint_and_deterministic():
    spec = _spec()
    qids = [f"q{i}" for i in range(100)]
    p1 = partition_queries(qids, spec, salt="in_corpus")
    p2 = partition_queries(qids, spec, salt="in_corpus")
    assert p1.calibration == p2.calibration and p1.sealed == p2.sealed, "deterministic"
    allp = set(p1.calibration) | set(p1.dev) | set(p1.sealed)
    assert len(allp) == 100, "partition covers all ids"
    assert not (set(p1.calibration) & set(p1.sealed)), "cal and sealed disjoint"
    assert not (set(p1.dev) & set(p1.sealed)), "dev and sealed disjoint"


def test_partition_salt_separates_pools():
    spec = _spec()
    qids = [f"q{i}" for i in range(100)]
    a = partition_queries(qids, spec, salt="in_corpus")
    b = partition_queries(qids, spec, salt="absent_evidence")
    assert a.sealed != b.sealed, "different pools get different shuffles under the same seed"


def test_partition_counts_track_fractions():
    spec = _spec(calibration_frac=0.5, dev_frac=0.25, sealed_frac=0.25)
    p = partition_queries([f"q{i}" for i in range(80)], spec)
    assert p.counts()["calibration"] == 40
    assert p.counts()["dev"] == 20 and p.counts()["sealed"] == 20


# ---- leakage assertions --------------------------------------------------------------
def test_integrity_passes_on_clean_split():
    spec = _spec()
    rows = synthetic_msmarco_rows(n_rows=280, seed=7)
    occ = iter_passage_occurrences(rows)
    fam = build_family_index(occ, spec)
    qr = build_qrels(occ, fam)
    alloc = allocate_families(fam, qr, spec)
    report = assert_split_integrity(alloc, fam)
    assert report["family_intersection"] == 0
    assert report["exact_hash_intersection"] == 0
    assert report["stable_doc_id_intersection"] == 0


def test_integrity_fires_on_injected_leak():
    """A family placed on BOTH sides must trip the assertion — proof the gate is real."""
    spec = _spec()
    rows = synthetic_msmarco_rows(n_rows=120, seed=8)
    occ = iter_passage_occurrences(rows)
    fam = build_family_index(occ, spec)
    some_fam = next(iter(fam.members))
    leaky = Allocation(indexed_families={some_fam}, heldout_families={some_fam},
                       unused_families=set(), indexed_docs=0, heldout_docs=0,
                       indexed_queries=[], heldout_queries=[], excluded_queries=[],
                       algorithm="test")
    try:
        assert_split_integrity(leaky, fam)
        assert False, "leak must raise"
    except AssertionError as e:
        assert "intersection" in str(e)


# ---- qrel remap + realizability ------------------------------------------------------
def test_qrel_realizability_positives_resolve():
    spec = _spec()
    rows = synthetic_msmarco_rows(n_rows=300, seed=9)
    occ = iter_passage_occurrences(rows)
    fam = build_family_index(occ, spec)
    qr = build_qrels(occ, fam)
    alloc = allocate_families(fam, qr, spec)
    idx_sdids = alloc.indexed_stable_doc_ids(fam)
    remapped = remap_qrels(qr, alloc, fam, alloc.indexed_queries)
    rep = assert_positives_resolve(remapped, idx_sdids, qr)
    assert rep["unresolved"] == 0 and rep["positives_checked"] > 0


def test_qrel_realizability_fires_on_missing():
    spec = _spec()
    rows = synthetic_msmarco_rows(n_rows=300, seed=10)
    occ = iter_passage_occurrences(rows)
    fam = build_family_index(occ, spec)
    qr = build_qrels(occ, fam)
    alloc = allocate_families(fam, qr, spec)
    remapped = remap_qrels(qr, alloc, fam, alloc.indexed_queries)
    try:
        assert_positives_resolve(remapped, [], qr)   # empty corpus -> nothing resolves
        assert False, "must raise when positives don't resolve"
    except AssertionError as e:
        assert "do not resolve" in str(e)


# ---- plan_split orchestration --------------------------------------------------------
def test_plan_split_reproducible_and_named():
    spec = _spec()
    rows = synthetic_msmarco_rows(n_rows=320, seed=11, multi_positive_rate=0.3)
    p1 = plan_split(rows, spec)
    p2 = plan_split(rows, spec)
    assert p1.realization.corpus_hash == p2.realization.corpus_hash, "corpus reproducible"
    assert p1.incorpus_partition.sealed == p2.incorpus_partition.sealed
    assert p1.collection_names() == [spec.collection_name("passage"), spec.collection_name("raptor")]
    # realization accounting adds up
    r = p1.realization
    assert r.indexed_queries + r.heldout_queries + r.excluded_queries == len(p1.qrels.families)
    assert r.manifest8 == spec.manifest8()


def test_documents_from_allocation_only_indexed():
    spec = _spec()
    rows = synthetic_msmarco_rows(n_rows=250, seed=12)
    plan = plan_split(rows, spec)
    docs = documents_from_allocation(plan.allocation, plan.family_index, spec)
    idx_sdids = set(plan.allocation.indexed_stable_doc_ids(plan.family_index))
    held = set(plan.allocation.heldout_stable_doc_ids(plan.family_index))
    doc_sdids = {d.stable_doc_id for d in docs}
    assert doc_sdids == idx_sdids, "materialized docs == indexed partition"
    assert not (doc_sdids & held), "no held-out passage may appear as an indexed document"
    assert all(d.stable_doc_id for d in docs), "every doc carries its stable_doc_id"
    # deterministic doc_ids across two runs
    docs2 = documents_from_allocation(plan.allocation, plan.family_index, spec)
    assert [d.doc_id for d in docs] == [d.doc_id for d in docs2]


# ---- multi-shard (14-language posture) -----------------------------------------------
_EN_SHARED = ("The grand fortress overlooks the western harbor and hosts annual maritime "
              "festivals attended by thousands of coastal visitors every winter season.")
_EN_OTHER = ("Semiconductor fabrication requires extreme cleanroom precision and multi-stage "
             "lithography processes refined over decades of industrial engineering research.")
_TR_HI = "यह किला पश्चिमी बंदरगाह के ऊपर स्थित है और हर वर्ष समुद्री उत्सव आयोजित करता है।"
_TR_TA = "இந்த கோட்டை மேற்கு துறைமுகத்தை கண்காணிக்கிறது மற்றும் கடல்சார் விழாக்களை நடத்துகிறது."


def _shard_row(qid, en, tr, sel=1):
    return {"query_id": qid, "query": f"query text for {qid}", "Eng_Query": f"eng {qid}",
            "passages": {"English_passages": [en], "Translated_passages": [tr],
                         "is_selected": [sel]}}


def test_multi_shard_english_dedup_unifies_family_across_shards():
    """The SAME English passage appearing in the hi shard and the ta shard must collapse into
    ONE cross-shard family (exact-hash rule), and BOTH shards' translations must join it via
    their own (prefixed) row provenance."""
    spec = _spec()
    shards = [("hi", [_shard_row("q1", _EN_SHARED, _TR_HI)]),
              ("ta", [_shard_row("q1", _EN_SHARED, _TR_TA)])]
    occ = ib.occurrences_from_shards(shards, spec)
    # row_ids are disjoint across shards
    assert {o.row_id for o in occ} == {"hi:0", "ta:0"}
    # 4 occurrences (2 en + 2 tr) but the two en occurrences share ONE stable_doc_id
    assert len(occ) == 4
    sdids = {o.stable_doc_id for o in occ}
    assert len(sdids) == 3, "identical English text across shards -> one canonical doc"
    fam = ib.build_family_index(occ, spec)
    fams = {fam.family_of[s] for s in sdids}
    assert len(fams) == 1, ("English + hindi translation + tamil translation must form ONE "
                            "unified family spanning both shards")


def test_multi_shard_row_prefix_prevents_false_translation_link():
    """Shard-local row indices collide without prefixes: (row 0, idx 0) in the hi shard and
    (row 0, idx 0) in the ta shard would provenance-link a Tamil translation to a Hindi row's
    English passage. Demonstrate the corruption on unprefixed input, then prove
    occurrences_from_shards prevents it."""
    spec = _spec()
    row_hi = _shard_row("qh", _EN_SHARED, _TR_HI)
    row_ta = _shard_row("qt", _EN_OTHER, _TR_TA)

    # THE BUG (unprefixed concatenation): both rows get row_id "0" -> provenance collision
    occ_bug = (ib.iter_passage_occurrences([row_hi], cfg="hi")
               + ib.iter_passage_occurrences([row_ta], cfg="ta"))
    fam_bug = ib.build_family_index(occ_bug, spec)
    hi_tr = next(o for o in occ_bug if o.variant.startswith("tr:") and o.text == _TR_HI)
    ta_en = next(o for o in occ_bug if o.variant == "en" and o.text == _EN_OTHER)
    assert fam_bug.family_of[hi_tr.stable_doc_id] == fam_bug.family_of[ta_en.stable_doc_id], \
        "bug demo: unprefixed shards falsely link a hi translation to the ta English family"

    # THE FIX: disjoint '<cfg>:' prefixes -> no false link, two separate families
    occ = ib.occurrences_from_shards([("hi", [row_hi]), ("ta", [row_ta])], spec)
    fam = ib.build_family_index(occ, spec)
    hi_en = next(o for o in occ if o.variant == "en" and o.text == _EN_SHARED)
    hi_tr = next(o for o in occ if o.variant.startswith("tr:") and o.text == _TR_HI)
    ta_en = next(o for o in occ if o.variant == "en" and o.text == _EN_OTHER)
    ta_tr = next(o for o in occ if o.variant.startswith("tr:") and o.text == _TR_TA)
    assert fam.family_of[hi_tr.stable_doc_id] == fam.family_of[hi_en.stable_doc_id]
    assert fam.family_of[ta_tr.stable_doc_id] == fam.family_of[ta_en.stable_doc_id]
    assert fam.family_of[hi_en.stable_doc_id] != fam.family_of[ta_en.stable_doc_id], \
        "unrelated rows in different shards must stay in separate families"


def test_indexed_all_mode_whole_corpus_minus_holdout():
    """indexed_docs_target=0 sentinel: indexed = every family not held out; no unused families;
    exclusions only from true indexed/heldout straddles; integrity holds."""
    rows = synthetic_msmarco_rows(n_rows=300, seed=13, multi_positive_rate=0.4)
    spec = _spec(indexed_docs_target=0, heldout_docs_target=120, size_tolerance=0.3)
    plan = plan_split(rows, spec)
    alloc, fam, r = plan.allocation, plan.family_index, plan.realization
    assert alloc.unused_families == set(), "whole-corpus mode leaves no family unused"
    assert r.indexed_docs == fam.total_docs - r.heldout_docs, \
        "realized indexed must be exactly total - holdout"
    for _, reason in alloc.excluded_queries:
        assert "unused" not in reason, f"no 'unused' exclusions may exist in ALL mode: {reason}"
    assert plan.integrity["family_intersection"] == 0
    # the sentinel is part of the hashed manifest: ALL-mode and 40k-mode differ
    assert spec.manifest8() != _spec().manifest8()


def test_indexed_all_mode_maximizes_realizability():
    """Same corpus: ALL mode realizes at least as many in-corpus queries as a small fixed
    target (the motivation for the mode)."""
    rows = synthetic_msmarco_rows(n_rows=300, seed=14)
    fixed = plan_split(rows, _spec(indexed_docs_target=250, heldout_docs_target=120,
                                   size_tolerance=0.3))
    whole = plan_split(rows, _spec(indexed_docs_target=0, heldout_docs_target=120,
                                   size_tolerance=0.3))
    assert len(whole.allocation.indexed_queries) >= len(fixed.allocation.indexed_queries)
    assert len(whole.allocation.excluded_queries) <= len(fixed.allocation.excluded_queries)


def test_cross_shard_query_variants_share_partition():
    """The hi and ta variants of ONE source query (same query_id, same English positives) must
    land in the SAME cal/dev/sealed partition — never straddle tuning and sealed."""
    rng_words = __import__("random").Random(99)
    rows_hi, rows_ta = [], []
    for i in range(40):
        en = (f"Answer passage {i}: " + " ".join(rng_words.sample(ib._SYN_VOCAB, 8))
              + f". Canonical reference {i} for cross shard partition testing.")
        rows_hi.append(_shard_row(f"q{i}", en, f"अनुच्छेद {i}: उत्तर पाठ {i} यहाँ है।"))
        rows_ta.append(_shard_row(f"q{i}", en, f"பத்தி {i}: பதில் உரை {i} இங்கே."))
    spec = _spec(indexed_docs_target=0, heldout_docs_target=30, size_tolerance=0.4,
                 min_heldout_queries=2)
    plan = ib.plan_split_multi([("hi", rows_hi), ("ta", rows_ta)], spec)
    group_of = plan.qrels.group_of
    for part in (plan.incorpus_partition, plan.absent_partition):
        where = {}
        for pname in ("calibration", "dev", "sealed"):
            for q in getattr(part, pname):
                g = group_of[q]
                assert where.setdefault(g, pname) == pname, \
                    f"group {g} straddles {where[g]} and {pname}"
    # sanity: at least one group realized with BOTH language variants
    counts = {}
    for q in plan.allocation.indexed_queries + plan.allocation.heldout_queries:
        counts[group_of[q]] = counts.get(group_of[q], 0) + 1
    assert any(v == 2 for v in counts.values()), "test needs multi-variant groups to be meaningful"


def test_cross_process_reproducibility():
    """P0 exit gate: stable IDs + partitions reproduce across two INDEPENDENT loader runs,
    even under different PYTHONHASHSEED (built-in hash() must not leak into any seed)."""
    import subprocess
    script = (
        "import sys; sys.path.insert(0, 'src')\n"
        "import index_build as ib, corpus_spec as cspec\n"
        "rows = ib.synthetic_msmarco_rows(n_rows=300, seed=2, multi_positive_rate=0.3)\n"
        "spec = cspec.CorpusBuildSpec(dataset_revision='x', indexed_docs_target=350,\n"
        "    heldout_docs_target=150, size_tolerance=0.3, min_heldout_queries=5, min_partition_queries=1)\n"
        "p = ib.plan_split(rows, spec)\n"
        "print(p.realization.corpus_hash, ':'.join(p.incorpus_partition.sealed[:5]),\n"
        "      ':'.join(p.absent_partition.sealed[:5]))\n")
    root = os.path.join(os.path.dirname(__file__), "..")
    outs = []
    for hashseed in ("0", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=hashseed)
        r = subprocess.run([sys.executable, "-c", script], cwd=root, env=env,
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        outs.append(r.stdout.strip())
    assert outs[0] == outs[1], f"non-reproducible across PYTHONHASHSEED:\n{outs[0]}\n{outs[1]}"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        passed += 1
        print(f"  PASS  {fn.__name__}")
    print(f"\n{passed}/{len(fns)} leakage-safe split tests passed")
