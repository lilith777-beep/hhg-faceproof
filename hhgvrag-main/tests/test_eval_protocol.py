"""
test_eval_protocol.py — the eval/ protocol scripts (P0.4 deliverables 5, 6, 8):
audit_data (reservoir sample + realized-corpus audit), sealed_test (mechanically-enforced
sealed command), and dry_run_build (manifest + counts, no build).

All sealed-test tests write into a TEMP base — no real eval/sealed_runs/ run is ever created,
so the command stays UNEXECUTED per the P0 exit gate.
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))

import corpus_spec as cspec        # noqa: E402
import index_build as ib           # noqa: E402
import audit_data as ad            # noqa: E402
import sealed_test as st           # noqa: E402
import dry_run_build as drb        # noqa: E402


# ======================================================================================
# audit_data
# ======================================================================================
def test_reservoir_sample_deterministic_and_sized():
    stream = list(range(1000))
    a = ad.reservoir_sample(stream, 20, seed=7)
    b = ad.reservoir_sample(stream, 20, seed=7)
    assert a == b, "same seed -> same sample"
    assert len(a) == 20 and len(set(a)) == 20
    assert ad.reservoir_sample(stream, 20, seed=8) != a, "different seed -> different sample"

def test_reservoir_sample_smaller_than_k():
    assert sorted(ad.reservoir_sample([1, 2, 3], 10, seed=0)) == [1, 2, 3]

def test_audit_records_basic_stats():
    recs = [{"text": "the quick brown fox jumps over the lazy dog every morning", "lang": "en",
             "variant": "en", "is_selected": 0} for _ in range(10)]
    rep = ad.audit_records(recs)
    assert rep["n"] == 10
    assert rep["language_retention"]["en"] == 1.0
    assert rep["length_words"]["min"] >= 10

def test_audit_flags_boilerplate_and_headings():
    recs = ([{"text": "Click here to subscribe to our newsletter and cookie policy terms",
              "lang": "en", "variant": "en", "is_selected": 0}] * 5
            + [{"text": "What Is A Refund", "lang": "en", "variant": "en", "is_selected": 0}] * 5)
    rep = ad.audit_records(recs)
    joined = " ".join(rep["warnings"])
    assert "boilerplate" in joined and "heading" in joined

def test_audit_translation_script_proxy_flags_romanized():
    # 'tr:hi' passages with NO Devanagari -> low script-match -> gate warning (amendment 4)
    recs = [{"text": "yeh romanized hindi without devanagari script at all here now",
             "lang": "hi", "variant": "tr:hi", "is_selected": 0} for _ in range(10)]
    rep = ad.audit_records(recs)
    assert rep["translation_quality"]["hi"]["script_match_rate"] == 0.0
    assert any("down-rank / gate" in w for w in rep["warnings"])

def test_audit_translation_script_proxy_passes_real_devanagari():
    recs = [{"text": "यह वास्तविक देवनागरी लिपि में एक हिंदी अनुच्छेद है जो सही है",
             "lang": "hi", "variant": "tr:hi", "is_selected": 0} for _ in range(10)]
    rep = ad.audit_records(recs)
    assert rep["translation_quality"]["hi"]["script_match_rate"] == 1.0

def test_audit_family_structure_and_adapter():
    rows = ib.synthetic_msmarco_rows(n_rows=200, seed=3)
    occ = ib.iter_passage_occurrences(rows)
    fam = ib.build_family_index(occ, cspec.CorpusBuildSpec(dataset_revision="a"))
    fs = ad.audit_family_structure(fam)
    assert fs["families"] > 0 and fs["unique_canonical"] >= fs["families"]
    recs = ad.records_from_occurrences(occ)
    assert len(recs) == len(occ) and "text" in recs[0]


# ======================================================================================
# sealed_test — mechanically-enforced guards
# ======================================================================================
def _dummy_eval(run_dir):
    return {"answer_coverage": 0.0, "note": "dummy eval (unit test only)"}

def test_worktree_clean_predicate():
    assert st.worktree_is_clean("") and st.worktree_is_clean("   \n")
    assert not st.worktree_is_clean(" M src/foo.py")

def test_preflight_requires_manifest():
    ok, reason = st.preflight("", "/tmp/x", "")
    assert not ok and "manifest" in reason

def test_preflight_rejects_bad_manifest():
    ok, reason = st.preflight("NOTAHEX!", "/tmp/x", "")
    assert not ok and "8-hex" in reason

def test_preflight_refuses_dirty_worktree():
    ok, reason = st.preflight("deadbeef", "/tmp/x", " M src/config.py")
    assert not ok and "dirty" in reason

def test_preflight_ok_when_clean_and_no_prior():
    base = tempfile.mkdtemp()
    try:
        ok, reason = st.preflight("deadbeef", base, "")
        assert ok and reason == "ok"
    finally:
        shutil.rmtree(base, ignore_errors=True)

def test_run_sealed_writes_immutable_artifacts():
    base = tempfile.mkdtemp()
    try:
        out = st.run_sealed("deadbeef", _dummy_eval, base=base, porcelain="",
                            collection_hashes={"msmarco_xi_40k_deadbeef__passage": "abc"},
                            model_revisions={"bge-m3": "rev1"}, seeds={"allocation": 42},
                            timestamp="20260817T120000Z")
        rd = out["run_dir"]
        assert os.path.exists(os.path.join(rd, "metadata.json"))
        assert os.path.exists(os.path.join(rd, "results.json"))
        assert os.path.exists(os.path.join(rd, "COMPLETED"))
        meta = out["metadata"]
        for key in ("manifest8", "git_sha", "collection_hashes", "model_revisions", "seeds",
                    "cmdline", "hardware", "env", "timestamp_utc"):
            assert key in meta, f"metadata missing {key}"
        assert meta["collection_hashes"]["msmarco_xi_40k_deadbeef__passage"] == "abc"
    finally:
        shutil.rmtree(base, ignore_errors=True)

def test_run_sealed_refuses_rerun_without_invalidation():
    base = tempfile.mkdtemp()
    try:
        st.run_sealed("deadbeef", _dummy_eval, base=base, porcelain="",
                      timestamp="20260817T120000Z")
        # second run for the SAME manifest must be refused
        try:
            st.run_sealed("deadbeef", _dummy_eval, base=base, porcelain="",
                          timestamp="20260817T130000Z")
            assert False, "re-run must be refused"
        except st.SealedTestRefused as e:
            assert "already has a COMPLETED" in str(e)
    finally:
        shutil.rmtree(base, ignore_errors=True)

def test_run_sealed_allows_rerun_with_invalidation():
    base = tempfile.mkdtemp()
    try:
        st.run_sealed("deadbeef", _dummy_eval, base=base, porcelain="",
                      timestamp="20260817T120000Z")
        # a human drops an invalidation record naming the reason
        with open(st.invalidation_path(base, "deadbeef"), "w") as f:
            f.write("re-run: corrected a metric bug, prior comparison invalidated")
        out = st.run_sealed("deadbeef", _dummy_eval, base=base, porcelain="",
                            timestamp="20260817T140000Z")
        assert os.path.exists(os.path.join(out["run_dir"], "COMPLETED"))
        assert len(st.completed_runs(base, "deadbeef")) == 2
    finally:
        shutil.rmtree(base, ignore_errors=True)

def test_run_sealed_refuses_dirty_and_writes_nothing():
    base = tempfile.mkdtemp()
    try:
        try:
            st.run_sealed("deadbeef", _dummy_eval, base=base, porcelain=" M src/harness.py")
            assert False, "dirty worktree must be refused"
        except st.SealedTestRefused:
            pass
        assert os.listdir(base) == [], "a refused run must write NOTHING"
    finally:
        shutil.rmtree(base, ignore_errors=True)

def test_sealed_run_never_writes_config():
    """The sealed command must never mutate config/thresholds, and must write only under base."""
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "src", "config.py")
    before = open(cfg_path, "rb").read()
    base = tempfile.mkdtemp()
    try:
        out = st.run_sealed("deadbeef", _dummy_eval, base=base, porcelain="",
                            timestamp="20260817T120000Z")
        assert out["run_dir"].startswith(base), "all output must be under the sealed base"
        assert open(cfg_path, "rb").read() == before, "config.py must be untouched"
    finally:
        shutil.rmtree(base, ignore_errors=True)

def test_env_snapshot_redacts_secrets():
    os.environ["HHGVRAG_TEST_SARVAM_API_KEY"] = "super-secret-value"
    try:
        snap = st.env_snapshot()
        flat = str(snap)
        assert "super-secret-value" not in flat, "secret VALUE must never appear"
        assert "HHGVRAG_TEST_SARVAM_API_KEY" not in flat, "secret NAME must not appear either"
        assert snap["secret_like_vars_present"] >= 1
    finally:
        del os.environ["HHGVRAG_TEST_SARVAM_API_KEY"]

def test_completed_runs_and_invalidation_helpers():
    base = tempfile.mkdtemp()
    try:
        assert st.completed_runs(base, "deadbeef") == []
        assert not st.invalidation_exists(base, "deadbeef")
        st.run_sealed("deadbeef", _dummy_eval, base=base, porcelain="",
                      timestamp="20260817T120000Z")
        assert len(st.completed_runs(base, "deadbeef")) == 1
    finally:
        shutil.rmtree(base, ignore_errors=True)


# ======================================================================================
# dry_run_build
# ======================================================================================
def test_dry_run_report_structure():
    spec = cspec.CorpusBuildSpec(dataset_revision="dr", indexed_docs_target=600,
                                 heldout_docs_target=200, size_tolerance=0.3,
                                 min_heldout_queries=5, min_partition_queries=1)
    rows = ib.synthetic_msmarco_rows(n_rows=400, seed=1)
    rep = drb.build_report(rows, spec)
    assert rep["manifest8"] == spec.manifest8()
    assert rep["collections_would_build"][0] == spec.collection_name("passage")
    assert rep["integrity"]["family_intersection"] == 0
    assert rep["realization"]["indexed_docs"] > 0
    assert "in_corpus_partition" in rep and "absent_evidence_partition" in rep

def test_dry_run_propagates_unrealizable():
    spec = cspec.CorpusBuildSpec(dataset_revision="dr", indexed_docs_target=100,
                                 heldout_docs_target=9_000_000, size_tolerance=0.01,
                                 min_heldout_queries=5)
    rows = ib.synthetic_msmarco_rows(n_rows=200, seed=1)
    try:
        drb.build_report(rows, spec)
        assert False, "should raise BuildConstraintError"
    except ib.BuildConstraintError:
        pass

def test_parse_indexed_arg():
    assert drb.parse_indexed("all") == 0 and drb.parse_indexed("ALL ") == 0
    assert drb.parse_indexed("40000") == 40000
    try:
        drb.parse_indexed("many")
        assert False, "non-numeric non-'all' must raise"
    except ValueError:
        pass

def test_build_report_multi_whole_corpus():
    """Multi-shard report in ALL mode: unified family graph, projection block, group counts."""
    shards = [("hi", ib.synthetic_msmarco_rows(n_rows=150, seed=21)),
              ("ta", ib.synthetic_msmarco_rows(n_rows=150, seed=22))]
    spec = cspec.CorpusBuildSpec(dataset_revision="m", indexed_docs_target=0,
                                 heldout_docs_target=120, size_tolerance=0.3,
                                 min_heldout_queries=2, min_partition_queries=1)
    rep = drb.build_report_multi(shards, spec)
    r = rep["realization"]
    assert r["indexed_docs"] == r["unique_canonical"] - r["heldout_docs"]
    assert rep["integrity"]["family_intersection"] == 0
    assert rep["projection"]["leaf_chunks"] > 0
    assert rep["projection"]["raptor_summaries"] > 0
    assert rep["query_groups"]["indexed"] > 0
    # both shards contributed rows (prefixed row ids -> source_rows spans shards)
    assert r["source_rows"] == 300

def test_build_projection_math():
    pj = drb.build_projection(100_000)
    assert pj["leaf_chunks"] == 115_000
    assert pj["raptor_summaries"] == round(115_000 / 4.4)
    assert pj["total_points"] == pj["leaf_chunks"] + pj["raptor_summaries"]
    assert pj["projected_build_hours_1strategy_plus_raptor"] > pj["embed_hours"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        passed += 1
        print(f"  PASS  {fn.__name__}")
    print(f"\n{passed}/{len(fns)} eval-protocol tests passed")
