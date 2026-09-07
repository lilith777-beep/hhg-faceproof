"""
data.html — researcher EDA dossier. Smoke-renders the SAMPLE (schema-faithful placeholder
that a ./data_stats.json fetch overrides in production), proves it is honestly labeled a
validation-split SAMPLE (never "full validation"), and proves every rendered field is inert
(the dossier is built purely by DOM construction / textContent, so a hostile data_stats.json
can never inject markup or execute script).
"""
import pathlib

DATA = pathlib.Path(__file__).resolve().parent.parent.parent / "web" / "data.html"
DATA_URL = DATA.as_uri()

TAG = '<img src=x onerror="window.__xss=1">'


def _open(harness):
    page = harness.new(goto=False, stub_health=False)
    page.goto(DATA_URL)
    page.wait_for_selector("#leak .leakcell", timeout=5000)
    return page


def test_data_renders_sample_dossier(harness):
    page = _open(harness)
    # frozen manifest + sealed stamp + scope wording are all on the page
    body = page.inner_text("body")
    assert "73ca3e90" in body                                   # manifest hash surfaced
    assert "SAMPLE" in page.inner_text("#banner").upper()       # honestly flagged as sample
    assert "sample" in (page.get_attribute("#banner", "class") or "")
    # leakage proof stamp: three cells, all exactly "0"
    zeros = [c.text_content().strip() for c in page.query_selector_all("#leak .leakcell .big")]
    assert zeros == ["0", "0", "0"], zeros
    # composition treemap actually drew tiles
    assert len(page.query_selector_all("#comp svg rect")) >= 8
    # per-language table has the detected languages
    assert len(page.query_selector_all("#langtable tbody tr")) >= 11
    # both eval figures rendered
    assert page.query_selector("#risk svg") is not None
    assert page.query_selector("#frontier svg") is not None
    assert page.errors == []


def test_data_is_labeled_validation_sample_never_full(harness):
    page = _open(harness)
    body = page.inner_text("body").lower()
    # NEVER the forbidden claim
    assert "full validation" not in body
    # ALWAYS the honest scope, with the exact traceable magnitudes
    assert ("validation-split sample" in body) or ("sample validation split" in body)
    assert "84k" in body
    assert "876k indexed" in body
    assert "10.1k held out" in body
    assert page.errors == []


def test_data_hostile_fields_stay_inert(harness):
    page = _open(harness)
    hostile = {
        "meta": {"generated": "2026-01-01", "git_sha": TAG},
        "manifest": TAG,
        "scope": {"source_languages": TAG, "corpus_hash": TAG, "public_wording": TAG},
        "realization": {
            "indexed_docs": 100, "heldout_docs": 10, "indexed_queries": 50,
            "heldout_queries": 5, "excluded_queries": 1, "n_families": 7,
            "indexed_families": 6, "heldout_families": 1, "calibration_queries": 3,
            "dev_queries": 2, "sealed_queries": 2,
            "in_corpus_partition": {"sealed": 1}, "absent_evidence_partition": {"sealed": 1},
            "excluded_reasons": {TAG: 1},
            "leakage": {"family_intersection": 0, "stable_doc_id_intersection": 0,
                        "exact_hash_intersection": 0, "indexed_exact_hashes": 100,
                        "heldout_exact_hashes": 10}},
        "composition": {"n_canonical_shards": 200, "english_canonical": 100,
                        "canonical_by_detected_lang": {TAG: 123}, "collapse_note": {TAG: TAG}},
        "lengths": {"passage_char_overall": {"p50": 1, "p90": 2, "p99": 3, "p100": 4},
                    "max_passage_chars": TAG, "passages_over_2000_chars": TAG},
        "queries": {"per_source_lang": {TAG: 5}},
        "qrels": {"coverage_by_lang": {TAG: {"total": 5, "with_positive": 4}}},
        "qdrant": {"chunk_tok": {"p50": 1, "p90": 2, "p99": 3, "p100": 4}},
        "eval": {"latency_ms": {"p50": 1, "p70": 2, "p90": 3, "p100": 4, "n": 5, "source": TAG},
                 "per_language": {TAG: {"recall10": TAG, "mrr10": TAG}},
                 "risk_coverage": [{"coverage": 1, "risk": 0.1}],
                 "accuracy_latency": [{"name": TAG, "latency_ms": 50, "accuracy": 0.8, "budget": True}]},
    }
    page.evaluate("(d) => renderDossier(d)", hostile)
    page.wait_for_timeout(150)
    # 1. nothing executed
    assert page.evaluate("() => window.__xss") is None
    # 2. the markup never became elements / handlers, anywhere on the page
    assert page.query_selector("img") is None
    assert page.query_selector("[onerror]") is None
    assert page.query_selector("script[src]") is None
    # 3. the payload survived as inert TEXT (proves it was escaped, not dropped)
    assert "<img" in page.inner_text("body")
    assert page.errors == []
