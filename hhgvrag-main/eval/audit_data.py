"""
audit_data.py — raw seeded reservoir sample + realized-corpus audit (P0.4 / deliverable 5).

This is a REPORT-ONLY tool (it never mutates the corpus, thresholds, or config — amendment 4:
transforms are audit-TRIGGERED and calibrated elsewhere). It answers the questions the freeze
review needs before authorizing transforms:
  * how big is the corpus, how duplicated, how large do near-dup families get
  * language / script retention (amendment 4: published by language)
  * passage length distribution (too-short / too-long tails)
  * boilerplate / heading / self-question prevalence (P1.1 cleaning trigger)
  * translation CPU-proxy quality by language (script-match rate -> down-rank/gate signal)
  * a seeded reservoir sample of raw passages for human inspection

Runnable two ways:
    python eval/audit_data.py --synthetic 4000                 # local, no dataset
    python eval/audit_data.py --qdrant-url http://localhost:6333 --collection msmarco_xi_..__passage
Every warning names a SUGGESTED action; applying it is a separate, calibrated decision.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import corpus_spec as cspec       # noqa: E402
import index_build as ib          # noqa: E402
from generation import ExtractiveGenerator   # noqa: E402  (reuse _is_question, don't duplicate)

_is_question = ExtractiveGenerator._is_question
_BOILERPLATE = ("click here", "cookie policy", "terms of service", "privacy policy",
                "subscribe to our", "sign up for our", "lorem ipsum")


# ---- seeded reservoir sample (Algorithm R) ---------------------------------------------
def reservoir_sample(items, k: int, seed: int = 0) -> list:
    """Uniform k-sample over a stream in one pass, deterministic under `seed`. Works when the
    corpus is far too large to hold twice; the sample is for human inspection, so it is drawn
    from the RAW stream before any selection bias."""
    import random
    rng = random.Random(seed)
    out: list = []
    for i, x in enumerate(items):
        if i < k:
            out.append(x)
        else:
            j = rng.randint(0, i)
            if j < k:
                out[j] = x
    return out


# ---- text-level audit ------------------------------------------------------------------
def _script_matches(text: str, lang: str) -> bool:
    """True if the text actually contains characters of its claimed script (translation CPU
    proxy: a romanized/garbled Hindi 'translation' with no Devanagari fails this)."""
    if lang == "en":
        return True
    for l, rx in ib._SCRIPT_PATTERNS:
        if l == lang:
            return bool(rx.search(text))
    return True   # unknown script -> don't penalize


def _looks_like_heading(text: str) -> bool:
    stripped = text.strip()
    if _is_question(stripped):
        return True
    words = stripped.split()
    # short, no terminal punctuation -> title/heading fragment
    return len(words) <= 6 and stripped[-1:] not in ".!?।॥۔"


def audit_records(records, label: str = "corpus", warn: dict = None) -> dict:
    """Core audit over a list of {text, lang, variant, is_selected} dicts. Pure + testable."""
    warn = warn or {}
    th_short = warn.get("min_words", 5)
    th_boiler = warn.get("boilerplate_rate", 0.02)
    th_heading = warn.get("heading_rate", 0.15)
    th_trans = warn.get("translation_script_match", 0.85)

    n = len(records)
    if n == 0:
        return {"label": label, "n": 0, "warnings": ["empty corpus"]}

    word_counts, langs, scripts = [], {}, {}
    boiler = heading = short = 0
    trans_total = trans_ok = 0
    trans_by_lang: dict = {}
    for r in records:
        text = r["text"]
        lang = r.get("lang", "en")
        wc = len(text.split())
        word_counts.append(wc)
        langs[lang] = langs.get(lang, 0) + 1
        low = text.lower()
        if any(bp in low for bp in _BOILERPLATE):
            boiler += 1
        if _looks_like_heading(text):
            heading += 1
        if wc < th_short:
            short += 1
        if str(r.get("variant", "")).startswith("tr:"):
            trans_total += 1
            ok = _script_matches(text, lang)
            trans_ok += 1 if ok else 0
            d = trans_by_lang.setdefault(lang, [0, 0])
            d[0] += 1
            d[1] += 1 if ok else 0

    word_counts.sort()
    def pct(p):
        return word_counts[min(len(word_counts) - 1, int(p * (len(word_counts) - 1)))]

    warnings = []
    if boiler / n > th_boiler:
        warnings.append(f"boilerplate rate {boiler/n:.1%} > {th_boiler:.0%} -> ACTION: enable "
                        f"heading/boilerplate cleaning (P1.1)")
    if heading / n > th_heading:
        warnings.append(f"heading/self-question rate {heading/n:.1%} > {th_heading:.0%} -> "
                        f"ACTION: strip heading/query-echo passages")
    trans_report = {}
    for lang, (tot, ok) in sorted(trans_by_lang.items()):
        rate = ok / tot if tot else 1.0
        trans_report[lang] = {"n": tot, "script_match_rate": round(rate, 3)}
        if rate < th_trans:
            warnings.append(f"translation script-match for '{lang}' {rate:.1%} < {th_trans:.0%} "
                            f"-> ACTION: down-rank / gate '{lang}' translations (amendment 4; "
                            f"calibrate the threshold, never hard-drop unconditionally)")

    return {
        "label": label,
        "n": n,
        "language_retention": {k: round(v / n, 4) for k, v in sorted(langs.items())},
        "length_words": {"min": word_counts[0], "median": pct(0.5), "p90": pct(0.9),
                         "max": word_counts[-1], "mean": round(statistics.mean(word_counts), 1)},
        "short_passages": {"n": short, "rate": round(short / n, 4)},
        "boilerplate": {"n": boiler, "rate": round(boiler / n, 4)},
        "heading_like": {"n": heading, "rate": round(heading / n, 4)},
        "translation_quality": trans_report,
        "warnings": warnings,
    }


def audit_family_structure(family_index: ib.FamilyIndex) -> dict:
    sizes = sorted(family_index.size.values(), reverse=True)
    multi = [s for s in sizes if s > 1]
    n_docs = sum(sizes)
    return {
        "unique_canonical": len(family_index.family_of),
        "families": len(family_index.members),
        "exact_dup_groups": sum(1 for v in family_index.exact_groups.values() if len(v) > 1),
        "family_size_max": sizes[0] if sizes else 0,
        "multi_member_families": len(multi),
        "near_dup_doc_rate": round(sum(multi) / n_docs, 4) if n_docs else 0.0,
        "warnings": ([f"largest family has {sizes[0]} members -> review SimHash threshold "
                      f"(possible over-merge / chaining)"] if sizes and sizes[0] > 20 else []),
    }


# ---- adapters --------------------------------------------------------------------------
def records_from_occurrences(occ) -> list:
    return [{"text": o.text, "lang": o.lang, "variant": o.variant, "is_selected": o.is_selected}
            for o in occ]


def records_from_payloads(payloads) -> list:
    out = []
    for pl in payloads:
        if pl.get("is_summary"):
            continue                      # audit LEAF evidence, never summaries
        out.append({"text": pl.get("text", ""), "lang": pl.get("language", "en"),
                    "variant": "tr:" + pl.get("language", "") if pl.get("language") not in
                    ("en", None) else "en", "is_selected": 0})
    return out


def audit_collection(client, collection: str, sample_k: int = 20, seed: int = 0) -> dict:
    payloads, offset = [], None
    while True:
        pts, offset = client.scroll(collection, limit=1024, offset=offset,
                                    with_payload=True, with_vectors=False)
        payloads.extend((p.payload or {}) for p in pts)
        if offset is None:
            break
    records = records_from_payloads(payloads)
    rep = audit_records(records, label=collection)
    rep["reservoir_sample"] = [r["text"][:160] for r in reservoir_sample(records, sample_k, seed)]
    return rep


# ---- CLI -------------------------------------------------------------------------------
def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Realized-corpus audit + reservoir sample (report-only)")
    ap.add_argument("--synthetic", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample", type=int, default=15)
    ap.add_argument("--qdrant-url", type=str, default=None)
    ap.add_argument("--collection", type=str, default=None)
    ap.add_argument("--indexed", type=int, default=800)
    ap.add_argument("--heldout", type=int, default=200)
    args = ap.parse_args()

    if args.qdrant_url:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=args.qdrant_url)
        report = audit_collection(client, args.collection, args.sample, args.seed)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    n = args.synthetic or 4000
    rows = ib.synthetic_msmarco_rows(n_rows=n, seed=args.seed)
    occ = ib.iter_passage_occurrences(rows)
    fam = ib.build_family_index(occ, cspec.CorpusBuildSpec(dataset_revision="audit"))
    records = records_from_occurrences(occ)
    report = {
        "raw_corpus": audit_records(records, label="raw"),
        "family_structure": audit_family_structure(fam),
        "reservoir_sample": [r["text"][:160] for r in reservoir_sample(records, args.sample, args.seed)],
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
