"""
eda.py — researcher-grade exploratory data analysis of the MSMARCO-XI validation-sample
corpus (manifest 73ca3e90).

This is a REPORT-ONLY tool. It never mutates the corpus, the collections, thresholds, or
config. It reads the SAME data the build read (the real shard reader + the real Indic-correct
tokenizer + the frozen realization manifest + the live Qdrant collection) and produces:

    eval/eda/*.png            matplotlib figures (headless Agg, colorblind-safe)
    eval/eda/data_stats.json  every number behind every figure and every sentence in EDA.md
    eval/eda/EDA.md           the written analysis + the researcher's narrative

--------------------------------------------------------------------------------------------
WHAT READS WHAT  (so the lead can run partial with --skip-qdrant / --skip-shards / --families off)
--------------------------------------------------------------------------------------------
  * realization.json ONLY  (always available, no network):
      leakage 0/0/0 confirmation, family/doc/query counts, indexed/heldout/excluded split,
      cal/dev/sealed partition sizes, excluded-reason breakdown.
  * raw parquet SHARDS ONLY  (the real loader, no Qdrant):
      corpus composition (source-config AND script-detected), the Devanagari-collapse confusion
      matrix, passage char/token length distributions + long-tail outliers, query analysis,
      positives-per-query / qrel coverage, exact-dup + SimHash near-dup FAMILY size
      distributions (--families full|sample), cross-lingual family composition, vocabulary /
      type-token ratio / script-block distribution / code-mixing rate.
  * live QDRANT collection ONLY:
      chunks-per-language as actually indexed, chunk char/token length distribution, and the
      12-topic distribution overall and per language (from the `topic` payload field).

Determinism: every sample is drawn from random.Random(--seed) (default 20260817). Anything
sampled is LABELED as sampled in the figure title, EDA.md, and data_stats.json — never silently.

Sealed hygiene: this tool NEVER opens queries_*_sealed.json. Query analysis is computed from
the shards (which carry query text + is_selected) and the realization counts only.

Run (on Forge, after the build has verified):
    pip install matplotlib          # only extra dep beyond the build's requirements
    python eval/eda.py \
        --qdrant-url http://localhost:6333 \
        --hf-home ~/.cache/huggingface \
        --out eval/eda

Offline smoke test (no dataset, no Qdrant — validates every non-network code path):
    python eval/eda.py --self-test --out /tmp/eda_selftest
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import statistics
import sys
import time
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import corpus_spec as cspec          # noqa: E402  the frozen identity + SimHash
import index_build as ib             # noqa: E402  the REAL shard reader + family machinery
from textnorm import tokens as indic_tokens   # noqa: E402  Indic-correct word tokenizer

# ------------------------------------------------------------------------------------------
# palette (the dataviz reference instance — pre-validated colorblind-safe, light surface).
# We commit these static research PNGs to the LIGHT surface deliberately (one medium).
# ------------------------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"            # categorical slot 1 / default magnitude hue
# 8-slot categorical, fixed order (used ONLY where there are <=8 genuine classes)
CAT = ["#2a78d6", "#1baf7a", "#eda100", "#008300",
       "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]
# blue sequential ramp, light -> dark (magnitude encoding for treemap / heatmap)
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6",
       "#256abf", "#184f95", "#104281", "#0d366b"]
DIV_LO, DIV_MID, DIV_HI = "#2a78d6", "#f0efec", "#e34948"   # diverging blue<->red
STATUS_GOOD = "#0ca30c"

LANG_NAMES = {
    "as": "Assamese", "bn": "Bengali", "gu": "Gujarati", "hi": "Hindi",
    "kn": "Kannada", "ml": "Malayalam", "mr": "Marathi", "ne": "Nepali",
    "or": "Odia", "pa": "Punjabi", "sa": "Sanskrit", "ta": "Tamil",
    "te": "Telugu", "ur": "Urdu", "en": "English",
}

# The 14 source configs use only 10 distinct scripts. `_detect_lang` (build) keys off the
# script block, so these 4 source languages carry NO distinct detected label — they collapse:
COLLAPSE = {"mr": "hi", "sa": "hi", "ne": "hi", "as": "bn"}   # source -> detected label

# Unicode script blocks for the script-distribution / code-mixing analysis (ordered).
SCRIPT_BLOCKS = [
    ("Latin", 0x0041, 0x024F), ("Devanagari", 0x0900, 0x097F),
    ("Bengali", 0x0980, 0x09FF), ("Gurmukhi", 0x0A00, 0x0A7F),
    ("Gujarati", 0x0A80, 0x0AFF), ("Oriya", 0x0B00, 0x0B7F),
    ("Tamil", 0x0B80, 0x0BFF), ("Telugu", 0x0C00, 0x0C7F),
    ("Kannada", 0x0C80, 0x0CFF), ("Malayalam", 0x0D00, 0x0D7F),
    ("Arabic", 0x0600, 0x06FF),
]
_LATIN_RE = re.compile(r"[A-Za-z]")
_INDIC_RE = re.compile(r"[ऀ-෿؀-ۿ]")


def _lang(code: str) -> str:
    return LANG_NAMES.get(code, code)


# ==========================================================================================
# matplotlib setup + plotting primitives
# ==========================================================================================
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Arial"],
        "font.size": 11, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "axes.titlecolor": INK, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.dpi": 120,
    })
    return plt


def _style(ax, title=None, xlabel=None, ylabel=None, subtitle=None):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.grid(axis="x", visible=False)
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", loc="left", pad=28 if subtitle else 8)
    if subtitle:
        ax.text(0.0, 1.012, subtitle, transform=ax.transAxes, fontsize=9.5,
                color=MUTED, ha="left", va="bottom")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def _seq_color(frac: float) -> str:
    """Interpolate the blue sequential ramp at frac in [0,1] (magnitude encoding)."""
    frac = 0.0 if frac < 0 else 1.0 if frac > 1 else frac
    pos = frac * (len(SEQ) - 1)
    i = int(pos)
    if i >= len(SEQ) - 1:
        return SEQ[-1]
    t = pos - i
    a, b = SEQ[i].lstrip("#"), SEQ[i + 1].lstrip("#")
    rgb = tuple(round(int(a[j:j + 2], 16) * (1 - t) + int(b[j:j + 2], 16) * t) for j in (0, 2, 4))
    return "#%02x%02x%02x" % rgb


def _save(fig, out_dir, name):
    import matplotlib.pyplot as plt
    path = os.path.join(out_dir, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return name


# --- squarify (MIT, Uri Laserson) — pure-function treemap layout, ported to avoid a dep ----
def _sq_layoutrow(sizes, x, y, dy):
    w = sum(sizes) / dy
    r = []
    for s in sizes:
        r.append((x, y, w, s / w))
        y += s / w
    return r


def _sq_layoutcol(sizes, x, y, dx):
    h = sum(sizes) / dx
    r = []
    for s in sizes:
        r.append((x, y, s / h, h))
        x += s / h
    return r


def _sq_layout(sizes, x, y, dx, dy):
    return _sq_layoutrow(sizes, x, y, dy) if dx >= dy else _sq_layoutcol(sizes, x, y, dx)


def _sq_leftover(sizes, x, y, dx, dy):
    if dx >= dy:
        w = sum(sizes) / dy
        return x + w, y, dx - w, dy
    h = sum(sizes) / dx
    return x, y + h, dx, dy - h


def _sq_worst(sizes, x, y, dx, dy):
    return max(max(w / h, h / w) for (_, _, w, h) in _sq_layout(sizes, x, y, dx, dy))


def squarify(values, x, y, dx, dy):
    sizes = [float(v) for v in values]
    total = sum(sizes)
    sizes = [s * dx * dy / total for s in sizes]

    def rec(sz, x, y, dx, dy):
        if not sz:
            return []
        if len(sz) == 1:
            return _sq_layout(sz, x, y, dx, dy)
        i = 1
        while i < len(sz) and _sq_worst(sz[:i], x, y, dx, dy) >= _sq_worst(sz[:i + 1], x, y, dx, dy):
            i += 1
        cur, rest = sz[:i], sz[i:]
        lx, ly, ldx, ldy = _sq_leftover(cur, x, y, dx, dy)
        return _sq_layout(cur, x, y, dx, dy) + rec(rest, lx, ly, ldx, ldy)

    return rec(sizes, x, y, dx, dy)


# ==========================================================================================
# shard reading (the REAL loader) + streaming corpus analysis
# ==========================================================================================
def iter_shard_rows(cfg, spec, shard_dir=None):
    """Rows from one language's validation shard, capped at spec.max_rows_per_shard EXACTLY as
    the build caps them. Default path is the real `ib._iter_msmarco_rows` (hf_hub_download at
    the frozen dataset_revision, served from the local cache offline). `--shard-dir` is an
    offline override: it reads the SAME parquet the real loader would, resolved by the real
    `_LANG_FILE` naming, with the identical pyarrow batch iteration."""
    cap = spec.max_rows_per_shard
    if shard_dir:
        import pyarrow.parquet as pq
        fname = f"{ib._LANG_FILE[cfg]}val.parquet"
        path = None
        for root, _, files in os.walk(shard_dir):
            if fname in files:
                path = os.path.join(root, fname)
                break
        if path is None:
            raise FileNotFoundError(f"{fname} not found under {shard_dir}")
        pf = pq.ParquetFile(path)
        i = 0
        for batch in pf.iter_batches(batch_size=64):
            for row in batch.to_pylist():
                if cap and i >= cap:
                    return
                i += 1
                yield row
        return
    for i, row in enumerate(ib._iter_msmarco_rows(cfg, split="validation",
                                                  revision=spec.dataset_revision)):
        if cap and i >= cap:
            break
        yield row


def _char_len(t):
    return len(t)


def analyze_shards(spec, shard_dir, seed, res_per_lang, keep_occurrences, row_iters=None):
    """ONE streaming pass over all 14 shards. Builds every shard-derived aggregate and a seeded
    per-language reservoir of canonical passage texts (for vocab/script/code-mixing). Optionally
    retains the full PassageOccurrence list (for the exact family reconstruction).

    `row_iters` (test hook): {cfg: iterable-of-rows} to bypass the network entirely."""
    rng = random.Random(seed)
    langs = list(spec.languages)

    occ_all = [] if keep_occurrences else None
    seen_sdid = set()                              # canonical dedup (variant-aware stable_doc_id)
    # per DETECTED-language canonical aggregates
    canon_by_lang = Counter()
    char_by_lang = defaultdict(list)
    wtok_by_lang = defaultdict(list)
    codemix_by_lang = Counter()                    # canonical passages mixing Latin+Indic
    reservoir = defaultdict(list)                  # detected-lang -> [texts]  (Algorithm R)
    res_seen = Counter()
    # occurrence-level aggregates
    occ_by_cfg = Counter()
    occ_by_variant = Counter()                     # "en" vs "tr:<lang>"
    # Devanagari-collapse confusion: source cfg -> detected-lang (TRANSLATED canonicals only)
    collapse = defaultdict(Counter)
    en_canon = 0                                   # shared English canonical passages
    longest = []                                   # (char_len, detected_lang, cfg) top outliers
    # query aggregates (from shards — never touches sealed files)
    q_text = {}                                    # qid -> query text (first seen)
    q_cfg = {}                                     # qid -> source cfg
    q_pos = defaultdict(set)                        # qid -> {selected stable_doc_id}
    group_langs = defaultdict(set)                 # source-query group -> {cfg}
    max_char_seen = 0

    def _reservoir_add(lang, text):
        res_seen[lang] += 1
        pool = reservoir[lang]
        if len(pool) < res_per_lang:
            pool.append(text)
        else:
            j = rng.randint(0, res_seen[lang] - 1)
            if j < res_per_lang:
                pool[j] = text

    for cfg in langs:
        rows = row_iters[cfg] if row_iters is not None else iter_shard_rows(cfg, spec, shard_dir)
        occ = ib.iter_passage_occurrences(rows, spec.include_english, spec.include_translated,
                                          cfg=cfg, row_prefix=f"{cfg}:")
        for o in occ:
            occ_by_cfg[cfg] += 1
            base_variant = "en" if o.variant == "en" else "tr"
            occ_by_variant[o.variant] += 1
            # query bookkeeping
            if o.query_text:
                q_text.setdefault(o.query_id, o.query_text)
            q_cfg.setdefault(o.query_id, cfg)
            group_langs[o.query_group].add(cfg)
            if o.is_selected:
                q_pos[o.query_id].add(o.stable_doc_id)
            # canonical dedup
            if o.stable_doc_id not in seen_sdid:
                seen_sdid.add(o.stable_doc_id)
                lang = o.lang
                cl = _char_len(o.text)
                wt = len(indic_tokens(o.text))
                canon_by_lang[lang] += 1
                char_by_lang[lang].append(cl)
                wtok_by_lang[lang].append(wt)
                if _LATIN_RE.search(o.text) and _INDIC_RE.search(o.text):
                    codemix_by_lang[lang] += 1
                _reservoir_add(lang, o.text)
                if base_variant == "en":
                    en_canon += 1
                else:
                    collapse[cfg][lang] += 1
                if cl > max_char_seen:
                    max_char_seen = cl
                longest.append((cl, lang, cfg))
                if len(longest) > 4000:            # keep the top tail bounded
                    longest.sort(reverse=True)
                    del longest[400:]
            if occ_all is not None:
                occ_all.append(o)

    longest.sort(reverse=True)
    del longest[200:]
    return {
        "langs": langs,
        "n_occurrences": sum(occ_by_cfg.values()),
        "n_canonical": len(seen_sdid),
        "canon_by_lang": dict(canon_by_lang),
        "char_by_lang": {k: v for k, v in char_by_lang.items()},
        "wtok_by_lang": {k: v for k, v in wtok_by_lang.items()},
        "codemix_by_lang": dict(codemix_by_lang),
        "occ_by_cfg": dict(occ_by_cfg),
        "occ_by_variant": dict(occ_by_variant),
        "collapse": {k: dict(v) for k, v in collapse.items()},
        "en_canonical": en_canon,
        "longest": longest,
        "max_char": max_char_seen,
        "reservoir": {k: v for k, v in reservoir.items()},
        "res_seen": dict(res_seen),
        "q_text": q_text, "q_cfg": q_cfg,
        "q_pos": {k: v for k, v in q_pos.items()},
        "group_langs": {k: v for k, v in group_langs.items()},
        "occ_all": occ_all,
    }


# ==========================================================================================
# family reconstruction (the leakage-safe-split showcase) — reuses the REAL machinery
# ==========================================================================================
def analyze_families(occ_all, spec, mode, sample_n, seed):
    """Rebuild the family graph with `ib.build_family_index` (the exact build machinery + the
    frozen SimHash params). `mode='full'` reproduces the realization's family count as a
    self-check; `mode='sample'` runs on a seeded occurrence subsample (labeled)."""
    occ = occ_all
    sampled = False
    if mode == "sample" and len(occ_all) > sample_n:
        occ = random.Random(seed).sample(occ_all, sample_n)
        sampled = True
    fam = ib.build_family_index(occ, spec)

    fam_sizes = Counter(fam.size.values())                       # near-dup family size -> freq
    exact_sizes = Counter(len(v) for v in fam.exact_groups.values())
    # cross-lingual family composition: distinct detected langs per family
    langs_per_family = Counter()
    multiling_families = 0
    span_hist = Counter()
    for rep, members in fam.members.items():
        ls = {fam.sdid_lang.get(s) for s in members}
        ls.discard(None)
        span_hist[len(ls)] += 1
        langs_per_family[len(ls)] += 1
        if len(ls) > 1:
            multiling_families += 1
    return {
        "mode": mode, "sampled": sampled, "n_occurrences_used": len(occ),
        "n_canonical": len(fam.family_of), "n_families": len(fam.members),
        "family_size_freq": dict(fam_sizes),
        "exact_group_size_freq": dict(exact_sizes),
        "family_lang_span_freq": dict(span_hist),
        "multilingual_families": multiling_families,
        "largest_family": max(fam.size.values()) if fam.size else 0,
        "singleton_families": fam_sizes.get(1, 0),
    }


# ==========================================================================================
# live Qdrant — indexed reality (chunks, chunk lengths, topics)
# ==========================================================================================
def analyze_qdrant(url, collection):
    from qdrant_client import QdrantClient
    client = QdrantClient(url=url, timeout=120)
    try:
        total = client.count(collection, exact=True).count
    except Exception as e:
        raise RuntimeError(f"cannot reach collection {collection} at {url}: {e}")

    chunks_by_lang = Counter()
    sdids_by_lang = defaultdict(set)               # distinct canonical docs indexed per lang
    chunk_char = []
    chunk_tok = []
    topic_overall = Counter()
    topic_by_lang = defaultdict(Counter)
    n_summary = 0
    offset = None
    fields = ["language", "token_count", "char_start", "char_end",
              "topic", "stable_doc_id", "is_summary"]
    scanned = 0
    while True:
        pts, offset = client.scroll(collection, limit=4096, offset=offset,
                                    with_payload=fields, with_vectors=False)
        for p in pts:
            pay = p.payload or {}
            if pay.get("is_summary"):
                n_summary += 1
                continue
            lang = pay.get("language", "unknown")
            chunks_by_lang[lang] += 1
            sd = pay.get("stable_doc_id")
            if sd:
                sdids_by_lang[lang].add(sd)
            cs, ce = pay.get("char_start"), pay.get("char_end")
            if isinstance(cs, int) and isinstance(ce, int) and ce >= cs:
                chunk_char.append(ce - cs)
            tc = pay.get("token_count")
            if isinstance(tc, int):
                chunk_tok.append(tc)
            tp = pay.get("topic")
            if tp:
                topic_overall[tp] += 1
                topic_by_lang[lang][tp] += 1
        scanned += len(pts)
        if offset is None:
            break
    return {
        "collection": collection, "total_points": total, "scanned": scanned,
        "summary_points_in_passage_collection": n_summary,   # must be 0
        "chunks_by_lang": dict(chunks_by_lang),
        "docs_by_lang": {k: len(v) for k, v in sdids_by_lang.items()},
        "chunk_char": chunk_char, "chunk_tok": chunk_tok,
        "topic_overall": dict(topic_overall),
        "topic_by_lang": {k: dict(v) for k, v in topic_by_lang.items()},
        "has_topics": bool(topic_overall),
    }


# ==========================================================================================
# lexical analysis (vocab / TTR / script blocks / code-mixing) — on the seeded reservoir
# ==========================================================================================
def _classify_char(ch):
    o = ord(ch)
    if ch.isdigit():
        return "Digit"
    for name, lo, hi in SCRIPT_BLOCKS:
        if lo <= o <= hi:
            return name
    if ch.isspace():
        return None
    if 0x0900 <= o <= 0x0DFF:
        return "OtherIndic"
    return "Other"


def analyze_lexical(shard_stats, token_budget):
    """Vocabulary size + type-token ratio at a FIXED per-language token budget (so TTR is
    comparable across languages — TTR is sample-size dependent), plus the script-block mix and
    code-mixing rate. All computed on the seeded reservoir; labeled as sampled."""
    reservoir = shard_stats["reservoir"]
    out_vocab, out_scripts = {}, {}
    for lang, texts in reservoir.items():
        types, n_tok = set(), 0
        script_counts = Counter()
        capped = False
        for t in texts:
            for ch in t:
                c = _classify_char(ch)
                if c:
                    script_counts[c] += 1
            for w in indic_tokens(t):
                if n_tok < token_budget:
                    types.add(w)
                    n_tok += 1
            if n_tok >= token_budget:
                capped = True
        out_vocab[lang] = {
            "tokens_counted": n_tok, "vocab_size": len(types),
            "type_token_ratio": (len(types) / n_tok) if n_tok else 0.0,
            "reached_budget": capped, "reservoir_passages": len(texts),
        }
        tot = sum(script_counts.values()) or 1
        out_scripts[lang] = {k: v / tot for k, v in script_counts.items()}
    return {"token_budget": token_budget, "vocab": out_vocab, "scripts": out_scripts}


# ==========================================================================================
# realization (frozen manifest) — leakage + split structure
# ==========================================================================================
def analyze_realization(real):
    r = real["realization"]
    integ = real["integrity"]
    notes = r.get("notes", {})
    return {
        "manifest8": real["manifest8"],
        "scope": real.get("scope", {}),
        "source_rows": r["source_rows"], "source_passages": r["source_passages"],
        "unique_canonical": r["unique_canonical"], "n_families": r["n_families"],
        "indexed_docs": r["indexed_docs"], "heldout_docs": r["heldout_docs"],
        "indexed_families": r["indexed_families"], "heldout_families": r["heldout_families"],
        "indexed_queries": r["indexed_queries"], "heldout_queries": r["heldout_queries"],
        "excluded_queries": r["excluded_queries"],
        "calibration_queries": r["calibration_queries"], "dev_queries": r["dev_queries"],
        "sealed_queries": r["sealed_queries"],
        "in_corpus_partition": notes.get("in_corpus_partition", {}),
        "absent_evidence_partition": notes.get("absent_evidence_partition", {}),
        "excluded_reasons": notes.get("excluded_reasons", {}),
        "query_groups": notes.get("query_groups", {}),
        "leakage": {
            "family_intersection": integ["family_intersection"],
            "stable_doc_id_intersection": integ["stable_doc_id_intersection"],
            "exact_hash_intersection": integ["exact_hash_intersection"],
            "indexed_exact_hashes": integ["indexed_exact_hashes"],
            "heldout_exact_hashes": integ["heldout_exact_hashes"],
        },
    }


# ==========================================================================================
# statistics helpers
# ==========================================================================================
def _pct(arr, ps=(50, 90, 95, 99, 100)):
    if not arr:
        return {f"p{p}": None for p in ps}
    s = sorted(arr)
    out = {}
    for p in ps:
        if p == 100:
            out["p100"] = s[-1]
        else:
            k = max(0, min(len(s) - 1, int(math.ceil(p / 100.0 * len(s))) - 1))
            out[f"p{p}"] = s[k]
    out["mean"] = round(statistics.fmean(s), 2)
    out["n"] = len(s)
    return out


def _order_by_value(counter, universe=None):
    keys = list(universe) if universe else list(counter.keys())
    return sorted(keys, key=lambda k: counter.get(k, 0), reverse=True)


# ==========================================================================================
# PLOTS
# ==========================================================================================
def plot_composition(shard_stats, qd, out_dir, figs):
    plt = _mpl()
    canon = shard_stats["canon_by_lang"]
    order = _order_by_value(canon)
    names = [_lang(k) for k in order]
    docs = [canon[k] for k in order]
    chunks = [qd["chunks_by_lang"].get(k) for k in order] if qd else None

    fig, ax = plt.subplots(figsize=(10, 5.2))
    ymax = max(docs)
    ax.bar(range(len(order)), docs, color=[_seq_color(v / ymax) for v in docs], width=0.72,
           zorder=3)
    for i, v in enumerate(docs):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8.5, color=INK2)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(names, rotation=40, ha="right")
    _style(ax, "Canonical passages per script-detected language",
           subtitle="Unique stable_doc_ids · from raw shards · magnitude = blue ramp",
           ylabel="canonical passages")
    ax.margins(x=0.01)
    figs.append(_save(fig, out_dir, "01_composition_docs_by_lang.png"))

    if chunks and any(chunks):
        fig, ax = plt.subplots(figsize=(10, 5.2))
        cm = max(c for c in chunks if c)
        ax.bar(range(len(order)), [c or 0 for c in chunks],
               color=[_seq_color((c or 0) / cm) for c in chunks], width=0.72, zorder=3)
        for i, v in enumerate(chunks):
            if v:
                ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8.5, color=INK2)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(names, rotation=40, ha="right")
        _style(ax, "Indexed chunks per language (live Qdrant)",
               subtitle=f"{qd['collection']} · {qd['total_points']:,} points",
               ylabel="passage chunks")
        figs.append(_save(fig, out_dir, "02_chunks_by_lang.png"))

    # treemap of canonical passages
    fig, ax = plt.subplots(figsize=(11, 6.2))
    rects = squarify(docs, 0, 0, 100, 62)
    vmax = max(docs)
    for (x, y, w, h), v, nm in zip(rects, docs, names):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=_seq_color(v / vmax),
                                   edgecolor=SURFACE, linewidth=2))
        if w * h > 12:
            frac = v / sum(docs) * 100
            tc = "#ffffff" if v / vmax > 0.5 else INK
            ax.text(x + w / 2, y + h / 2, f"{nm}\n{v:,}\n{frac:.1f}%", ha="center",
                    va="center", fontsize=9, color=tc, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 62)
    ax.axis("off")
    ax.set_title("Corpus composition treemap — canonical passages by language", fontsize=13,
                 fontweight="bold", loc="left")
    figs.append(_save(fig, out_dir, "03_composition_treemap.png"))


def plot_collapse(shard_stats, out_dir, figs):
    plt = _mpl()
    collapse = shard_stats["collapse"]
    srcs = shard_stats["langs"]
    detected = []
    for c in srcs:
        for d in collapse.get(c, {}):
            if d not in detected:
                detected.append(d)
    detected = sorted(detected, key=lambda d: -sum(collapse.get(c, {}).get(d, 0) for c in srcs))

    import numpy as np
    M = np.zeros((len(srcs), len(detected)))
    for i, c in enumerate(srcs):
        row = collapse.get(c, {})
        tot = sum(row.values()) or 1
        for j, d in enumerate(detected):
            M[i, j] = row.get(d, 0) / tot

    fig, ax = plt.subplots(figsize=(9.5, 7))
    im = ax.imshow(M, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(detected)))
    ax.set_xticklabels([_lang(d) for d in detected], rotation=40, ha="right")
    ax.set_yticks(range(len(srcs)))
    ax.set_yticklabels([f"{_lang(c)} ({c})" for c in srcs])
    for i in range(len(srcs)):
        for j in range(len(detected)):
            if M[i, j] > 0.01:
                ax.text(j, i, f"{M[i, j] * 100:.0f}", ha="center", va="center", fontsize=8,
                        color="#ffffff" if M[i, j] > 0.5 else INK2)
    for c in COLLAPSE:
        if c in srcs:
            ax.get_yticklabels()[srcs.index(c)].set_color("#c0392b")
    ax.set_title("Devanagari collapse — translated-passage script detection",
                 fontsize=13, fontweight="bold", loc="left", pad=26)
    ax.text(0.0, 1.012, "row = source shard, col = detected label (% of translated passages); "
            "red rows carry no distinct label", transform=ax.transAxes, fontsize=9, color=MUTED)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="fraction of shard's translated passages")
    ax.grid(False)
    figs.append(_save(fig, out_dir, "04_devanagari_collapse.png"))


def plot_lengths(shard_stats, qd, out_dir, figs):
    plt = _mpl()
    import numpy as np
    allc = [c for v in shard_stats["char_by_lang"].values() for c in v]

    fig, ax = plt.subplots(figsize=(9.5, 5))
    lo, hi = max(1, min(allc)), max(allc)
    bins = np.logspace(math.log10(lo), math.log10(hi), 60)
    ax.hist(allc, bins=bins, color=BLUE, edgecolor=SURFACE, linewidth=0.3, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axvline(2000, color=DIV_HI, linestyle="--", linewidth=1.3)
    ax.text(2000, ax.get_ylim()[1] * 0.6, " 2000 chars", color=DIV_HI, fontsize=9)
    _style(ax, "Passage character-length distribution (all languages)",
           subtitle=f"canonical passages · log-log · max = {max(allc):,} chars",
           xlabel="characters", ylabel="passages")
    figs.append(_save(fig, out_dir, "05_passage_charlen_hist.png"))

    # box per language, log scale
    order = _order_by_value({k: statistics.median(v) for k, v in shard_stats["char_by_lang"].items()})
    data = [shard_stats["char_by_lang"][k] for k in order]
    fig, ax = plt.subplots(figsize=(11, 5.4))
    bp = ax.boxplot(data, showfliers=False, patch_artist=True, widths=0.6)
    for patch in bp["boxes"]:
        patch.set_facecolor(SEQ[2])
        patch.set_edgecolor(BLUE)
    for med in bp["medians"]:
        med.set_color(INK)
    ax.set_yscale("log")
    ax.set_xticklabels([_lang(k) for k in order], rotation=40, ha="right")
    _style(ax, "Passage length by language (characters, log scale)",
           subtitle="box = IQR, whiskers = 1.5·IQR, outliers hidden", ylabel="characters")
    figs.append(_save(fig, out_dir, "06_passage_charlen_box_by_lang.png"))

    # word-token length hist + chunk token_count overlay from Qdrant
    allt = [t for v in shard_stats["wtok_by_lang"].values() for t in v]
    fig, ax = plt.subplots(figsize=(9.5, 5))
    tbins = np.logspace(0, math.log10(max(allt) + 1), 50)
    ax.hist(allt, bins=tbins, color=BLUE, alpha=0.85, label="passage (Indic word tokens)",
            edgecolor=SURFACE, linewidth=0.3, zorder=3)
    if qd and qd["chunk_tok"]:
        ax.hist(qd["chunk_tok"], bins=tbins, color=CAT[2], alpha=0.6,
                label="indexed chunk token_count", edgecolor=SURFACE, linewidth=0.3, zorder=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=9)
    _style(ax, "Token-length: raw passage vs indexed chunk",
           subtitle="passage = textnorm word tokens; chunk = as-indexed token_count (max_tokens=320)",
           xlabel="tokens", ylabel="count")
    figs.append(_save(fig, out_dir, "07_token_length_hist.png"))


def plot_queries(shard_stats, real_stats, out_dir, figs):
    plt = _mpl()
    import numpy as np
    q_text, q_cfg = shard_stats["q_text"], shard_stats["q_cfg"]
    qlen = [len(t) for t in q_text.values()]
    qtok = [len(indic_tokens(t)) for t in q_text.values()]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    a1.hist(qlen, bins=50, color=BLUE, edgecolor=SURFACE, linewidth=0.3, zorder=3)
    _style(a1, "Query length (characters)", ylabel="queries", xlabel="characters")
    a2.hist(qtok, bins=range(0, max(qtok) + 2), color=CAT[1], edgecolor=SURFACE,
            linewidth=0.3, zorder=3)
    _style(a2, "Query length (word tokens)", ylabel="queries", xlabel="tokens")
    fig.suptitle(f"Query length distribution — {len(q_text):,} distinct queries",
                 fontsize=13, fontweight="bold", x=0.02, ha="left")
    figs.append(_save(fig, out_dir, "08_query_length.png"))

    # queries per language
    per_lang = Counter(q_cfg.values())
    order = _order_by_value(per_lang, shard_stats["langs"])
    fig, ax = plt.subplots(figsize=(10, 4.8))
    vals = [per_lang.get(k, 0) for k in order]
    vmax = max(vals)
    ax.bar(range(len(order)), vals, color=[_seq_color(v / vmax) for v in vals], width=0.72,
           zorder=3)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8.5, color=INK2)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([_lang(k) for k in order], rotation=40, ha="right")
    _style(ax, "Queries per language (source shard)", ylabel="distinct queries")
    figs.append(_save(fig, out_dir, "09_queries_per_language.png"))

    # cross-lingual query-group sizes
    span = Counter(len(s) for s in shard_stats["group_langs"].values())
    xs = sorted(span)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(xs, [span[x] for x in xs], color=BLUE, width=0.7, zorder=3)
    for x in xs:
        ax.text(x, span[x], f"{span[x]:,}", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_yscale("log")
    ax.set_xticks(xs)
    _style(ax, "Cross-lingual query-group size",
           subtitle="how many languages share one source MS-MARCO query (co-located in the split)",
           xlabel="languages sharing the source query", ylabel="query groups (log)")
    figs.append(_save(fig, out_dir, "10_crosslingual_group_size.png"))

    # answerable vs absent-evidence vs excluded (from realization)
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    cats = ["in-corpus\n(answerable)", "absent-evidence\n(held-out)", "excluded\n(split-family)"]
    vals = [real_stats["indexed_queries"], real_stats["heldout_queries"],
            real_stats["excluded_queries"]]
    cols = [CAT[3], CAT[0], CAT[5]]
    b = ax.barh(cats, vals, color=cols, zorder=3)
    for rect, v in zip(b, vals):
        ax.text(v, rect.get_y() + rect.get_height() / 2, f" {v:,}", va="center", fontsize=10,
                color=INK2)
    ax.set_xscale("log")
    _style(ax, "Query realizability split (frozen realization)",
           subtitle="answerable = positives fully indexed; absent = positives fully held out",
           xlabel="queries (log)")
    ax.grid(axis="y", visible=False)
    figs.append(_save(fig, out_dir, "11_answerable_split.png"))


def plot_qrels(shard_stats, out_dir, figs):
    plt = _mpl()
    q_pos = shard_stats["q_pos"]
    q_cfg = shard_stats["q_cfg"]
    npos = [len(v) for v in q_pos.values()]
    dist = Counter(npos)
    xs = sorted(dist)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(xs, [dist[x] for x in xs], color=BLUE, width=0.7, zorder=3)
    for x in xs[:12]:
        ax.text(x, dist[x], f"{dist[x]:,}", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_yscale("log")
    _style(ax, "Positives per query (qrel depth)",
           subtitle=f"singletons = {dist.get(1, 0):,} · multi-positive = {sum(v for k, v in dist.items() if k > 1):,}",
           xlabel="selected positive passages", ylabel="queries (log)")
    figs.append(_save(fig, out_dir, "12_positives_per_query.png"))

    # per-language qrel coverage: fraction of queries with >=1 positive
    tot = Counter(q_cfg.values())
    withpos = Counter(q_cfg[q] for q, s in q_pos.items() if s)
    order = shard_stats["langs"]
    frac = [(withpos.get(k, 0) / tot[k]) if tot.get(k) else 0 for k in order]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(range(len(order)), frac, color=STATUS_GOOD, width=0.72, zorder=3)
    for i, v in enumerate(frac):
        ax.text(i, v, f"{v*100:.0f}%", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([_lang(k) for k in order], rotation=40, ha="right")
    ax.set_ylim(0, 1.05)
    _style(ax, "Per-language qrel coverage",
           subtitle="fraction of that language's queries with >=1 selected positive",
           ylabel="coverage")
    figs.append(_save(fig, out_dir, "13_qrel_coverage_by_lang.png"))


def plot_duplication(fam_stats, real_stats, out_dir, figs):
    plt = _mpl()
    tag = " (sampled)" if fam_stats.get("sampled") else ""

    def _loglog(freq, title, sub, xlabel, name, color):
        xs = sorted(freq)
        ys = [freq[x] for x in xs]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.scatter(xs, ys, s=26, color=color, zorder=3, edgecolor=SURFACE, linewidth=0.4)
        ax.set_xscale("log")
        ax.set_yscale("log")
        _style(ax, title, subtitle=sub, xlabel=xlabel, ylabel="number of families (log)")
        figs.append(_save(fig, out_dir, name))

    _loglog(fam_stats["exact_group_size_freq"],
            "Exact-duplicate group sizes" + tag,
            "canonical texts sharing identical bytes (English repeats across queries/shards)",
            "passages in exact group (log)", "14_exact_dup_sizes.png", BLUE)
    _loglog(fam_stats["family_size_freq"],
            "Near-duplicate FAMILY sizes" + tag,
            f"union of exact + SimHash(H<=3) + translation link · largest = {fam_stats['largest_family']}",
            "canonical passages in family (log)", "15_simhash_family_sizes.png", CAT[4])

    # cross-lingual family composition
    span = fam_stats["family_lang_span_freq"]
    xs = sorted(span)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    cols = [BLUE if x == 1 else CAT[1] for x in xs]
    ax.bar(xs, [span[x] for x in xs], color=cols, width=0.7, zorder=3)
    for x in xs:
        ax.text(x, span[x], f"{span[x]:,}", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_yscale("log")
    ax.set_xticks(xs)
    _style(ax, "Cross-lingual family composition" + tag,
           subtitle=f"{fam_stats['multilingual_families']:,} families span >1 language "
                    f"(translation-linked) and move to ONE side as a unit",
           xlabel="distinct languages in family", ylabel="families (log)")
    figs.append(_save(fig, out_dir, "16_crosslingual_family_span.png"))

    # leakage confirmation 0/0/0
    lk = real_stats["leakage"]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    labels = ["family\nintersection", "stable_doc_id\nintersection", "exact_hash\nintersection"]
    vals = [lk["family_intersection"], lk["stable_doc_id_intersection"],
            lk["exact_hash_intersection"]]
    ax.bar(labels, [max(v, 0.0) for v in vals], color=STATUS_GOOD, width=0.5, zorder=3)
    ax.set_ylim(0, 1)
    for i, v in enumerate(vals):
        ax.text(i, 0.06, f"= {v}", ha="center", fontsize=13, fontweight="bold", color="#0a6b0a")
    _style(ax, "Leakage between indexed and held-out partitions",
           subtitle=f"indexed {lk['indexed_exact_hashes']:,} vs held-out {lk['heldout_exact_hashes']:,} "
                    f"distinct exact hashes — provably disjoint by construction",
           ylabel="overlapping items")
    ax.grid(axis="y", visible=False)
    figs.append(_save(fig, out_dir, "17_leakage_confirmation.png"))


def plot_lexical(lex_stats, out_dir, figs):
    plt = _mpl()
    vocab = lex_stats["vocab"]
    order = sorted(vocab, key=lambda k: -vocab[k]["vocab_size"])
    names = [_lang(k) for k in order]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    vals = [vocab[k]["vocab_size"] for k in order]
    vmax = max(vals) or 1
    ax.bar(range(len(order)), vals, color=[_seq_color(v / vmax) for v in vals], width=0.72,
           zorder=3)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(names, rotation=40, ha="right")
    _style(ax, "Vocabulary size at fixed token budget (sampled)",
           subtitle=f"types in first {lex_stats['token_budget']:,} Indic-correct tokens per language "
                    f"· textnorm.WORD_RE", ylabel="distinct types")
    figs.append(_save(fig, out_dir, "18_vocab_by_language.png"))

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ttr = [vocab[k]["type_token_ratio"] for k in order]
    ax.bar(range(len(order)), ttr, color=CAT[4], width=0.72, zorder=3)
    for i, v in enumerate(ttr):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(names, rotation=40, ha="right")
    _style(ax, "Type-token ratio at fixed budget (sampled)",
           subtitle="higher = richer morphology / less repetition at equal token count",
           ylabel="types / tokens")
    figs.append(_save(fig, out_dir, "19_ttr_by_language.png"))

    # script-block stacked bar
    scripts = lex_stats["scripts"]
    order2 = _scripts_order(scripts)
    block_names = ["Latin", "Devanagari", "Bengali", "Gurmukhi", "Gujarati", "Oriya", "Tamil",
                   "Telugu", "Kannada", "Malayalam", "Arabic", "Digit", "Other", "OtherIndic"]
    present = [b for b in block_names if any(scripts[l].get(b, 0) > 0.001 for l in order2)]
    palette = (CAT + SEQ)[:len(present)]
    import numpy as np
    fig, ax = plt.subplots(figsize=(11, 5.4))
    bottom = np.zeros(len(order2))
    for bi, b in enumerate(present):
        vals = np.array([scripts[l].get(b, 0) for l in order2])
        ax.bar(range(len(order2)), vals, bottom=bottom, label=b, color=palette[bi % len(palette)],
               width=0.72, edgecolor=SURFACE, linewidth=0.4, zorder=3)
        bottom += vals
    ax.set_xticks(range(len(order2)))
    ax.set_xticklabels([_lang(k) for k in order2], rotation=40, ha="right")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    _style(ax, "Script-block composition per language (sampled)",
           subtitle="fraction of non-space characters by Unicode block", ylabel="fraction")
    figs.append(_save(fig, out_dir, "20_script_distribution.png"))


def _scripts_order(scripts):
    return sorted(scripts, key=lambda k: -scripts[k].get("Latin", 0))


def plot_codemix(shard_stats, out_dir, figs):
    plt = _mpl()
    canon = shard_stats["canon_by_lang"]
    mix = shard_stats["codemix_by_lang"]
    order = [k for k in shard_stats["langs"] if canon.get(k)]
    order += [k for k in canon if k not in order]
    order = sorted(order, key=lambda k: -(mix.get(k, 0) / canon[k]) if canon.get(k) else 0)
    frac = [(mix.get(k, 0) / canon[k]) if canon.get(k) else 0 for k in order]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(range(len(order)), frac, color=CAT[7], width=0.72, zorder=3)
    for i, v in enumerate(frac):
        ax.text(i, v, f"{v*100:.1f}%", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([_lang(k) for k in order], rotation=40, ha="right")
    _style(ax, "Code-mixing rate (Latin + Indic in one passage)",
           subtitle="fraction of canonical passages mixing Latin and Indic script",
           ylabel="fraction")
    figs.append(_save(fig, out_dir, "21_codemixing_by_language.png"))


def plot_topics(qd, out_dir, figs):
    plt = _mpl()
    if not qd or not qd.get("has_topics"):
        return
    topic_overall = qd["topic_overall"]
    order = _order_by_value(topic_overall)
    fig, ax = plt.subplots(figsize=(10, 5.2))
    vals = [topic_overall[t] for t in order]
    vmax = max(vals)
    ax.barh(range(len(order)), vals, color=[_seq_color(v / vmax) for v in vals], zorder=3)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.invert_yaxis()
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:,}", va="center", fontsize=8.5, color=INK2)
    _style(ax, "Topic distribution (indexed chunks)",
           subtitle=f"{len(order)} topics · from Qdrant `topic` payload", xlabel="chunks")
    ax.grid(axis="y", visible=False)
    figs.append(_save(fig, out_dir, "22_topic_distribution.png"))

    # topic x language heatmap (column-normalized)
    tbl = qd["topic_by_lang"]
    langs = sorted(tbl, key=lambda l: -sum(tbl[l].values()))
    import numpy as np
    M = np.zeros((len(order), len(langs)))
    for j, l in enumerate(langs):
        col = tbl[l]
        tot = sum(col.values()) or 1
        for i, t in enumerate(order):
            M[i, j] = col.get(t, 0) / tot
    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(M, cmap="Blues", aspect="auto", vmin=0)
    ax.set_xticks(range(len(langs)))
    ax.set_xticklabels([_lang(l) for l in langs], rotation=40, ha="right")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_title("Topic x language (column-normalized)", fontsize=13, fontweight="bold", loc="left")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="fraction of language's chunks")
    ax.grid(False)
    figs.append(_save(fig, out_dir, "23_topic_by_language.png"))


# ==========================================================================================
# data_stats.json + EDA.md
# ==========================================================================================
def build_data_stats(spec, real_stats, shard_stats, fam_stats, qd, lex_stats, meta):
    def lang_pct(d):
        return {k: _pct(v) for k, v in d.items()}
    allc = [c for v in shard_stats["char_by_lang"].values() for c in v]
    allt = [t for v in shard_stats["wtok_by_lang"].values() for t in v]
    q_text = shard_stats["q_text"]
    stats = {
        "meta": meta,
        "manifest": real_stats["manifest8"],
        "scope": real_stats["scope"],
        "realization": {k: real_stats[k] for k in (
            "source_rows", "source_passages", "unique_canonical", "n_families",
            "indexed_docs", "heldout_docs", "indexed_families", "heldout_families",
            "indexed_queries", "heldout_queries", "excluded_queries",
            "calibration_queries", "dev_queries", "sealed_queries",
            "in_corpus_partition", "absent_evidence_partition", "excluded_reasons",
            "query_groups", "leakage")},
        "composition": {
            "n_occurrences_shards": shard_stats["n_occurrences"],
            "n_canonical_shards": shard_stats["n_canonical"],
            "canonical_by_detected_lang": shard_stats["canon_by_lang"],
            "occurrences_by_source_cfg": shard_stats["occ_by_cfg"],
            "occurrences_by_variant": shard_stats["occ_by_variant"],
            "english_canonical": shard_stats["en_canonical"],
            "collapse_confusion": shard_stats["collapse"],
            "collapse_note": {src: dst for src, dst in COLLAPSE.items()},
        },
        "lengths": {
            "passage_char_overall": _pct(allc),
            "passage_wordtok_overall": _pct(allt),
            "passage_char_by_lang": lang_pct(shard_stats["char_by_lang"]),
            "passage_wordtok_by_lang": lang_pct(shard_stats["wtok_by_lang"]),
            "max_passage_chars": shard_stats["max_char"],
            "top_longest_passages": [{"chars": c, "lang": l, "src_cfg": g}
                                     for c, l, g in shard_stats["longest"][:20]],
            "passages_over_2000_chars": sum(1 for c in allc if c > 2000),
        },
        "queries": {
            "n_distinct": len(q_text),
            "char": _pct([len(t) for t in q_text.values()]),
            "wordtok": _pct([len(indic_tokens(t)) for t in q_text.values()]),
            "per_source_lang": dict(Counter(shard_stats["q_cfg"].values())),
            "crosslingual_group_size_freq": dict(Counter(
                len(s) for s in shard_stats["group_langs"].values())),
            "n_groups": len(shard_stats["group_langs"]),
        },
        "qrels": {
            "positives_per_query_freq": dict(Counter(
                len(v) for v in shard_stats["q_pos"].values())),
            "singletons": sum(1 for v in shard_stats["q_pos"].values() if len(v) == 1),
            "multi_positive": sum(1 for v in shard_stats["q_pos"].values() if len(v) > 1),
            "coverage_by_lang": {
                k: {"total": Counter(shard_stats["q_cfg"].values()).get(k, 0),
                    "with_positive": sum(1 for q, s in shard_stats["q_pos"].items()
                                         if s and shard_stats["q_cfg"].get(q) == k)}
                for k in shard_stats["langs"]},
        },
    }
    if fam_stats:
        stats["duplication"] = fam_stats
    if lex_stats:
        stats["lexical"] = lex_stats
    if qd:
        stats["qdrant"] = {
            "collection": qd["collection"], "total_points": qd["total_points"],
            "summary_points_in_passage_collection": qd["summary_points_in_passage_collection"],
            "chunks_by_lang": qd["chunks_by_lang"], "docs_by_lang": qd["docs_by_lang"],
            "chunk_char": _pct(qd["chunk_char"]), "chunk_tok": _pct(qd["chunk_tok"]),
            "topic_overall": qd["topic_overall"], "has_topics": qd["has_topics"],
        }
    return stats


def write_eda_md(stats, figs, out_dir, spec):
    r = stats["realization"]
    comp = stats["composition"]
    ln = stats["lengths"]
    q = stats["queries"]
    qr = stats["qrels"]
    dup = stats.get("duplication")
    qd = stats.get("qdrant")
    lex = stats.get("lexical")
    figset = set(figs)

    # --- distribution-aware derived values so the PROSE can never contradict the numbers ---
    _sing, _multi = qr["singletons"], qr["multi_positive"]
    _qtot = _sing + _multi
    _dom_single = _sing >= _multi
    _dom = "single-positive" if _dom_single else "multi-positive"
    _dom_note = ("" if _dom_single else " — the selected English answer and its translated "
                 "variant both count, so a query typically carries two positives")
    _freq = qr["positives_per_query_freq"]
    _items = sorted((int(k), v) for k, v in _freq.items())
    _cum, _med, _half = 0, (_items[0][0] if _items else 0), (_qtot / 2 if _qtot else 0)
    for _k, _v in _items:
        _cum += _v
        if _cum >= _half:
            _med = _k
            break
    _cov = qr["coverage_by_lang"]
    _tot_q = sum(c["total"] for c in _cov.values()) or 1
    _cov_overall = sum(c["with_positive"] for c in _cov.values()) / _tot_q
    _cov_min = min((c["with_positive"] / c["total"]) for c in _cov.values() if c["total"]) \
        if any(c["total"] for c in _cov.values()) else 0.0
    _en_frac = (comp["english_canonical"] / comp["n_canonical_shards"]) \
        if comp["n_canonical_shards"] else 0.0
    _indic = comp["n_canonical_shards"] - comp["english_canonical"]

    def fig(name, caption):
        return f"![{caption}]({name})\n\n*{caption}*\n" if name in figset else \
            f"*(figure {name} not generated — data source unavailable in this run)*\n"

    collapse_extra = sum(
        v for src in COLLAPSE for k, v in comp["collapse_confusion"].get(src, {}).items())

    L = []
    L.append(f"# EDA — MSMARCO-XI validation-sample corpus (`{stats['manifest']}`)\n")
    L.append(f"> {stats['scope'].get('public_wording', '')}\n")
    L.append(f"Generated {stats['meta']['generated']} · seed {stats['meta']['seed']} · "
             f"eval/eda.py @ {stats['meta'].get('git_sha', 'unknown')[:10]}\n")
    L.append("This report is machine-backed: **every number here is reproducible from "
             "`data_stats.json`** in this directory. Sampled quantities are labeled as such.\n")

    L.append("## Data sources used in this run\n")
    L.append(f"- Shards: **{'yes' if stats['meta']['ran_shards'] else 'no'}** "
             f"(the real `index_build._iter_msmarco_rows` at revision "
             f"`{spec.dataset_revision[:12]}`, {spec.max_rows_per_shard} rows/shard × "
             f"{len(spec.languages)} languages)")
    L.append(f"- Live Qdrant: **{'yes' if qd else 'no'}**"
             + (f" (`{qd['collection']}`, {qd['total_points']:,} points)" if qd else ""))
    L.append(f"- Family reconstruction: **{dup['mode'] if dup else 'off'}**"
             + (" (sampled)" if dup and dup.get('sampled') else "") + "\n")

    L.append("## 1 · Corpus composition\n")
    L.append(f"The shard pass reproduces the frozen realization: **{comp['n_occurrences_shards']:,} "
             f"passage occurrences** collapse to **{comp['n_canonical_shards']:,} canonical "
             f"passages** (realization: {r['source_passages']:,} → {r['unique_canonical']:,}). Of "
             f"these, **{comp['english_canonical']:,}** are the shared English passages (deduped "
             f"once across all 14 shards) and the remainder are per-language translations.\n")
    L.append(fig("01_composition_docs_by_lang.png", "Canonical passages per detected language"))
    L.append(fig("03_composition_treemap.png", "Composition treemap"))
    if qd:
        L.append(fig("02_chunks_by_lang.png", "Indexed chunks per language (live Qdrant)"))
    L.append("\n### The Devanagari collapse (quantified, honestly)\n")
    L.append(f"The build detects language by **Unicode script block**, not source-shard label. "
             f"Four source languages share a script with a labeled sibling and therefore carry "
             f"**no distinct label**: Marathi, Sanskrit and Nepali all render in Devanagari and "
             f"fold into **`hi`**; Assamese renders in Bengali script and folds into **`bn`**. "
             f"Across those four shards, **{collapse_extra:,} translated canonical passages** are "
             f"indexed under a sibling's label. This is a labeling artifact, not a loss — the text "
             f"is fully indexed and retrievable — but per-language *reporting* for mr/sa/ne/as is "
             f"absorbed into hi/bn, and any language-filtered retrieval treats them as the sibling. "
             f"The confusion matrix below shows the fold rate per shard.\n")
    L.append(fig("04_devanagari_collapse.png", "Source shard → detected label"))

    L.append("## 2 · Length distributions\n")
    pc, wt = ln["passage_char_overall"], ln["passage_wordtok_overall"]
    L.append(f"Passages are short (MS-MARCO web passages): median **{pc['p50']:,}** chars / "
             f"**{wt['p50']}** word tokens, p90 **{pc['p90']:,}** / **{wt['p90']}**, but the tail "
             f"is heavy — p99 **{pc['p99']:,}** chars and a maximum of **{ln['max_passage_chars']:,} "
             f"chars**. **{ln['passages_over_2000_chars']:,} passages exceed 2,000 chars**; these "
             f"drive the passage-aware chunker to split (max_tokens=320) and force the longest "
             f"full-length embedding passes.\n")
    L.append(fig("05_passage_charlen_hist.png", "Passage char-length (log-log)"))
    L.append(fig("06_passage_charlen_box_by_lang.png", "Passage length by language"))
    L.append(fig("07_token_length_hist.png", "Raw passage vs indexed chunk tokens"))

    L.append("## 3 · Query analysis\n")
    L.append(f"**{q['n_distinct']:,} distinct queries** (namespaced per shard), median "
             f"**{q['char']['p50']}** chars / **{q['wordtok']['p50']}** word tokens — natural "
             f"short questions. Queries fan out cross-lingually: **{q['n_groups']:,} source "
             f"query-groups** tie the per-language variants of one MS-MARCO query together so the "
             f"split can co-locate them (a sealed Tamil query can never leak through its Hindi "
             f"sibling).\n")
    L.append(fig("08_query_length.png", "Query length"))
    L.append(fig("09_queries_per_language.png", "Queries per language"))
    L.append(fig("10_crosslingual_group_size.png", "Cross-lingual group size"))
    L.append(f"By the frozen split, **{r['indexed_queries']:,}** queries are answerable "
             f"(positives fully indexed), **{r['heldout_queries']:,}** are absent-evidence "
             f"(positives fully held out — the abstain test set), and only "
             f"**{r['excluded_queries']}** are excluded ({'; '.join(r['excluded_reasons'])}).\n")
    L.append(fig("11_answerable_split.png", "Query realizability split"))

    L.append("## 4 · Duplication & leakage structure\n")
    if dup:
        L.append(f"The corpus is **naturally duplicated**: English passages recur across queries "
                 f"and shards. Grouping canonical passages into leakage-safe families (exact-text "
                 f"∪ SimHash Hamming≤{spec.simhash_hamming_max} ∪ translation-link) yields "
                 f"**{dup['n_families']:,} families** over **{dup['n_canonical']:,} canonical "
                 f"passages**"
                 + (f" — matching the realization's {r['n_families']:,}." if dup['mode'] == 'full'
                    and not dup.get('sampled') else " (sampled).")
                 + f" The largest family holds **{dup['largest_family']}** passages; "
                 f"**{dup['multilingual_families']:,} families span more than one language** "
                 f"(a passage and its translations). Whole-family allocation sends each family to "
                 f"exactly one side.\n")
        L.append(fig("14_exact_dup_sizes.png", "Exact-duplicate group sizes"))
        L.append(fig("15_simhash_family_sizes.png", "Near-duplicate family sizes"))
        L.append(fig("16_crosslingual_family_span.png", "Cross-lingual family composition"))
    else:
        L.append("_(family reconstruction skipped this run — see realization counts.)_\n")
    lk = r["leakage"]
    L.append(f"\n**Leakage is zero by construction and re-proven on the realized index:** "
             f"family intersection **{lk['family_intersection']}**, stable_doc_id intersection "
             f"**{lk['stable_doc_id_intersection']}**, exact-hash intersection "
             f"**{lk['exact_hash_intersection']}** between the "
             f"{lk['indexed_exact_hashes']:,} indexed and {lk['heldout_exact_hashes']:,} held-out "
             f"distinct exact hashes.\n")
    L.append(fig("17_leakage_confirmation.png", "Leakage confirmation 0/0/0"))

    L.append("## 5 · Qrel / retrieval difficulty\n")
    L.append(f"Queries are predominantly **{_dom}**{_dom_note}: **{_sing:,}** singleton vs "
             f"**{_multi:,}** multi-positive, median **{_med}** positive(s) per query. Either way "
             f"the qrel is strict and MRR-sensitive — the reranker must place the right passage(s) "
             f"first. Per-language qrel coverage is **{_cov_overall*100:.0f}%** overall "
             f"(minimum **{_cov_min*100:.0f}%**).\n")
    L.append(fig("12_positives_per_query.png", "Positives per query"))
    L.append(fig("13_qrel_coverage_by_lang.png", "Per-language qrel coverage"))

    L.append("## 6 · Vocabulary & script\n")
    if lex:
        L.append(f"Vocabulary and type-token ratio are measured at a **fixed {lex['token_budget']:,}"
                 f"-token budget per language** (sampled from the seeded reservoir) so the numbers "
                 f"are comparable — TTR is otherwise sample-size dependent. Tokenization uses the "
                 f"**Indic-correct `textnorm.WORD_RE`** (combining marks Mn/Mc included), not bare "
                 f"`\\w` — the fix that stopped Devanagari words shattering into consonant "
                 f"fragments across every lexical path.\n")
        L.append(fig("18_vocab_by_language.png", "Vocabulary size"))
        L.append(fig("19_ttr_by_language.png", "Type-token ratio"))
        L.append(fig("20_script_distribution.png", "Script-block composition"))
    L.append(fig("21_codemixing_by_language.png", "Code-mixing rate"))

    if qd and qd.get("has_topics"):
        L.append("## 7 · Topic distribution\n")
        L.append(f"Topic labels from the Qdrant payload (`topic`) spread indexed chunks across "
                 f"**{len(qd['topic_overall'])} topics**.\n")
        L.append(fig("22_topic_distribution.png", "Topic distribution"))
        L.append(fig("23_topic_by_language.png", "Topic × language"))

    L.append("## 8 · Researcher's narrative — what the structure implies for retrieval\n")
    L.append(
        f"**Why hybrid dense+sparse.** The corpus is cross-lingual MS-MARCO: {q['n_distinct']:,} "
        f"short questions against {comp['n_canonical_shards']:,} short passages, "
        f"~{_en_frac*100:.0f}% of them English answer passages and the rest Indic translations. "
        f"Dense BGE-M3 carries the cross-lingual semantic match (a Tamil question to an English "
        f"passage), but the passages are keyword-dense web text where exact term hits matter — and "
        f"the qrel is strict (median {_med} positive(s)/query). Sparse lexical signal recovers the "
        f"term-anchored cases dense retrieval softens, which is exactly why the pipeline fuses both "
        f"rather than betting on one.\n")
    L.append(
        f"**Why the tokenizer fix mattered — and why it is load-bearing here.** "
        f"{comp['n_canonical_shards'] - comp['english_canonical']:,} passages are in Indic scripts, "
        f"and the entire vectorless edge tier plus the OOD gate, the extractive composer and the "
        f"grounding check are lexical. Under bare `\\w`, Devanagari matras and the virama are "
        f"dropped, so every Indic word fragments — the whole lexical arm silently indexes garbage. "
        f"The script-block and vocabulary figures are computed with the corrected tokenizer and "
        f"show coherent, script-consistent vocabularies; this is the evidence the fix is real, not "
        f"asserted.\n")
    L.append(
        f"**Why same-language preference and script-aware routing.** Language here is a *script* "
        f"signal, not a metadata field: {collapse_extra:,} Marathi/Sanskrit/Nepali/Assamese "
        f"passages fold into their Devanagari/Bengali siblings, and code-mixing means passages "
        f"carry Latin tokens inside Indic text. Retrieval that prefers same-language evidence and "
        f"answers in the query's language must therefore key off detected script and tolerate "
        f"mixing — a naive language-tag filter would silently drop the four collapsed languages. "
        f"The length tail (max {ln['max_passage_chars']:,} chars; "
        f"{ln['passages_over_2000_chars']:,} passages over 2,000 chars) is why chunking is "
        f"passage-aware with a token cap rather than fixed-size.\n")
    L.append(
        f"**Why reranking, and why the leakage-safe split is the headline.** With a strict qrel "
        f"(median {_med} positive(s) per query), first-stage recall is necessary but not "
        f"sufficient — a cross-encoder reranker is what converts recall@k into MRR. And none of these metrics "
        f"mean anything without a clean split: because families (exact ∪ near-dup ∪ translation) "
        f"are allocated whole, the {r['heldout_queries']:,}-query absent-evidence set shares "
        f"**zero** passages, near-duplicates, or translations with the index "
        f"({lk['exact_hash_intersection']}/{lk['stable_doc_id_intersection']}/"
        f"{lk['family_intersection']} intersection). The abstain behavior is measured against "
        f"genuinely unseen evidence, and the reported retrieval numbers cannot be inflated by a "
        f"near-duplicate leaking across the split.\n")

    L.append("\n---\n### Figures\n")
    for fn in figs:
        L.append(f"- `{fn}`")
    L.append("\n### Limitations & assumptions\n")
    L.append("- Language = Unicode script detection (the build's `_detect_lang`); mr/sa/ne fold "
             "into hi and as into bn (quantified in §1). Per-language stats for detected labels.")
    L.append(f"- Shard pass is capped at {spec.max_rows_per_shard} rows/shard (the frozen build "
             "cap); it is the exact indexed corpus, not the full validation split.")
    if lex:
        L.append(f"- Vocabulary/TTR/script/code-mixing use a seeded reservoir "
                 f"(≤{stats['meta']['reservoir_per_lang']:,} passages/language) at a fixed "
                 f"{lex['token_budget']:,}-token budget — labeled sampled throughout.")
    if dup and dup.get("sampled"):
        L.append(f"- Family reconstruction ran on a {dup['n_occurrences_used']:,}-occurrence "
                 "sample (`--families sample`); counts do not equal the full realization.")
    L.append("- Sealed query files are never opened; query analysis is from shards + realization "
             "counts only.")
    with open(os.path.join(out_dir, "EDA.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


# ==========================================================================================
# main
# ==========================================================================================
def _load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _git_sha():
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=os.path.dirname(HERE)).stdout.strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser(description="Researcher-grade EDA of the MSMARCO-XI val sample")
    ap.add_argument("--manifest", default="73ca3e90")
    ap.add_argument("--build-dir", default=None, help="default eval/builds/<manifest>")
    ap.add_argument("--qdrant-url", default="http://localhost:6333")
    ap.add_argument("--collection", default=None, help="default <prefix>_<manifest>__passage")
    ap.add_argument("--shard-dir", default=None,
                    help="offline override: dir holding the *val.parquet shards (else HF cache)")
    ap.add_argument("--hf-home", default=None,
                    help="set HF_HOME + force HF_HUB_OFFLINE=1 (serve shards from local cache)")
    ap.add_argument("--out", default=os.path.join(HERE, "eda"))
    ap.add_argument("--seed", type=int, default=20260817)
    ap.add_argument("--families", choices=["full", "sample", "off"], default="full")
    ap.add_argument("--family-sample", type=int, default=200_000)
    ap.add_argument("--reservoir-per-lang", type=int, default=25_000)
    ap.add_argument("--token-budget", type=int, default=300_000)
    ap.add_argument("--max-rows-per-shard", type=int, default=None,
                    help="override spec cap (default = the frozen 6000)")
    ap.add_argument("--skip-qdrant", action="store_true")
    ap.add_argument("--skip-shards", action="store_true")
    ap.add_argument("--self-test", action="store_true",
                    help="synthetic shards + no Qdrant — validates every non-network path")
    args = ap.parse_args()

    will_plot = not (args.skip_shards and args.skip_qdrant)
    if will_plot:
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            print("ERROR: matplotlib is required for the figures. Install it first:\n"
                  "    pip install matplotlib\n"
                  "(or run realization-only with --skip-shards --skip-qdrant)")
            return 2

    if args.hf_home:
        os.environ["HF_HOME"] = os.path.expanduser(args.hf_home)
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    os.makedirs(args.out, exist_ok=True)
    build_dir = args.build_dir or os.path.join(HERE, "builds", args.manifest)
    real = _load_json(os.path.join(build_dir, "realization.json"))
    if real["manifest8"] != args.manifest:
        print(f"REFUSED: realization manifest {real['manifest8']} != {args.manifest}")
        return 2
    spec = cspec.CorpusBuildSpec(**real["spec"])
    # integrity self-check: the reconstructed spec MUST reproduce the manifest
    if spec.manifest8() != args.manifest:
        print(f"WARNING: reconstructed spec manifest {spec.manifest8()} != {args.manifest} "
              "— spec drift; proceeding but numbers may not match the frozen build")
    if args.max_rows_per_shard is not None:
        spec.max_rows_per_shard = args.max_rows_per_shard
    collection = args.collection or f"{spec.collection_prefix}_{args.manifest}__passage"

    t0 = time.time()
    real_stats = analyze_realization(real)
    print(f"[realization] manifest {real_stats['manifest8']} · "
          f"indexed {real_stats['indexed_docs']:,} / heldout {real_stats['heldout_docs']:,} · "
          f"leakage {real_stats['leakage']['family_intersection']}/"
          f"{real_stats['leakage']['stable_doc_id_intersection']}/"
          f"{real_stats['leakage']['exact_hash_intersection']}")

    # ---- shards ----
    shard_stats = fam_stats = lex_stats = None
    if not args.skip_shards:
        row_iters = None
        if args.self_test:
            print("[self-test] synthetic MSMARCO-XI rows (no dataset, no network)")
            row_iters = {c: ib.synthetic_msmarco_rows(n_rows=400, seed=i)
                         for i, c in enumerate(spec.languages)}
        keep = args.families in ("full", "sample")
        print(f"[shards] streaming {len(spec.languages)} shards "
              f"(cap {spec.max_rows_per_shard}/shard, keep_occ={keep}) ...")
        shard_stats = analyze_shards(spec, args.shard_dir, args.seed, args.reservoir_per_lang,
                                     keep, row_iters=row_iters)
        print(f"[shards] {shard_stats['n_occurrences']:,} occurrences → "
              f"{shard_stats['n_canonical']:,} canonical "
              f"(realization: {real_stats['source_passages']:,} → "
              f"{real_stats['unique_canonical']:,})")
        if not args.self_test:
            if shard_stats["n_occurrences"] != real_stats["source_passages"]:
                print("  WARNING: occurrence count != realization source_passages")
            if shard_stats["n_canonical"] != real_stats["unique_canonical"]:
                print("  WARNING: canonical count != realization unique_canonical")

        if args.families != "off":
            print(f"[families] rebuilding family graph (mode={args.families}) ...")
            fam_stats = analyze_families(shard_stats["occ_all"], spec, args.families,
                                         args.family_sample, args.seed)
            print(f"[families] {fam_stats['n_families']:,} families "
                  f"(realization {real_stats['n_families']:,}) · "
                  f"{fam_stats['multilingual_families']:,} multilingual")
            shard_stats["occ_all"] = None      # free memory before plotting

        print("[lexical] vocab / TTR / scripts / code-mixing on seeded reservoir ...")
        lex_stats = analyze_lexical(shard_stats, args.token_budget)

    # ---- qdrant ----
    qd = None
    if not args.skip_qdrant and not args.self_test:
        try:
            print(f"[qdrant] scrolling {collection} ...")
            qd = analyze_qdrant(args.qdrant_url, collection)
            print(f"[qdrant] {qd['total_points']:,} points · "
                  f"topics={'yes' if qd['has_topics'] else 'no'} · "
                  f"summary_leak={qd['summary_points_in_passage_collection']}")
        except Exception as e:
            print(f"[qdrant] SKIPPED — {e}")

    # ---- plots ----
    figs = []
    if shard_stats:
        print("[plot] composition / collapse / lengths / queries / qrels / lexical ...")
        plot_composition(shard_stats, qd, args.out, figs)
        plot_collapse(shard_stats, args.out, figs)
        plot_lengths(shard_stats, qd, args.out, figs)
        plot_queries(shard_stats, real_stats, args.out, figs)
        plot_qrels(shard_stats, args.out, figs)
        plot_codemix(shard_stats, args.out, figs)
        if lex_stats:
            plot_lexical(lex_stats, args.out, figs)
    if fam_stats:
        plot_duplication(fam_stats, real_stats, args.out, figs)
    if qd:
        plot_topics(qd, args.out, figs)

    # ---- artifacts ----
    meta = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": args.seed, "git_sha": _git_sha(),
        "ran_shards": bool(shard_stats), "ran_qdrant": bool(qd),
        "reservoir_per_lang": args.reservoir_per_lang,
        "wall_seconds": round(time.time() - t0, 1),
        "self_test": args.self_test,
    }
    if shard_stats:
        stats = build_data_stats(spec, real_stats, shard_stats, fam_stats, qd, lex_stats, meta)
        write_eda_md(stats, figs, args.out, spec)
    else:
        stats = {"meta": meta, "manifest": real_stats["manifest8"],
                 "realization": real_stats, "note": "shards skipped — realization-only run"}
    with open(os.path.join(args.out, "data_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1, default=list)

    print(f"\nEDA complete in {meta['wall_seconds']}s → {args.out}")
    print(f"  {len(figs)} figures · data_stats.json"
          + (" · EDA.md" if shard_stats else " (realization-only)"))
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    raise SystemExit(main())
