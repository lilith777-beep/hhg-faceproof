"""
test_tokenizer_offsets.py — P1.1 frozen-transform invariant matrix for the offset-mapping
tokenizer + offset-correct chunking.

The production chunker slices text by CHAR offsets, NEVER by naive token-string search. These
tests prove the invariants across scripts (English / Devanagari / Urdu), pathological inputs
(repeats / emoji / combining marks / punctuation), and windowing edges (exact token ceilings,
monotonic spans, chunk.text == source[span], overlap bounds) — using BOTH the CPU
ApproxOffsetTokenizer and a fake SentencePiece-style tokenizer whose token STRINGS are not
source substrings (the '▁foo' hazard), so passing can only mean offset slicing, not string search.
"""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import chunking as ck                                    # noqa: E402
from chunking import (ApproxOffsetTokenizer, FixedSizeChunker, Document,  # noqa: E402
                      chunk_by_offsets)
from embeddings import HashEmbedder, HFOffsetTokenizer   # noqa: E402


class FakeSPTokenizer:
    """Mimics a SentencePiece tokenizer: sub-word pieces with REAL char offsets, but token
    strings that carry a leading '▁' marker (so a piece string is NOT a substring of the
    source). If chunking relied on token-string `.find()` it would desync; using offsets it
    cannot."""

    def encode_offsets(self, text: str) -> list:
        offs = []
        for m in re.finditer(r"\S+", text or ""):
            s, w = m.start(), m.group()
            i = 0
            while i < len(w):
                j = min(i + 3, len(w))        # <=3-char sub-words
                offs.append((s + i, s + j))
                i = j
        return offs

    def piece_strings(self, text: str) -> list:
        """What a real SP tokenizer would EMIT as strings (with the ▁ word-start marker)."""
        out = []
        for m in re.finditer(r"\S+", text or ""):
            w = m.group()
            i = 0
            while i < len(w):
                piece = w[i:i + 3]
                out.append(("▁" + piece) if i == 0 else piece)   # ▁ prefix on word-start
                i = j if (j := min(i + 3, len(w))) else i + 3
        return out


# ---- corpus fixtures across scripts + hazards -------------------------------------------
CASES = {
    "english": "The quick brown fox jumps over the lazy dog near the quiet river bank today.",
    "devanagari": "गोवा भारत के पश्चिमी तट पर एक राज्य है जो अपने समुद्र तटों के लिए प्रसिद्ध है।",
    "urdu": "گوا بھارت کی مغربی ساحل پر ایک ریاست ہے جو اپنے ساحلوں کے لیے مشہور ہے۔",
    "repeats": "buffalo buffalo buffalo buffalo buffalo buffalo buffalo buffalo buffalo",
    "emoji": "I love Goa 🏖️ and RAG 🤖 systems 🚀 running under 200ms ⚡ every single time 🎯 now.",
    "combining": "café résumé naïve jalãpeño coöperate señor",
    "punctuation": "Well... really?! Yes -- indeed; (see: e.g., foo/bar) [note] {a=b} 100% sure!!!",
}

TOKENIZERS = {"approx": ApproxOffsetTokenizer(), "fake_sp": FakeSPTokenizer()}


def _invariants(name, text, ot, size, overlap):
    doc = Document(doc_id=f"d-{name}", text=text, language="xx", stable_doc_id=f"sid-{name}")
    chunks = FixedSizeChunker(size=size, overlap=overlap, offset_tokenizer=ot).chunk(doc)
    if not text.strip():
        assert chunks == []
        return
    assert chunks, f"{name}: non-empty text must produce chunks"
    prev_start = -1
    for c in chunks:
        # 1) chunk.text == source[span] — the core offset-correctness invariant
        assert c.text == text[c.char_start:c.char_end], f"{name}: chunk.text != source[span]"
        # 2) spans within bounds
        assert 0 <= c.char_start <= c.char_end <= len(text), f"{name}: span out of bounds"
        # 3) exact token ceiling
        assert c.token_count <= size, f"{name}: token_count {c.token_count} > size {size}"
        # 4) monotonic non-decreasing starts
        assert c.char_start >= prev_start, f"{name}: non-monotonic span"
        prev_start = c.char_start
        # 5) stable_doc_id propagated to every chunk
        assert c.stable_doc_id == f"sid-{name}"


def test_invariants_all_scripts_all_tokenizers():
    for tname, ot in TOKENIZERS.items():
        for name, text in CASES.items():
            for size, overlap in ((4, 1), (8, 3), (16, 0)):
                _invariants(f"{name}/{tname}", text, ot, size, overlap)


def test_fake_sp_strings_are_not_substrings_but_offsets_are_correct():
    """Proves the hazard is real: the emitted piece strings ('▁foo') are NOT source substrings,
    yet every offset span slices back to real text."""
    text = CASES["english"]
    sp = FakeSPTokenizer()
    pieces = sp.piece_strings(text)
    assert any(p.startswith("▁") for p in pieces), "fake SP must emit ▁-marked pieces"
    assert any(p not in text for p in pieces), "at least one piece string must NOT be a substring"
    for (s, e) in sp.encode_offsets(text):
        assert text[s:e] and not text[s:e].startswith("▁")   # offsets slice clean text


def test_overlap_bounds_exact():
    """Consecutive windows overlap by EXACTLY `overlap` tokens (until the final short window)."""
    text = " ".join(f"w{i}" for i in range(30))
    offs = ApproxOffsetTokenizer().encode_offsets(text)
    size, overlap = 7, 3
    windows = chunk_by_offsets(text, offs, size, overlap)
    for a, b in zip(windows, windows[1:]):
        _, _, _, a_ti, a_tj = a
        _, _, _, b_ti, b_tj = b
        assert b_ti == a_ti + (size - overlap), "stride must be size-overlap"
        assert a_tj - b_ti == overlap, f"overlap must be exactly {overlap} tokens"


def test_chunk_by_offsets_edges():
    # empty
    assert chunk_by_offsets("", [], 4, 1) == []
    # single token
    out = chunk_by_offsets("hello", [(0, 5)], 4, 1)
    assert len(out) == 1 and out[0][0] == "hello"
    # exact ceiling: a window never exceeds `size` tokens
    text = "a b c d e f g h"
    offs = ApproxOffsetTokenizer().encode_offsets(text)
    for (ctext, cs, ce, ti, tj) in chunk_by_offsets(text, offs, 3, 1):
        assert tj - ti <= 3 and ctext == text[cs:ce]
    # bad params rejected
    for bad in ((0, 0), (3, 3), (3, 5), (3, -1)):
        try:
            chunk_by_offsets(text, offs, *bad)
            assert False, f"expected ValueError for size/overlap {bad}"
        except ValueError:
            pass


def test_repeats_resolve_to_distinct_spans():
    """Repeated tokens are the classic naive-.find() failure (every 'buffalo' finds index 0).
    Offset slicing gives DISTINCT, advancing spans."""
    text = CASES["repeats"]
    chunks = FixedSizeChunker(size=2, overlap=0,
                              offset_tokenizer=ApproxOffsetTokenizer()).chunk(
        Document(doc_id="r", text=text))
    starts = [c.char_start for c in chunks]
    assert starts == sorted(starts) and len(set(starts)) == len(starts), \
        "repeated tokens must map to distinct advancing spans"
    for c in chunks:
        assert c.text == text[c.char_start:c.char_end]


def test_full_coverage_no_gaps_zero_overlap():
    """With overlap=0 the concatenated spans cover exactly the tokenized region (no lost text
    between tokens beyond the original inter-token whitespace)."""
    text = CASES["punctuation"]
    ot = ApproxOffsetTokenizer()
    offs = ot.encode_offsets(text)
    windows = chunk_by_offsets(text, offs, 5, 0)
    # first char of first window == first token start; last char == last token end
    assert windows[0][1] == offs[0][0]
    assert windows[-1][2] == offs[-1][1]


def test_hf_offset_tokenizer_adapter_shape():
    """HFOffsetTokenizer wraps a return_offsets_mapping tokenizer and drops zero-width spans."""
    class _Stub:
        def __call__(self, text, return_offsets_mapping=False, add_special_tokens=True):
            # simulate HF: special tokens map to (0,0); real pieces to true spans
            return {"offset_mapping": [(0, 0), (0, 3), (4, 9), (0, 0)]}
    ot = HFOffsetTokenizer(_Stub())
    assert ot.encode_offsets("foo bar") == [(0, 3), (4, 9)], "zero-width spans must be dropped"
    assert ot.encode_offsets("") == []


def test_embedder_exposes_offset_tokenizer():
    ot = HashEmbedder(dim=32).get_offset_tokenizer()
    assert hasattr(ot, "encode_offsets")
    assert ot.encode_offsets("hello world") == [(0, 5), (6, 11)]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        passed += 1
        print(f"  PASS  {fn.__name__}")
    print(f"\n{passed}/{len(fns)} tokenizer-offset invariant tests passed")
