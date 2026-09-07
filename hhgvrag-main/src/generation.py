"""
generation.py — grounded answer generation.

`build_prompt` forces the model to answer ONLY from the numbered context, cite passages as [n],
and say it doesn't know when the context is insufficient. Backends:
- `ExtractiveGenerator` — deterministic, grounded, CPU (picks the best context sentence + cites).
  Used to test the ANSWER path without a model.
- `EchoHallucinator`   — returns off-context text; used to test that the grounding guardrail
  catches ungrounded answers.
- `LocalLLMGenerator`  — the real 3B (Qwen2.5-3B / Llama-3.2-3B) on Modal GPU. Lazy import.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

from textnorm import INDIC_STOP, WORD_RE as _TOK   # Indic-correct: matras/virama attached
_SENT = re.compile(r"(?<=[\.\!\?।॥۔؟])\s+")   # ؟ = Urdu question mark (U+061F)

SYSTEM = (
    "You answer strictly from the numbered CONTEXT passages. Cite the passages you use as [n]. "
    "If the context does not contain the answer, reply exactly: I don't have grounded information "
    "on that. Keep the answer to 1-3 sentences."
)

# v2 contextual prompt — engineered for grounded multilingual QA on a small model.
# Kept as a stable constant: an identical system prefix lets vLLM's prefix cache skip
# re-prefilling it on every request.
SYSTEM_V2 = (
    "You are a precise question-answering assistant over a retrieval corpus.\n"
    "Rules — follow ALL of them:\n"
    "1. Answer ONLY with facts stated in the numbered CONTEXT passages. Never add outside "
    "knowledge, never guess.\n"
    "2. Cite every fact with its passage number in square brackets, e.g. [1] or [2][3].\n"
    "3. If the context does not contain the answer, reply exactly: I don't have grounded "
    "information on that.\n"
    "4. Reply in the SAME LANGUAGE AND SCRIPT as the QUESTION — a romanized (Latin-script) "
    "Hindi/Tamil/Bengali question gets its answer in that language WRITTEN IN LATIN script; "
    "(Hindi question → Hindi answer, English → "
    "English, etc.), even when the context passages are in another language.\n"
    "5. Be COMPREHENSIVE and well-structured. Lead with the direct answer, then synthesize "
    "ALL the relevant grounded facts the passages provide — supporting detail, mechanism, "
    "categories/types, examples, numbers, and key qualifications. Aim for a thorough answer "
    "of 5-8 sentences (two short paragraphs) whenever the passages support it; NEVER stop at "
    "a single definitional sentence if more grounded detail is available. No preamble, no "
    "restating the question, no repetition — density, not padding.\n"
    "6. If passages conflict, state the answer from the most specific passage and cite it."
)


def build_messages(query: str, contexts: Sequence) -> list:
    """Chat-format prompt (system + user) for chat-template backends (vLLM / HF chat)."""
    lines = ["CONTEXT:"]
    for i, c in enumerate(contexts, 1):
        lines.append(f"[{i}] {c.text}")
    lines += ["", f"QUESTION: {query}", "ANSWER:"]
    return [{"role": "system", "content": SYSTEM_V2},
            {"role": "user", "content": "\n".join(lines)}]


@dataclass
class GenOutput:
    text: str
    cited_chunk_ids: list = field(default_factory=list)


@runtime_checkable
class Generator(Protocol):
    def generate(self, query: str, contexts: Sequence) -> GenOutput: ...


def build_prompt(query: str, contexts: Sequence) -> str:
    lines = [SYSTEM, "", "CONTEXT:"]
    for i, c in enumerate(contexts, 1):
        lines.append(f"[{i}] {c.text}")
    lines += ["", f"QUESTION: {query}", "ANSWER:"]
    return "\n".join(lines)


def parse_citations(text: str, contexts: Sequence) -> list:
    ids = []
    for m in re.findall(r"\[(\d+)\]", text):
        i = int(m) - 1
        if 0 <= i < len(contexts):
            cid = contexts[i].chunk_id
            if cid not in ids:
                ids.append(cid)
    return ids


def _content(text: str) -> set:
    return {t for t in _TOK.findall(text.lower()) if len(t) > 1}


# ---- extractive backend (the <200ms live path) -----------------------------------------
_STOPWORDS = set(
    "a an the of to in on for and or is are was were be been being this that these those "
    "it its as at by with from into about over under how what when where who whom which why "
    "do does did can could should would will may might i you he she they we me my your "
    "get take much many long there here also very more most".split()
) | INDIC_STOP   # Indic function words scored as "overlap" before — a wrong-topic
                 # same-language sentence could beat correct evidence on के/हैं alone
# a sentence that ASKS is a heading or a query echo, never an answer
_INTERROGATIVE = ("what", "how", "why", "when", "where", "which", "who", "whom", "whose",
                  "is", "are", "can", "do", "does", "did", "will", "would", "should",
                  "क्या", "कैसे", "कब", "कहाँ", "कौन", "क्यों", "कितना", "कितने")
_DEF_QUERY = re.compile(
    r"\s*(?:what\s+(?:is|are)|define|definition\s+of|meaning\s+of)\s+"
    r"(?:a\s+|an\s+|the\s+)?([\w\s-]{2,40}?)\s*\??\s*$", re.IGNORECASE)
# abbreviation-aware sentence assembly: the raw splitter cuts after any period, so
# "established by charter (i.e. by an ad hoc act)" fractures mid-thought — merge fragments
# whose tail is a known abbreviation or an unclosed parenthetical back onto the next piece
_ABBREV_TAIL = re.compile(
    r"(?:\b(?:i\.e|e\.g|etc|vs|viz|approx|no|inc|ltd|corp|co|mr|mrs|ms|dr|st|jr|sr|"
    r"u\.s|u\.k|a\.m|p\.m)\.|\([^)]*)$", re.IGNORECASE)


def _split_sents(text: str) -> list:
    parts = [s.strip() for s in _SENT.split(text) if s.strip()]
    out: list = []
    for p in parts:
        if out and _ABBREV_TAIL.search(out[-1]):
            out[-1] = f"{out[-1]} {p}"
        else:
            out.append(p)
    return out


# meta-framing prose describes content instead of stating it — never a good answer
# (defense-in-depth: leaf-evidence semantics already keep RAPTOR summaries out entirely)
# MS MARCO passages are web scrapes that often begin mid-list ("2 Any impact on your credit
# score...", "1. If you have...", "A: The Celebi disc..."). Verbatim extraction copies those
# enumeration artifacts, which then collide with our [n] citation markers into numbering
# soup. Strip ONLY unambiguous list/QA prefixes: 1-3 digits (4-digit years spared) followed
# by an Uppercase start ("2 Any" strips; "2 million" survives), bullets, and Q:/A: tags.
_LIST_ARTIFACT = re.compile(
    r"^\s*(?:\[?\d{1,3}\]?[\.\):]?\s+(?=[A-Z])|[•▪·*]\s+|-\s+(?=[A-Z])|[QqAa](?:ns(?:wer)?)?:\s+)")

_META_SENT = re.compile(
    r"^(?:the|these|this)\s+(?:passages?|text|document|section|excerpt|article)s?\s+"
    r"(?:discuss|describe|mention|state|explain|provide|contain|cover|talk)", re.IGNORECASE)


def _stem(t: str) -> str:
    """Cheap plural/suffix fold so 'corporation' matches 'corporations' (scoring only)."""
    if len(t) > 4 and t.endswith("es"):
        return t[:-2]
    if len(t) > 3 and t.endswith("s"):
        return t[:-1]
    return t


def _content_ex(text: str) -> set:
    return {_stem(t) for t in _TOK.findall(text.lower())
            if len(t) > 1 and t not in _STOPWORDS}


class ExtractiveGenerator:
    """Grounded MULTI-SENTENCE extraction: a professional answer, not a one-line snippet.

    Composition: (1) score every sentence in the top passages — stopword-filtered overlap,
    hard penalties for interrogative/heading echoes and meta-framing, a definitional bonus,
    reranker-order decay; (2) LEAD with the best answer sentence; (3) greedily add up to
    `max_support` supporting sentences by MMR (marginal relevance × novelty vs already-chosen
    text) so the answer gains CONTEXT without repetition or padding; (4) cite each segment's
    source passage inline as [n]. Every character remains verbatim source text — the
    zero-fabrication posture is structural, and it stays sub-2ms CPU."""

    def __init__(self, max_len_chars: int = 1300, max_support: int = 5,
                 support_floor: float = 0.26, novelty_lambda: float = 0.55):
        # defaults per user-locked "descriptive 4-6 sentences": definition + mechanism +
        # context + qualification, all verbatim+cited, still <2ms to compose
        self.max_len_chars = max_len_chars
        self.max_support = max_support
        self.support_floor = support_floor      # support must score ≥ floor × lead score
        self.novelty_lambda = novelty_lambda    # MMR balance: relevance vs novelty

    @staticmethod
    def _is_question(sentence: str) -> bool:
        s = sentence.strip()
        if s.endswith("?") or s.endswith("?।") or s.endswith("؟"):
            return True
        first = (s.split() or [""])[0].lower().strip(",.:;\"'")
        return first in _INTERROGATIVE and len(s.split()) <= 12

    def _candidates(self, query: str, contexts: Sequence) -> list:
        """Score all sentences -> [(score, chunk_idx, sent_idx, sentence)], best first."""
        q = _content_ex(query)
        if not q:
            q = _content(query)
        m = _DEF_QUERY.match(query.lower())
        subject = m.group(1).strip() if m else None
        sub_re = (re.compile(rf"^(?:a |an |the )?{re.escape(subject)}e?s? "
                             rf"(?:is|are|refers to|means|was|were)\b")
                  if subject else None)
        # asker-language preference: for Indic queries, same-language passages win when
        # comparable — a Hindi question should get a Hindi answer whenever the corpus has one
        # (cross-lingual fallback still answers when only another language covers it)
        from router import detect_language_ex
        qlang = detect_language_ex(query).lang   # LID: romanized-hi counts as hi here, so
        # a Hinglish asker also gets Hindi-evidence preference
        out = []
        for i, c in enumerate(contexts[:5]):
            same_lang = (qlang != "en"
                         and (getattr(c, "payload", None) or {}).get("language") == qlang)
            for j, s in enumerate(_split_sents(c.text)):
                s = _LIST_ARTIFACT.sub("", s).strip()
                ctoks = _content_ex(s)
                if not ctoks:
                    continue
                score = len(q & ctoks) / max(1, len(q))
                if self._is_question(s):
                    score *= 0.1
                if _META_SENT.match(s):
                    score *= 0.2
                if len(ctoks) < 4:
                    score *= 0.4
                if sub_re is not None and sub_re.match(s.lower()):
                    score *= 2.5
                if same_lang:
                    score *= 1.45          # answer in the asker's language when possible
                score *= 1.0 / (1.0 + 0.15 * i)
                out.append((score, i, j, s))
        out.sort(key=lambda t: (-t[0], t[1], t[2]))
        return out

    def generate(self, query: str, contexts: Sequence) -> GenOutput:
        cands = self._candidates(query, contexts)
        if not cands or cands[0][0] <= 0:
            # NO evidence sentence matches: return EMPTY so the grounding gate converts this
            # to a localized abstain — a canned English IDK here once leaked through grounding
            # as a fake ANSWER (overlap on its own filler words)
            return GenOutput(text="", cited_chunk_ids=[])

        lead_score, lead_i, lead_j, lead = cands[0]
        picked = [(lead_i, lead_j, lead)]
        chosen_toks = set(_content_ex(lead))

        # elaboration prior: sentences that FOLLOW the lead in its own passage are natural
        # supporting context even when they don't repeat the query's tokens
        boosted = []
        for score, i, j, s in cands[1:]:
            if i == lead_i and j == lead_j + 1:
                score = max(score, lead_score * 0.55)
            elif i == lead_i and j == lead_j + 2:
                score = max(score, lead_score * 0.40)
            elif i != lead_i and j == 0 and score > 0:
                score = max(score, lead_score * 0.32)   # other passages' openers: mild prior
            boosted.append((score, i, j, s))
        boosted.sort(key=lambda t: (-t[0], t[1], t[2]))

        # MMR support selection: relevant AND novel, never below the quality floor
        for score, i, j, s in boosted:
            if len(picked) > self.max_support:
                break
            if score < self.support_floor * lead_score:
                break
            stoks = _content_ex(s)
            if not stoks:
                continue
            novelty = 1.0 - (len(stoks & chosen_toks) / len(stoks))
            if (self.novelty_lambda * (score / lead_score)
                    + (1 - self.novelty_lambda) * novelty) < 0.55 or novelty < 0.35:
                continue
            picked.append((i, j, s))
            chosen_toks |= stoks

        # order: lead first, then supports by (chunk rank, position); cite per segment,
        # merging adjacent same-chunk sentences under one citation
        supports = sorted(picked[1:], key=lambda t: (t[0], t[1]))
        segments, cited = [(lead_i, lead)], [lead_i]
        for i, _, s in supports:
            if segments[-1][0] == i:
                segments[-1] = (i, segments[-1][1] + " " + s)
            else:
                segments.append((i, s))
                if i not in cited:
                    cited.append(i)

        text, used = "", []
        for i, seg in segments:
            piece = f"{seg} [{i + 1}]"
            if not text and len(piece) > self.max_len_chars:
                # an oversized LEAD (e.g. an unsplit danda-less chunk) must keep its
                # citation marker — a blind tail-slice used to cut "[n]" off entirely
                seg = seg[:self.max_len_chars - 8].rstrip()
                piece = f"{seg} [{i + 1}]"
            if text and len(text) + len(piece) + 1 > self.max_len_chars:
                break
            text = f"{text} {piece}".strip()
            if i not in used:
                used.append(i)
        return GenOutput(text=text[:self.max_len_chars].rstrip(),
                         cited_chunk_ids=[contexts[i].chunk_id for i in used])


def verbatim_lead(query: str, contexts: Sequence, max_chars: int = 600,
                  max_sents: int = 3, min_score: float = None) -> GenOutput:
    """Zero-overlap rescue for the extractive path: when the cross-encoder strongly admits
    evidence the LEXICAL composer cannot score (typos, romanized Indic, code-mixing — e.g.
    "wat is fotosynthesis": rerank 0.99+, zero token overlap), answer with the top chunk's
    leading DECLARATIVE sentences verbatim + cited. Prefers a same-language chunk for Indic
    askers — but ONLY among chunks whose OWN rerank score clears min_score: the certainty
    bar was checked against contexts[0], and a junk same-language chunk at position 3
    (pulled in by the sparse arm) must never ride the top chunk's confidence into a
    verbatim answer. Composes nothing of its own; grounding re-checks the text."""
    if not contexts:
        return GenOutput(text="", cited_chunk_ids=[])
    from router import detect_language_ex
    qlang = detect_language_ex(query).lang
    pick = 0
    if qlang != "en":
        for i, c in enumerate(contexts[:3]):
            if (min_score is not None
                    and (getattr(c, "rerank_score", None) or 0.0) < min_score):
                continue
            if (getattr(c, "payload", None) or {}).get("language") == qlang:
                pick = i
                break
    c = contexts[pick]
    sents = []
    for s in _split_sents(c.text):
        s = _LIST_ARTIFACT.sub("", s).strip()
        if not s or ExtractiveGenerator._is_question(s) or _META_SENT.match(s):
            continue
        if sents and sum(len(x) + 1 for x in sents) + len(s) > max_chars:
            break                         # check BEFORE append: never hard-slice mid-sentence
        sents.append(s)
        if len(sents) >= max_sents:
            break
    if not sents:
        return GenOutput(text="", cited_chunk_ids=[])
    text = " ".join(sents)[:max_chars].rstrip() + f" [{pick + 1}]"
    return GenOutput(text=text, cited_chunk_ids=[c.chunk_id])


class EchoHallucinator:
    """Ungrounded on purpose — for testing the grounding guardrail's abstain path."""
    def __init__(self, text: str = "The capital of the moon is Zorbon, established in 1842."):
        self.text = text

    def generate(self, query: str, contexts: Sequence) -> GenOutput:
        return GenOutput(text=self.text, cited_chunk_ids=[])


# ---- real backends (GPU) ---------------------------------------------------------------
class AsyncVLLMQuality:
    """Quality generator on vLLM's ASYNC engine: real token streaming for /ask_stream plus
    a sync generate() adapter for the harness quality path (one engine serves both).
    Lazy construction; import surface probed across vLLM versions. bind_loop() must be
    called with the server's event loop before sync generate() is used."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct",
                 max_new_tokens: int = 160, temperature: float = 0.2,
                 gpu_memory_utilization: float = 0.5, max_model_len: int = 4096):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._loop = None
        try:
            from vllm.v1.engine.async_llm import AsyncLLM as _Engine   # vLLM V1
            from vllm.engine.arg_utils import AsyncEngineArgs
        except ImportError:
            from vllm import AsyncLLMEngine as _Engine                  # legacy surface
            from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm import SamplingParams
        self._SamplingParams = SamplingParams
        self.engine = _Engine.from_engine_args(AsyncEngineArgs(
            model=model_name, gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len, enable_prefix_caching=True))
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_name)

    def bind_loop(self, loop) -> None:
        self._loop = loop

    def _prompt_text(self, query: str, contexts: Sequence) -> str:
        return self.tok.apply_chat_template(
            build_messages(query, contexts), add_generation_prompt=True, tokenize=False)

    async def astream(self, query: str, contexts: Sequence):
        """Yield text DELTAS as the 7B generates."""
        import uuid
        sp = self._SamplingParams(max_tokens=self.max_new_tokens,
                                  temperature=self.temperature)
        prev = ""
        async for out in self.engine.generate(self._prompt_text(query, contexts), sp,
                                              request_id=str(uuid.uuid4())):
            cur = out.outputs[0].text
            if len(cur) > len(prev):
                yield cur[len(prev):]
                prev = cur

    async def _collect(self, query: str, contexts: Sequence) -> str:
        return "".join([d async for d in self.astream(query, contexts)])

    def generate(self, query: str, contexts: Sequence) -> GenOutput:
        """Sync adapter for the harness quality path (runs on a stage thread)."""
        import asyncio
        if self._loop is None:
            raise RuntimeError("AsyncVLLMQuality.bind_loop() not called")
        fut = asyncio.run_coroutine_threadsafe(self._collect(query, contexts), self._loop)
        try:
            text = fut.result(timeout=30).strip()
        except Exception:
            fut.cancel()   # abandon the engine request too — a timed-out generation must
            raise          # not keep burning the GPU shared with BGE-M3 + the reranker
        return GenOutput(text=text, cited_chunk_ids=parse_citations(text, contexts))


class VLLMGenerator:
    """Quality-mode generator on vLLM: continuous batching + prefix caching of the stable
    SYSTEM_V2 prefix + fast prefill. On a 48GB A6000, Qwen2.5-7B-Instruct fits at ~15GB
    (gpu_memory_utilization caps it so BGE-M3 + reranker share the card). Lazy import."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct",
                 max_new_tokens: int = 160, temperature: float = 0.2,
                 gpu_memory_utilization: float = 0.5, max_model_len: int = 4096):
        from vllm import LLM, SamplingParams
        self._SamplingParams = SamplingParams
        self.llm = LLM(model=model_name, gpu_memory_utilization=gpu_memory_utilization,
                       max_model_len=max_model_len, enable_prefix_caching=True)
        self.tok = self.llm.get_tokenizer()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def generate(self, query: str, contexts: Sequence) -> GenOutput:
        text_in = self.tok.apply_chat_template(
            build_messages(query, contexts), add_generation_prompt=True, tokenize=False)
        sp = self._SamplingParams(max_tokens=self.max_new_tokens,
                                  temperature=self.temperature)
        out = self.llm.generate([text_in], sp)
        text = out[0].outputs[0].text.strip()
        return GenOutput(text=text, cited_chunk_ids=parse_citations(text, contexts))


class LocalLLMGenerator:
    """Qwen2.5-3B / Llama-3.2-3B-Instruct via transformers (or vLLM). Lazy import."""
    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct",
                 max_new_tokens: int = 160, temperature: float = 0.2):
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto")
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def generate(self, query: str, contexts: Sequence) -> GenOutput:
        # two-step template->text->tokenize: apply_chat_template(return_tensors=...) broke on
        # transformers 5.x; this pattern (same as raptor.load_hf_summarizer) works everywhere
        text_in = self.tok.apply_chat_template(
            build_messages(query, contexts), add_generation_prompt=True, tokenize=False)
        enc = self.tok(text_in, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **enc, max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0, temperature=max(self.temperature, 1e-4),
            pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        text = self.tok.decode(out[0][enc["input_ids"].shape[1]:],
                               skip_special_tokens=True).strip()
        return GenOutput(text=text, cited_chunk_ids=parse_citations(text, contexts))
