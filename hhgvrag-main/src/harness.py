"""
harness.py — the structured orchestrator (requirement #5). NOT prompt-in/text-out.

Stages run as timed, retried, DEADLINE-ENFORCED, fallback-guarded tool-calls with typed I/O;
every run yields a `QueryTrace` (per-stage latency + guardrail decisions) that feeds the
P50/P70/P100 analytics. Decision flow: safety → retrieve → OOD gate → generate (+fallback) →
grounding → answer|abstain. Every entrypoint returns a typed RAGResponse — it never raises.
"""
from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from typing import Optional

from generation import GenOutput, verbatim_lead
from guardrails import grounding_check, ood_gate
from retrieval import expand_to_leaves, is_summary_payload
from router import detect_language, detect_language_ex, is_romanized_indic
from schemas import (Citation, Decision, OODResult, Query, QueryTrace, RAGResponse,
                     StageTiming)

ABSTAIN_MSG = "I don't have grounded information on that in my knowledge base."
# absolute cross-encoder bar for the zero-overlap verbatim rescue (10b): deliberately NOT
# the per-tier rerank_strong — composing from evidence the query shares no tokens with is
# only safe at certainty (BGE typo cases: 0.99+), never at a collapsed fixture bar
_RESCUE_CERTAINTY = 0.90
UNSAFE_MSG = "I can't help with that request."
ERROR_MSG = "Something went wrong while answering; please try again."

# fixed system messages LOCALIZED to the asker's language — an abstain in the wrong language
# is still a wrong-language answer. Keyed by router script-detection; 'as' shares Bengali
# script so it resolves to 'bn' strings; unknown -> English.
_L10N = {
    "hi": {"abstain": "मेरे ज्ञान आधार में इस विषय पर प्रमाणित जानकारी उपलब्ध नहीं है।",
           "unsafe": "मैं इस अनुरोध में सहायता नहीं कर सकता।",
           "error": "कुछ त्रुटि हुई; कृपया पुनः प्रयास करें।",
           "greet": "नमस्ते! मुझसे प्रश्न पूछिए — मैं अपने ज्ञान आधार से स्रोतों सहित उत्तर दूँगा।"},
    "bn": {"abstain": "আমার জ্ঞানভাণ্ডারে এই বিষয়ে প্রামাণিক তথ্য নেই।",
           "unsafe": "আমি এই অনুরোধে সাহায্য করতে পারি না।",
           "error": "কিছু ত্রুটি ঘটেছে; অনুগ্রহ করে আবার চেষ্টা করুন।",
           "greet": "নমস্কার! আমাকে প্রশ্ন করুন — আমি উৎসসহ উত্তর দেব।"},
    "ta": {"abstain": "எனது அறிவுத் தளத்தில் இது குறித்த உறுதிப்படுத்தப்பட்ட தகவல் இல்லை.",
           "unsafe": "இந்தக் கோரிக்கைக்கு என்னால் உதவ முடியாது.",
           "error": "பிழை ஏற்பட்டது; மீண்டும் முயற்சிக்கவும்.",
           "greet": "வணக்கம்! கேள்வி கேளுங்கள் — ஆதாரங்களுடன் பதில் தருகிறேன்."},
    "te": {"abstain": "నా విజ్ఞాన స్థావరంలో దీనిపై ధృవీకరించబడిన సమాచారం లేదు.",
           "unsafe": "ఈ అభ్యర్థనలో నేను సహాయం చేయలేను.",
           "error": "లోపం సంభవించింది; దయచేసి మళ్లీ ప్రయత్నించండి.",
           "greet": "నమస్తే! ప్రశ్న అడగండి — మూలాలతో సమాధానం ఇస్తాను."},
    "mr": {"abstain": "माझ्या ज्ञानकोशात या विषयावर प्रमाणित माहिती उपलब्ध नाही.",
           "unsafe": "मी या विनंतीस मदत करू शकत नाही.",
           "error": "त्रुटी आली; कृपया पुन्हा प्रयत्न करा.",
           "greet": "नमस्कार! प्रश्न विचारा — मी स्रोतांसह उत्तर देईन."},
    "gu": {"abstain": "મારા જ્ઞાનકોશમાં આ વિષય પર પ્રમાણિત માહિતી ઉપલબ્ધ નથી.",
           "unsafe": "હું આ વિનંતીમાં મદદ કરી શકતો નથી.",
           "error": "ભૂલ થઈ; કૃપા કરીને ફરી પ્રયાસ કરો."},
    "kn": {"abstain": "ನನ್ನ ಜ್ಞಾನಕೋಶದಲ್ಲಿ ಈ ವಿಷಯದ ದೃಢೀಕೃತ ಮಾಹಿತಿ ಇಲ್ಲ.",
           "unsafe": "ಈ ಕೋರಿಕೆಗೆ ನಾನು ಸಹಾಯ ಮಾಡಲಾರೆ.",
           "error": "ದೋಷ ಸಂಭವಿಸಿದೆ; ದಯವಿಟ್ಟು ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ."},
    "ml": {"abstain": "എന്റെ വിജ്ഞാനശേഖരത്തിൽ ഇതിനെക്കുറിച്ച് സ്ഥിരീകരിച്ച വിവരങ്ങളില്ല.",
           "unsafe": "ഈ അഭ്യർത്ഥനയിൽ എനിക്ക് സഹായിക്കാനാവില്ല.",
           "error": "പിശക് സംഭവിച്ചു; ദയവായി വീണ്ടും ശ്രമിക്കുക."},
    "pa": {"abstain": "ਮੇਰੇ ਗਿਆਨ ਭੰਡਾਰ ਵਿੱਚ ਇਸ ਵਿਸ਼ੇ ਬਾਰੇ ਪ੍ਰਮਾਣਿਤ ਜਾਣਕਾਰੀ ਨਹੀਂ ਹੈ।",
           "unsafe": "ਮੈਂ ਇਸ ਬੇਨਤੀ ਵਿੱਚ ਮਦਦ ਨਹੀਂ ਕਰ ਸਕਦਾ।",
           "error": "ਗਲਤੀ ਹੋਈ; ਕਿਰਪਾ ਕਰਕੇ ਮੁੜ ਕੋਸ਼ਿਸ਼ ਕਰੋ।"},
    "or": {"abstain": "ମୋର ଜ୍ଞାନଭଣ୍ଡାରରେ ଏ ବିଷୟରେ ପ୍ରମାଣିତ ସୂଚନା ନାହିଁ।",
           "unsafe": "ମୁଁ ଏହି ଅନୁରୋଧରେ ସାହାଯ୍ୟ କରିପାରିବି ନାହିଁ।",
           "error": "ତ୍ରୁଟି ଘଟିଛି; ଦୟାକରି ପୁଣି ଚେଷ୍ଟା କରନ୍ତୁ।"},
    "ur": {"abstain": "میرے علمی ذخیرے میں اس موضوع پر مصدقہ معلومات موجود نہیں۔",
           "unsafe": "میں اس درخواست میں مدد نہیں کر سکتا۔",
           "error": "خرابی پیش آئی؛ براہِ کرم دوبارہ کوشش کریں۔"},
    "ne": {"abstain": "मेरो ज्ञान आधारमा यस विषयमा प्रमाणित जानकारी उपलब्ध छैन।",
           "unsafe": "म यो अनुरोधमा सहयोग गर्न सक्दिनँ।",
           "error": "त्रुटि भयो; कृपया फेरि प्रयास गर्नुहोस्।"},
    # LID v2 (2026-08-17): romanized variants — an asker who TYPES in Latin script gets
    # fixed messages in the same representation. Keys are "<lang>-r"; _localized falls back
    # to the native table, then to English.
    "hi-r": {"abstain": "Mere gyaan aadhaar mein is vishay par pramaanit jaankari uplabdh "
                        "nahin hai.",
             "unsafe": "Main is anurodh mein sahayata nahin kar sakta.",
             "error": "Kuchh truti hui; kripya punah prayas karein.",
             "greet": "Namaste! Mujhse prashn poochhiye — main apne gyaan aadhaar se "
                      "sroton sahit uttar doonga."},
    "bn-r": {"abstain": "Amar gyanbhandare ei bishoye pramanik tothyo nei.",
             "unsafe": "Ami ei anurodhe sahajyo korte pari na.",
             "error": "Kichhu truti ghotechhe; onugroho kore abar cheshta korun.",
             "greet": "Nomoskar! Amake proshno korun — ami utsoshoho uttor debo."},
    "ta-r": {"abstain": "Enadhu arivuth thalathil idhu kuriththa urudhippaduththappatta "
                        "thagaval illai.",
             "unsafe": "Indha korikkaiku ennal udhava mudiyadhu.",
             "error": "Pizhai erpattadhu; meendum muyarchikkavum.",
             "greet": "Vanakkam! Kelvi kelunga — aadharangaludan badhil tharugiren."},
}


def _localized(kind: str, lang: Optional[str], fallback: str) -> str:
    key = lang or ""
    table = _L10N.get(key)
    if table and kind in table:
        return table[kind]
    if "-" in key:                       # "hi-r" -> native "hi" before English fallback
        base = _L10N.get(key.split("-", 1)[0])
        if base and kind in base:
            return base[kind]
    return fallback

# for the LLM quality path: an explicit reply-language instruction beats hoping the model
# follows a system rule — with mixed-language context it mirrors the passages otherwise
_LANG_NAMES = {"en": "English", "hi": "Hindi", "bn": "Bengali", "ta": "Tamil",
               "te": "Telugu", "mr": "Marathi", "gu": "Gujarati", "kn": "Kannada",
               "ml": "Malayalam", "pa": "Punjabi", "or": "Odia", "ur": "Urdu"}

_CONTENT = re.compile(r"\w", re.UNICODE)


def _has_content(text: str) -> bool:
    """True if the query has at least one word character — guards the zero-vector edge case."""
    return bool(text and _CONTENT.search(text))


class RAGHarness:
    # deterministic failures (config/programming errors) — retrying them just wastes budget
    _DETERMINISTIC = (ValueError, KeyError, TypeError, AttributeError, IndexError)

    def __init__(self, *, embedder, retriever, generator, safety, collection: str,
                 ood_threshold: float = 0.32, grounding_min_support: float = 0.5,
                 budget_ms: int = 200, stage_timeouts: Optional[dict] = None,
                 top_k: int = 8, stt=None, nli=None,
                 normalizer=None, router=None, reranker=None,
                 rerank_min: float = 0.2, rerank_candidates: int = 40,
                 rerank_veto: float = 0.0, rerank_strong: float = 0.80,
                 response_cache=None, session_ctx=None,
                 quality_generator=None, pageindex=None, transliterator=None):
        self.embedder = embedder
        self.retriever = retriever
        self.generator = generator
        self.safety = safety
        self.collection = collection
        self.ood_threshold = ood_threshold
        self.grounding_min_support = grounding_min_support
        self.budget_ms = budget_ms
        self.stage_timeouts = stage_timeouts or {}
        self.top_k = top_k
        self.stt = stt
        self.nli = nli
        self.normalizer = normalizer      # R3 noisy-ASR layer (optional)
        self.router = router              # R2 query router (optional)
        self.reranker = reranker          # R1 cross-encoder (optional)
        self.rerank_min = rerank_min
        self.rerank_candidates = rerank_candidates
        self.rerank_veto = rerank_veto
        self.rerank_strong = rerank_strong   # lexical-rescue bar when coverage is partial
        self.response_cache = response_cache  # R5a semantic response cache (optional)
        self.session_ctx = session_ctx        # R5c session context (optional)
        self.transliterator = transliterator  # typed romanized-Indic -> native (optional)
        self.quality_generator = quality_generator  # H8: LLM for quality mode (optional)
        self.pageindex = pageindex        # vectorless tree retrieval (optional, per-query toggle)
        # 16 workers: stage deadlines start at SUBMIT, so under concurrent requests a small pool
        # makes queue-wait eat the deadline and fire spurious timeouts. Threads are cheap; the
        # GPU serializes heavy work anyway. Orphaned (timed-out) stages also stop clogging slots.
        self._pool = ThreadPoolExecutor(max_workers=16, thread_name_prefix="hhgvrag-stage")
        # ADMISSION CONTROL (stress-evidenced 2026-08-16): ≥30 simultaneous cold requests
        # pile every stage behind the pool/GPU and 100% hit deadline->ERROR. Capping in-flight
        # requests lets admitted ones finish fast (Little's law); the rest fail FAST with an
        # honest over-capacity ERROR instead of burning a worker to die at the deadline.
        import threading as _th
        self._admit = _th.BoundedSemaphore(8)
        # voice admission, separate from the request semaphore: STT runs BEFORE answer()'s
        # admission, so without this a Sarvam brownout lets unlimited voice requests pile up
        # (each pinning a server threadpool token) and text traffic goes down with them
        self._voice_admit = _th.BoundedSemaphore(6)
        # typed-romanized transliteration is a Sarvam network call — cap concurrency so a
        # brownout can't pin pool workers/admission slots and starve text queries (the exact
        # cascade STT was moved off the pool to avoid); non-blocking acquire = circuit breaker
        self._xlit_admit = _th.BoundedSemaphore(4)

    def verify_ready(self) -> None:
        """Fail fast (at startup, not per-request) if the live collection was never built."""
        client = getattr(self.retriever, "client", None)
        if client is not None and hasattr(client, "collection_exists"):
            if not client.collection_exists(self.collection):
                raise RuntimeError(
                    f"collection '{self.collection}' not found — run build_index first")

    # --- staged tool-call: time + retry(backoff) + ENFORCED deadline ---------------------
    def _run_with_deadline(self, fn, timeout_ms):
        """Run fn under a hard wall-clock deadline; TimeoutError if it overruns."""
        if not timeout_ms:
            return fn()
        fut = self._pool.submit(fn)
        try:
            return fut.result(timeout=timeout_ms / 1000.0)
        except FuturesTimeout:
            # cancel() kills QUEUED-not-started fns outright (a started thread still runs
            # to completion — Python can't kill it) — without this, every timeout under
            # load left a zombie in the 16-worker pool and queue-wait ate later deadlines
            fut.cancel()
            raise TimeoutError(f"stage exceeded {timeout_ms}ms")

    # a stage never gets less than this, even with the budget exhausted — finishing a query
    # ~15ms over budget beats converting it to an ERROR at the finish line
    _STAGE_FLOOR_MS = 15.0

    def _stage(self, trace: QueryTrace, name: str, fn, retries: int = 1, deadline=None):
        last = None
        for attempt in range(retries + 1):
            timeout = self.stage_timeouts.get(name)
            if deadline is not None:
                # end-to-end budget enforcement: no stage may outlive the request deadline,
                # INCLUDING stages with no configured per-stage timeout (embed, cache, route…)
                remaining = max((deadline - time.perf_counter()) * 1000.0, self._STAGE_FLOOR_MS)
                timeout = min(timeout, remaining) if timeout else remaining
            t0 = time.perf_counter()
            try:
                res = self._run_with_deadline(fn, timeout)
                ms = (time.perf_counter() - t0) * 1000
                note = "over-budget" if (timeout and ms > timeout) else None
                trace.add(StageTiming(stage=name, ms=round(ms, 3), ok=True,
                                      retries=attempt, note=note))
                return res
            except Exception as e:  # noqa: BLE001 — bounded retry then propagate
                last = e
                ms = (time.perf_counter() - t0) * 1000
                is_timeout = isinstance(e, TimeoutError)
                # record EVERY attempt so the trace/latency don't undercount retries
                trace.add(StageTiming(stage=name, ms=round(ms, 3), ok=False, retries=attempt,
                                      note=("timeout" if is_timeout else str(e)[:80])))
                # retry only transient failures — never a timeout (already blew budget) or a
                # deterministic config/programming error (would just fail again)
                if attempt < retries and not is_timeout and not isinstance(e, self._DETERMINISTIC):
                    time.sleep(0.004 * (attempt + 1))
                    continue
                raise last

    def answer(self, query: Query, trace: Optional[QueryTrace] = None) -> RAGResponse:
        trace = trace or QueryTrace(query=query.text, decision=Decision.ANSWER)
        # admission: wait at most half the budget for a slot — beyond that the request could
        # not finish in budget anyway; fail FAST and honest instead of thrashing the pool
        # (stress-evidenced: 30 simultaneous cold requests previously hit 100% deadline-ERROR)
        # wait up to ONE full budget for a slot: the admitted request's own deadline starts
        # after admission, so served-latency is unaffected; burst throughput ~doubles
        # (live 30-burst: 7 served @100ms wait; ~2 waves more expected @200ms)
        if not self._admit.acquire(timeout=self.budget_ms / 1000.0):
            trace.decision = Decision.ERROR
            trace.total_ms = round(float(self.budget_ms), 3)
            return RAGResponse(answer=_localized("error", detect_language(query.text), ERROR_MSG),
                               decision=Decision.ERROR, abstained=True,
                               reason="over capacity — please retry", trace=trace)
        try:
            return self._answer_admitted(query, trace)
        finally:
            self._admit.release()

    def _answer_admitted(self, query: Query, trace: QueryTrace) -> RAGResponse:
        # the request-level deadline: budget starts NOW (post-STT), and every budget-scoped
        # stage below is capped at the remaining budget. Quality-mode generation is exempt —
        # it is the explicitly out-of-budget path, separately timed and honestly reported.
        deadline = time.perf_counter() + (self.budget_ms / 1000.0)
        lid = detect_language_ex(query.text)  # LID v2: language + script + romanized flag
        qlang = f"{lid.lang}-r" if lid.romanized else lid.lang   # localization key — every
        # fixed message matches the asker's language AND representation (Latin vs native)
        try:
            # 1) input safety
            safety = self._stage(trace, "safety", lambda: self.safety.check(query.text),
                                 deadline=deadline)
            trace.safety = safety
            if not safety.safe:
                return self._finish(trace, _localized("unsafe", qlang, UNSAFE_MSG), Decision.REFUSE_UNSAFE, abstain=True)
            if not _has_content(query.text):
                trace.ood = OODResult(in_domain=False, top_score=0.0, threshold=self.ood_threshold)
                return self._finish(trace, _localized("abstain", qlang, ABSTAIN_MSG), Decision.ABSTAIN_OOD, abstain=True)

            # 2) normalize / repair the (possibly noisy) transcript — R3.
            #    ENHANCER: a normalizer failure degrades to the raw query, never ERROR.
            base_k = query.top_k or self.top_k
            clean, eff_k, ood_relax = query.text, base_k, 0.0
            if self.normalizer is not None:
                try:
                    nq = self._stage(trace, "normalize", lambda: self.normalizer.normalize(
                        query.text, confidence=query.asr_confidence, base_top_k=base_k),
                        deadline=deadline)
                    clean, eff_k, ood_relax = nq.text, nq.top_k, nq.ood_relax
                    trace.normalized = clean if clean != query.text else None
                except Exception:
                    clean, eff_k, ood_relax = query.text, base_k, 0.0
                if not _has_content(clean):
                    trace.ood = OODResult(in_domain=False, top_score=0.0, threshold=self.ood_threshold)
                    return self._finish(trace, _localized("abstain", qlang, ABSTAIN_MSG), Decision.ABSTAIN_OOD, abstain=True)

            # 3) route: language + intent — R2 (greeting/meta get an intentional reply, not RAG).
            #    ENHANCER: a router failure degrades to plain qa routing, never ERROR.
            filters, collection = None, self.collection
            if self.router is not None:
                try:
                    route = self._stage(trace, "route", lambda: self.router.route(clean),
                                        deadline=deadline)
                    trace.route = route
                    if route.intent in ("chitchat", "meta"):
                        return self._finish(trace, self._small_talk(route.intent, qlang),
                                            Decision.SMALL_TALK, abstain=False)
                    filters = route.filters or None
                    collection = route.collection or self.collection
                except Exception:
                    filters, collection = None, self.collection

            # 3b) typed romanized-Indic -> native script (UPSTREAM normalization, like STT).
            #     Replaces the slow, weaker romanized dual-arm with the fast native path.
            #     Placed AFTER route so romanized greetings still reach small-talk. Gated by the
            #     STRICT is_romanized_indic (low FP — English/brand queries are NOT flagged, so
            #     we never transliterate English into Devanagari garbage; the gate is the ONLY
            #     defense against that mode, since garbage Devanagari passes every output check).
            #     retrieval_romanized drives the cache key + dual-arm skip; lid.romanized stays
            #     for messages (qlang) + quality-mode reply-script.
            retrieval_romanized = lid.romanized
            is_rom, rom_lang = is_romanized_indic(clean)
            if (is_rom and self.transliterator is not None
                    and self._xlit_admit.acquire(blocking=False)):
                try:
                    _t0 = time.perf_counter()
                    native = None
                    try:
                        # deadline=None: this is NOT the 200ms retrieval budget (~370ms Sarvam
                        # call would always time out) — it runs off-budget like STT.
                        native = self._stage(
                            trace, "transliterate",
                            lambda: self.transliterator.transliterate(clean, rom_lang),
                            deadline=None)
                    except Exception:
                        native = None            # fail -> keep romanized + dual-arm (no regression)
                    finally:
                        # forgive EXACTLY the transliteration wall-time so retrieval keeps its
                        # full budget; in finally so a translit timeout is forgiven too, else
                        # downstream stages face an already-blown deadline -> spurious ERROR.
                        deadline += time.perf_counter() - _t0
                    # commit to native-only ONLY if the result is genuinely native (re-LID
                    # rejects still-romanized/partial output -> those keep the resilient dual-arm)
                    if native and native != clean and not detect_language_ex(native).romanized:
                        # a romanized-Latin unsafe query evades the native-script safety lexicon;
                        # the transliterated form hits it — re-check before committing
                        sfx = self.safety.check(native)
                        if not sfx.safe:
                            trace.safety = sfx
                            return self._finish(trace, _localized("unsafe", qlang, UNSAFE_MSG),
                                                Decision.REFUSE_UNSAFE, abstain=True)
                        clean = native
                        retrieval_romanized = False
                finally:
                    self._xlit_admit.release()

            # 4) embed the query once (explicit — for cache check + retrieval) — R5a
            query_embed = self._stage(trace, "embed",
                                      lambda: self.embedder.embed_query(clean),
                                      deadline=deadline)

            # 5) session expansion decides CACHEABILITY first — R5c. A session-expanded
            #    query's answer depends on conversation history: caching it under the bare
            #    embedding would serve session A's contextual answer to session B asking the
            #    same words. Context-dependent turns bypass the cache entirely (get AND put).
            use_quality = bool(query.quality_mode and self.quality_generator)
            trace.quality_mode = use_quality
            session_id = query.session_id
            expanded = clean
            if self.session_ctx is not None and session_id:
                expanded = self.session_ctx.rewrite(session_id, clean)
            ctx_changed = expanded != clean
            mode = "pageindex" if (query.retrieval_mode == "pageindex"
                                   and self.pageindex is not None) else "hybrid"
            trace.retrieval_mode = mode
            cacheable = (self.response_cache is not None and not use_quality
                         and not ctx_changed)

            # 6) semantic response cache check — R5a (timed: the scan is real latency).
            # Key carries (lang, romanized): BGE-M3 aligns cross-lingual paraphrases by
            # DESIGN, so "what is diabetes" and "मधुमेह क्या है" can clear the cosine bar —
            # without the language dimension the Hindi asker gets the cached English answer.
            # retrieval_romanized (not lid.romanized): after transliteration the query IS
            # native, so a romanized asker and a native asker of the same question share the
            # entry (same native embedding + key), which is correct and doubles cache reuse.
            cache_key = (eff_k, mode, lid.lang, retrieval_romanized)
            if cacheable:
                try:
                    cached = self._stage(trace, "cache",
                                         lambda: self.response_cache.get(
                                             query_embed.dense, key=cache_key),
                                         deadline=deadline)
                except Exception:
                    cached = None
                if cached is not None:
                    trace.cache_hit = True
                    if self.session_ctx is not None and session_id:
                        self.session_ctx.add_turn(session_id, clean, cached.answer)
                    return self._finish(trace, cached.answer, cached.decision,
                                        citations=cached.citations)

            # 7) context re-embed for retrieval (timed; degrades to the plain embedding)
            retrieval_embed = query_embed
            if ctx_changed:
                try:
                    retrieval_embed = self._stage(
                        trace, "embed_ctx", lambda: self.embedder.embed_query(expanded),
                        deadline=deadline)
                except Exception:
                    retrieval_embed = query_embed

            # 7) retrieve (wide candidate pool when a reranker will trim it).
            #    hybrid = Qdrant dense+sparse (default) | pageindex = vectorless tree descent
            pool = max(eff_k, self.rerank_candidates) if self.reranker is not None else eff_k
            if mode == "pageindex":
                retrieved = self._stage(trace, "retrieve",
                                        lambda: self.pageindex.search(expanded, top_k=pool),
                                        deadline=deadline)
            else:
                retrieved = self._stage(trace, "retrieve", lambda: self.retriever.search(
                    collection, retrieval_embed, top_k=pool, filters=filters),
                    deadline=deadline)

            # 7a-roman) romanized dual-arm (LID v2): retrieve BOTH the original Hinglish
            # and its content-word residual, merge per-chunk on best dense score — the gate
            # then judges the best evidence either arm found. Thresholds untouched; a
            # failure in the second arm degrades silently to single-arm.
            q_for_rank = clean
            # only when transliteration did NOT happen (transliterator absent, or it failed/
            # no-op'd) — a successful transliteration already put us on the native path.
            if retrieval_romanized and mode == "hybrid":
                from router import romanized_residual
                residual = romanized_residual(expanded, lid.lang)
                if residual:
                    q_for_rank = residual
                    try:
                        res_embed = self._stage(trace, "embed_roman",
                                                lambda: self.embedder.embed_query(residual),
                                                deadline=deadline)
                        arm2 = self._stage(trace, "retrieve_roman",
                                           lambda: self.retriever.search(
                                               collection, res_embed, top_k=pool,
                                               filters=filters), deadline=deadline)
                        best: dict = {}
                        for r in list(retrieved) + list(arm2):
                            prev = best.get(r.chunk_id)
                            if prev is None or (r.dense_score or r.score or 0.0) > \
                                    (prev.dense_score or prev.score or 0.0):
                                best[r.chunk_id] = r
                        retrieved = sorted(
                            best.values(),
                            key=lambda r: (r.dense_score or r.score or 0.0),
                            reverse=True)[:pool]
                    except Exception:
                        pass

            # 7b) P0.3 leaf-evidence expansion: summaries NAVIGATE, leaves are EVIDENCE.
            #     Summary hits expand to leaf descendants; everything downstream (rerank,
            #     gate, generate, ground, cite) sees leaves only. Empty expansion ⇒ abstain
            #     — never fall back to summary prose as evidence.
            if any(is_summary_payload(r.payload) for r in retrieved):
                try:
                    retrieved = self._stage(
                        trace, "leaf_expand",
                        lambda: expand_to_leaves(self.retriever, collection, retrieved, pool),
                        deadline=deadline)
                except Exception:
                    # expansion failed/timed out: degrade to the directly-retrieved LEAVES
                    # (summaries stay excluded — protocol) — never ERROR the query
                    retrieved = [r for r in retrieved if not is_summary_payload(r.payload)]
                if not retrieved:
                    trace.ood = OODResult(in_domain=False, top_score=0.0,
                                          threshold=self.ood_threshold, signal="no_leaf_evidence")
                    return self._finish(trace, _localized("abstain", qlang, ABSTAIN_MSG), Decision.ABSTAIN_OOD, abstain=True)
            trace.n_retrieved = len(retrieved)
            if not retrieved:
                trace.ood = OODResult(in_domain=False, top_score=0.0,
                                      threshold=self.ood_threshold, signal="none")
                return self._finish(trace, _localized("abstain", qlang, ABSTAIN_MSG), Decision.ABSTAIN_OOD, abstain=True)

            # 8) cross-encoder rerank — R1 (reorder + calibrated answerability signal).
            #    ENHANCER: a rerank failure/timeout degrades to the fused-RRF order, never ERROR.
            if self.reranker is not None:
                try:
                    retrieved = self._stage(trace, "rerank",
                                            lambda: self.reranker.rerank(q_for_rank, retrieved, top_n=eff_k),
                                            deadline=deadline)
                    trace.rerank_top = max((r.rerank_score or 0.0) for r in retrieved) if retrieved else 0.0
                except Exception:
                    retrieved = retrieved[:eff_k]   # un-reranked, trimmed to the requested depth

            # 9) off-topic / answerability gate: dense+lexical (relaxed by low ASR confidence),
            #    OR admitted by a confident reranker
            ood = ood_gate(retrieved, max(0.0, self.ood_threshold - ood_relax),
                           query=q_for_rank)
            trace.ood = ood
            # the cross-encoder cuts BOTH ways: it can ADMIT a query the dense gate missed
            # (rescue, >= rerank_min) and VETO one the dense gate waved through when it is
            # certain nothing retrieved is relevant (< rerank_veto). Veto only applies when
            # a rerank score actually exists (a degraded rerank must not fail queries).
            # Admission policy. The calibrated PRIMARY signal is the dense τ. When a
            # rerank score exists, ANY admission beyond dense — lexical overlap OR the
            # pure rerank rescue — must be cross-encoder-confirmed at one of two bars:
            #   FULL content-token coverage -> rerank_min: the true noisy-ASR rescue
            #     ("wat is fotosynthesis" -> normalized, every content token covered).
            #   PARTIAL coverage -> rerank_strong: a real doc answers although dense
            #     missed (live: Wakanda-royals doc reranked 0.916). rerank_min alone is
            #     NOT enough — question-SHAPE junk fools it when only the entity token
            #     is missing (live: a "who is the king (of gods)" forum post scored
            #     0.525 for "king of Narnia"; westeros/Gondor junk 0.036-0.075).
            # No reranker configured (edge tier): the gate stands alone, exactly as
            # before — that tier's calibration never assumed a cross-check existed.
            # Reranker configured but FAILED this request (timeout/error): dense-arm
            # admissions stand, but a LEXICAL admission whose confirmation bar cannot be
            # applied must abstain — a transient rerank timeout must not silently flip
            # "needs 0.80 confirmation" into "answers at 0.34 token overlap".
            if self.reranker is None:
                dense_ok = ood.in_domain
                rescue = False
            elif trace.rerank_top is None:
                dense_ok = ood.in_domain and ood.signal == "dense"
                rescue = False
                if ood.signal == "lexical" and not dense_ok:
                    ood.signal = "lexical_unconfirmed"
            else:
                dense_ok = ood.in_domain and ood.signal == "dense"
                full_cover = (ood.lexical_top or 0.0) >= 0.999
                rescue = (trace.rerank_top >= self.rerank_strong
                          or (full_cover and trace.rerank_top >= self.rerank_min))
                if ood.signal == "lexical" and not (dense_ok or rescue):
                    ood.signal = "lexical_unconfirmed"
            if (dense_ok and self.rerank_veto > 0.0 and trace.rerank_top is not None
                    and trace.rerank_top < self.rerank_veto):
                dense_ok = False
                ood.signal = "rerank_veto"
            answerable = dense_ok or rescue
            if not answerable:
                return self._finish(trace, _localized("abstain", qlang, ABSTAIN_MSG), Decision.ABSTAIN_OOD, abstain=True)

            # 10) generate — extractive by default, LLM if quality mode (H8). Quality mode is
            #     the EXPLICITLY out-of-budget path: its own stage name ("generate_quality",
            #     seconds-scale timeout from config) and NO request deadline — the 150ms
            #     "generate" cap would kill every 3B call and orphan it on the GPU.
            # belt-and-braces: no summary ever reaches the generator or citations, even if
            # a future retrieval path regresses. Empty after filter ⇒ abstain (NO fallback).
            evidence = [r for r in retrieved if not is_summary_payload(r.payload)]
            if not evidence:
                return self._finish(trace, _localized("abstain", qlang, ABSTAIN_MSG), Decision.ABSTAIN_OOD, abstain=True)
            retrieved = evidence

            gen_backend = self.quality_generator if use_quality else self.generator
            try:
                if use_quality:
                    q_for_llm = clean
                    lang_name = _LANG_NAMES.get(lid.lang)
                    if lang_name and lid.romanized:
                        # the asker TYPED romanized — answer in their language, their script
                        q_for_llm = (f"{clean}\n(Reply in {lang_name} written in LATIN "
                                     f"script (romanized), matching how the user typed.)")
                    elif lang_name:
                        q_for_llm = f"{clean}\n(Reply in {lang_name}.)"
                    gen = self._stage(trace, "generate_quality",
                                      lambda: gen_backend.generate(q_for_llm, retrieved))
                else:
                    # q_for_rank == clean except for romanized queries, where the glossed
                    # content-carrier lets the extractive composer score English evidence
                    # ("lakshan" scores nothing against "symptoms"; its gloss scores)
                    gen = self._stage(trace, "generate",
                                      lambda: gen_backend.generate(q_for_rank, retrieved),
                                      deadline=deadline)
            except Exception:
                gen = self._extractive_fallback(trace, retrieved)
                trace.quality_mode = False   # honest: the answer the user gets is extractive

            # 10b) zero-overlap rescue — the admission gates said YES but the LEXICAL
            #      composer scored nothing (typo / romanized / code-mixed signature: rerank
            #      0.99+, zero token overlap). Once evidence is admitted, composition must
            #      not re-gate lexically — but composing verbatim from evidence the query
            #      shares NO tokens with demands CERTAINTY, not just admission:
            #      (a) the dense arm must have independently agreed (dense_ok — a lexical
            #          two-bar admission implies token overlap, so it can't be this case),
            #      (b) rerank_top must clear an ABSOLUTE certainty bar, not the per-tier
            #          rerank_strong (build_local_harness collapses that to 0.2 for the
            #          lexical stub, where 'strong' means nothing). BGE typo cases sit at
            #          0.99+; junk shape-matches don't. Grounding re-checks the text below.
            if (not use_quality and not (gen.text or "").strip()
                    and dense_ok
                    and trace.rerank_top is not None
                    and trace.rerank_top >= max(self.rerank_strong, _RESCUE_CERTAINTY)):
                bar = max(self.rerank_strong, _RESCUE_CERTAINTY)
                vlead = verbatim_lead(clean, retrieved, min_score=bar)
                if (vlead.text or "").strip():
                    gen = vlead
                    trace.add(StageTiming(stage="generate_fallback", ms=0.0, ok=True,
                                          note="verbatim lead (rerank-confirmed, zero overlap)"))

            # 11) grounding / anti-hallucination
            grounding = self._stage(
                trace, "ground_check",
                lambda: grounding_check(gen.text, retrieved, self.grounding_min_support,
                                        cited_ids=gen.cited_chunk_ids, nli=self.nli),
                deadline=None if use_quality else deadline)
            trace.grounding = grounding
            if not grounding.grounded:
                return self._finish(trace, _localized("abstain", qlang, ABSTAIN_MSG), Decision.ABSTAIN_UNGROUNDED, abstain=True)

            # 12) grounded answer — cache store + session record
            citations = self._citations(grounding.cited_chunk_ids or gen.cited_chunk_ids, retrieved)
            response = self._finish(trace, gen.text, Decision.ANSWER, citations=citations)
            if cacheable:
                self.response_cache.put(query_embed.dense, response, key=cache_key)
            if self.session_ctx is not None and session_id:
                self.session_ctx.add_turn(session_id, clean, response.answer)
            return response

        except Exception as e:  # noqa: BLE001 — total failure -> safe fallback
            return self._finish(trace, _localized("error", qlang, ERROR_MSG), Decision.ERROR, abstain=True,
                                reason=str(e)[:120])

    def answer_voice(self, audio: bytes, language: Optional[str] = None,
                     session_id: Optional[str] = None,
                     quality_mode: bool = False,
                     retrieval_mode: Optional[str] = None) -> RAGResponse:
        trace = QueryTrace(query="", decision=Decision.ANSWER)
        # transcript failed -> no text to detect from; the caller's language hint is all we have
        qlang = (language or "en").split("-")[0].lower()
        if self.stt is None:
            # a 500 here would make the frontend dead-mark a backend that answers text fine;
            # no-STT is a per-tier capability, not a backend failure — degrade typed + honest
            return self._finish(trace, _localized("error", qlang, ERROR_MSG), Decision.ERROR,
                                abstain=True, reason="stt: not configured on this tier")
        # STT runs DIRECTLY on the caller's thread, never on the shared 16-worker stage pool:
        # a Sarvam stall would otherwise hold pool workers for its full HTTP timeout and
        # starve every stage of every TEXT query too (verified cascade at 16 concurrent).
        # The HTTP client's own timeout is the guard; _voice_admit caps concurrent stalls.
        if not self._voice_admit.acquire(blocking=False):
            return self._finish(trace, _localized("error", qlang, ERROR_MSG), Decision.ERROR,
                                abstain=True, reason="over capacity (voice)")
        try:
            t0 = time.perf_counter()
            try:
                try:
                    t = self.stt.transcribe(audio, language)
                except Exception as e:  # noqa: BLE001
                    # retry taxonomy: ONE retry for a transient upstream 5xx only.
                    # Timeouts and 4xx (bad audio / bad key / 429) are deterministic —
                    # re-uploading the clip doubles Sarvam spend for the same failure.
                    code = getattr(getattr(e, "response", None), "status_code", 0)
                    if not (500 <= int(code or 0) < 600):
                        raise
                    t = self.stt.transcribe(audio, language)
                trace.add(StageTiming(stage="stt",
                                      ms=round((time.perf_counter() - t0) * 1000, 2),
                                      ok=True))
            except Exception as e:  # noqa: BLE001 — STT failure degrades typed, never 500
                trace.add(StageTiming(stage="stt",
                                      ms=round((time.perf_counter() - t0) * 1000, 2),
                                      ok=False, note=str(e)[:80]))
                return self._finish(trace, _localized("error", qlang, ERROR_MSG),
                                    Decision.ERROR, abstain=True,
                                    reason=f"stt: {str(e)[:110]}")
        finally:
            self._voice_admit.release()
        trace.query = t.text
        return self.answer(
            Query(text=t.text[:4000], language=t.language,
                  asr_confidence=getattr(t, "confidence", 1.0),
                  session_id=session_id, quality_mode=quality_mode,
                  retrieval_mode=retrieval_mode), trace=trace)

    # --- helpers ------------------------------------------------------------------------
    def _small_talk(self, intent: str, qlang: str = "en") -> str:
        if intent != "meta":
            g = _localized("greet", qlang, "")
            if g:
                return g
        if intent == "meta":
            return ("I'm a voice RAG assistant: ask a question and I retrieve grounded passages "
                    "from my knowledge base and answer with citations — or tell you when I don't "
                    "have the information.")
        return "Hi! Ask me a question and I'll answer from my knowledge base, with sources."

    def _extractive_fallback(self, trace: QueryTrace, retrieved) -> GenOutput:
        # verbatim_lead filters interrogative/list/meta sentences and splits danda-correctly;
        # the old first-". "-sentence cut could echo the passage's own heading as the answer
        # and never split Devanagari at all
        trace.add(StageTiming(stage="generate_fallback", ms=0.0, ok=True,
                              note="extractive fallback"))
        out = verbatim_lead(trace.query or "", retrieved, max_chars=400, max_sents=2)
        if (out.text or "").strip():
            return out
        top = retrieved[0]
        return GenOutput(text=top.text[:300], cited_chunk_ids=[top.chunk_id])

    def _citations(self, ids, retrieved) -> list:
        by_id = {r.chunk_id: r for r in retrieved}
        out = []
        for cid in ids:
            r = by_id.get(cid)
            if r:
                out.append(Citation(chunk_id=cid, doc_id=r.payload.get("doc_id"),
                                    passage_id=r.payload.get("passage_id"), text=r.text))
        return out

    def _finish(self, trace: QueryTrace, answer: str, decision: Decision,
                citations=None, abstain: bool = False, reason: Optional[str] = None) -> RAGResponse:
        trace.decision = decision
        trace.total_ms = round(trace.retrieval_to_output_ms, 3)
        return RAGResponse(answer=answer, decision=decision, citations=citations or [],
                           abstained=abstain, reason=reason, trace=trace)
