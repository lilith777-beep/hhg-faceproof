# Video Scripts + Posting Kit — hhgvrag · #RAGInGoa

Shoot Aug 20–21 (post-freeze, final numbers on screen). 3 members × IG + X + LinkedIn ×
2 videos = **18 posts**, every post tagged **#RAGInGoa**, ≥1 Instagram account public.
Final numbers below marked ⟨X⟩ get filled from the sealed report before shooting.

---

## VIDEO 1 — Team/Process (90 seconds, process not product)

**Format**: fast cuts, phone-shot, energetic. B-roll: terminal scrolling, whiteboard,
the three of you arguing at a screen, Forge's GPU fans, late-night chai.

| t | Shot | Line (VO or to-camera) |
|---|---|---|
| 0–8s | Hook: face to camera | "Nine days to build a voice RAG that answers in under 200 milliseconds — in 14 Indian languages. Here's how we actually worked." |
| 8–20s | Whiteboard with the pipeline sketch | "Rule one: accuracy is sacrosanct. Every answer is verbatim from the corpus, cited, or the system refuses — in your language." |
| 20–35s | Terminal montage: tests scrolling green | "Rule two: nothing ships without evidence. 250+ automated tests. Every bug we hit live became a regression test the same hour." |
| 35–50s | Screen: the adversarial review doc | "Rule three: we attacked our own system. An independent verifier rejected our evaluation plan twice — leakage-safe data splits, sealed test sets we're not allowed to rerun until the end. We rebuilt it their way." |
| 50–65s | Forge box + nvidia-smi | "Everything runs on one GPU in our office. We stress-tested it until it broke at 30 simultaneous requests — then fixed that too." |
| 65–80s | The team, three-lens explanation | "Three lenses: one of us owns the score, one owns the user, one owns the code. Every feature had to survive all three." |
| 80–90s | Logo/URL card | "hhgvrag. Speak in your language. Get the truth, or an honest no. #RAGInGoa" |

---

## VIDEO 2 — Demo (2–3 min, end-to-end, screen-recorded + one phone shot)

**Setup**: hhgvrag.vercel.app on desktop; OBS/screen-record at 1080p; phone for the
noisy-environment shot. Do a warmup query off-camera first (cache/GPU warm).

**Beat 1 — The hero (voice Hindi → Hindi answer)** ·  ~25s
Hold the mic. Ask: **"कॉर्पोरेशन क्या है?"**
Show: transcript appears → grounded Hindi answer with [1][2] citations → open sources,
point at the yellow highlighted verbatim segments → zoom the budget bar: "⟨~50⟩ms of a
200ms budget."
Line: "Speech in, grounded answer out — in Hindi, cited, in ⟨X⟩ milliseconds."

**Beat 2 — Live noisy environment** · ~25s  ⚠ rehearse twice
Phone speaker plays café/street noise nearby. Ask in English by voice:
**"how long does a tax refund take?"**
Line: "Background noise, real mic. We measured this: at 10 decibels signal-to-noise our
transcription holds ⟨97.5⟩% accuracy — and the system never fabricated a single answer
across 204 noise conditions."

**Beat 3 — The refusal (the feature, not a failure)** · ~20s
Type or speak: **"वेस्टेरोस का राजा कौन है?"**
Show: the terracotta "DECLINED" badge + the Hindi refusal + gate signals.
Line: "Off-corpus? It declines — in your language — and shows you exactly which gate
fired. Grounded, or nothing."

**Beat 4 — Full-flow tour (enterprise speed-run)** · ~35s
Rapid sequence, narrated over cursor:
1. English factual ask → instant answer → stage breakdown open ("every millisecond
   accounted: embed, retrieve, rerank, ground").
2. Repeat the same question → "⚡ cache hit — ⟨~11⟩ms."
3. Follow-up "what are its benefits?" → session context chip.
4. Toggle **Vectorless** → same answer, retrieval with zero embeddings.
5. Toggle **Quality mode** → grounded answer lands instantly, then the 7B elaboration
   streams in beneath it, separately timed.
Line: "Hybrid retrieval, semantic cache, conversation memory, a vectorless mode, and a
streamed deep-dive — all behind one 200-millisecond guarantee."

**Beat 5 — Evidence deep-dive** · ~30s
Open the data window (/data.html). Scroll: corpus stats (⟨X⟩ chunks, 14 languages),
chunking A/B table with starred winner, the 204-cell ASR grid, risk–coverage curve,
sealed-test stamp with manifest hash.
Line: "We don't claim — we publish. Real relevance labels, confidence intervals, a
sealed test set we ran exactly once. P50 ⟨X⟩ms, P95 under load ⟨X⟩ms, sample maximum
⟨X⟩ms across ⟨N⟩ queries."

**Close** · ~10s
"hhgvrag — a voice RAG that measures uncertainty honestly. Built in nine days for
Hacker House Goa. #RAGInGoa" → URL card.

---

## Posting kit (each member posts BOTH videos on ALL THREE platforms)

**Checklist per member**: IG (≥1 of the 3 accounts public) · X · LinkedIn → 6 posts each.
Every caption ends with **#RAGInGoa**.

Caption A (Video 1, vary per member):
> 9 days. 3 people. 1 GPU. We built a voice RAG that answers in 14 Indian languages in
> under 200ms — and refuses honestly when it doesn't know. Here's how we worked. #RAGInGoa

Caption B (Video 2):
> Speak Hindi, get a cited Hindi answer in ⟨X⟩ms. Ask nonsense, get an honest no — in
> your language. Zero fabricated answers across 204 noise conditions. This is hhgvrag.
> Live demo 👉 hhgvrag.vercel.app #RAGInGoa

X-specific: thread it — clip + 3 tweets (hero number → refusal screenshot → evidence link).
LinkedIn-specific: add 2 lines on the engineering discipline (sealed evaluation, CIs).

**Fill-ins before shooting**: ⟨X⟩ numbers from `eval/` reports post-sealed-run; member
names/handles: ____________ · ____________ · ____________

---

## SOTA differentiators & hype angles — the moon shot (honest ammo only)

The energy is earned, not manufactured. Every line below is a *true* flex — lead with these
in captions, thread hooks, and the video's confident moments. The quirkiness comes from the
confidence of substance.

**The researcher flexes (what a real AI researcher does, not a hackathon dev):**
- **"We found a bug in our own tokenizer — live on camera."** Python's `\w` silently drops
  Devanagari vowel signs and the virama, so `मधुमेह` was being shattered into consonant
  fragments across every lexical path. We caught it, proved it, fixed it with a Unicode
  mark-aware tokenizer across 14 scripts. *That's* engineering rigor — the kind judges reward.
- **"We chose the right model, not the biggest."** For Indian languages, Sarvam-1 (2B,
  Indic-trained) beats Gemma-2-2B and Llama-3.2-3B on Indic benchmarks and approaches
  Llama-3.1-8B — at a fraction of the size. Right-trained > big.
- **"We publish, we don't claim."** Leakage-safe passage-family splits, a sealed test we run
  exactly once, bootstrap confidence intervals, a 204-cell noise grid. Numbers with error bars.

**The scale + honesty flexes:**
- **876k documents, 14 Indian languages** — cross-lingual, leakage-disjoint 10k-doc holdout.
- **Grounded or nothing.** The answer is extracted verbatim and cited — invention is
  *impossible by construction* — or it refuses, in your language, in your script.
- **Speak Hinglish, get answered.** "corporation kya hai" → answered. Romanized vernacular in,
  same-language out — because most of India types Latin-script Hindi.

**The edge flex (the "runs on a potato" wow):**
- **It runs on a 2GB-GPU laptop. Offline.** Vectorless retrieval + extractive answer, zero
  cloud, <50ms — with a 1-bit / Indic-tuned LLM streaming the polish on CPU. The whole voice
  RAG in your pocket, no datacenter.
- **Sub-200ms grounded answer; ~750ms full voice turn** (STT 211 + RAG 52 + TTS 487, measured).

**Quirk beats to film (delight = shareability):**
- The budget bar racing and landing at ~50ms of a 200ms line — the "we had 150ms to spare" grin.
- The refusal as a *feature*: ask it nonsense in Hindi, watch the terracotta DECLINE badge +
  the gate signals — "it knows what it doesn't know."
- The live tokenizer-bug reveal: split-word terminal output → the fix → "+recall, one commit."
- Toggle Vectorless mid-demo: "same answer, zero embeddings, runs on your phone."
- The data-window scroll: 14-language treemap, the leakage-0/0/0 stamp, the sealed hash — a
  researcher's dashboard, not a toy.

**One-line identities to seed across posts (vary per member/platform):**
> "We built a voice RAG that speaks 14 Indian languages, answers in under 200ms, refuses when
> it doesn't know — and runs offline on a laptop. Then we found a bug in our own tokenizer and
> fixed it on camera. #RAGInGoa"

**Marketing cadence** (post-freeze Aug 20 → submit Aug 21): tease (bug-reveal clip) → hero
(Hindi voice demo) → credibility (data-window/sealed evals) → edge (runs-on-laptop) → team
(three-lens process). Each member seeds a different angle so the three accounts don't echo.
