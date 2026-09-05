# FaceProof pipeline context

## Objective

Given one explicitly consented face image, discover a genuinely live public social-media post,
verify whether its media contains the same face, separately measure whether it is a transformed
copy of the source image, and
anchor the resulting evidence fingerprint on an Ethereum-compatible chain. FaceProof does not name
the person and does not claim that blockchain makes a biometric decision true.

## Current v0.3 graph

```text
consented query image
  -> decode + bounded input + exactly-one-face quality gate
  -> YuNet landmarks -> aligned SFace 128-D unit face descriptor
  -> SSCD DISC-Mixup 512-D unit copy descriptor
  -> live bounded Mastodon hashtag enumeration
  -> bounded parallel preview downloads
  -> per-media face descriptors + SSCD descriptors + pHash
  -> exact FAISS IndexFlatIP face search UNION exact FAISS SSCD search UNION pHash rescue
  -> full-resolution face + SSCD verification
  -> SHA-256 + pHash + AKAZE/RANSAC content corroboration
  -> typed identity/content decision + ambiguity gate + human confirmation
  -> canonical evidence SHA-256
  -> Ethereum transaction calldata
  -> transaction, receipt, canonical-block and tamper re-verification
```

## Recent decisions

1. **Identity and copying are separate claims.** Face similarity answers “same person?”; SSCD,
   SHA, pHash and AKAZE answer “same or derived visual content?”. Scores are never combined with an
   arbitrary weighted average.
2. **Exact retrieval first.** The bounded gallery is at most 200 media items, so FAISS
   `IndexFlatIP` gives exact cosine rankings. Approximate HNSW/IVF would add avoidable Recall@K loss.
3. **Union retrieval protects both modes.** The top face and SSCD lists are admitted first in
   round-robin order so neither learned lane can evict the other; close pHash candidates then fill
   any remaining finalist capacity as a deterministic rescue.
4. **Face verification remains mandatory.** An SSCD-positive/face-negative result is
   `copy_face_conflict`, not a verified identity match and cannot become the selected proof.
5. **SSCD is required in production.** `auto` exists only as an explicitly evidenced legacy
   fallback. The pinned model checksum and runtime are recorded in evidence; descriptors are not.
6. **Thresholds are provisional until real calibration.** Face `0.50` and SSCD `0.75` are starting
   operating points, not universal accuracy claims.
7. **Search is genuine but scoped.** The software enumerates a live Mastodon hashtag; it is not a
   global reverse-face index and cannot see private, deleted, old-out-of-window or unfederated media.
8. **Blockchain proves integrity, not truth.** It proves that exact evidence bytes were committed
   in a canonical transaction. Human confirmation remains mandatory.

## Decision states

| Face | Content | State | Eligible as verified match? |
|---|---|---|---|
| pass | pass | `same_identity_and_content` | yes |
| pass | fail | `same_identity_different_content` | yes, identity claim only |
| fail | pass | `copy_face_conflict` | no; manual investigation |
| fail | fail | `rejected` | no |

## What is measured versus pending

Measured now: deterministic software tests, pinned-model integrity, synthetic same/different
fixture separation, exact vector ranking, live-source enumeration, chain inclusion checks and
tamper rejection.

Pending real data: identity-disjoint ROC/DET, TAR at fixed FAR, 1:N FPIR/FNIR, preview Recall@K,
pose/occlusion/demographic slices, SSCD precision/recall on social transforms, and confidence
intervals. No “SOTA” or zero-error claim is permitted until those reports exist.

## Research decisions still open

- face parsing/segmentation model and its weight license;
- pose and occlusion quality model;
- whether to retain SFace or benchmark a better licensed embedding model;
- public testnet versus persisted local-Anvil state;
- target operational FPIR/FNIR and latency budgets.

## Final release gates

- Real consenting query plus multiple same-person and non-match calibration images.
- Live shared hashtag containing multiple unrelated image posts.
- Face threshold calibrated on identity-positive/negative data; SSCD threshold separately
  calibrated on derived-copy/unrelated-image data, both with held-out evaluation.
- Target post appears inside both the source window and full-verification candidate union.
- No unresolved face/content conflict or ambiguity.
- Evidence includes exact model hashes and runtime versions.
- Anchor and independent re-verification pass; a tampered evidence copy fails.
