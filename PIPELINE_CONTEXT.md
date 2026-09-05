# FaceProof v0.3 — concise pipeline context

## Purpose and boundary

FaceProof is a bounded, consent-governed research CLI. It discovers image media inside an
operator-selected Mastodon scope, measures two independent questions, preserves every terminal
disposition, and anchors a reviewed evidence commitment. It is not a public face-identification
database, does not infer a name from an account, and does not claim that a blockchain makes a
match true.

Biometric processing is permitted only when the manifest records consent for the represented
participant and the particular enrollment/candidate role. Public visibility is not permission.
Copy discovery may still run on permitted images whose faces are not eligible; that path remains
copy-only and identity stays unknown.

## Input roles

```text
enrollment_face_images          copy_reference_images
(consented identity examples)   (originals whose copies are sought)
             \                    /
              \  explicitly may /
               \ be same file  /
                v              v
                   one run
```

A newly captured selfie is never silently reused as the original for unrelated older images.
Multiple-face enrollment requires an explicit face index. Face embeddings remain memory-only.

## Actual execution graph

```text
validated JSONL consent/evaluation manifest + frozen model lock + policy
                 |
     +-----------+------------------+
     |                              |
enrollment images               copy references
     |                              |
safe decode -> YuNet            safe decode -> SSCD TorchScript
5 landmarks -> SFace            RGB, direct 320x320 -> 512-D unit vector
128-D unit vectors                   |
     +--------------+---------------+
                    |
authorized Mastodon hashtag/account enumeration
real pagination -> boosts normalized with wrapper provenance -> every image attachment
                    |
hardened instance-CDN fetch: HTTPS + approved host + DNS and connected-peer checks
+ redirect revalidation + timeout + streamed byte cap + signature/type agreement
+ OpenCV decode validation + EXIF orientation + 40 MP cap
                    |
preview analysis (consent-eligible faces only)
  exact FAISS IndexFlatIP face top-K DISTINCT MEDIA
  UNION exact FAISS IndexFlatIP SSCD top-K DISTINCT MEDIA
  (+ pHash diagnostic rescue)
                    |
default truth path: ALL originals                 optimized path: preview union
                    |                                  (not promoted yet)
full-resolution SFace + SSCD + quality evidence
BiSeNet regions + 5-point SQPnP coarse pose + blur/exposure/resolution
SHA-256 exact bytes + pHash diagnostic + AKAZE/RANSAC geometry diagnostic
                    |
independent axes: SUPPORTED / NOT_SUPPORTED / BORDERLINE / UNASSESSABLE
execution: NOT_RUN / ERROR remain separate
                    |
claim state table + source/candidate region association where geometry permits
                    |
DRAFT evidence + complete disposition ledger
 -> named human review of explicit claims
 -> random 32-byte nonce
 -> exact sorted UTF-8 JSON bytes (not claimed RFC 8785)
 -> version-domain SHA-256
                    |
pinned minimal digest registry on Anvil/testnet
 -> chain ID + bytecode + storage + commitment log + tx/receipt/canonical block verification
 -> local Anvil stop/restart/load-state verification
```

## Decision logic

| Face axis | Copy axis | Output |
|---|---|---|
| supported | supported | both supported, pending human confirmation |
| supported | not supported | identity-only supported |
| unassessable | supported | copy-only; identity unknown |
| reliably not supported | supported | conflict review |
| either borderline | any | review |
| both not supported | both | no accepted match |
| missing/stale calibration or out-of-policy bounds | any | review-only |

Low cosine is “unsupported at this operating point,” not proof of different identity. SSCD,
pHash, and geometry never overwrite face cosine. Poor face quality never deletes copy evidence.
Posts are not forced into a winner/runner-up identity margin; multiple posts may all be valid.

## Recent decisions

1. YuNet/SFace remains the face baseline; no model replacement without comparative evidence.
2. Official standalone SSCD is loaded with the existing PyTorch runtime; the training stack is
   not installed. The upstream 0.75 example is not an operational policy threshold.
3. Exact FAISS removes approximate-index error; evaluation still measures preview and model misses.
4. BiSeNet parsing is region evidence, not universal hand/sticker occlusion truth. Unsupported
   regions are `UNKNOWN`.
5. Five-landmark SQPnP is approximate coarse pose. Synthetic tests validate convention only;
   real pose-bin validation remains required.
6. Full-resolution-all is the default until frozen-data preview Recall@K earns promotion.
7. Evidence finalization happens after review; the receipt is outside the manifest to avoid a
   circular hash.
8. Local state persistence and public-testnet proof are distinct. Mainnet is prohibited and a
   public-testnet write needs explicit CLI authorization.

## What is proven now vs externally blocked

Software tests prove deterministic state handling, fail-closed model behavior, exact retrieval
against a NumPy oracle, manifest leakage rejection, tamper detection, registry storage, and a fresh
process loading persisted local Anvil state. Synthetic examples prove integration only.

Real operating thresholds, false-positive identification rate, true-positive identification rate,
preview Recall@K, quality false rejection, region visibility error, and real pose-bin error remain
blocked until the identity-disjoint consented corpus is supplied. A live multi-post Mastodon proof
remains blocked until controlled posts, permission records, and the configured account/tag scope
exist. No SOTA or zero-error claim is made before those reports pass.
