# FaceProof v0.3 implementation checklist

Executable facts as of 2026-09-07. External prerequisites remain `BLOCKED`, never `PASS`.

## Frozen Phase 0 baseline

- `ruff check src tests` — PASS.
- `pytest` — PASS, 53 tests before v0.3 changes.
- Old synthetic pipeline: SFace positive `0.936`, negative `0.404`; SSCD transformed positive
  `0.631`, negative `0.257`; transient Anvil anchor/tamper check passed.
- `faceproof source probe --tag cats` — integration-only PASS: five pages, 200 statuses and 200
  image attachments from a live public hashtag. It was not consented face evaluation.
- Old-pipeline per-query real corpus capture — `BLOCKED_REAL_CALIBRATION_DATA` (no eligible corpus).

## Implementation

- [x] Explicit `enrollment_face_images` and `copy_reference_images`; multi-face selection is explicit.
- [x] Manifest consent is mandatory before enrollment/candidate biometric processing.
- [x] Central hardened fetch/decode bounds, redirect/DNS/connected-peer SSRF checks, and ledger errors.
- [x] One model lock pins YuNet, SFace, SSCD, BiSeNet, preprocessing, terms, and pose template.
- [x] Exact normalized face/SSCD FAISS indexes aggregate distinct media and retain responsible rows.
- [x] BiSeNet region evidence fails to `UNKNOWN`; five-point SQPnP and deterministic quality are raw
  evidence until separately validated.
- [x] Identity/source-family/hash/copy-lineage leakage checks and one actual-pipeline manifest runner.
- [x] Search FPIR/FNIR/TPIR, confidence bounds, preview K metrics, bytes/latency/errors, and quality
  validation metrics are implemented.
- [x] Independent axis/state policy and complete conflict/unassessable/review state-table tests.
- [x] Exhaustive full-resolution-all is default; preview union is explicitly unpromoted optimization.
- [x] Geometry records reprojection/distribution and associates source face region only when valid.
- [x] Evidence finalizes after named review, with nonce, immutable artifact hashes, exact-byte digest.
- [x] Minimal digest registry verifies code/storage/tx/receipt/block; mainnet prohibited.
- [x] Anvil `--state` persistence verified after stop and fresh-process restart.
- [x] Authorized-account Mastodon pagination, multi-image posts, boost provenance, bounded retries.
- [x] Current lint/unit/synthetic/model checks run green (see commands below).
- [x] Malformed parser tensors, landmarks, geometry, crops, pose solutions, and non-finite quality
  inputs fail to `UNKNOWN`/`NEEDS_REVIEW`; a partial eye/mouth observation cannot clear a pair.
- [x] Evaluation excludes `UNASSESSABLE`, `NOT_RUN`, and `ERROR` axes from performance
  denominators while retaining every failure in the terminal-disposition ledger.
- [x] Calibration is search-level and participant/source-family clustered; the locked-test report
  must name the exact frozen policy and exact calibration-decision hash before final binding.
- [x] Concurrent media downloads are reassembled in provider order, making tied rankings and
  evidence deterministic regardless of response completion order.

## Ten blocker statuses

1. Face segmentation/visibility — **IMPLEMENTED; BLOCKED_REAL_VISIBILITY_VALIDATION**. Parser load and
   fail-closed behavior pass; real region confusion/false-clear/unknown targets need annotations.
2. Head pose — **IMPLEMENTED; BLOCKED_REAL_POSE_VALIDATION**. Synthetic convention and degeneracy
   tests pass; real coarse-bin error/range needs labeled faces.
3. Candidate quality — **IMPLEMENTED; BLOCKED_REAL_QUALITY_CALIBRATION**. Raw components and
   `NEEDS_REVIEW` work; normalization, false rejection and coverage need development data.
4. Identity-disjoint calibration — **IMPLEMENTED RUNNER; BLOCKED_REAL_CALIBRATION_DATA**.
5. Open-set 1:N evaluation — **IMPLEMENTED RUNNER; BLOCKED_REAL_CALIBRATION_DATA**.
6. Thresholds/conflicts — **STATE LOGIC PASS; BLOCKED_CALIBRATED_POLICY**. No operational threshold
   is guessed; provisional policy is review-only.
7. Preview Recall@K — **IMPLEMENTED; BLOCKED_FROZEN_PREVIEW_EVALUATION**. Exhaustive stays default.
8. Persistent blockchain — **LOCAL_PERSISTENCE_VERIFIED** with actual Anvil registry and fresh
   process; public testnet remains optional/unrun.
9. Live Mastodon demonstration — **CLIENT TESTED; BLOCKED_LIVE_MASTODON** pending controlled posts,
   their permissions, tag/account configuration, and actual live run.
10. Evidence/provenance — **PASS (software/synthetic)**: exact bytes, random nonce, model/policy/
    report/runtime/source/dependency hashes, ledger and tamper checks.

## Latest executed commands and outcomes

```text
.venv\Scripts\ruff.exe check src\faceproof tests
  PASS
.venv\Scripts\python.exe -m pytest tests -q
  PASS, 117 tests
.venv\Scripts\faceproof.exe models status
  PASS: all four artifacts match pinned SHA-256
.venv\Scripts\faceproof.exe doctor
  Local model/runtime checks ready; Mastodon tag/accounts correctly reported needed
.venv\Scripts\faceproof.exe acceptance --skip-chain
  PASS (synthetic integration only)
.venv\Scripts\faceproof.exe acceptance
  PASS: actual registry, all verification checks, tamper rejection, LOCAL_PERSISTENCE_VERIFIED
.venv\Scripts\python.exe -m pip wheel . --no-deps --no-build-isolation --wheel-dir dist
  PASS; wheel content scan found no env, model, artifact, demo-input, ONNX, PT, or OBJ payloads
  (`pip wheel` with build isolation was first BLOCKED by the sandbox package index; no check was
  relabeled as passing—the already installed pinned Hatchling produced the verified wheel.)
```

## Focused performance/equivalence evidence

- Pinned SSCD descriptors for seven varied fixtures were bit-for-bit identical before/after
  batching (maximum absolute delta `0.0`). A 16-image synthetic 720p CPU batch improved from
  `5833.6 ms` to `5391.4 ms` median (`~7.6%`); this is a local engineering measurement, not a
  universal latency claim.
- Exact-media retrieval for 20,000 vectors x 8 queries improved from `1266.9 ms` to `295.7 ms`
  median (`4.28x`) with the same selected-vector checksum. NumPy-oracle and deterministic-tie
  regression tests cover empty indexes, duplicate media, invalid vectors, and K above index size.
- A supplied external folder with one copy-positive and two copy-negative samples was rejected as
  `BLOCKED_REAL_CALIBRATION_DATA` (minimum 2/5), with a clean CLI error rather than a traceback.
  Its raw SSCD integration scores were `0.511585` positive and `0.101933`/`0.094695` negative;
  these three observations are not a threshold or an accuracy estimate.

## External next inputs

1. Consented images and signed/documented permission references for every biometric face.
2. Completed JSONL split/lineage manifest with manual visibility and pose-bin annotations.
3. At least six authorized Mastodon media cases across three posts; account handles and hashtag.
4. Human reviewer ID. Optional public-testnet RPC/test funds only if that separate demo is desired.

The implementation is SOTA-oriented in auditability and evaluation discipline, but no SOTA,
zero-error, public-chain timestamp, ownership, capture-time, or network-wide-search claim is
currently authorized by the available evidence. Comparative held-out results are the remaining
gate for that label.
