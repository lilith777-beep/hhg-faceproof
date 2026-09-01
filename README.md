# FaceProof

FaceProof is a consent-first, zero-paid-service pipeline for **HH Goa 2026 Task 3**:

```text
one-face scan
  -> live public Mastodon media enumeration
  -> preview face/image shortlist
  -> full-resolution SFace + pHash + AKAZE verification
  -> canonical evidence SHA-256
  -> local Anvil Ethereum transaction
  -> independent transaction/block re-verification
```

It does **not** infer a person's name. It asks whether a face in a consenting input is
visually consistent with a face in a genuinely discovered public post. Embeddings remain in
memory; the input image and embedding are never written to the evidence or blockchain.

## Why this design

There is no honest, free, open-source global reverse-face web index. Claiming otherwise would hide
a proprietary index, scrape platforms against their rules, or use a pre-picked result. FaceProof
uses a reproducible scoped search instead: the public/hashtag timeline of an operator-selected
open-source Mastodon instance. The API is queried live during every run, and returned post/media
identifiers, timestamps, endpoint, counts, and fetch time are preserved as provenance.

For the recording, publish a consenting image with a unique hashtag such as
`#HHGoaFaceProofNeel2026`, wait until the instance returns it, and configure that tag. This is a
genuine live search, not a hardcoded result, while remaining lawful, free, bounded, and repeatable.

## Requirement mapping

| Task requirement | Implementation |
|---|---|
| Detect/encode face | OpenCV YuNet + SFace, run locally with pinned model checksums |
| Genuine social search | Live Mastodon public/hashtag timeline API; no result URL in code/config |
| Accuracy | Single-face/quality gate; preview-first SFace; full-resolution SFace; pHash and AKAZE content corroboration; top-candidate margin and ambiguity block |
| Blockchain upload | Versioned SHA-256 evidence payload in Ethereum-compatible transaction calldata |
| Re-verification | Recompute digest, fetch transaction/receipt/block, verify payload, chain, status, block hash, sender, recipient, and confirmations |
| No paid software | Python/OpenCV/Mastodon/Foundry Anvil; local chain is the default |

## Quick start

Python 3.11 is the tested baseline.

```powershell
cd faceproof
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
faceproof models install
faceproof doctor
```

Install Foundry from its official installer, then start a local chain in a second terminal:

```powershell
anvil --chain-id 31337
```

Anvil's first unlocked development account is used by default. Do not expose Anvil to a network or
use its well-known development keys for real funds. A Sepolia/private-key profile is shown only as
an optional commented block in `.env.example`; it is not required for submission.

## Configure and prove genuine search

Set a unique tag in `.env`:

```dotenv
FACEPROOF_MASTODON_INSTANCE=https://mastodon.social
FACEPROOF_MASTODON_TAG=HHGoaFaceProofNeel2026
```

Post the consenting reference/near-reference image publicly with that hashtag. Then prove the live
connector sees current social data before running biometrics:

```powershell
faceproof source probe
```

The probe prints the exact API endpoint, fetch scope, pages/statuses/media scanned, and a current
post URL. No face bytes are sent to Mastodon; the client only enumerates public post metadata/media,
then downloads bounded candidates for local matching.

## Run end to end

Use only your own face or a person who explicitly agreed:

```powershell
faceproof run .\demo-input\consented-face.jpg --i-have-consent
```

The CLI shows the selected post, author, canonical URI, face score/threshold, independent image
signals, ambiguity state, and image SHA-256. It always requests human confirmation before the chain
write. Even `--yes` cannot anchor an ambiguous result.

Separate stages are useful for inspection:

```powershell
faceproof discover .\demo-input\consented-face.jpg --i-have-consent
faceproof anchor .\artifacts\RUN_ID\evidence.json
faceproof verify .\artifacts\RUN_ID\evidence.json .\artifacts\RUN_ID\anchor.json
```

Tamper test: copy `evidence.json`, alter one character in a value, and verify the copy against the
original receipt. Verification must fail because its recomputed digest no longer equals transaction
calldata.

## Accuracy model

FaceProof deliberately separates two claims:

1. **Same face:** SFace cosine similarity exceeds the configured threshold.
2. **Same visual content:** exact SHA-256, close DCT perceptual hash, or sufficient AKAZE/RANSAC
   geometric inliers corroborate that the post contains the same/derived image.

Candidates are enumerated up to a configured bound, previewed concurrently, locally encoded, then
only the best finalists are downloaded at full resolution. Ranking favors exact image, corroborated
content, face score, geometric inlier ratio, and perceptual distance—in that order. If two
uncorroborated faces are closer than the configured margin, the output is marked ambiguous and the
blockchain write is blocked.

`FACEPROOF_FACE_THRESHOLD=0.50` is a conservative project starting point above the upstream SFace
demo's 0.363 threshold, **not a universal accuracy guarantee**. Before the final recording,
calibrate it on consented same-person images spanning
pose/light/age and non-match faces representative of the demo. Record false-accept and false-reject
rates and set the threshold explicitly; the evidence states whether the threshold came from the
environment or the default. Human review remains mandatory.

```powershell
faceproof calibrate-threshold reference.jpg .\calibration\positive .\calibration\negative
```

Use at least two other consenting images of the subject and five representative non-match faces.
The generated report contains only filenames, hashes, scores, measured error rates, and the
recommended balanced-accuracy threshold—not embeddings or image bytes. Keep the calibration images
outside Git.

## Evidence and blockchain semantics

`evidence.json` includes:

- query image SHA-256/byte count, face box, pinned detector/encoder IDs;
- live source endpoint/scope/time, page/status/media counts, bounded failures;
- canonical post/media IDs, public URL, author handle, timestamp, text and text hash;
- candidate image hash/type/retrieved URL, face count/index/box/score/threshold;
- exact-image, pHash, AKAZE inliers, selection margin, ambiguity, and top candidates;
- explicit claims and non-claims.

Failures preserve only a URL hash and error class. Evidence excludes original image bytes,
embeddings, secrets, and private keys.

The blockchain proves that a precise evidence fingerprint was included in a transaction in a
canonical block. It does not prove legal identity, post authorship/truth, or that the remote post
will remain online. Local Anvil is acceptable under the task's local/simulated-chain clause and is
reproducible for the recording; its state disappears when stopped unless you use Anvil state
persistence.

## Verification and tests

```powershell
ruff check src tests
pytest
faceproof models status
faceproof source probe
faceproof chain status
```

The deterministic suite covers canonical hashing, tamper rejection, transaction/block checks,
consent, mismatch rejection, source parsing/provenance, SSRF and download limits, preview/full
selection, pHash stability, AKAZE transforms, configuration validation, and ambiguity metadata.
Mocked connector tests are never described as live search; `source probe` is the live integration
gate.

Before recording:

- [ ] Use a unique live hashtag and show `faceproof source probe` discovering it.
- [ ] Use a sharp, well-lit, single-face scan with explicit consent.
- [ ] Confirm the returned post manually and ensure `ambiguous` is `False`.
- [ ] Show the evidence component signals, not only a face score.
- [ ] Run Anvil locally and show its transaction log/transaction hash.
- [ ] Show every `faceproof verify` check passing.
- [ ] Modify an evidence copy and show verification fail.
- [ ] Run tests/lint and the repository secret scan before publishing.

## Known limitations and safety

- Search covers only public image posts visible to the selected Mastodon instance/timeline and
  bounded pages. It is not global web search and cannot see private/deleted/unfederated content.
- Face matching is probabilistic and may vary with pose, light, occlusion, age, image quality, and
  population. SFace's packaged weight provenance/training-data documentation should be treated as a
  model-card limitation; do not claim demographic fairness without measuring it.
- A unique hashtag is the reliable open-source demo path. Public firehose search is broader but can
  miss older content within the bounded scan.
- A public post/image may disappear later. The chain still verifies the evidence hash, but this
  project intentionally does not republish copyrighted image bytes.
- The fetcher requires public HTTPS destinations, rejects private/reserved addresses, limits
  redirects and bytes, validates image MIME/signatures, and should still run least-privileged.
- Consent is mandatory. Do not use this for surveillance, stalking, doxxing, or covert identity
  inference.

## Repository hygiene and license

```powershell
git status --short
git ls-files | Select-String -Pattern '(\.env$|artifacts/|models/.+\.onnx|private|credential)'
```

The second command should print nothing sensitive. Generated models, inputs, `.env`, artifacts,
cache/build directories, and credentials stay untracked. Project source is MIT licensed. See
`THIRD_PARTY.md` for upstream components, licenses, and model caveats.
