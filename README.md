# FaceProof

FaceProof is a consent-first, zero-paid-service pipeline for **HH Goa 2026 Task 3**:

```text
one-face scan
  -> live public Mastodon media enumeration
  -> YuNet faces + SFace identity descriptors
  -> SSCD copy descriptors
  -> exact FAISS cosine retrieval on both descriptor spaces
  -> full-resolution SFace + SSCD + pHash + AKAZE verification
  -> explicit identity/content decision matrix + human confirmation
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

For the recording, publish a consenting image under a shared hashtag containing several unrelated
image posts (for example the event's public tag), wait until the instance returns it, and configure
that tag. The matcher must choose the target from multiple live candidates. A unique tag is useful
only for connector diagnosis; do not use a one-result tag as the final search demonstration.

## Requirement mapping

| Task requirement | Implementation |
|---|---|
| Detect/encode face | OpenCV YuNet + SFace, run locally with pinned model checksums |
| Copy detection | Meta SSCD DISC-Mixup descriptors, pinned by SHA-256 and run locally |
| Genuine social search | Live Mastodon public/hashtag timeline API; no result URL in code/config |
| Retrieval | Exact cosine search with FAISS `IndexFlatIP` over normalized SFace and SSCD vectors |
| Accuracy | Single-face gate; dual retrieval; full-resolution re-verification; exact hash, SSCD, pHash and AKAZE/RANSAC corroboration; conflict and ambiguity blocks |
| Blockchain upload | Versioned SHA-256 evidence payload in Ethereum-compatible transaction calldata |
| Re-verification | Recompute digest, fetch transaction/receipt/block, verify payload, chain, status, block hash, sender, recipient, and confirmations |
| No paid software | Python/OpenCV/PyTorch/SSCD/FAISS/Mastodon/Foundry Anvil; local chain is the default |

## Quick start

Python 3.11 is the tested baseline.

```powershell
cd faceproof
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,vision]"
Copy-Item .env.example .env
faceproof models install
faceproof doctor
```

Run the complete bundled synthetic acceptance proof with one command:

```powershell
faceproof acceptance
```

It verifies a fictional same-person pair against a different-person negative, writes evidence,
starts a localhost Anvil process when available, anchors and re-verifies the digest, confirms a
tampered copy is rejected, then stops the Anvil process it started. This proves the software path;
it is deliberately labeled synthetic and does not replace the required genuine live-post run.

For a short human checklist, open [`DO_THIS_NEXT.md`](DO_THIS_NEXT.md).

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
FACEPROOF_MASTODON_TAG=HHGoa2026
```

Post the consenting reference/near-reference image publicly with that shared hashtag. Ensure the
timeline contains multiple image posts, then prove the connector sees current social data before
running biometrics:

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

1. **Same identity:** SFace cosine similarity exceeds the configured face threshold.
2. **Same/derived content:** exact SHA-256, SSCD cosine, close DCT perceptual hash, or sufficient
   AKAZE/RANSAC geometric support corroborates the image relationship.

Every bounded preview is encoded into both spaces. Two exact FAISS `IndexFlatIP` indexes retrieve
the top face and top copy candidates independently; pHash adds a deterministic rescue lane. The
union is downloaded at full resolution and every learned score is recomputed before a result can
be selected. Copy similarity without a face match is recorded as `copy_face_conflict`, never as a
person match. Near-tied faces are blocked unless content uniquely disambiguates them or both hits
are the exact same remote asset.

SSCD uses the official DISC-Mixup ResNet-50 TorchScript checkpoint, 320×320 RGB/ImageNet
preprocessing, L2-normalized 512-dimensional descriptors, and cosine similarity. The default
`FACEPROOF_SSCD_THRESHOLD=0.75` follows the upstream DISC guidance; it is not silently lowered to
make the synthetic fixture pass. The fixture's derived edit is recovered by SFace + AKAZE while
SSCD remains an independent measured signal.

Face detection/encoding uses the decoded source image. AKAZE geometric corroboration is separately
bounded to a 512-pixel working edge: the placeholder pair retained 35+ RANSAC inliers at a 0.44+
ratio while reducing the warm benchmark from about 3.0 seconds to 0.23 seconds on the development
machine. Cold CLI startup remains hardware-dependent and is reported in evidence timings.

`FACEPROOF_FACE_THRESHOLD=0.50` is a conservative project starting point above the upstream SFace
demo's 0.363 threshold, **not a universal accuracy guarantee**. Before the final recording,
calibrate it on consented same-person images spanning
pose/light/age and non-match faces representative of the demo. Record false-accept and false-reject
rates and set the threshold explicitly; the evidence states whether the threshold came from the
environment or the default. Human review remains mandatory.

```powershell
faceproof calibrate-threshold reference.jpg .\calibration\positive .\calibration\negative
```

Use substantially more than the two-positive/five-negative smoke-test minimum. For a credible
final report, include varied pose, lighting, crop, compression and occlusion, plus hard negatives
that resemble the subject. Report sample counts and uncertainty; never call the bundled synthetic
pair a population-level accuracy metric.
The generated report contains only filenames, hashes, scores, measured error rates, and the
recommended balanced-accuracy threshold—not embeddings or image bytes. Keep the calibration images
outside Git.

Calibrate copy detection separately. Copy positives must be transformations of the exact reference
(resize, crop, recompression, overlays, screenshots), not merely other photos of the same person:

```powershell
faceproof calibrate-copy reference.jpg .\copy-calibration\positive .\copy-calibration\negative
```

Use unrelated and visually similar hard-negative images, then validate the chosen SSCD operating
point on a held-out transform set before changing `FACEPROOF_SSCD_THRESHOLD`.

YuNet detection uses `FACEPROOF_DETECTION_THRESHOLD=0.80`. The upstream demo commonly uses 0.90,
but the placeholder acceptance set exposed false rejections at 0.90 (positive confidences 0.866 and
0.872). At 0.80 both positives were detected while the SFace verifier still separated the
same-person pair (0.936) from the different-person pair (0.404). Re-check this gate on real,
consented validation images before submission.

## Evidence and blockchain semantics

`evidence.json` includes:

- query image SHA-256/byte count, face box, pinned detector/encoder/SSCD IDs and checksums;
- live source endpoint/scope/time, page/status/media counts, bounded failures;
- canonical post/media IDs, public URL, author handle, timestamp, text and text hash;
- candidate image hash/type/retrieved URL, face count/index/box/score/threshold;
- exact-image, SSCD, pHash, AKAZE inliers, dual-FAISS ranks/channels, decision, selection margin,
  ambiguity, conflicts, runtime versions, timings, and top candidates;
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
consent, mismatch/conflict rejection, SSCD loading, exact FAISS ranking, source parsing/provenance,
SSRF and download limits, preview/full selection, pHash stability, AKAZE transforms, configuration
validation, evidence semantics, and ambiguity metadata.
Mocked connector tests are never described as live search; `source probe` is the live integration
gate.

Before recording:

- [ ] Use a shared live hashtag with multiple image posts and show `source probe` scanning them.
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
- SSCD is a copy-detection model, not a face-recognition model. Its upstream repository is archived;
  FaceProof pins the model checksum and keeps SFace identity verification as a separate mandatory
  gate. CPU SSCD inference is accurate but slower; a compatible CUDA PyTorch build is recommended
  for larger candidate sets.
- There is no explicit face segmentation, pose normalization, liveness detection, or multi-view
  identity ensemble yet. Those are research/calibration gaps, not hidden features. See
  [`PIPELINE_CONTEXT.md`](PIPELINE_CONTEXT.md).
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
