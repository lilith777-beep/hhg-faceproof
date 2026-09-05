# FaceProof

FaceProof v0.3 is an open-source, consent-governed CLI for HH Goa 2026 Task 3:

```text
consented face references + explicit image-copy references
  -> authorized Mastodon media discovery
  -> independent face and copy verification
  -> reviewed evidence sealing
  -> persistent local Ethereum commitment and fresh-process verification
```

There is no paid API and no website. The face baseline is OpenCV YuNet + SFace. Copy retrieval uses
the official SSCD DISC-Mixup TorchScript artifact. Both descriptor types use separate exact
`faiss.IndexFlatIP` indexes. BiSeNet face parsing and five-landmark SQPnP provide conservative
quality evidence; neither is allowed to manufacture identity confidence.

Read [PIPELINE_CONTEXT.md](PIPELINE_CONTEXT.md) for the visual system graph and recent decisions.

## Safety boundary

- Every enrollment face and every candidate face needs a manifest record with documented biometric
  permission. Public visibility and ownership of a posting account are not consent.
- Image-copy discovery is a separate permitted process. A candidate without biometric permission
  can be evaluated for copying, but its face is not encoded and identity remains unknown.
- `enrollment_face_images` and `copy_reference_images` are separate CLI inputs. The same file may be
  declared for both roles; it is never reused silently.
- Face embeddings are memory-only by default. Evidence contains scores, boxes, model hashes and
  claims—not embeddings, private images, tokens, keys, or identifying URLs on chain.
- FaceProof does not infer identity labels from usernames and does not automate takedowns.

## Install

Requires Python 3.11–3.14. Anvil from Foundry is needed only for blockchain checks.

```powershell
cd faceproof
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,vision]"
faceproof models install
Copy-Item .env.example .env
```

`model-lock.json` pins every learned artifact, source revision, SHA-256, preprocessing/output
contract, and reviewed code/weight terms. `requirements.lock` records the verified environment.
Models are checksum-verified before loading and never auto-update during a scan.

## Consent/evaluation manifest

The JSONL manifest is the control plane for permission and identity-disjoint evaluation. Each row
contains:

- `image_id`, relative `path`, exact `sha256`;
- one or more roles: `enrollment_face`, `candidate_face`, `copy_reference`, `candidate_copy`;
- `participant_ids`, `biometric_consent`, and `consent_ref` for biometric roles;
- `capture_session_id`, `source_image_family_id`, and `split` (`development`, `calibration`, or
  `test`);
- face annotations with their own participant/consent references;
- optional `media_id`, `canonical_uri`, `copy_parent_image_id`, and evaluator-only labels.

Loading fails on missing permission, changed bytes, duplicate IDs, cross-split identities,
cross-split source families, exact-byte leakage, or copy lineage crossing a split. Real calibration
also needs independent same-person photographs; synthetic edits do not create new identities.

## Configure authorized Mastodon discovery

```dotenv
FACEPROOF_MASTODON_INSTANCE=https://mastodon.social
FACEPROOF_MASTODON_TAG=HHGoa2026
FACEPROOF_MASTODON_ALLOWED_ACCOUNTS=controlled_account,second_controlled_account
```

The client performs real pagination, supports multi-image posts, normalizes boosts while retaining
wrapper provenance, honors bounded 429/5xx retries, and uses `preview_url` only for preview
experiments. Originals use the attachment `url`. Image fetching is restricted to the configured
instance host and revalidates DNS and the connected socket peer after every redirect. No Mastodon
authorization header is sent to media hosts.

Publishing is deliberately not automated. Create the six consented cases on at least three
controlled posts (original, edited copy, independent same-person photo, different consenting
person, occlusion, and composite/conflict), include one multi-image post, and use a small page size
in the recording to visibly exercise pagination.

## Policy/evaluation lifecycle

Full-resolution-all is authoritative and the default. Preview top-K is only an optimization until
its recall target passes on frozen data.

```powershell
# 1. Review-only development/calibration execution using the actual pipeline
faceproof evaluate-manifest .\private-data\manifest.jsonl `
  --policy .\policy\provisional-review-only.json --split calibration `
  --output .\private-results\calibration-evaluation.json

# 2. Select separate face/copy thresholds from calibration only
faceproof calibrate-manifest-report .\private-results\calibration-evaluation.json `
  --policy-id faceproof-study-1 --output .\private-results\calibration-decision.json

# 3. Freeze a pre-test policy, then run the untouched test split
faceproof freeze-policy .\private-results\calibration-decision.json `
  --output .\private-results\policy-pretest.json
faceproof evaluate-manifest .\private-data\manifest.jsonl `
  --policy .\private-results\policy-pretest.json --split test `
  --output .\private-results\test-evaluation.json

# 4. Bind the immutable test report without changing thresholds
faceproof freeze-policy .\private-results\calibration-decision.json `
  --test-report .\private-results\test-evaluation.json `
  --output .\private-results\policy-final.json

# 5. Compare old face-only, dual, and dual-plus-quality on the same frozen split
faceproof compare-manifest .\private-data\manifest.jsonl `
  --policy .\private-results\policy-final.json --split test `
  --output-dir .\private-results\comparison
```

The runner reports media, expected face-vector, identity, and query-reference counts; search-level
FPIR/FNIR/TPIR with confidence intervals; incorrect known returns; preview K=5/10/20/50 truth
recall and exhaustive-verifier retention; bytes, latency, failures, and every terminal disposition.
An unknown query counts once if it returns any accepted wrong candidate. Pair counts are not
misreported as independent searches.

The default research targets are provisional: FPIR upper 95% bound <=1%, TPIR lower 95% bound
>=90%, and preview candidate-recall lower 95% bound >=99% for identity and copy positives. Failure
to meet them is a blocked result, not a threshold adjustment on the test set.

## One end-to-end live command

After the manifest, controlled posts, and frozen policy are ready:

```powershell
faceproof run .\private-data\enrollment.jpg `
  --copy-reference .\private-data\original.jpg `
  --manifest .\private-data\manifest.jsonl `
  --policy .\private-results\policy-final.json `
  --reviewer "reviewer-id" --i-have-consent
```

If an enrollment image has multiple faces, add exactly one `--face-index N` for that image. The run
discovers media, performs exhaustive full-resolution verification, writes a draft, asks the named
reviewer to confirm explicit claims, seals exact evidence bytes, anchors the digest, and verifies
the registry/transaction/receipt/block.

Public-testnet writes require both an explicit command and `--allow-public-testnet`. Ethereum
mainnet is prohibited. The default local chain is Anvil chain 31337.

## Evidence and blockchain meaning

After review, the private v2 manifest includes a random 32-byte nonce, input roles/hashes, consent
references, observation times, source/media IDs, boxes/associations, raw quality and similarity
scores, thresholds and axis states, human review, actual model/policy/report hashes, runtime and
dependency/source fingerprints, and the full terminal ledger. It is encoded once as sorted UTF-8
JSON with fixed separators and `allow_nan=False`; this is not claimed as RFC 8785.

The version-domain SHA-256 of those exact bytes is stored in a pinned minimal digest-registry
contract. The receipt stays outside the evidence. Verification checks chain identity, deployed
bytecode, storage, commitment log, transaction input/result, receipt and canonical block. Synthetic acceptance
stops Anvil, starts a fresh process from persisted state, and verifies the old commitment without
redeploying or re-anchoring. This is `LOCAL_PERSISTENCE_VERIFIED`, not an independent public
timestamp or proof that the reviewed claims are true.

## Checks

```powershell
ruff check src tests
pytest -q
faceproof models status
faceproof doctor
faceproof acceptance --skip-chain
faceproof acceptance
```

The bundled images are fictional synthetic integration fixtures. They test software wiring only
and are never presented as accuracy evidence.

## Known blockers

The repository is code-complete for the bounded pipeline, but real claims remain blocked until:

- an identity-disjoint, permission-documented development/calibration/test corpus exists;
- visibility and real pose-bin annotations validate the quality policy;
- enough independent known/unknown searches support the declared confidence bounds;
- preview Recall@K passes without hiding exhaustive verifier misses;
- at least six real media cases across three authorized Mastodon posts are discoverable;
- the human reviewer confirms each explicit claim; and
- optional public-testnet RPC/funds are explicitly authorized (local persistence needs neither).

See [IMPLEMENTATION_CHECKLIST.md](IMPLEMENTATION_CHECKLIST.md) for exact current statuses. No SOTA,
zero-error, ownership, capture-time, or network-wide-search claim is made from component choice or
the six-case demo.

## License

Project code is MIT licensed. See `THIRD_PARTY.md` and `model-lock.json` for dependency/model terms.
