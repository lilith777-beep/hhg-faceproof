# Do this next — one small box at a time

The software path is ready. These are human/data gates; do not do them all at once.

## Step 1 — permissions

- [ ] Make one private consent-reference document per participant.
- [ ] Confirm it permits **both enrollment and candidate biometric matching** where applicable.
- [ ] Give each document an opaque ID (do not commit the document).

Stop. Take a break.

## Step 2 — photos and split manifest

- [ ] Put consented files in a private folder outside Git.
- [ ] Include independent same-person photos, hard different-person negatives, originals/edits,
  similar noncopies/backgrounds, crops, small faces, masks/glasses/hair/hands/overlays/profiles,
  and a consented composite/conflict.
- [ ] Assign participant, capture-session, and source-family IDs.
- [ ] Assign each identity and every derivative family to exactly one of development, calibration,
  or locked test.
- [ ] Add manual eyes/nose/mouth visibility and coarse pose-bin annotations.
- [ ] Build the JSONL manifest described in `README.md`, using actual SHA-256 values.

Stop. Send Codex only: **manifest ready** and its local path. Do not send private photos in chat.

## Step 3 — controlled Mastodon posts (Sudarshana)

- [ ] Use controlled accounts whose represented people gave documented biometric permission.
- [ ] Publish at least six labeled media cases across at least three public posts.
- [ ] Include one multi-image post and: original, edited copy, independent same-person photo,
  different consenting person, heavy occlusion, and composite/conflict.
- [ ] Use one shared hashtag and check every post in an incognito window.
- [ ] Record the instance, hashtag, authorized account handles/IDs, post IDs, media IDs, and hashes
  in the private evaluator manifest.

Stop. Send Codex: **posts ready**, instance, hashtag, and authorized handles. Never send credentials.

## Step 4 — reviewer

- [ ] Choose the human reviewer ID that will appear in final evidence.
- [ ] Decide the private-media retention/deletion period.

Then Codex can run calibration, freeze policy, run the untouched test once, verify preview recall,
run the genuine live discovery, seal reviewed evidence, and record the persistent-chain demo.

Repository: `https://github.com/BlueBlaze6335/hhg-faceproof`. Do not submit until the locked report,
live multi-post ledger, screen recording, and public-repo secret/privacy audit all pass.
