# Do this next

The code path is consolidated. These are the remaining human-data gates. Do one box at a time.

## 1. Put the photos in folders

- [ ] Put one clear, single-face photo at `demo-input/real/query.jpg`.
- [ ] Put varied consenting same-person photos in `calibration/positive/` (pose, light, crop,
      compression, expression, and appropriate occlusion).
- [ ] Put representative consenting different-person and look-alike hard-negative photos in
      `calibration/negative/`.
- [ ] Keep a separate `copy-calibration/positive/` set of transformed copies of the exact query
      and `copy-calibration/negative/` set of unrelated images. Do not mix these labels with the
      face-identity folders.

Stop here. Tell Codex: **photos ready**.

## 2. Make one public Mastodon post

- [ ] Post the query or same-person image publicly.
- [ ] Add a shared hashtag with multiple image posts, such as the current event hashtag.
- [ ] Open the post in a private/incognito window to confirm it is public.
- [ ] Copy the instance name, hashtag, and post URL.

Stop here. Send Codex those three values.

## 3. Send Codex the values

- [ ] Reply with **photos ready**, Mastodon instance, hashtag, and public post URL.

The approved public repository is `hhg-faceproof`. Codex can then run calibration, genuine live
search, full blockchain verification, and a clean public-repo audit. Do not submit the form until
the screen recording and repository have both passed an incognito check.
