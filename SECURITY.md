# Security policy

FaceProof processes biometric-derived data. Use it only for the bounded purpose and people covered
by documented, informed permission. Public visibility or control of a posting account is not
permission to encode a face. Do not use FaceProof for indiscriminate identification, surveillance,
access control, automatic enforcement, or other high-impact decisions.

Face and SSCD descriptors stay in memory by default. Evidence and blockchain data must never
contain embeddings, access tokens, private keys, private images, or identifying source URLs on
chain. Private media needs a documented retention/deletion policy; a hash does not make retention
safe or guarantee future availability.

Never commit `.env`, credentials, consent records, source/evaluation photos, downloaded weights,
generated evidence, chain state, or artifacts. Default Anvil keys are public development keys:
never fund them or expose Anvil outside loopback. Mainnet transactions are prohibited. An optional
public-testnet run must use explicit CLI authorization and a disposable testnet-only signer.

The media fetcher accepts HTTPS from approved hosts, checks public DNS and the connected peer on
every hop, bounds redirects/time/streamed bytes, validates type signatures, and enforces decode and
pixel limits. Loopback media access exists only as an explicit test configuration.

For credential, biometric-data, SSRF, or consent-boundary defects, contact the repository owner
privately rather than placing exploit details or personal data in a public issue.
