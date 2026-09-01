# Security policy

FaceProof processes biometric data. Use it only with explicit, informed consent and only for
matching-content discovery—not identity inference, surveillance, access control, or high-impact
decisions.

Never commit `.env`, private keys, source/calibration photos, downloaded models, or the generated
`artifacts/` directory. The default local Anvil keys are public development keys: never fund them
with real assets and never expose Anvil outside localhost. If using the optional public-testnet
profile, use a fresh testnet-only account.

If a security defect could expose credentials, biometric data, or permit server-side request
forgery, do not open a public issue containing exploit details. Contact the repository owner
privately through the security-reporting channel configured on GitHub.
