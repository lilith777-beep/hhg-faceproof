# Third-party components

FaceProof source is MIT licensed. Dependencies remain under their own licenses.

| Component | Role | License/provenance note |
|---|---|---|
| Python | runtime | PSF License |
| OpenCV / opencv-python | image, YuNet/SFace APIs, AKAZE | Apache-2.0 |
| OpenCV Zoo YuNet model | face detection | Upstream model directory declares MIT |
| OpenCV Zoo SFace model | face encoding | Upstream directory declares Apache-2.0; packaged-weight training provenance is not sufficiently documented for a fairness claim |
| Meta SSCD DISC-Mixup model/code | copy detection | MIT; official repository is archived, so FaceProof pins the 512-D TorchScript checkpoint checksum and documents the maintenance risk |
| PyTorch | local SSCD inference | BSD-3-Clause |
| FAISS | exact cosine vector retrieval | MIT |
| NumPy | numerical operations | BSD-3-Clause |
| httpx | bounded HTTP client | BSD-3-Clause |
| Typer / Rich | CLI | MIT |
| web3.py | Ethereum RPC/transactions | MIT |
| python-dotenv | local configuration | BSD-3-Clause |
| Mastodon | live public social source/server | AGPL-3.0; FaceProof only calls its documented HTTP API |
| Foundry Anvil | local Ethereum-compatible node | MIT OR Apache-2.0 |

Model binaries are downloaded from pinned upstream OpenCV Zoo and Meta URLs by `faceproof models
install`, verified by SHA-256, ignored by Git, and not redistributed in this repository.
