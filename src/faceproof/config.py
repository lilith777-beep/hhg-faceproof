from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ANVIL_CHAIN_ID = 31_337


@dataclass(frozen=True, slots=True)
class Settings:
    project_root: Path
    model_dir: Path
    artifact_dir: Path
    mastodon_instance: str
    mastodon_tag: str | None
    mastodon_max_pages: int
    mastodon_page_size: int
    rpc_url: str | None
    private_key: str | None
    signer_mode: str
    expected_signer: str | None
    expected_chain_id: int
    explorer_tx_url: str
    face_threshold: float
    face_threshold_source: str
    prefilter_face_threshold: float
    ambiguity_margin: float
    phash_max_distance: int
    akaze_min_inliers: int
    akaze_min_inlier_ratio: float
    max_candidates: int
    max_finalists: int
    request_timeout_s: float = 12.0
    max_image_bytes: int = 12 * 1024 * 1024
    max_redirects: int = 3
    min_confirmations: int = 1

    @classmethod
    def load(cls, project_root: Path | None = None) -> Settings:
        root = (project_root or Path.cwd()).resolve()
        threshold_text = os.getenv("FACEPROOF_FACE_THRESHOLD")
        threshold = float(threshold_text or "0.50")
        if not 0.0 < threshold < 1.0:
            raise ValueError("FACEPROOF_FACE_THRESHOLD must be between 0 and 1")

        max_candidates = int(os.getenv("FACEPROOF_MAX_CANDIDATES", "200"))
        if not 1 <= max_candidates <= 1000:
            raise ValueError("FACEPROOF_MAX_CANDIDATES must be between 1 and 1000")

        page_size = int(os.getenv("FACEPROOF_MASTODON_PAGE_SIZE", "40"))
        if not 1 <= page_size <= 40:
            raise ValueError("FACEPROOF_MASTODON_PAGE_SIZE must be between 1 and 40")

        max_pages = int(os.getenv("FACEPROOF_MASTODON_MAX_PAGES", "5"))
        if not 1 <= max_pages <= 25:
            raise ValueError("FACEPROOF_MASTODON_MAX_PAGES must be between 1 and 25")

        max_finalists = int(os.getenv("FACEPROOF_MAX_FINALISTS", "20"))
        if not 1 <= max_finalists <= max_candidates:
            raise ValueError("FACEPROOF_MAX_FINALISTS must be between 1 and max candidates")

        prefilter = float(os.getenv("FACEPROOF_PREFILTER_FACE_THRESHOLD", "0.30"))
        if not -1.0 <= prefilter < threshold:
            raise ValueError("FACEPROOF_PREFILTER_FACE_THRESHOLD must be below the final threshold")

        ambiguity_margin = float(os.getenv("FACEPROOF_AMBIGUITY_MARGIN", "0.05"))
        if not 0.0 <= ambiguity_margin <= 0.5:
            raise ValueError("FACEPROOF_AMBIGUITY_MARGIN must be between 0 and 0.5")

        signer_mode = os.getenv("FACEPROOF_SIGNER_MODE", "unlocked")
        if signer_mode not in {"unlocked", "private-key"}:
            raise ValueError("FACEPROOF_SIGNER_MODE must be 'unlocked' or 'private-key'")

        return cls(
            project_root=root,
            model_dir=root / "models",
            artifact_dir=root / "artifacts",
            mastodon_instance=os.getenv("FACEPROOF_MASTODON_INSTANCE", "https://mastodon.social"),
            mastodon_tag=os.getenv("FACEPROOF_MASTODON_TAG") or None,
            mastodon_max_pages=max_pages,
            mastodon_page_size=page_size,
            rpc_url=os.getenv("FACEPROOF_RPC_URL", "http://127.0.0.1:8545") or None,
            private_key=os.getenv("FACEPROOF_PRIVATE_KEY") or None,
            signer_mode=signer_mode,
            expected_signer=os.getenv("FACEPROOF_EXPECTED_SIGNER") or None,
            expected_chain_id=int(os.getenv("FACEPROOF_CHAIN_ID", str(ANVIL_CHAIN_ID))),
            explorer_tx_url=os.getenv(
                "FACEPROOF_EXPLORER_TX_URL",
                "",
            ),
            face_threshold=threshold,
            face_threshold_source=(
                "FACEPROOF_FACE_THRESHOLD"
                if threshold_text
                else "conservative project default (upstream SFace demo uses 0.363)"
            ),
            prefilter_face_threshold=prefilter,
            ambiguity_margin=ambiguity_margin,
            phash_max_distance=int(os.getenv("FACEPROOF_PHASH_MAX_DISTANCE", "12")),
            akaze_min_inliers=int(os.getenv("FACEPROOF_AKAZE_MIN_INLIERS", "10")),
            akaze_min_inlier_ratio=float(os.getenv("FACEPROOF_AKAZE_MIN_INLIER_RATIO", "0.25")),
            max_candidates=max_candidates,
            max_finalists=max_finalists,
        )
