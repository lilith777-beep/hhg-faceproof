from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated
from urllib.parse import urlsplit

import httpx
import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .acceptance import _local_anvil_path, run_chain_acceptance, run_placeholder_acceptance
from .calibration import calibrate, calibrate_copy
from .chain import EthereumAnchor
from .config import Settings
from .copydetect import (
    SSCD,
    SSCDDescriptorEngine,
    file_sha256,
    install_sscd_model,
    load_sscd_engine,
)
from .dataset import load_manifest
from .errors import FaceProofError, NoVerifiedMatch
from .evaluation import (
    ManifestImageFetcher,
    ManifestMediaSource,
    calibrate_from_evaluation,
    run_manifest_evaluation,
)
from .face_analysis import (
    PARSER_FILENAME,
    PARSER_SHA256,
    FaceParser,
    FaceQualityAssessor,
    install_parser_model,
)
from .faces import MODEL_SPECS, FaceEngine, ModelSpec, _file_sha256, download_models
from .integrity import (
    evidence_digest,
    evidence_file_digest,
    finalize_evidence,
    read_json,
    write_json,
)
from .pipeline import DiscoveryPipeline
from .policy import freeze_policy, load_policy
from .remote import SafeImageFetcher
from .search import CandidateSource, MastodonMediaSource

load_dotenv()
console = Console()
app = typer.Typer(
    name="faceproof",
    no_args_is_help=True,
    help="Consent-first public-post face matching with local Ethereum verification.",
)
models_app = typer.Typer(no_args_is_help=True, help="Install and inspect pinned face models.")
source_app = typer.Typer(no_args_is_help=True, help="Probe the live public-media source.")
chain_app = typer.Typer(no_args_is_help=True, help="Inspect the local blockchain verifier.")
app.add_typer(models_app, name="models")
app.add_typer(source_app, name="source")
app.add_typer(chain_app, name="chain")


def _settings() -> Settings:
    return Settings.load(Path.cwd())


def _source(
    settings: Settings, *, instance: str | None = None, tag: str | None = None
) -> MastodonMediaSource:
    return MastodonMediaSource(
        instance=instance or settings.mastodon_instance,
        tag=tag if tag is not None else settings.mastodon_tag,
        max_pages=settings.mastodon_max_pages,
        page_size=settings.mastodon_page_size,
        max_media=settings.max_candidates,
        timeout_s=settings.request_timeout_s,
        allowed_accounts=frozenset(settings.mastodon_allowed_accounts) or None,
    )


def _pipeline(
    settings: Settings,
    *,
    instance: str | None = None,
    tag: str | None = None,
    manifest: Path | None = None,
    policy_path: Path | None = None,
    source_override: CandidateSource | None = None,
    fetcher_override: object | None = None,
    enable_copy: bool = True,
    enable_quality: bool = True,
) -> DiscoveryPipeline:
    face_engine = FaceEngine(
        settings.model_dir, detection_threshold=settings.detection_threshold
    )
    selected_policy = policy_path or (
        settings.project_root / "policy" / "provisional-review-only.json"
    )
    policy = load_policy(selected_policy, project_root=settings.project_root)
    parser = FaceParser(settings.model_dir, face_engine.cv2) if enable_quality else None
    return DiscoveryPipeline(
        settings=settings,
        face_engine=face_engine,
        candidate_source=source_override
        or _source(settings, instance=instance, tag=tag),
        image_fetcher=fetcher_override
        or SafeImageFetcher(
            timeout_s=settings.request_timeout_s,
            max_bytes=settings.max_image_bytes,
            max_redirects=settings.max_redirects,
            allowed_hosts=frozenset(
                {
                    str(urlsplit(instance or settings.mastodon_instance).hostname).lower(),
                    *settings.media_allowed_hosts,
                }
            ),
        ),
        copy_engine=_copy_engine(settings) if enable_copy else None,
        consent_manifest=load_manifest(manifest) if manifest is not None else None,
        policy=policy,
        quality_assessor=(
            FaceQualityAssessor(
                face_engine.cv2,
                parser,
                pose_policy_validated=policy.pose_policy_validated,
                visibility_policy_validated=policy.visibility_policy_validated,
            )
            if enable_quality
            else None
        ),
        progress=lambda message: console.print(f"[cyan]->[/cyan] {message}"),
    )


def _copy_engine(settings: Settings) -> SSCDDescriptorEngine | None:
    engine = load_sscd_engine(
        settings.model_dir,
        mode=settings.sscd_mode,
        device=settings.sscd_device,
        batch_size=settings.sscd_batch_size,
    )
    if engine is None and settings.sscd_mode != "off":
        console.print(
            "[yellow]SSCD unavailable; continuing with the legacy face/pHash path. "
            "Install the vision extra and models for the production ensemble.[/yellow]"
        )
    return engine


def _show_match(bundle: dict) -> None:
    match = bundle["match"]
    table = Table(title="FaceProof public-post candidate assessment", show_header=False)
    table.add_column("Field", style="dim")
    table.add_column("Value", overflow="fold")
    table.add_row("Post", str(match["post_url"]))
    table.add_row("Author", str(match["author_handle"]))
    table.add_row("Canonical URI", str(match["canonical_uri"]))
    table.add_row("Face similarity", f"{float(match['face_similarity']):.3f}")
    face_threshold = match.get("face_threshold")
    table.add_row(
        "Face threshold",
        f"{float(face_threshold):.3f}" if face_threshold is not None else "not calibrated",
    )
    table.add_row("Face match", str(bool(match.get("face_match", True))))
    if match.get("sscd_similarity") is not None:
        table.add_row("SSCD similarity", f"{float(match['sscd_similarity']):.3f}")
        sscd_threshold = match.get("sscd_threshold")
        table.add_row(
            "SSCD threshold",
            f"{float(sscd_threshold):.3f}"
            if sscd_threshold is not None
            else "not calibrated",
        )
    table.add_row("Decision", str(match.get("decision", "legacy_face_match")))
    table.add_row("Same image/content", str(bool(match["same_content"])))
    table.add_row("Ambiguous", str(bool(match["ambiguous"])))
    table.add_row("Image SHA-256", str(match["candidate_image_sha256"]))
    console.print(table)


@models_app.command("install")
def install_models() -> None:
    """Download pinned YuNet/SFace/SSCD binaries and verify their checksums."""
    settings = _settings()
    for path in download_models(settings.model_dir):
        console.print(f"[green]OK[/green] {path.name}  sha256={_file_sha256(path)}")
    path = install_sscd_model(settings.model_dir)
    console.print(f"[green]OK[/green] {path.name}  sha256={file_sha256(path)}")
    path = install_parser_model(settings.model_dir)
    console.print(f"[green]OK[/green] {path.name}  sha256={file_sha256(path)}")


@models_app.command("status")
def model_status() -> None:
    """Check that every local model matches the pinned checksum."""
    settings = _settings()
    failed = False
    for spec in MODEL_SPECS:
        path = settings.model_dir / spec.filename
        actual = _file_sha256(path) if path.exists() else "missing"
        valid = actual == spec.sha256
        failed = failed or not valid
        status = "[green]OK[/green]" if valid else "[red]FAIL[/red]"
        console.print(f"{status} {spec.filename}: {actual}")
    path = settings.model_dir / SSCD.filename
    actual = file_sha256(path) if path.exists() else "missing"
    valid = actual == SSCD.sha256
    failed = failed or not valid
    status = "[green]OK[/green]" if valid else "[red]FAIL[/red]"
    console.print(f"{status} {SSCD.filename}: {actual}")
    path = settings.model_dir / PARSER_FILENAME
    actual = file_sha256(path) if path.exists() else "missing"
    valid = actual == PARSER_SHA256
    failed = failed or not valid
    status = "[green]OK[/green]" if valid else "[red]FAIL[/red]"
    console.print(f"{status} {PARSER_FILENAME}: {actual}")
    if failed:
        raise typer.Exit(1)


@source_app.command("probe")
def source_probe(
    instance: Annotated[str | None, typer.Option(help="Mastodon HTTPS origin.")] = None,
    tag: Annotated[str | None, typer.Option(help="Optional hashtag without #.")] = None,
) -> None:
    """Perform a genuine, read-only live search and show its bounded scope."""
    settings = _settings()
    batch = _source(settings, instance=instance, tag=tag).discover()
    table = Table(title="Live open social-source probe", show_header=False)
    table.add_column("Field", style="dim")
    table.add_column("Value", overflow="fold")
    table.add_row("Provider", batch.provider)
    table.add_row("Scope", batch.scope)
    table.add_row("Endpoint", batch.endpoint)
    table.add_row("Pages", str(batch.pages_fetched))
    table.add_row("Statuses", str(batch.statuses_scanned))
    table.add_row("Image media", str(batch.media_scanned))
    if batch.hits:
        table.add_row("Example live post", batch.hits[0].post_url)
    console.print(table)


@chain_app.command("status")
def chain_status() -> None:
    """Check the configured chain, expected network, and local Anvil binary."""
    settings = _settings()
    connected = False
    actual_chain = "unreachable"
    try:
        anchor = EthereumAnchor(settings)
        actual_chain = str(anchor.chain_id)
        connected = anchor.chain_id == settings.expected_chain_id
    except FaceProofError:
        pass
    table = Table(title="Blockchain verifier")
    table.add_column("Check")
    table.add_column("State")
    table.add_row("Anvil executable", _local_anvil_path(settings) or "not found")
    table.add_row("RPC", settings.rpc_url or "not configured")
    table.add_row("Expected chain", str(settings.expected_chain_id))
    table.add_row("Actual chain", actual_chain)
    table.add_row("Ready", "[green]yes[/green]" if connected else "[yellow]no[/yellow]")
    console.print(table)


@app.command()
def doctor() -> None:
    """Validate the local, open-source execution prerequisites."""
    settings = _settings()
    checks = {
        "YuNet model pinned": _model_is_valid(settings.model_dir, MODEL_SPECS[0]),
        "SFace model pinned": _model_is_valid(settings.model_dir, MODEL_SPECS[1]),
        "SSCD model pinned": (
            file_sha256(settings.model_dir / SSCD.filename) == SSCD.sha256
            if (settings.model_dir / SSCD.filename).exists()
            else False
        ),
        "Face parser pinned": (
            file_sha256(settings.model_dir / PARSER_FILENAME) == PARSER_SHA256
            if (settings.model_dir / PARSER_FILENAME).exists()
            else False
        ),
        "Face parser loads and infers in OpenCV": _parser_loads(settings),
        "FAISS runtime": _module_available("faiss"),
        "PyTorch runtime": _module_available("torch"),
        "Mastodon HTTPS source configured": settings.mastodon_instance.startswith("https://"),
        "Shared Mastodon hashtag configured": bool(settings.mastodon_tag),
        "Authorized Mastodon accounts configured": bool(settings.mastodon_allowed_accounts),
        "Private-media retention policy specified": (
            settings.private_media_retention_policy != "UNSPECIFIED"
        ),
        "Ethereum RPC configured": bool(settings.rpc_url),
        "Signer configured": settings.signer_mode == "unlocked" or bool(settings.private_key),
    }
    table = Table(title="FaceProof doctor")
    table.add_column("Check")
    table.add_column("State")
    for name, passed in checks.items():
        table.add_row(name, "[green]ready[/green]" if passed else "[yellow]needed[/yellow]")
    console.print(table)
    if not all(checks.values()):
        console.print("Copy .env.example to .env and fill only the missing values.")


def _model_is_valid(model_dir: Path, spec: ModelSpec) -> bool:
    path = model_dir / spec.filename
    return path.exists() and _file_sha256(path) == spec.sha256


def _parser_loads(settings: Settings) -> bool:
    try:
        engine = FaceEngine(settings.model_dir, detection_threshold=settings.detection_threshold)
        parser = FaceParser(settings.model_dir, engine.cv2)
        fixture = Path(__file__).with_name("fixtures") / "query.png"
        image = engine.decode(fixture.read_bytes())
        face = engine.require_single_query_face(image)
        result = parser.parse(image, face)
        return result.semantic_mask is not None and result.semantic_mask.ndim == 2
    except FaceProofError:
        return False


def _module_available(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None


@app.command()
def acceptance(
    skip_chain: Annotated[
        bool,
        typer.Option(help="Run only local biometric/evidence checks without Anvil."),
    ] = False,
) -> None:
    """Run the bundled synthetic biometric, evidence, chain, and tamper gates."""
    settings = _settings()
    biometric = run_placeholder_acceptance(settings)
    checks = dict(biometric["checks"])
    chain_result: dict | None = None
    if not skip_chain:
        chain_result = run_chain_acceptance(settings, biometric["evidence_path"])
        checks["all_chain_checks"] = all(chain_result["verification_checks"].values())
        checks["tamper_rejected"] = bool(chain_result["tamper_rejected"])

    table = Table(title="Synthetic release acceptance")
    table.add_column("Check")
    table.add_column("Result")
    for name, passed in checks.items():
        table.add_row(
            name.replace("_", " "), "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
        )
    console.print(table)
    console.print(
        f"Positive {biometric['positive_score']:.3f} | "
        f"negative {biometric['negative_score']:.3f} | "
        f"threshold {biometric['threshold']:.3f}"
    )
    if biometric.get("sscd_positive_score") is not None:
        console.print(
            f"SSCD positive {biometric['sscd_positive_score']:.3f} | "
            f"negative {biometric['sscd_negative_score']:.3f} | "
            "retrieval FAISS exact cosine"
        )
    console.print(f"Evidence: {biometric['evidence_path']}")
    if chain_result:
        console.print(f"Transaction: {chain_result['transaction_hash']}")
    console.print(
        "[yellow]Synthetic acceptance is not the final live-post proof. "
        "A consenting human image and genuinely discovered public post remain required.[/yellow]"
    )


@app.command()
def calibrate_threshold(
    reference: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    positives: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
    negatives: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("calibration.json"),
) -> None:
    """Measure a threshold from consented positive and representative negative faces."""
    settings = _settings()
    report = calibrate(
        FaceEngine(settings.model_dir, detection_threshold=settings.detection_threshold),
        reference,
        positives,
        negatives,
        output.resolve(),
    )
    metrics = report["metrics"]
    console.print(
        Panel.fit(
            f"Recommended threshold: {report['recommended_threshold']}\n"
            f"Balanced accuracy: {metrics['balanced_accuracy']:.3f}\n"
            f"False accept rate: {metrics['false_accept_rate']:.3f}\n"
            f"False reject rate: {metrics['false_reject_rate']:.3f}\n"
            f"Report: {output.resolve()}\n\n"
            "Set FACEPROOF_FACE_THRESHOLD to the recommendation only after reviewing samples.",
            title="Face threshold calibration",
        )
    )


@app.command("calibrate-copy")
def calibrate_copy_threshold(
    reference: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    positives: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
    negatives: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("copy-calibration.json"),
) -> None:
    """Measure SSCD on derived-copy positives and unrelated-image negatives."""
    settings = _settings()
    face_engine = FaceEngine(settings.model_dir, detection_threshold=settings.detection_threshold)
    copy_engine = load_sscd_engine(
        settings.model_dir,
        mode="required",
        device=settings.sscd_device,
        batch_size=settings.sscd_batch_size,
    )
    if copy_engine is None:  # pragma: no cover - required mode always raises instead
        raise NoVerifiedMatch("SSCD is required for copy calibration")
    report = calibrate_copy(
        copy_engine,
        face_engine.decode,
        reference,
        positives,
        negatives,
        output.resolve(),
    )
    metrics = report["metrics"]
    console.print(
        Panel.fit(
            f"Recommended SSCD threshold: {report['recommended_threshold']}\n"
            f"Balanced accuracy: {metrics['balanced_accuracy']:.3f}\n"
            f"False accept rate: {metrics['false_accept_rate']:.3f}\n"
            f"False reject rate: {metrics['false_reject_rate']:.3f}\n"
            f"Report: {output.resolve()}\n\n"
            "Only adopt this threshold after held-out transform evaluation.",
            title="Copy threshold calibration",
        )
    )


@app.command("evaluate-manifest")
def evaluate_manifest(
    manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    policy: Annotated[Path, typer.Option("--policy", exists=True, dir_okay=False)],
    split: Annotated[str, typer.Option("--split")] = "test",
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("evaluation.json"),
    mode: Annotated[str, typer.Option("--mode")] = "dual-quality",
) -> None:
    """Run the actual exhaustive pipeline over one identity-disjoint manifest split."""
    settings = _settings()
    dataset = load_manifest(manifest)
    if mode not in {"baseline", "dual", "dual-quality"}:
        raise typer.BadParameter("mode must be baseline, dual, or dual-quality")
    source = ManifestMediaSource(dataset, split)
    pipeline = _pipeline(
        settings,
        manifest=manifest,
        policy_path=policy,
        source_override=source,
        fetcher_override=ManifestImageFetcher(dataset),
        enable_copy=mode != "baseline",
        enable_quality=mode == "dual-quality",
    )
    report = run_manifest_evaluation(
        dataset, pipeline, split=split, output=output.resolve()
    )
    console.print_json(json.dumps(report["metrics"], ensure_ascii=False))
    console.print(f"[green]Evaluation report:[/green] {output.resolve()}")


@app.command("compare-manifest")
def compare_manifest(
    manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    policy: Annotated[Path, typer.Option("--policy", exists=True, dir_okay=False)],
    split: Annotated[str, typer.Option("--split")] = "test",
    output_dir: Annotated[Path, typer.Option("--output-dir")] = Path("comparison"),
) -> None:
    """Run baseline, dual, and dual-plus-quality through the same frozen manifest."""
    settings = _settings()
    dataset = load_manifest(manifest)
    destination = output_dir.resolve()
    for mode in ("baseline", "dual", "dual-quality"):
        pipeline = _pipeline(
            settings,
            manifest=manifest,
            policy_path=policy,
            source_override=ManifestMediaSource(dataset, split),
            fetcher_override=ManifestImageFetcher(dataset),
            enable_copy=mode != "baseline",
            enable_quality=mode == "dual-quality",
        )
        output = destination / f"{split}-{mode}.json"
        report = run_manifest_evaluation(
            dataset, pipeline, split=split, output=output
        )
        report["comparison_mode"] = mode
        write_json(output, report)
        console.print(f"[green]{mode}:[/green] {output}")


@app.command("freeze-policy")
def freeze_policy_command(
    calibration_report: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    test_report: Annotated[
        Path | None, typer.Option("--test-report", exists=True, dir_okay=False)
    ] = None,
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("policy.json"),
    max_gallery_media: Annotated[int, typer.Option("--max-gallery-media")] = 200,
    max_enrollment_references: Annotated[int, typer.Option("--max-enrollment-references")] = 8,
    max_copy_references: Annotated[int, typer.Option("--max-copy-references")] = 8,
) -> None:
    """Freeze calibrated operating limits and immutable report/model hashes."""
    settings = _settings()
    freeze_policy(
        calibration_report,
        test_report,
        project_root=settings.project_root,
        output=output.resolve(),
        max_gallery_media=max_gallery_media,
        max_enrollment_references=max_enrollment_references,
        max_copy_references=max_copy_references,
    )
    console.print(f"[green]Frozen policy:[/green] {output.resolve()}")


@app.command("calibrate-manifest-report")
def calibrate_manifest_report(
    evaluation: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    policy_id: Annotated[str, typer.Option("--policy-id")],
    output: Annotated[Path, typer.Option("--output", "-o")] = Path(
        "calibration-decision.json"
    ),
    fpir_upper_target: Annotated[float, typer.Option("--fpir-upper-target")] = 0.01,
    tpir_lower_target: Annotated[float, typer.Option("--tpir-lower-target")] = 0.90,
) -> None:
    """Freeze thresholds from a calibration-split pipeline report, never test data."""
    calibrate_from_evaluation(
        evaluation,
        output.resolve(),
        policy_id=policy_id,
        fpir_upper_target=fpir_upper_target,
        tpir_lower_target=tpir_lower_target,
    )
    console.print(f"[green]Frozen calibration decision:[/green] {output.resolve()}")


@app.command()
def discover(
    image: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    copy_reference: Annotated[
        list[Path],
        typer.Option("--copy-reference", help="Explicit original image to search for copies."),
    ],
    manifest: Annotated[Path, typer.Option("--manifest", exists=True, dir_okay=False)],
    face_index: Annotated[list[int] | None, typer.Option("--face-index")] = None,
    policy: Annotated[Path | None, typer.Option("--policy", dir_okay=False)] = None,
    i_have_consent: Annotated[
        bool,
        typer.Option(
            "--i-have-consent",
            help="Assert that this is your face or the person explicitly agreed to the search.",
        ),
    ] = False,
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
    instance: Annotated[str | None, typer.Option(help="Mastodon HTTPS origin.")] = None,
    tag: Annotated[str | None, typer.Option(help="Optional hashtag without #.")] = None,
) -> None:
    """Detect one face, search public media, verify locally, and write evidence."""
    settings = _settings()
    bundle, evidence_path = _pipeline(
        settings, instance=instance, tag=tag, manifest=manifest, policy_path=policy
    ).discover(
        image,
        copy_reference_paths=copy_reference,
        enrollment_face_indices=face_index,
        consent_asserted=i_have_consent,
        output_path=output,
    )
    _show_match(bundle.to_dict())
    console.print(f"[green]Evidence written:[/green] {evidence_path}")
    console.print(f"[green]Evidence SHA-256:[/green] {evidence_digest(bundle.to_dict())}")
    if bundle.match["ambiguous"]:
        console.print("[yellow]Do not anchor yet: the leading candidates are ambiguous.[/yellow]")


@app.command()
def anchor(
    evidence: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
    allow_public_testnet: Annotated[
        bool, typer.Option("--allow-public-testnet")
    ] = False,
) -> None:
    """Store a finalized evidence digest in the pinned registry and save its receipt."""
    value = read_json(evidence)
    if bool((value.get("match") or {}).get("ambiguous")):
        raise NoVerifiedMatch("refusing to anchor an ambiguous match")
    settings = _settings()
    destination = output or evidence.with_name("anchor.json")
    receipt = EthereumAnchor(settings).anchor(
        evidence, destination, allow_public_testnet=allow_public_testnet
    )
    location = receipt.explorer_url or f"RPC chain {receipt.chain_id}"
    console.print(
        Panel.fit(
            f"[green]Anchored[/green]\nTransaction: {receipt.transaction_hash}\n"
            f"Block: {receipt.block_number}\nLocation: {location}",
            title="Ethereum proof",
        )
    )
    console.print(f"Receipt written: {destination}")


@app.command("finalize-evidence")
def finalize_evidence_command(
    draft: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    reviewer: Annotated[str, typer.Option("--reviewer")],
    confirm_claim: Annotated[list[str], typer.Option("--confirm-claim")],
    notes: Annotated[str, typer.Option("--notes")] = "",
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
) -> None:
    """Seal a human-reviewed evidence draft into exact immutable bytes."""
    destination = output or draft.with_name("evidence.final.json")
    finalize_evidence(
        draft,
        destination,
        reviewer=reviewer,
        confirmed_claims=confirm_claim,
        notes=notes,
    )
    console.print(f"[green]Final evidence:[/green] {destination}")
    console.print(f"[green]Domain-separated SHA-256:[/green] {evidence_file_digest(destination)}")


@app.command()
def verify(
    evidence: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    receipt: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
) -> None:
    """Recompute evidence and verify it against the actual on-chain transaction."""
    report = EthereumAnchor(_settings()).verify(evidence, receipt)
    table = Table(title="Independent on-chain re-verification")
    table.add_column("Check")
    table.add_column("Result")
    for name, passed in report.checks.items():
        table.add_row(
            name.replace("_", " "), "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
        )
    for name, status in report.reproduction.items():
        table.add_row(f"reproduction: {name.replace('_', ' ')}", status)
    console.print(table)
    console.print(
        Panel.fit(
            f"[bold green]VERIFIED[/bold green]\n{report.evidence_sha256}\n"
            f"{report.confirmations} confirmation(s)",
            title="FaceProof",
        )
    )


@app.command()
def run(
    image: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    copy_reference: Annotated[list[Path], typer.Option("--copy-reference")],
    manifest: Annotated[Path, typer.Option("--manifest", exists=True, dir_okay=False)],
    reviewer: Annotated[str, typer.Option("--reviewer")],
    face_index: Annotated[list[int] | None, typer.Option("--face-index")] = None,
    policy: Annotated[Path | None, typer.Option("--policy", dir_okay=False)] = None,
    i_have_consent: Annotated[bool, typer.Option("--i-have-consent")] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
    allow_public_testnet: Annotated[
        bool, typer.Option("--allow-public-testnet")
    ] = False,
    instance: Annotated[str | None, typer.Option(help="Mastodon HTTPS origin.")] = None,
    tag: Annotated[str | None, typer.Option(help="Optional hashtag without #.")] = None,
) -> None:
    """Run face scan -> live post discovery -> chain anchor -> verification."""
    settings = _settings()
    bundle, evidence_path = _pipeline(
        settings, instance=instance, tag=tag, manifest=manifest, policy_path=policy
    ).discover(
        image,
        copy_reference_paths=copy_reference,
        enrollment_face_indices=face_index,
        consent_asserted=i_have_consent,
    )
    bundle_dict = bundle.to_dict()
    _show_match(bundle_dict)
    if bundle.match["ambiguous"]:
        raise NoVerifiedMatch("top candidates are ambiguous; inspect evidence before anchoring")
    if not yes and not typer.confirm("Confirm this post and anchor its evidence fingerprint?"):
        console.print(f"Evidence preserved without blockchain write: {evidence_path}")
        raise typer.Exit()

    final_path = evidence_path.with_name("evidence.final.json")
    finalize_evidence(
        evidence_path,
        final_path,
        reviewer=reviewer,
        confirmed_claims=list(bundle.claims["reviewable_claims"]),
        notes="confirmed interactively by the named reviewer",
    )
    receipt_path = evidence_path.with_name("anchor.json")
    anchor_client = EthereumAnchor(settings)
    receipt = anchor_client.anchor(
        final_path, receipt_path, allow_public_testnet=allow_public_testnet
    )
    report = anchor_client.verify(final_path, receipt_path)
    console.print(
        Panel.fit(
            "\n".join(
                [
                    "[bold green]END-TO-END VERIFIED[/bold green]",
                    f"Post: {bundle_dict['match']['post_url']}",
                    f"Evidence: {report.evidence_sha256}",
                    f"Transaction: {receipt.transaction_hash}",
                    f"Block: {report.block_number}",
                ]
            ),
            title="Face scan -> public post -> blockchain proof",
        )
    )


@app.command("show-evidence")
def show_evidence(
    evidence: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
) -> None:
    """Print saved evidence and its reproducible SHA-256."""
    value = read_json(evidence)
    console.print_json(json.dumps(value, ensure_ascii=False))
    console.print(f"SHA-256: {evidence_digest(value)}")


def main() -> None:
    try:
        app()
    except (FaceProofError, httpx.HTTPError) as exc:
        console.print(f"[bold red]FaceProof stopped:[/bold red] {exc}")
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
