from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Annotated

import httpx
import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .calibration import calibrate
from .chain import EthereumAnchor
from .config import Settings
from .errors import FaceProofError, NoVerifiedMatch
from .faces import MODEL_SPECS, FaceEngine, ModelSpec, _file_sha256, download_models
from .integrity import evidence_digest, read_json
from .pipeline import DiscoveryPipeline
from .remote import SafeImageFetcher
from .search import MastodonMediaSource

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
    )


def _pipeline(
    settings: Settings, *, instance: str | None = None, tag: str | None = None
) -> DiscoveryPipeline:
    return DiscoveryPipeline(
        settings=settings,
        face_engine=FaceEngine(settings.model_dir),
        candidate_source=_source(settings, instance=instance, tag=tag),
        image_fetcher=SafeImageFetcher(
            timeout_s=settings.request_timeout_s,
            max_bytes=settings.max_image_bytes,
            max_redirects=settings.max_redirects,
        ),
        progress=lambda message: console.print(f"[cyan]->[/cyan] {message}"),
    )


def _show_match(bundle: dict) -> None:
    match = bundle["match"]
    table = Table(title="Locally verified public-post candidate", show_header=False)
    table.add_column("Field", style="dim")
    table.add_column("Value", overflow="fold")
    table.add_row("Post", str(match["post_url"]))
    table.add_row("Author", str(match["author_handle"]))
    table.add_row("Canonical URI", str(match["canonical_uri"]))
    table.add_row("Face similarity", f"{float(match['face_similarity']):.3f}")
    table.add_row("Threshold", f"{float(match['face_threshold']):.3f}")
    table.add_row("Same image/content", str(bool(match["same_content"])))
    table.add_row("Ambiguous", str(bool(match["ambiguous"])))
    table.add_row("Image SHA-256", str(match["candidate_image_sha256"]))
    console.print(table)


@models_app.command("install")
def install_models() -> None:
    """Download pinned YuNet/SFace binaries and verify their checksums."""
    settings = _settings()
    for path in download_models(settings.model_dir):
        console.print(f"[green]OK[/green] {path.name}  sha256={_file_sha256(path)}")


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
    table.add_row("Anvil executable", shutil.which("anvil") or "not on PATH")
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
        "Mastodon HTTPS source configured": settings.mastodon_instance.startswith("https://"),
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
        FaceEngine(settings.model_dir), reference, positives, negatives, output.resolve()
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


@app.command()
def discover(
    image: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
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
    bundle, evidence_path = _pipeline(settings, instance=instance, tag=tag).discover(
        image, consent_asserted=i_have_consent, output_path=output
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
) -> None:
    """Write an evidence digest to Ethereum transaction calldata and save its receipt."""
    value = read_json(evidence)
    if bool((value.get("match") or {}).get("ambiguous")):
        raise NoVerifiedMatch("refusing to anchor an ambiguous match")
    settings = _settings()
    destination = output or evidence.with_name("anchor.json")
    receipt = EthereumAnchor(settings).anchor(evidence, destination)
    location = receipt.explorer_url or f"RPC chain {receipt.chain_id}"
    console.print(
        Panel.fit(
            f"[green]Anchored[/green]\nTransaction: {receipt.transaction_hash}\n"
            f"Block: {receipt.block_number}\nLocation: {location}",
            title="Ethereum proof",
        )
    )
    console.print(f"Receipt written: {destination}")


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
    i_have_consent: Annotated[bool, typer.Option("--i-have-consent")] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
    instance: Annotated[str | None, typer.Option(help="Mastodon HTTPS origin.")] = None,
    tag: Annotated[str | None, typer.Option(help="Optional hashtag without #.")] = None,
) -> None:
    """Run face scan -> live post discovery -> chain anchor -> verification."""
    settings = _settings()
    bundle, evidence_path = _pipeline(settings, instance=instance, tag=tag).discover(
        image, consent_asserted=i_have_consent
    )
    bundle_dict = bundle.to_dict()
    _show_match(bundle_dict)
    if bundle.match["ambiguous"]:
        raise NoVerifiedMatch("top candidates are ambiguous; inspect evidence before anchoring")
    if not yes and not typer.confirm("Confirm this post and anchor its evidence fingerprint?"):
        console.print(f"Evidence preserved without blockchain write: {evidence_path}")
        raise typer.Exit()

    receipt_path = evidence_path.with_name("anchor.json")
    anchor_client = EthereumAnchor(settings)
    receipt = anchor_client.anchor(evidence_path, receipt_path)
    report = anchor_client.verify(evidence_path, receipt_path)
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
