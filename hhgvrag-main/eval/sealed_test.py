"""
sealed_test.py — the mechanically-enforced sealed evaluation command (amendment 6 + P0 exit gate).

The sealed test runs EXACTLY ONCE per frozen manifest, after freeze, and its outcome is immutable.
This module is the enforcement wrapper; the actual metric computation is injected by the lead on
Forge (`--eval-module mod:fn`) so this command "exists UNEXECUTED" at the P0 gate. It:
  1. REQUIRES the frozen --manifest hash.
  2. REFUSES a dirty git worktree (uncommitted protocol changes) via `git status --porcelain`.
  3. Writes eval/sealed_runs/<manifest8>_<timestamp>/ with immutable metadata (git SHA, collection
     hashes, model revisions, hardware, seeds, cmdline, redacted env) + a COMPLETED marker.
  4. REFUSES a re-run for a manifest that already has a COMPLETED run, unless an explicit
     <manifest8>.invalidation record exists (naming why).
  5. NEVER writes config / thresholds — outcomes cannot feed back into tuning.

Secret hygiene (P0.1): the env snapshot is an allowlist of non-secret keys only; it never dumps
values (or even names) of arbitrary environment variables.
"""
from __future__ import annotations

import argparse
import glob
import importlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SEALED_BASE = os.path.join(_REPO_ROOT, "eval", "sealed_runs")
_ENV_ALLOWLIST = ("CONDA_DEFAULT_ENV", "CUDA_VISIBLE_DEVICES", "HOSTNAME", "COMPUTERNAME",
                  "NUMBER_OF_PROCESSORS", "OS", "VIRTUAL_ENV")
_SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL")


class SealedTestRefused(Exception):
    """Raised when a guard fails — the sealed run is refused, nothing is written."""


# ---- pure guards (unit-testable without git / fs) --------------------------------------
def worktree_is_clean(porcelain: str) -> bool:
    return (porcelain or "").strip() == ""


def completed_runs(base: str, manifest8: str) -> list:
    """Run dirs for this manifest that carry a COMPLETED marker."""
    return sorted(os.path.dirname(m)
                  for m in glob.glob(os.path.join(base, f"{manifest8}_*", "COMPLETED")))


def invalidation_path(base: str, manifest8: str) -> str:
    return os.path.join(base, f"{manifest8}.invalidation")


def invalidation_exists(base: str, manifest8: str) -> bool:
    return os.path.exists(invalidation_path(base, manifest8))


def preflight(manifest8: str, base: str, porcelain: str) -> tuple:
    """The full guard, pure given the porcelain string + filesystem state. Returns (ok, reason)."""
    if not manifest8:
        return False, "missing --manifest: the frozen manifest hash is required"
    if len(manifest8) != 8 or any(c not in "0123456789abcdef" for c in manifest8.lower()):
        return False, f"--manifest '{manifest8}' is not an 8-hex manifest8"
    if not worktree_is_clean(porcelain):
        first = (porcelain.strip().splitlines() or [""])[0]
        return False, (f"dirty git worktree (uncommitted protocol changes) -> commit or stash "
                       f"before a sealed run; first change: {first!r}")
    prior = completed_runs(base, manifest8)
    if prior and not invalidation_exists(base, manifest8):
        return False, (f"manifest {manifest8} already has a COMPLETED sealed run "
                       f"({os.path.basename(prior[0])}); a sealed test runs ONCE. To re-run, a "
                       f"human must create {invalidation_path(base, manifest8)} recording WHY.")
    return True, "ok"


# ---- environment / hardware capture (secret-safe) --------------------------------------
def env_snapshot() -> dict:
    """Allowlisted, non-secret env only. Records the COUNT of other vars, never their names."""
    kept = {k: os.environ[k] for k in _ENV_ALLOWLIST if k in os.environ}
    redacted = sum(1 for k in os.environ if any(m in k.upper() for m in _SECRET_MARKERS))
    return {"allowlisted": kept, "other_var_count": len(os.environ) - len(kept),
            "secret_like_vars_present": redacted}


def hardware_snapshot() -> dict:
    u = platform.uname()
    hw = {"system": u.system, "release": u.release, "machine": u.machine,
          "processor": u.processor, "python": platform.python_version()}
    try:                                # best-effort GPU line on Forge; absent locally
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            hw["gpu"] = out.stdout.strip()
    except Exception:
        pass
    return hw


def git_porcelain(repo_root: str = _REPO_ROOT) -> str:
    return subprocess.run(["git", "status", "--porcelain"], cwd=repo_root,
                          capture_output=True, text=True).stdout


def git_sha(repo_root: str = _REPO_ROOT) -> str:
    r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root,
                       capture_output=True, text=True)
    return r.stdout.strip() or "UNKNOWN"


def build_metadata(manifest8: str, *, repo_root: str = _REPO_ROOT, spec_json: str = "",
                   collection_hashes: dict = None, model_revisions: dict = None,
                   seeds: dict = None, argv=None, timestamp: str = "") -> dict:
    return {
        "manifest8": manifest8,
        "timestamp_utc": timestamp,
        "git_sha": git_sha(repo_root),
        "spec_json": spec_json,
        "collection_hashes": collection_hashes or {},
        "model_revisions": model_revisions or {},
        "seeds": seeds or {},
        "cmdline": argv if argv is not None else sys.argv,
        "hardware": hardware_snapshot(),
        "env": env_snapshot(),
        "protocol": "sealed-once; outcomes never feed tuning; config never written",
    }


# ---- the orchestrated run --------------------------------------------------------------
def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def run_sealed(manifest8: str, eval_fn, *, base: str = SEALED_BASE, repo_root: str = _REPO_ROOT,
               spec_json: str = "", collection_hashes: dict = None, model_revisions: dict = None,
               seeds: dict = None, porcelain: str = None, argv=None, timestamp: str = None) -> dict:
    """Guard -> create immutable run dir -> write metadata -> run eval ONCE -> mark COMPLETED.

    `eval_fn(run_dir) -> dict` is the injected metric computation (the lead's harness eval on
    Forge). It receives the run dir to drop artifacts into and returns a results dict. This
    wrapper NEVER writes config/thresholds and NEVER opens sealed outcomes for tuning.
    """
    porcelain = git_porcelain(repo_root) if porcelain is None else porcelain
    ok, reason = preflight(manifest8, base, porcelain)
    if not ok:
        raise SealedTestRefused(reason)

    ts = timestamp or _now()
    os.makedirs(base, exist_ok=True)
    run_dir = os.path.join(base, f"{manifest8}_{ts}")
    os.makedirs(run_dir, exist_ok=False)   # collision on the same manifest+second is a hard error

    meta = build_metadata(manifest8, repo_root=repo_root, spec_json=spec_json,
                          collection_hashes=collection_hashes, model_revisions=model_revisions,
                          seeds=seeds, argv=argv, timestamp=ts)
    with open(os.path.join(run_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    results = eval_fn(run_dir) if eval_fn is not None else {}
    with open(os.path.join(run_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(os.path.join(run_dir, "COMPLETED"), "w", encoding="utf-8") as f:
        json.dump({"manifest8": manifest8, "timestamp_utc": ts, "git_sha": meta["git_sha"]},
                  f, indent=2)
    return {"run_dir": run_dir, "metadata": meta, "results": results}


def _load_eval_fn(spec: str):
    """--eval-module 'package.module:callable' -> the injected eval callable."""
    mod_name, _, fn_name = spec.partition(":")
    mod = importlib.import_module(mod_name)
    return getattr(mod, fn_name)


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Run the sealed evaluation ONCE (mechanically enforced)")
    ap.add_argument("--manifest", required=True, help="frozen 8-hex manifest8 (REQUIRED)")
    ap.add_argument("--base", default=SEALED_BASE)
    ap.add_argument("--spec-json", default="", help="path to the frozen CorpusBuildSpec canonical JSON")
    ap.add_argument("--collection-hashes", default="", help="path to JSON {collection: hash}")
    ap.add_argument("--model-revisions", default="", help="path to JSON {model: revision}")
    ap.add_argument("--eval-module", default="",
                    help="dotted 'module:callable' that computes sealed metrics; WITHOUT it the "
                         "command only preflights (stays UNEXECUTED for the P0 gate)")
    args = ap.parse_args(argv)

    porcelain = git_porcelain()
    ok, reason = preflight(args.manifest, args.base, porcelain)
    print(f"preflight: {'OK' if ok else 'REFUSED'} -- {reason}")
    if not ok:
        return 3
    if not args.eval_module:
        print("READY (unexecuted). Wire --eval-module <module:callable> on Forge to run once.")
        return 0

    def _read_json(path):
        if not path:
            return {}
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    spec_json = ""
    if args.spec_json:
        with open(args.spec_json, encoding="utf-8") as f:
            spec_json = f.read()
    try:
        out = run_sealed(args.manifest, _load_eval_fn(args.eval_module), base=args.base,
                         spec_json=spec_json,
                         collection_hashes=_read_json(args.collection_hashes),
                         model_revisions=_read_json(args.model_revisions),
                         argv=(argv if argv is not None else sys.argv))
    except SealedTestRefused as e:
        print(f"REFUSED: {e}")
        return 3
    print(f"SEALED RUN COMPLETE: {out['run_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
