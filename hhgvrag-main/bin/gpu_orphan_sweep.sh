#!/usr/bin/env bash
# gpu_orphan_sweep.sh — surgically reap hhgvrag GPU orphans (the renamed vLLM engine).
#
# WHY THIS EXISTS
#   vLLM spawns an "EngineCore" child that calls setproctitle("VLLM::EngineCore"). A crash or a
#   non-clean stop can leave that child alive, still pinning ~24 GB of A6000 VRAM, with a name
#   that NO name-based matcher (pkill -f vllm, pkill EngineCore) will reliably catch. The next
#   server start then fails to allocate VRAM.
#
#   The GPU is SHARED with the user's desktop (Xorg, gnome-shell, chrome, rustdesk). Broad kills
#   are unacceptable. The ONLY safe identity is the executable path: the hhgvrag engine's
#   /proc/<pid>/exe always resolves UNDER the conda env prefix below. Nothing on the desktop does.
#
# WHAT IT KILLS (the entire allowlist — all must hold):
#   1. /proc/<pid>/exe resolves under CONDA_PREFIX_DIR  (the hard safety gate)
#   2. pid is NOT the live server MainPID and NOT a descendant of it (never kill the running server)
#   3. pid is NOT this script or its parent
#
# CAVEAT (documented operational rule): because the gate is "any process running the hhgvrag
#   conda Python", a manual `preflight`/`poststop` run WILL also target an unrelated script you
#   launched from that same env (e.g. a manual `python src/build_real_index.py`). The live server
#   tree is auto-protected (rule 2), but ad-hoc conda-env jobs are not. Do not restart the service
#   while a separate conda-env Python job is running, or pass --dry-run first. See docs/RUNBOOK.md.
#
# MODES
#   preflight   run before ExecStart: clear ghosts from a prior crashed instance
#   poststop    run after ExecStopPost: reap any child that survived control-group SIGTERM
#   (manual)    with --dry-run and no mode: just list what WOULD match
#
# FLAGS
#   --dry-run   print what WOULD be killed and exit 0 without killing anything (REQUIRED for drills)
#
# Exit code is 0 for every valid invocation (a sweep hiccup must never block server startup).
# Only a usage error exits non-zero (2).

set -u

CONDA_PREFIX_DIR="/home/padmanabha/anaconda3/envs/hhgvrag313"
SERVICE="hhgvrag.service"
TAG="hhgvrag-sweep"

DRY_RUN=0
MODE=""
for a in "$@"; do
  case "$a" in
    --dry-run) DRY_RUN=1 ;;
    preflight|poststop) MODE="$a" ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)
      echo "usage: $0 [--dry-run] <preflight|poststop>" >&2; exit 2 ;;
  esac
done
if [ -z "$MODE" ]; then
  if [ "$DRY_RUN" = "1" ]; then MODE="manual"; else
    echo "usage: $0 [--dry-run] <preflight|poststop>" >&2; exit 2
  fi
fi

log() { logger -t "$TAG" -- "$*" 2>/dev/null || true; echo "[$TAG] $*"; }

SELF=$$
PARENT=${PPID:-0}

# --- resolve the live server's process tree so we never touch it -----------------------------
# In a systemd ExecStartPre/ExecStopPost context $MAINPID is set by systemd. When run by hand we
# ask systemd directly. A dead/absent MainPID protects nothing (correct: poststop must reap the
# reparented engine child).
MAIN_PID="${MAINPID:-0}"
case "$MAIN_PID" in ''|*[!0-9]*) MAIN_PID=0 ;; esac
if [ "$MAIN_PID" -eq 0 ]; then
  q=$(systemctl show "$SERVICE" -p MainPID --value 2>/dev/null || echo 0)
  case "$q" in ''|*[!0-9]*) q=0 ;; esac
  MAIN_PID="$q"
fi

PROTECTED=" $SELF $PARENT "
add_tree() {
  root="$1"
  case "$root" in ''|*[!0-9]*|0) return 0 ;; esac
  kill -0 "$root" 2>/dev/null || return 0
  case "$PROTECTED" in *" $root "*) return 0 ;; esac
  PROTECTED="$PROTECTED$root "
  for cd in /proc/[0-9]*; do
    cpid=${cd#/proc/}
    ppid=$(awk '/^PPid:/{print $2; exit}' "/proc/$cpid/status" 2>/dev/null) || continue
    [ "$ppid" = "$root" ] && add_tree "$cpid"
  done
}
if [ "$MAIN_PID" -gt 0 ] 2>/dev/null; then add_tree "$MAIN_PID"; fi

is_conda_exe() {
  exe=$(readlink -f "/proc/$1/exe" 2>/dev/null) || return 1
  case "$exe" in "$CONDA_PREFIX_DIR"/*) return 0 ;; *) return 1 ;; esac
}
is_protected() { case "$PROTECTED" in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

log "start mode=$MODE dry_run=$DRY_RUN prefix=$CONDA_PREFIX_DIR protected_main=$MAIN_PID"

# --- build the target set --------------------------------------------------------------------
TARGETS=""
for cd in /proc/[0-9]*; do
  pid=${cd#/proc/}
  is_protected "$pid" && continue
  is_conda_exe "$pid" || continue
  exe=$(readlink -f "/proc/$pid/exe" 2>/dev/null || echo '?')
  comm=$(cat "/proc/$pid/comm" 2>/dev/null || echo '?')
  cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | cut -c1-160)
  log "MATCH pid=$pid comm=$comm exe=$exe cmd=[$cmd]"
  TARGETS="$TARGETS $pid"
done

TARGETS=$(echo "$TARGETS" | tr -s ' ')
TARGETS=${TARGETS# }
TARGETS=${TARGETS% }

if [ -z "$TARGETS" ]; then
  log "no conda-prefix orphans found; nothing to sweep"
  exit 0
fi

if [ "$DRY_RUN" = "1" ]; then
  log "DRY-RUN: would terminate: $TARGETS (no action taken)"
  exit 0
fi

# --- graceful, then forceful; re-verify identity right before every signal (pid-reuse guard) --
for pid in $TARGETS; do
  is_protected "$pid" && continue
  if is_conda_exe "$pid"; then
    log "SIGTERM pid=$pid"
    kill -TERM "$pid" 2>/dev/null || true
  fi
done
sleep 3
for pid in $TARGETS; do
  is_protected "$pid" && continue
  if kill -0 "$pid" 2>/dev/null && is_conda_exe "$pid"; then
    log "SIGKILL pid=$pid (survived SIGTERM)"
    kill -KILL "$pid" 2>/dev/null || true
  fi
done

log "complete mode=$MODE swept=[$TARGETS]"
exit 0
