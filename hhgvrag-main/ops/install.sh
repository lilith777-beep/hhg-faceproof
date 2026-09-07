#!/usr/bin/env bash
# install.sh — install + start the hhgvrag systemd stack so it survives a Forge reboot.
#
#   sudo bash ~/hhgvrag/ops/install.sh
#
# Idempotent: safe to re-run. Copies units + journald/logrotate config, does a ONE-TIME cutover
# from the legacy nohup server/tunnel to systemd, enables and starts everything in dependency
# order, then prints a verification checklist.
#
# ONE-TIME CUTOVER (first run only): the current `nohup ... server.py` and the `~/bin/cloudflared
# tunnel --url` quick tunnel are stopped and replaced by systemd units. Expect ~2-3 min backend
# downtime while models reload. Qdrant data (bind mount) and the Tailscale Funnel URL are preserved.
# The unrelated `engageai-cockpit` cloudflared tunnel and all other containers are left untouched.
#
# Flags:  --no-migrate  skip stopping the legacy nohup processes (assume already stopped)

set -u

REPO="/home/padmanabha/hhgvrag"
OWNER="padmanabha"
GROUP="padmanabha"
UNIT_SRC="$REPO/ops/units"
UNIT_DST="/etc/systemd/system"
CONDA_PREFIX_DIR="/home/padmanabha/anaconda3/envs/hhgvrag313"
ENV_FILE="/home/padmanabha/.config/hhgvrag.env"
LOG_DIR="$REPO/logs"
SWEEP="$REPO/bin/gpu_orphan_sweep.sh"

# ALL unit files to copy (hhgvrag-health.service is timer-triggered — copied but NOT enabled).
UNITS="qdrant-hhgvrag.service hhgvrag.service hhgvrag-funnel.service hhgvrag-cloudflared.service hhgvrag-health.service hhgvrag-health.timer"
# Units that have an [Install] section and are enabled/started (health.service excluded — no [Install]).
ENABLE_UNITS="qdrant-hhgvrag.service hhgvrag.service hhgvrag-funnel.service hhgvrag-cloudflared.service hhgvrag-health.timer"
MIGRATE=1
for a in "$@"; do case "$a" in --no-migrate) MIGRATE=0 ;; esac; done

c_g=$'\033[32m'; c_r=$'\033[31m'; c_y=$'\033[33m'; c_0=$'\033[0m'
ok()   { printf '%s[ ok ]%s %s\n'   "$c_g" "$c_0" "$*"; }
warn() { printf '%s[warn]%s %s\n'   "$c_y" "$c_0" "$*"; }
die()  { printf '%s[fail]%s %s\n'   "$c_r" "$c_0" "$*"; exit 1; }
step() { printf '\n== %s ==\n' "$*"; }

# --- 0. must be root -------------------------------------------------------------------------
[ "$(id -u)" -eq 0 ] || die "must run as root:  sudo bash $REPO/ops/install.sh"

# --- 1. preconditions ------------------------------------------------------------------------
step "preconditions"
[ -d "$REPO" ]                || die "repo not found at $REPO"
[ -d "$UNIT_SRC" ]            || die "unit sources not found at $UNIT_SRC (run from a full checkout)"
[ -x "$CONDA_PREFIX_DIR/bin/python" ] || die "conda python missing at $CONDA_PREFIX_DIR/bin/python"
if [ -f "$ENV_FILE" ]; then
  perm=$(stat -c '%a' "$ENV_FILE" 2>/dev/null || echo '?')
  [ "$perm" = "600" ] && ok "env file present, chmod $perm (contents never read here)" \
                       || warn "env file present but chmod is $perm (expected 600) — fix: chmod 600 $ENV_FILE"
else
  die "secrets env file missing: $ENV_FILE  (create it per docs/RUNBOOK.md, chmod 600)"
fi
command -v docker    >/dev/null 2>&1 || die "docker not found"
command -v tailscale >/dev/null 2>&1 || warn "tailscale not found — funnel unit will fail until installed"
[ -x /home/padmanabha/bin/cloudflared ] || warn "~/bin/cloudflared missing — cloudflared failover unit will fail"
[ -d /home/padmanabha/qdrant_storage ]  || warn "qdrant_storage bind dir missing (will be created empty by docker)"
ok "preconditions checked"

# --- 2. make bin scripts executable + prepare logs dir ---------------------------------------
step "bin scripts + logs dir"
chmod +x "$REPO"/bin/*.sh 2>/dev/null && ok "chmod +x $REPO/bin/*.sh" || warn "could not chmod bin scripts"
install -d -o "$OWNER" -g "$GROUP" "$LOG_DIR" && ok "logs dir $LOG_DIR"
for f in incidents.log cf_url.txt cloudflared.log; do
  [ -e "$LOG_DIR/$f" ] || { : > "$LOG_DIR/$f"; }
  chown "$OWNER:$GROUP" "$LOG_DIR/$f" 2>/dev/null || true
done
ok "log files pre-created (owned by $OWNER)"

# --- 3. one-time cutover: stop legacy nohup server + quick tunnel -----------------------------
if [ "$MIGRATE" -eq 1 ]; then
  step "cutover: stop legacy nohup processes (if any)"

  # 3a. legacy server holding :8000 that is NOT already under hhgvrag.service
  legacy_srv=""
  for pid in $(ss -lptnH 'sport = :8000' 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u); do
    grep -q 'hhgvrag.service' "/proc/$pid/cgroup" 2>/dev/null && continue
    legacy_srv="$legacy_srv $pid"
  done
  if [ -n "${legacy_srv// /}" ]; then
    for pid in $legacy_srv; do
      cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | cut -c1-80)
      warn "stopping legacy server pid=$pid [$cmd]"
      kill -TERM "$pid" 2>/dev/null || true
    done
    sleep 3
    # reap the renamed vLLM engine child left behind (exe-under-conda-prefix allowlist)
    [ -x "$SWEEP" ] && "$SWEEP" preflight || warn "sweep script not executable"
  else
    ok "no legacy nohup server on :8000"
  fi

  # 3b. legacy cloudflared QUICK tunnel (has --url), excluding systemd-managed + engageai named tunnel
  for pid in $(pgrep -f 'cloudflared.*tunnel.*--url.*localhost:8000' 2>/dev/null); do
    grep -q 'hhgvrag-cloudflared.service' "/proc/$pid/cgroup" 2>/dev/null && continue
    cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | cut -c1-80)
    warn "stopping legacy cloudflared quick tunnel pid=$pid [$cmd]"
    kill -TERM "$pid" 2>/dev/null || true
  done

  # 3c. wait for :8000 to free up
  for i in $(seq 1 15); do
    ss -lntH 'sport = :8000' 2>/dev/null | grep -q ':8000' || { ok "port 8000 is free"; break; }
    sleep 1
  done
else
  step "cutover skipped (--no-migrate)"
fi

# --- 4. install unit + config files ----------------------------------------------------------
step "install units + journald/logrotate config"
for u in $UNITS; do
  install -m 0644 "$UNIT_SRC/$u" "$UNIT_DST/$u" && ok "installed $u" || die "failed to install $u"
done
install -d /etc/systemd/journald.conf.d
install -m 0644 "$REPO/ops/journald-hhgvrag.conf" /etc/systemd/journald.conf.d/hhgvrag.conf \
  && ok "journald cap -> /etc/systemd/journald.conf.d/hhgvrag.conf (SystemMaxUse=2G)"
install -m 0644 "$REPO/ops/logrotate-hhgvrag" /etc/logrotate.d/hhgvrag \
  && ok "logrotate -> /etc/logrotate.d/hhgvrag"

systemctl restart systemd-journald 2>/dev/null && ok "journald restarted (cap applied)" || warn "journald restart failed"
systemctl daemon-reload && ok "daemon-reload" || die "daemon-reload failed"

# --- 5. enable + start in dependency order ---------------------------------------------------
step "enable + start (ordered)"
systemctl enable $ENABLE_UNITS >/dev/null 2>&1 && ok "enabled units (health.service is timer-triggered)" || warn "enable reported an issue"

systemctl start qdrant-hhgvrag.service || warn "qdrant-hhgvrag start returned non-zero"
printf '   waiting for qdrant /readyz'
for i in $(seq 1 60); do
  curl -sf -m 3 http://localhost:6333/readyz >/dev/null 2>&1 && { echo " ready"; break; }
  printf '.'; sleep 2
done

systemctl start hhgvrag.service || warn "hhgvrag start returned non-zero (check: journalctl -u hhgvrag)"
printf '   waiting for hhgvrag /health (model load, up to 5 min)'
for i in $(seq 1 150); do
  curl -sf -m 3 http://localhost:8000/health >/dev/null 2>&1 && { echo " up"; break; }
  printf '.'; sleep 2
done

systemctl start hhgvrag-funnel.service    || warn "funnel re-arm returned non-zero"
systemctl start hhgvrag-health.timer      || warn "health timer start returned non-zero"
systemctl start hhgvrag-cloudflared.service || warn "cloudflared failover start returned non-zero"

# --- 6. verification checklist ---------------------------------------------------------------
step "VERIFICATION CHECKLIST"
fails=0
for u in qdrant-hhgvrag.service hhgvrag.service hhgvrag-funnel.service hhgvrag-cloudflared.service hhgvrag-health.timer; do
  st=$(systemctl is-active "$u" 2>/dev/null); [ -n "$st" ] || st=inactive
  if [ "$st" = "active" ]; then ok "$u: $st"; else warn "$u: $st"; fails=$((fails+1)); fi
done

if curl -sf -m 5 http://localhost:6333/readyz >/dev/null 2>&1; then ok "qdrant /readyz: ready"; else warn "qdrant /readyz: NOT ready"; fails=$((fails+1)); fi

hbody=$(curl -sf -m 8 http://localhost:8000/health 2>/dev/null || echo '')
if printf '%s' "$hbody" | grep -q '"status":"ok"'; then
  ok "backend /health: $(printf '%s' "$hbody" | cut -c1-80)"
else
  warn "backend /health: NOT ok (models may still be loading — recheck: curl localhost:8000/health)"; fails=$((fails+1))
fi

if tailscale funnel status 2>/dev/null | grep -q '127.0.0.1:8000'; then
  ok "tailscale funnel: armed for :8000"
else
  warn "tailscale funnel: NOT armed (run: sudo tailscale funnel --bg 8000)"; fails=$((fails+1))
fi

cfurl=$(cat "$LOG_DIR/cf_url.txt" 2>/dev/null || echo '')
[ -n "$cfurl" ] && ok "cloudflared failover URL: $cfurl" || warn "cloudflared URL not captured yet (see $LOG_DIR/cf_url.txt shortly)"

step "RESULT"
if [ "$fails" -eq 0 ]; then
  ok "hhgvrag systemd stack is UP and reboot-survivable."
else
  warn "$fails check(s) not green yet. If backend is still loading models, re-run the checklist in a minute:"
  echo "     systemctl is-active hhgvrag qdrant-hhgvrag hhgvrag-funnel; curl -s localhost:8000/health"
fi
cat <<'NEXT'

Next (human) steps — see docs/RUNBOOK.md:
  1. Judge-now drill (public path):   bash ~/hhgvrag/bin/drill_judge_now.sh
  2. Ghost-kill drill (dry-run first): bash ~/hhgvrag/bin/gpu_orphan_sweep.sh --dry-run preflight
  3. Reboot drill (schedule a window): sudo reboot   then re-run drill_judge_now.sh after ~4 min
NEXT
exit 0
