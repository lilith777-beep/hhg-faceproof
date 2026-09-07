#!/usr/bin/env bash
# uninstall.sh — remove the hhgvrag systemd stack.
#
#   sudo bash ~/hhgvrag/ops/uninstall.sh              # stop+disable+remove units and config
#   sudo bash ~/hhgvrag/ops/uninstall.sh --reset-funnel   # ALSO tear down the Tailscale Funnel
#
# Leaves DATA intact: qdrant_storage, logs/, and the secrets env file are NOT touched. By default
# the Tailscale Funnel stays armed (removing it would kill the public judge URL) — pass
# --reset-funnel to also `tailscale funnel reset`.
#
# After this, the backend is no longer managed. To go back to the manual mode, start the server by
# hand per docs/RUNBOOK.md (or just re-run ops/install.sh).

set -u
UNIT_DST="/etc/systemd/system"
UNITS="hhgvrag-cloudflared.service hhgvrag-health.timer hhgvrag-health.service hhgvrag-funnel.service hhgvrag.service qdrant-hhgvrag.service"
RESET_FUNNEL=0
for a in "$@"; do case "$a" in --reset-funnel) RESET_FUNNEL=1 ;; esac; done

c_g=$'\033[32m'; c_y=$'\033[33m'; c_0=$'\033[0m'
ok()   { printf '%s[ ok ]%s %s\n' "$c_g" "$c_0" "$*"; }
warn() { printf '%s[warn]%s %s\n' "$c_y" "$c_0" "$*"; }

[ "$(id -u)" -eq 0 ] || { echo "must run as root: sudo bash $0"; exit 1; }

echo "== stopping + disabling units =="
for u in $UNITS; do
  systemctl stop "$u" 2>/dev/null && ok "stopped $u" || warn "$u not running"
  systemctl disable "$u" 2>/dev/null && ok "disabled $u" || true
done

echo "== removing unit + config files =="
for u in $UNITS; do
  [ -e "$UNIT_DST/$u" ] && rm -f "$UNIT_DST/$u" && ok "removed $UNIT_DST/$u"
done
rm -f /etc/systemd/journald.conf.d/hhgvrag.conf && ok "removed journald drop-in" || true
rm -f /etc/logrotate.d/hhgvrag && ok "removed logrotate file" || true

systemctl daemon-reload && ok "daemon-reload"
systemctl restart systemd-journald 2>/dev/null && ok "journald restarted" || true

if [ "$RESET_FUNNEL" -eq 1 ]; then
  tailscale funnel reset 2>/dev/null && ok "tailscale funnel reset (public URL torn down)" || warn "funnel reset failed"
else
  warn "Tailscale Funnel left ARMED (public URL still live). Use --reset-funnel to tear it down."
fi

echo
warn "Qdrant container 'qdrant' may still be running under docker; stop with: docker stop qdrant"
ok "uninstall complete. Data (qdrant_storage, logs, env file) left intact."
exit 0
