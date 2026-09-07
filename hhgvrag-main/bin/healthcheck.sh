#!/usr/bin/env bash
# healthcheck.sh — liveness probe + bounded self-heal for hhgvrag.service.
#
# Run by hhgvrag-health.timer (every 60s, as root so it may restart the unit). It detects the
# case systemd's own Restart= cannot: the process is UP but the pipeline is HUNG (models wedged,
# GPU stall) so /health stops answering while the service still shows "active".
#
# POLICY
#   * Probe http://localhost:8000/health.
#   * Require 3 CONSECUTIVE failures before acting (transient blips do not restart a live judge demo).
#   * 5-MINUTE COOLDOWN keyed on the unit's ActiveEnterTimestampMonotonic: never restart a service
#     that entered "active" less than 5 min ago. A fresh (re)start reloads BGE-M3 + reranker +
#     vLLM-7B + warmup (can take minutes) during which /health may not answer yet — restarting then
#     would thrash. The cooldown gives the new instance time to come up.
#   * While the unit is "activating" we do not probe at all (still warming up).
#   * Every decision is written to journald (logger) and to the incidents log.
#
# State:     /home/padmanabha/hhgvrag/logs/.health_fail_count
# Incidents: /home/padmanabha/hhgvrag/logs/incidents.log

set -u

SERVICE="hhgvrag.service"
URL="http://localhost:8000/health"
LOG_DIR="/home/padmanabha/hhgvrag/logs"
STATE_FILE="$LOG_DIR/.health_fail_count"
INCIDENTS="$LOG_DIR/incidents.log"
FAIL_THRESHOLD=3
COOLDOWN_S=300                    # 5 minutes
TAG="hhgvrag-health"

mkdir -p "$LOG_DIR" 2>/dev/null || true

ts() { date '+%Y-%m-%dT%H:%M:%S%z'; }
note() {                                  # note <level> <message>
  lvl="$1"; shift
  logger -t "$TAG" -- "$lvl $*" 2>/dev/null || true
  printf '%s %-5s %s\n' "$(ts)" "$lvl" "$*" >> "$INCIDENTS" 2>/dev/null || true
  chown padmanabha:padmanabha "$INCIDENTS" 2>/dev/null || true
}
read_count() { c=$(cat "$STATE_FILE" 2>/dev/null || echo 0); case "$c" in ''|*[!0-9]*) c=0 ;; esac; echo "$c"; }
write_count() { echo "$1" > "$STATE_FILE" 2>/dev/null || true; chown padmanabha:padmanabha "$STATE_FILE" 2>/dev/null || true; }

# is-active prints the state AND exits non-zero when not active, so capture stdout then default.
active_state=$(systemctl is-active "$SERVICE" 2>/dev/null); [ -n "$active_state" ] || active_state=unknown

# Still coming up — do not probe, do not count. Reset so we start clean once it is active.
if [ "$active_state" = "activating" ]; then
  write_count 0
  logger -t "$TAG" -- "INFO service activating; skip probe" 2>/dev/null || true
  exit 0
fi

# Probe. r.ok + expected body.
body=$(curl -s -f -m 8 "$URL" 2>/dev/null)
rc=$?
if [ "$rc" -eq 0 ] && printf '%s' "$body" | grep -q '"status":"ok"'; then
  prev=$(read_count)
  if [ "$prev" -ne 0 ]; then note INFO "recovered after $prev consecutive failure(s); /health ok"; fi
  write_count 0
  exit 0
fi

# Failure path.
count=$(( $(read_count) + 1 ))
write_count "$count"
note WARN "health probe failed ($count/$FAIL_THRESHOLD) rc=$rc state=$active_state"

[ "$count" -lt "$FAIL_THRESHOLD" ] && exit 0

# Threshold reached — consult the cooldown before restarting.
# ActiveEnterTimestampMonotonic is microseconds since boot (CLOCK_MONOTONIC); /proc/uptime is seconds
# since boot on the same clock. Work in integer seconds (small numbers, no overflow).
enter_us=$(systemctl show "$SERVICE" -p ActiveEnterTimestampMonotonic --value 2>/dev/null || echo 0)
case "$enter_us" in ''|*[!0-9]*) enter_us=0 ;; esac

if [ "$enter_us" -eq 0 ]; then
  # Never became active (crash-looping / failed). systemd's Restart= owns this; we only record it.
  note ERROR "$FAIL_THRESHOLD consecutive failures and unit never reached active (state=$active_state); leaving to systemd Restart="
  exit 0
fi

enter_s=$(( enter_us / 1000000 ))
now_s=$(awk '{printf "%d", $1}' /proc/uptime 2>/dev/null || echo 0)
elapsed_s=$(( now_s - enter_s ))
if [ "$elapsed_s" -lt "$COOLDOWN_S" ]; then
  note WARN "$FAIL_THRESHOLD consecutive failures but within 5-min cooldown (active ${elapsed_s}s ago); NOT restarting"
  exit 0
fi

note ERROR "$FAIL_THRESHOLD consecutive failures, unit active for ${elapsed_s}s (> cooldown); restarting $SERVICE"
if systemctl restart "$SERVICE" 2>/dev/null; then
  note INFO "restart issued for $SERVICE"
else
  note ERROR "restart command FAILED for $SERVICE"
fi
write_count 0
exit 0
