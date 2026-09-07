#!/usr/bin/env bash
# cf_url_capture.sh — ExecStartPost helper for hhgvrag-cloudflared.service.
#
# A cloudflared quick tunnel prints an EPHEMERAL https://<random>.trycloudflare.com URL once at
# startup. This waits (bounded) for that line to appear in the cloudflared logfile and writes the
# URL to logs/cf_url.txt so operators (and drill_judge_now.sh) can find the current failover URL
# without scraping journald. Always exits 0 — a missed capture must never fail the tunnel unit.

set -u
LOG_DIR="/home/padmanabha/hhgvrag/logs"
CF_LOG="$LOG_DIR/cloudflared.log"
OUT="$LOG_DIR/cf_url.txt"
DEADLINE=60          # seconds
TAG="hhgvrag-cf-url"

mkdir -p "$LOG_DIR" 2>/dev/null || true

i=0
while [ "$i" -lt "$DEADLINE" ]; do
  url=$(grep -oE 'https://[a-z0-9][a-z0-9.-]*\.trycloudflare\.com' "$CF_LOG" 2>/dev/null | head -1)
  if [ -n "$url" ]; then
    printf '%s\n' "$url" > "$OUT" 2>/dev/null || true
    chown padmanabha:padmanabha "$OUT" 2>/dev/null || true
    logger -t "$TAG" -- "quick-tunnel URL captured: $url" 2>/dev/null || true
    echo "[$TAG] $url"
    exit 0
  fi
  i=$((i+1))
  sleep 1
done

logger -t "$TAG" -- "no trycloudflare URL within ${DEADLINE}s (tunnel may still be connecting)" 2>/dev/null || true
echo "[$TAG] no URL captured within ${DEADLINE}s"
exit 0
