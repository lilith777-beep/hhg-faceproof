#!/usr/bin/env bash
# drill_judge_now.sh — "a judge is clicking the public link RIGHT NOW: is it working?"
#
# Exercises the PUBLIC path (Tailscale Funnel — the URL judges actually use), not localhost, so it
# catches tunnel/DNS/cert breakage that a local curl would miss. No sudo, no state changes.
#
#   ./drill_judge_now.sh                 # hits the primary Funnel URL
#   ./drill_judge_now.sh <base-url>      # e.g. a fresh cloudflared URL from logs/cf_url.txt
#
# Checks, each with a PRE-VERIFIED expected decision (validated live against the 20k msmarco index):
#   1. GET  /health                                            -> status ok
#   2. POST /ask_text  EN answerable  ("...asthma")            -> decision = answer
#   3. POST /ask_text  HI answerable  (Devanagari, diabetes)   -> decision = answer   (replies in Hindi)
#   4. POST /ask_text  OOD            ("2027 cricket world cup")-> decision = abstain* (correctly declines)
#
# Exit 0 iff all four pass. Wall times INCLUDE tunnel RTT and are for situational awareness only —
# the <200 ms budget is retrieval->output measured on-box (eval/latency.py), not this path.

set -u

BASE="${1:-https://goquest-z790-aorus-elite-ax.tail16e418.ts.net}"
BASE="${BASE%/}"
PASS=0; FAIL=0

green() { printf '  \033[32mPASS\033[0m %s\n' "$1"; PASS=$((PASS+1)); }
red()   { printf '  \033[31mFAIL\033[0m %s\n' "$1"; FAIL=$((FAIL+1)); }

decision_of() { printf '%s' "$1" | grep -oE '"decision":"[^"]*"' | head -1 | cut -d'"' -f4; }

# The '\n' prefix on -w is REQUIRED: some curl builds drop the write-out entirely when it is glued
# to a body that has no trailing newline.
ask() {                                   # ask <json-payload>
  curl -s -m 25 -w '\n@@%{http_code}@@%{time_total}' \
    -X POST "$BASE/ask_text" -H 'Content-Type: application/json' --data-raw "$1" 2>/dev/null
}
# meta <response> -> "code time" (robust param-expansion split; no cut field-counting)
meta() {
  tok=$(printf '%s' "$1" | grep -oE '@@[0-9]+@@[0-9.]+' | tail -1); tok=${tok#@@}
  printf '%s %s' "${tok%%@@*}" "${tok##*@@}"
}

echo "== hhgvrag judge-now drill =="
echo "target: $BASE"
echo "time:   $(date '+%Y-%m-%dT%H:%M:%S%z')"
echo

# 1) health -----------------------------------------------------------------------------------
h=$(curl -s -m 15 -w '\n@@%{http_code}@@%{time_total}' "$BASE/health" 2>/dev/null)
hm=$(meta "$h"); hcode=${hm% *}; htime=${hm#* }
if printf '%s' "$h" | grep -q '"status":"ok"'; then
  coll=$(printf '%s' "$h" | grep -oE '"collection":"[^"]*"' | cut -d'"' -f4)
  green "health http=$hcode ${htime}s collection=$coll"
else
  red "health unreachable/!ok http=${hcode:-none} (raw: $(printf '%s' "$h" | head -c 120))"
fi

# 2) EN answerable ----------------------------------------------------------------------------
r=$(ask '{"text":"what are the symptoms of asthma"}'); m=$(meta "$r"); code=${m% *}; t=${m#* }; d=$(decision_of "$r")
[ "$d" = "answer" ] && green "EN  '...symptoms of asthma' -> $d (http=$code ${t}s)" \
                    || red "EN  '...symptoms of asthma' -> ${d:-<none>} (expected answer, http=$code)"

# 3) HI answerable (replies in Hindi) ---------------------------------------------------------
r=$(ask '{"text":"मधुमेह के लक्षण क्या हैं"}'); m=$(meta "$r"); code=${m% *}; t=${m#* }; d=$(decision_of "$r")
[ "$d" = "answer" ] && green "HI  'मधुमेह के लक्षण...' -> $d (http=$code ${t}s)" \
                    || red "HI  'मधुमेह के लक्षण...' -> ${d:-<none>} (expected answer, http=$code)"

# 4) OOD -> abstain ---------------------------------------------------------------------------
r=$(ask '{"text":"who won the 2027 cricket world cup final"}'); m=$(meta "$r"); code=${m% *}; t=${m#* }; d=$(decision_of "$r")
case "$d" in
  abstain*) green "OOD '2027 cricket world cup' -> $d (http=$code ${t}s)" ;;
  *)        red   "OOD '2027 cricket world cup' -> ${d:-<none>} (expected abstain*, http=$code)" ;;
esac

echo
echo "result: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || { echo "JUDGE-NOW DRILL FAILED — see docs/RUNBOOK.md (tunnel recovery / failover)"; exit 1; }
echo "JUDGE-NOW DRILL GREEN — public path healthy and answering correctly."
exit 0
