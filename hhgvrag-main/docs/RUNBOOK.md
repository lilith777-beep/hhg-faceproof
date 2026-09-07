# hhgvrag — Operations RUNBOOK (Forge)

Operational guide for the live Voice-RAG backend on **Forge** (Ubuntu 22.04, RTX A6000, user
`padmanabha`). This is the incident + maintenance reference. When something is on fire, jump to
[60-second "judge is clicking right now"](#60-second-judge-is-clicking-right-now).

> **Golden rules**
> 1. Manage the backend with **systemctl only**. Never `pkill`/`kill -9` the server or vLLM — the
>    engine child is renamed and name-based kills are unsafe (see [ghost-kill](#ghost-kill-the-money-section)).
> 2. The GPU is **shared with the user's desktop**. Never touch Xorg / gnome-shell / chrome /
>    rustdesk (see [DO-NOT-KILL](#do-not-kill-list)).
> 3. **Never print secrets.** Keys live in `/home/padmanabha/.config/hhgvrag.env` (chmod 600).
> 4. No collection flips or index rebuilds-in-place — versioned build + config flip only
>    (see [rebuild/rollback](#rebuild--rollback)).

---

## Topology

```
 judge browser
      │  https://hhgvrag.vercel.app         (frontend: static, Vercel project "hhgvrag")
      ▼
 PUBLIC URL (primary):  https://goquest-z790-aorus-elite-ax.tail16e418.ts.net
      │  Tailscale Funnel  →  127.0.0.1:8000
      ▼
 ┌──────────────────────────── Forge (padmanabha) ─────────────────────────────┐
 │  hhgvrag.service ──► python src/server.py --qdrant-url http://localhost:6333 │
 │      │  conda env: /home/padmanabha/anaconda3/envs/hhgvrag313 (py3.13)       │
 │      │  models: BGE-M3 + bge-reranker-v2-m3 + Qwen2.5-7B(vLLM) + Sarvam STT  │
 │      ▼                                                                        │
 │  qdrant-hhgvrag.service ──► docker "qdrant" :6333  (HNSW, bind mount)        │
 │      volume: /home/padmanabha/qdrant_storage → /qdrant/storage               │
 │                                                                              │
 │  hhgvrag-funnel.service     (oneshot: re-arm Tailscale Funnel)               │
 │  hhgvrag-cloudflared.service (WARM FAILOVER: rotating trycloudflare URL)     │
 │  hhgvrag-health.timer       (60s liveness probe + bounded self-heal)         │
 └──────────────────────────────────────────────────────────────────────────────┘

 FAILOVER URL (rotating):  read /home/padmanabha/hhgvrag/logs/cf_url.txt
 Modal (src/modal_app.py):  BUILT fallback code only — NO index is built there. NOT a hot standby.
```

**Key paths**

| What | Path |
|---|---|
| Repo | `/home/padmanabha/hhgvrag` |
| Conda python | `/home/padmanabha/anaconda3/envs/hhgvrag313/bin/python` |
| Conda prefix (sweep allowlist) | `/home/padmanabha/anaconda3/envs/hhgvrag313/` |
| Secrets env (chmod 600) | `/home/padmanabha/.config/hhgvrag.env` |
| Qdrant data (bind mount) | `/home/padmanabha/qdrant_storage` |
| Logs | `/home/padmanabha/hhgvrag/logs/` (`incidents.log`, `cf_url.txt`, `cloudflared.log`) |
| Units | `/etc/systemd/system/{qdrant-hhgvrag,hhgvrag,hhgvrag-funnel,hhgvrag-cloudflared,hhgvrag-health}.{service,timer}` |
| Live collection (current) | `msmarco_xi__raptor` |

**Install / uninstall**

```bash
sudo bash ~/hhgvrag/ops/install.sh          # idempotent; does the one-time nohup→systemd cutover
sudo bash ~/hhgvrag/ops/uninstall.sh         # stop+disable+remove units (leaves data + funnel)
sudo bash ~/hhgvrag/ops/uninstall.sh --reset-funnel   # also tears down the public Funnel URL
```

---

## Start / stop (systemctl only — never pkill)

```bash
# status of the whole stack
systemctl status hhgvrag qdrant-hhgvrag hhgvrag-funnel hhgvrag-cloudflared hhgvrag-health.timer

# start / stop / restart the backend (control-group kill catches the renamed vLLM child;
# ExecStopPost sweeps any survivor by exe-path allowlist)
sudo systemctl start   hhgvrag
sudo systemctl stop    hhgvrag
sudo systemctl restart hhgvrag        # ~2-3 min: reloads BGE-M3 + reranker + vLLM-7B + warmup

# Qdrant (data is on the bind mount; restarting the container is non-destructive)
sudo systemctl restart qdrant-hhgvrag

# follow logs
journalctl -u hhgvrag -f
journalctl -u qdrant-hhgvrag -f
tail -f ~/hhgvrag/logs/incidents.log     # health-probe decisions
```

**Readiness expectations after a start/restart**

- `systemctl start hhgvrag` blocks through the `/readyz` gate (up to 120s) then returns once the
  process is exec'd. **Models still load for ~1-3 min afterward** — `/health` is the real ready signal:
  ```bash
  curl -s localhost:8000/health      # {"status":"ok","collection":"msmarco_xi__raptor","budget_ms":200}
  ```
- If `hhgvrag` won't start, it is almost always (a) Qdrant not ready, or (b) a GPU ghost pinning
  VRAM → see [ghost-kill](#ghost-kill-the-money-section).

**NEVER do this** (leaves a 24 GB VRAM ghost and/or kills desktop processes):

```bash
pkill -f server.py        # ✗ orphans the renamed VLLM::EngineCore child
pkill -f vllm             # ✗ does NOT match the renamed child at all
pkill -9 python           # ✗ may hit unrelated desktop/user python
kill -9 <any nvidia pid>  # ✗ never kill a pid you haven't exe-verified
```

---

## Ghost-kill (the money section)

**The problem.** vLLM runs an engine subprocess that calls `setproctitle("VLLM::EngineCore")`. On a
crash or non-clean stop it can survive, still pinning **~24 GB of A6000 VRAM**. Its process name is
useless for matching (`pkill -f vllm` misses it entirely; `pkill EngineCore` is both unreliable and
dangerous). The next server start then dies allocating VRAM.

**The only safe identity is the executable path.** Every hhgvrag process (the server python *and*
its vLLM engine child) has `/proc/<pid>/exe` resolving **under the conda prefix**
`/home/padmanabha/anaconda3/envs/hhgvrag313/`. **Nothing on the desktop does.** That is the allowlist.

### Preferred: the sweep script (always dry-run first)

```bash
# 1. SEE what would be killed — touches nothing:
bash ~/hhgvrag/bin/gpu_orphan_sweep.sh --dry-run preflight

# 2. If (and only if) the dry-run lists the ghost and nothing you care about, sweep for real:
bash ~/hhgvrag/bin/gpu_orphan_sweep.sh preflight
```

The script kills a pid **only if all hold**: exe under the conda prefix, **not** the live server's
MainPID or any descendant of it (the running server tree is auto-protected — it queries
`systemctl show hhgvrag -p MainPID`), and not the script itself. It SIGTERMs, waits 3s, then SIGKILLs
survivors, re-verifying identity before every signal (pid-reuse guard). Every action goes to
`journalctl -t hhgvrag-sweep`.

> **Caveat:** the allowlist is "any process running the hhgvrag conda Python." The *live server tree*
> is protected, but an **ad-hoc job you launched from that same env** (e.g. a manual
> `python src/build_real_index.py`) is **not** — a manual sweep will target it too. Don't sweep while a
> separate conda-env job is running, and always `--dry-run` first.

### Manual fallback (script unavailable)

```bash
# List every GPU compute pid and classify by exe. Only conda-prefix pids are ghost candidates.
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
  exe=$(readlink -f /proc/$pid/exe 2>/dev/null)
  case "$exe" in
    /home/padmanabha/anaconda3/envs/hhgvrag313/*) echo "GHOST-CANDIDATE  $pid  $exe" ;;
    *)                                            echo "KEEP (do NOT kill) $pid  $exe" ;;
  esac
done

# For a confirmed GHOST-CANDIDATE that is NOT the live server (check: systemctl show hhgvrag -p MainPID):
kill -TERM <ghost-pid> ; sleep 3 ; kill -KILL <ghost-pid> 2>/dev/null   # force only if it survived
nvidia-smi   # confirm VRAM freed
```

### DO-NOT-KILL list

These share the GPU and MUST never be touched (verified live; none resolve under the conda prefix):

| Process | exe path | Notes |
|---|---|---|
| Xorg | `/usr/lib/xorg/Xorg` | desktop display server |
| GNOME shell | `/usr/bin/gnome-shell` | desktop |
| Chrome | `/opt/google/chrome/chrome` (`--type=gpu-process`) | user's browser |
| RustDesk | `/usr/share/rustdesk/rustdesk` | remote desktop — killing it can lock out access |
| `engageai-cockpit` tunnel | `/home/padmanabha/.local/bin/cloudflared tunnel run engageai-cockpit` | **different project**; never `pkill cloudflared` |
| other containers | docker: `show-memory-*`, `postgres:*`, `neo4j:*`, `github-mcp-*` | unrelated services |

Rule of thumb: **if `readlink -f /proc/<pid>/exe` is not under `/home/padmanabha/anaconda3/envs/hhgvrag313/`, do not kill it.**

---

## Rebuild / rollback

Indexes are **versioned and immutable**. Never rebuild a live collection in place; build a new one,
verify, then flip. Old collections are retained so rollback is a config flip + restart.

**Naming:** `msmarco_xi_40k_<manifest8>__<strategy>` (e.g. `msmarco_xi_40k_a1b2c3d4__raptor`), where
`<manifest8>` is the first 8 chars of the immutable CorpusBuildSpec manifest hash. (The current live
collection predates the versioned scheme and is `msmarco_xi__raptor`.)

**Build → verify → flip (promotion is the integration lead's call; Ops executes the restart):**

```bash
# 1. BUILD a new versioned collection (Data/lead own the build; runs ~2h wall). Data lands in the
#    Qdrant bind mount and is additive — existing collections are untouched.
#    (per src/build_real_index.py / the CorpusBuildSpec; DOES NOT touch the live collection)

# 2. VERIFY the new collection before any flip:
curl -s localhost:6333/collections | python3 -m json.tool         # new collection present?
curl -s localhost:6333/collections/msmarco_xi_40k_<manifest8>__raptor | python3 -m json.tool  # points_count

# 3. FLIP (integration lead edits src/config.py: collection_prefix / live_strategy / use_raptor to
#    point at the new suffix — Ops does NOT decide promotion), then Ops restarts:
sudo systemctl restart hhgvrag
curl -s localhost:8000/health      # confirm "collection" is the NEW one

# 4. Post-action verification (below). Keep the OLD collection — do not delete until a rollback drill passes.
```

**Rollback (fast):** the previous collection is still in Qdrant — revert the `src/config.py` collection
setting to the old suffix and `sudo systemctl restart hhgvrag`. Confirm `/health` shows the old
collection and run the [post-action verification](#post-action-verification). No data restore needed.

---

## Tunnel recovery + failover

**Primary = Tailscale Funnel** (stable URL judges use). **Failover = cloudflared quick tunnel**
(rotating URL in `logs/cf_url.txt`). Modal is **not** a standby (no index there).

### Funnel is down (public URL 502/timeout, but `curl localhost:8000/health` is OK)

```bash
tailscale funnel status                       # expect: / proxy http://127.0.0.1:8000
sudo systemctl restart hhgvrag-funnel         # idempotent re-arm
# or arm by hand (exact observed command):
sudo tailscale funnel --bg 8000
tailscale funnel status                       # re-verify
curl -s https://goquest-z790-aorus-elite-ax.tail16e418.ts.net/health
```

If Tailscale itself is unhealthy: `sudo systemctl restart tailscaled` then re-arm the funnel.

### Fail over to cloudflared (Funnel unrecoverable)

```bash
sudo systemctl restart hhgvrag-cloudflared    # (re)start the quick tunnel
sleep 8
cat ~/hhgvrag/logs/cf_url.txt                 # the current public URL, e.g. https://<words>.trycloudflare.com
bash ~/hhgvrag/bin/drill_judge_now.sh "$(cat ~/hhgvrag/logs/cf_url.txt)"   # validate it
```

Then point the frontend at the new URL — see [stale-frontend-URL fix](#stale-frontend-url-fix). Note
the quick-tunnel URL **rotates on every restart**, so update the frontend each time you restart it.

### Backend itself is down (`curl localhost:8000/health` fails)

Not a tunnel problem — see [Start/stop](#start--stop-systemctl-only--never-pkill) and
[ghost-kill](#ghost-kill-the-money-section). `sudo systemctl restart hhgvrag` and watch `journalctl -u hhgvrag -f`.

---

## Stale-frontend-URL fix

The frontend (`hhgvrag.vercel.app`) stores the backend URL in `localStorage` (key
`hhgvrag_backend_v2` after the P0.2 migration) and health-checks it, falling back once to the compiled
default. If a judge sees "backend offline" while `curl` to the backend works, the cached URL is stale.

- **Fastest (per browser):** open `hhgvrag.vercel.app`, use the **Backend URL** field, paste the
  current URL (Funnel primary, or `cf_url.txt` if failed over), submit. Or clear site data /
  `localStorage.removeItem('hhgvrag_backend_v2')` in the console and reload to pick up the compiled default.
- **Permanent (all users):** the compiled default in `web/` must be the **Funnel** URL. If you failed
  over to cloudflared, the frontend needs the new `cf_url.txt` URL — coordinate with the UI owner to
  redeploy `web/` on Vercel (Ops does not edit `web/`). Because cloudflared URLs rotate, prefer
  restoring the Funnel over baking a quick-tunnel URL into the deploy.

---

## 60-second "judge is clicking right now"

Backend health is measured **on-box**; the public path is measured **through the tunnel**.

```bash
# 0. One command that answers "is the public path working RIGHT NOW?" (health + EN + HI + OOD):
bash ~/hhgvrag/bin/drill_judge_now.sh
```

If you have literally 60 seconds, do this in order and stop at the first failure:

1. **Backend alive?** `curl -s localhost:8000/health` → must show `"status":"ok"`.
   - No → `sudo systemctl restart hhgvrag`; if it won't come up, [ghost-kill](#ghost-kill-the-money-section).
2. **Public path alive?** `curl -s https://goquest-z790-aorus-elite-ax.tail16e418.ts.net/health`.
   - No (but step 1 OK) → [Funnel recovery](#tunnel-recovery--failover); if unrecoverable, fail over to cloudflared.
3. **Answering correctly?** `bash ~/hhgvrag/bin/drill_judge_now.sh` (EN answer / HI answer / OOD abstain).
4. **Frontend showing offline** but backend OK → [stale-frontend-URL fix](#stale-frontend-url-fix).

> **Modal is NOT a hot standby.** `src/modal_app.py` is built fallback *code* but **no index has been
> built on Modal** — you cannot cut over to it in seconds. The real failover is
> **Funnel → cloudflared quick tunnel** (same Forge backend, different public URL).

---

## Post-action verification

Run after ANY restart, rebuild, flip, rollback, or tunnel change. Three queries with **known**
expected decisions (validated live against the 20k index). The scripted drill checks all three:

```bash
bash ~/hhgvrag/bin/drill_judge_now.sh          # against the Funnel (or pass a URL for cloudflared)
```

Or by hand against the backend:

```bash
BASE=http://localhost:8000    # or the public URL

# 1. English, answerable  -> decision "answer"
curl -s -X POST $BASE/ask_text -H 'Content-Type: application/json' \
  --data-raw '{"text":"what are the symptoms of asthma"}' | grep -o '"decision":"[^"]*"' | head -1

# 2. Hindi, answerable  -> decision "answer"  (replies in Hindi)
curl -s -X POST $BASE/ask_text -H 'Content-Type: application/json' \
  --data-raw '{"text":"मधुमेह के लक्षण क्या हैं"}' | grep -o '"decision":"[^"]*"' | head -1

# 3. Out-of-domain  -> decision "abstain_ood"  (system correctly declines)
curl -s -X POST $BASE/ask_text -H 'Content-Type: application/json' \
  --data-raw '{"text":"who won the 2027 cricket world cup final"}' | grep -o '"decision":"[^"]*"' | head -1
```

Expected: `answer`, `answer`, `abstain_ood`. Any deviation → the index/collection or a model didn't
load correctly; check `/health`'s `collection`, then `journalctl -u hhgvrag`.

Decision vocabulary (from `src/schemas.py`): `answer`, `abstain_ood`, `abstain_ungrounded`,
`refuse_unsafe`, `small_talk`, `error`.

---

## Drills (run these after install and before the contest)

| Drill | Command | Pass criteria |
|---|---|---|
| Judge-now | `bash ~/hhgvrag/bin/drill_judge_now.sh` | 4/4 green |
| Ghost-kill (dry) | `bash ~/hhgvrag/bin/gpu_orphan_sweep.sh --dry-run preflight` | lists only conda-prefix pids |
| Ghost-kill (live) | start a throwaway conda-env python that grabs GPU, then `... preflight` | orphan gone, desktop + live server untouched |
| Reboot survival | schedule a window → `sudo reboot` → wait ~4 min | all units active, Funnel up, drill green |
| Health self-heal | `sudo systemctl kill -s SIGKILL hhgvrag` (simulate hang is harder) | timer logs incident, restarts past cooldown |

**Reboot drill detail:** after `sudo reboot`, on reconnect check
`systemctl is-active qdrant-hhgvrag hhgvrag hhgvrag-funnel hhgvrag-cloudflared hhgvrag-health.timer`,
then `curl localhost:8000/health`, then `drill_judge_now.sh`. First query should be in budget once
`/health` is ok (models finished loading).

---

## Post-contest: KEY ROTATION (required)

`SARVAM_API_KEY` and `HF_TOKEN` were **exposed in session transcripts** (and previously in the process
argv before the systemd `EnvironmentFile` migration). Treat both as compromised.

- **Rotate both keys after the Aug 22 deadline** (post-contest rotation is the fallback; rotating
  earlier is better if operationally possible without breaking the live demo).
- Procedure: issue new keys in the Sarvam and Hugging Face dashboards → update
  `/home/padmanabha/.config/hhgvrag.env` in place (keep `chmod 600`) → `sudo systemctl restart hhgvrag`
  → run [post-action verification](#post-action-verification) → revoke the old keys.
- **Never print keys** during any of this. Do not `cat` the env file, do not echo values, do not paste
  them into logs, tickets, or chat. `EnvironmentFile` keeps them out of `ps`/argv; keep it that way.

---

## Appendix: what each unit does

| Unit | Type | Role |
|---|---|---|
| `qdrant-hhgvrag.service` | simple | docker-wraps Qdrant (digest-pinned v1.19.0) on :6333; `docker rm -f` cutover; systemd owns restart |
| `hhgvrag.service` | simple | the server; `/readyz` gate + preflight sweep (ExecStartPre); control-group kill + poststop sweep; `Restart=on-failure`, 5-in-600s start limit |
| `hhgvrag-funnel.service` | oneshot | idempotent re-arm of `tailscale funnel --bg 8000` (root) |
| `hhgvrag-cloudflared.service` | simple | warm failover quick tunnel; writes rotating URL to `logs/cf_url.txt` |
| `hhgvrag-health.service`/`.timer` | oneshot/timer | 60s probe; 3-consecutive-fail + 5-min cooldown → `systemctl restart hhgvrag`; logs to `incidents.log` |

Journald is capped globally at `SystemMaxUse=2G` (`/etc/systemd/journald.conf.d/hhgvrag.conf`).
File logs rotate via `/etc/logrotate.d/hhgvrag` (copytruncate, 7 days).
