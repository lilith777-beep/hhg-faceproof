# FORGE — on-prem GPU deploy runbook

Target: the Forge box — i9 14th gen · 64GB RAM · **RTX A6000 48GB**. The full stack
(BGE-M3 ~1.2GB + reranker ~1.2GB + Qwen2.5-3B ~6.5GB) uses ~11GB VRAM — huge headroom.
Total hosting cost: **$0**. Modal (`src/modal_app.py`) remains the built fallback.

## 1. One-time setup (on Forge)

```bash
git clone <repo-url> && cd HHG/hhgvrag        # or copy the hhgvrag/ folder
python3.11 -m venv .venv
# Windows: .venv\Scripts\activate      Linux: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-gpu.txt
```

Set the secrets as environment variables (NEVER commit them):

```bash
# Windows (persistent):  setx SARVAM_API_KEY "..." && setx HF_TOKEN "..."
# Linux (~/.bashrc):     export SARVAM_API_KEY=... ; export HF_TOKEN=...
```

## 2. Build the index (~20-40 min on the A6000, one time)

```bash
python src/build_real_index.py --max-docs 20000 --languages hi
# faster first smoke run:  --max-docs 2000 --no-llm-raptor
```

Collections land in `./qdrant_data` (gitignored). Rebuild any time; restart the server after.

## 3. Run the server

```bash
python src/server.py --host 0.0.0.0 --port 8000
# verify locally:  curl http://localhost:8000/health
```

Keep it alive across logouts: Linux → `tmux` / systemd unit; Windows → NSSM or a
scheduled task running the venv python. One process, one GPU, done.

## 4. Public URL (judges click this until Aug 22)

**Cloudflare Tunnel** (free, stable URL, no port-forwarding, survives NAT):

```bash
# install cloudflared, then:
cloudflared tunnel --url http://localhost:8000
```

It prints `https://<random>.trycloudflare.com` — works instantly, zero account.
For a URL that never rotates: `cloudflared tunnel login` → named tunnel + free
`*.cfargotunnel.com` or your own domain. (ngrok also works but free URLs rotate.)

## 5. Frontend

Deploy `web/` on Vercel (static). In the UI's Backend field, paste the tunnel URL.
The frontend stores it in localStorage — set it once in the deployed page.

## 6. Latency evidence (P50/P70/P100)

```bash
python eval/latency.py --url http://localhost:8000    # writes eval/latency_report.md
python eval/calibrate_ood.py                          # M7: real τ with BGE-M3
```

Run the latency sweep ON Forge against localhost (measures the pipeline, not the tunnel).
The report + the live trace panel in the UI are the submission's latency proof.

## Failure modes

- `collection not found` → run step 2 first (server fail-fast is intentional).
- Voice 200s but text works → SARVAM_API_KEY missing/placeholder.
- First model download slow → HF_TOKEN set? Weights cache in `~/.cache/huggingface`.
- Port 8000 busy → `--port 8001` and re-point the tunnel.
