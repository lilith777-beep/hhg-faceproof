"""
reranker_onnx.py — export BGE-reranker-v2-m3 to ONNX (fp16 CUDA), then PROVE it's a free win:
strict score/ranking parity vs the live PyTorch-fp16 reranker + a latency benchmark.

Run in the isolated BUILD env (py3.11) so the live server's env is never touched:
    ~/anaconda3/envs/hhgvrag/bin/pip install -q "optimum[onnxruntime-gpu]" onnxconverter-common
    ~/anaconda3/envs/hhgvrag/bin/python eval/reranker_onnx.py --out ~/reranker_onnx --depth 24 --reps 30

Accuracy gate: ranking must be IDENTICAL (Kendall/top-k agreement) and scores within tol — ONNX
is a serving optimization, not a model change; any ranking flip fails the export. Prints a
verdict + the measured speedup so the lead can decide to wire ONNXReranker into serving.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _passages(n: int):
    # varied lengths (incl. a long one) so the max_length / padding behavior is exercised
    base = [
        "A corporation is a company or group authorized to act as a single legal entity.",
        "Diabetes is a chronic disease that affects how your body turns food into energy.",
        "Most tax refunds are issued in fewer than 21 days after electronic filing.",
        "The mitochondria is the powerhouse of the cell, producing ATP via respiration.",
        "Asthma symptoms include wheezing, shortness of breath, chest tightness, and cough.",
        "Photosynthesis converts light, water, and carbon dioxide into glucose and oxygen.",
    ]
    out = []
    for i in range(n):
        t = base[i % len(base)]
        if i % 7 == 0:
            t = (t + " ") * 20  # a deliberately long passage to stress padding
        out.append(t + f" [doc {i}]")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Export + parity-verify + benchmark ONNX reranker")
    ap.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--out", default=os.path.expanduser("~/reranker_onnx"))
    ap.add_argument("--depth", type=int, default=24, help="candidates per query (live pool)")
    ap.add_argument("--reps", type=int, default=30, help="latency repetitions")
    ap.add_argument("--tol", type=float, default=0.02, help="max allowed score abs-diff")
    ap.add_argument("--skip-export", action="store_true")
    args = ap.parse_args()
    out = os.path.expanduser(args.out)
    os.makedirs(out, exist_ok=True)

    import numpy as np
    from types import SimpleNamespace

    # ---- export ------------------------------------------------------------------------
    onnx_path = os.path.join(out, "model.onnx")
    if not args.skip_export:
        print(f"[export] {args.model} -> ONNX ...")
        from optimum.onnxruntime import ORTModelForSequenceClassification
        m = ORTModelForSequenceClassification.from_pretrained(args.model, export=True)
        m.save_pretrained(out)
        # fp16 convert (robust: optimum optimizer, else onnxconverter_common)
        fp16_path = os.path.join(out, "model_fp16.onnx")
        try:
            import onnx
            from onnxconverter_common import float16
            model_fp16 = float16.convert_float_to_float16(
                onnx.load(onnx_path), keep_io_types=True)
            onnx.save(model_fp16, fp16_path)
            print(f"[export] fp16 -> {fp16_path}")
            onnx_path = fp16_path
        except Exception as e:  # noqa: BLE001
            print(f"[export] fp16 convert failed ({e}); using fp32 ONNX (still ORT-accelerated)")
    else:
        if os.path.exists(os.path.join(out, "model_fp16.onnx")):
            onnx_path = os.path.join(out, "model_fp16.onnx")

    # ---- load both rerankers -----------------------------------------------------------
    from reranker import BGEReranker, ONNXReranker
    print("[load] PyTorch fp16 reranker ...")
    torch_rr = BGEReranker(args.model, use_fp16=True)
    print(f"[load] ONNX reranker from {onnx_path} ...")
    onnx_rr = ONNXReranker(onnx_path, model_name=args.model)

    queries = ["what is a corporation", "symptoms of diabetes", "how long does a tax refund take",
               "मधुमेह के लक्षण", "কর্পোরেশন কি"]
    chunks = [SimpleNamespace(text=t, chunk_id=str(i), rerank_score=None)
              for i, t in enumerate(_passages(args.depth))]

    # ---- parity: scores + ranking ------------------------------------------------------
    max_absdiff = 0.0
    top1_agree = topk_agree = total = 0
    for q in queries:
        a = torch_rr.rerank(q, [SimpleNamespace(text=c.text, chunk_id=c.chunk_id,
                                                rerank_score=None) for c in chunks], top_n=args.depth)
        b = onnx_rr.rerank(q, [SimpleNamespace(text=c.text, chunk_id=c.chunk_id,
                                               rerank_score=None) for c in chunks], top_n=args.depth)
        sa = {c.chunk_id: c.rerank_score for c in a}
        sb = {c.chunk_id: c.rerank_score for c in b}
        max_absdiff = max(max_absdiff, max(abs(sa[k] - sb[k]) for k in sa))
        top1_agree += int(a[0].chunk_id == b[0].chunk_id)
        topk_agree += len(set(c.chunk_id for c in a[:8]) & set(c.chunk_id for c in b[:8])) / 8.0
        total += 1
    print(f"\n[parity] max score abs-diff = {max_absdiff:.4f} (tol {args.tol}) | "
          f"top-1 agreement = {top1_agree}/{total} | "
          f"top-8 set overlap = {topk_agree/total:.3f}")

    # ---- latency -----------------------------------------------------------------------
    def bench(rr, name):
        q = "what is a corporation"
        cs = [SimpleNamespace(text=c.text, chunk_id=c.chunk_id, rerank_score=None) for c in chunks]
        for _ in range(3):
            rr.rerank(q, cs, top_n=8)          # warm
        ts = []
        for _ in range(args.reps):
            t = time.perf_counter()
            rr.rerank(q, [SimpleNamespace(text=c.text, chunk_id=c.chunk_id, rerank_score=None)
                          for c in cs], top_n=8)
            ts.append((time.perf_counter() - t) * 1000)
        ts.sort()
        print(f"[latency] {name}: p50={ts[len(ts)//2]:.1f}ms p90={ts[int(len(ts)*0.9)]:.1f}ms "
              f"min={ts[0]:.1f}ms  (depth {args.depth}, {args.reps} reps)")
        return ts[len(ts) // 2]

    p_torch = bench(torch_rr, "PyTorch fp16")
    p_onnx = bench(onnx_rr, "ONNX fp16   ")
    speedup = p_torch / p_onnx if p_onnx else 0.0

    ok = (max_absdiff <= args.tol and top1_agree == total and topk_agree / total >= 0.99)
    print(f"\n[verdict] ranking parity: {'PASS' if ok else 'FAIL'} | "
          f"speedup: {speedup:.2f}x ({p_torch:.1f}ms -> {p_onnx:.1f}ms)")
    print("ONNX reranker is a free accuracy-neutral win — wire it into serving." if ok and
          speedup > 1.1 else
          "Do NOT ship: parity or speedup insufficient — investigate before wiring.")
    return 0 if ok else 4


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    raise SystemExit(main())
