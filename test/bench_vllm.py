#!/usr/bin/env python3
# bench_vllm_verify.py
import argparse, os, time, math, hashlib
import numpy as np

# ---- Environment sanity ----
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ.setdefault("VLLM_ENGINE_LOGGING_LEVEL", "WARNING")
# If you previously hit OOM in FlashInfer sampler during 1-token decode, uncomment:
# os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

def _pad_or_repeat_to_len(ids, target_len):
    if not ids: ids = [1]
    if len(ids) >= target_len: return ids[:target_len]
    reps = math.ceil(target_len / len(ids))
    return (ids * reps)[:target_len]

def _build_prefix_ids(tok, b, n, prompt):
    try:
        pref = tok.encode(prompt, add_special_tokens=False)
    except Exception:
        pref = tok.encode(" ", add_special_tokens=False) or [1]
    pref_ids = _pad_or_repeat_to_len(pref, n)
    prefix = np.tile(np.array(pref_ids, dtype=np.int32)[None, :], (b, 1))  # (B,N)
    return prefix

def _vocab_meta(tok):
    vs = getattr(tok, "vocab_size", None)
    if vs is None:
        try: vs = tok.vocab_size
        except Exception: vs = 32000
    reserved = set()
    for name in ("bos_token_id","eos_token_id","pad_token_id","unk_token_id"):
        tid = getattr(tok, name, None)
        if tid is not None:
            try: reserved.add(int(tid))
            except: pass
    low_floor = 32  # avoid tiny control ids
    return vs, reserved, low_floor

def _rng_from_hash(*parts):
    h = hashlib.sha256(":".join(map(str, parts)).encode()).digest()
    s1 = int.from_bytes(h[0:8], "little"); s2 = int.from_bytes(h[8:16], "little")
    return np.random.default_rng((s1, s2))

def _draw_k_ids(vocab_size, reserved, low_floor, k, iter_idx, seq_idx):
    rng = _rng_from_hash("verify", iter_idx, seq_idx)
    ids = []
    tries = 0
    while len(ids) < k:
        x = int(rng.integers(low_floor, vocab_size-1))
        if x not in reserved:
            ids.append(x)
        tries += 1
        if tries > 10_000 and len(ids) < k:
            # extremely unlikely; fallback
            stride = 97
            base = (low_floor + iter_idx*131 + seq_idx*233) % (vocab_size - low_floor) + low_floor
            while len(ids) < k:
                base = low_floor + ((base + stride) % (vocab_size - low_floor))
                if base not in reserved: ids.append(int(base))
            break
    return np.array(ids, dtype=np.int32)

def main():
    ap = argparse.ArgumentParser(description="vLLM verify-step microbenchmark (B×K tokens with cached N-prefix)")
    ap.add_argument("--model-id", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("-b", "--batch", type=int, default=8)
    ap.add_argument("-n", "--n-prefix", type=int, default=2048)
    ap.add_argument("-k", "--k-verify", type=int, default=64)   # K>=64 gives a cleaner signal
    ap.add_argument("-s", "--steps", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--prompt", type=str, default="Why is the sky blue? ")
    ap.add_argument("--dtype", type=str, default="auto", choices=["auto","fp16","bfloat16","float16","float32"])
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--tp-size", type=int, default=1)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    # Device info (optional)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<not-set>")
    try:
        import torch
        ngpu = torch.cuda.device_count()
    except Exception:
        ngpu = -1
    print(f"\n[vLLM] model: {args.model_id}")
    print(f"[vLLM] dtype={args.dtype}  gpu_mem={args.gpu_mem:.2f}  tp={args.tp_size}  CUDAs={visible}  #gpus={ngpu}")

    llm = LLM(
        model=args.model_id,
        trust_remote_code=True,
        dtype=args.dtype,
        gpu_memory_utilization=float(args.gpu_mem),
        enable_prefix_caching=True,           # critical
        tensor_parallel_size=int(args.tp_size),
    )
    tok = llm.get_tokenizer()
    vocab_size, reserved, low_floor = _vocab_meta(tok)

    B, N, K = args.batch, args.n_prefix, args.k_verify
    prefix = _build_prefix_ids(tok, B, N, args.prompt)
    prefill_prompts = [TokensPrompt(prompt_token_ids=prefix[i].tolist()) for i in range(B)]

    # Warm the prefix cache (compute N and capture KV)
    sp_prefill = SamplingParams(max_tokens=1, detokenize=False, temperature=0.0, top_k=1, top_p=1.0, seed=0)
    t0 = time.perf_counter()
    _ = llm.generate(prefill_prompts, sp_prefill)
    t_prefill = time.perf_counter() - t0
    print(f"[vLLM] prefill: N={N}, B={B}   time={t_prefill*1000:.1f} ms")

    total_iters = args.warmup + args.steps

    # Build UNIQUE verify prompts per (iter, seq) to defeat batch de-dup
    verify_prompts_all = []
    for i in range(total_iters):
        prompts_i = []
        for j in range(B):
            v_row = _draw_k_ids(vocab_size, reserved, low_floor, K, i, j)
            ids = np.concatenate([prefix[j], v_row]).tolist()
            prompts_i.append(TokensPrompt(prompt_token_ids=ids))
        verify_prompts_all.append(prompts_i)

    # 1-token decode params (greedy). Unique seeds per call to defeat result caching.
    def sp_decode(seed):
        return SamplingParams(max_tokens=1, detokenize=False, temperature=0.0, top_k=1, top_p=1.0, seed=seed)

    # Baseline: decode 1 token from cached prefix-only
    base_times = []
    for i in range(total_iters):
        t1 = time.perf_counter()
        _ = llm.generate(prefill_prompts, sp_decode(1_000_000 + i))
        dt = time.perf_counter() - t1
        if i >= args.warmup:
            base_times.append(dt)
    baseline = float(np.mean(base_times)) if base_times else 0.0

    # Verify+decode: prefix + K unique ids per (iter, seq)
    verify_times = []
    for i in range(total_iters):
        t1 = time.perf_counter()
        _ = llm.generate(verify_prompts_all[i], sp_decode(2_000_000 + i))
        dt = time.perf_counter() - t1
        if i >= args.warmup:
            verify_times.append(dt)
    vpd = float(np.mean(verify_times)) if verify_times else 0.0

    verify_only = max(vpd - baseline, 0.0)
    tps = (B * K) / verify_only if verify_only > 0 else float("inf")

    def _stats(xs):
        xs = np.array(xs, dtype=np.float64)
        if xs.size == 0: return (0.0, 0.0, 0.0)
        return (float(np.mean(xs)), float(np.median(xs)), float(np.std(xs)))

    b_mean, b_med, b_std = _stats(base_times)
    v_mean, v_med, v_std = _stats(verify_times)
    print(f"[vLLM] baseline decode (prefix-only):    mean={b_mean*1000:.3f} ms | median={b_med*1000:.3f} ms | std={b_std*1000:.3f} ms")
    print(f"[vLLM] verify+decode (prefix+K unique):   mean={v_mean*1000:.3f} ms | median={v_med*1000:.3f} ms | std={v_std*1000:.3f} ms")

    if verify_only <= 0:
        print("\n[!] verify_only <= 0 ms. Likely cache hit or overhead swamping signal.")
        print("    • Increase --k to 128/256 and/or increase -b.")
        print("    • Make sure VLLM_* logs aren’t hiding errors.")
    print(f"\n[vLLM] verify-only: K={K}, B={B}, steps={args.steps}, warmup={args.warmup}")
    if verify_only > 0:
        print(f"[vLLM] avg step: {verify_only*1000:.3f} ms   → {tps:.1f} tok/s (B×K per step)")
    else:
        print(f"[vLLM] avg step: 0.000 ms   → inf tok/s (B×K per step)  [INVALID; see warning above]")

if __name__ == "__main__":
    main()
