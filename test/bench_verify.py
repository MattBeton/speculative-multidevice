# bench_verify.py
# Benchmarks "spec verify" forwards: given a prefilled KV of length N for each
# of B sequences, feed K tokens per sequence at once (teacher-forced) and time
# the forward pass. Repeats for S steps and reports average.
#
# Backends:
#   - MLX  (Apple Silicon)
#   - HF/Transformers (CUDA)
#   - vLLM (CUDA; uses Automatic Prefix Caching to avoid re-prefill of N)
#
# Examples:
#   # MLX on Mac:
#   python bench_verify.py mlx \
#       --model-path "$(ls -d ~/.cache/huggingface/hub/models--mlx-community--Llama-3.2-3B-Instruct/snapshots/* | head -n1)" \
#       -b 32 -n 2048 -k 8 -s 50 --warmup 10
#
#   # HF on CUDA:
#   python bench_verify.py hf \
#       --model-id "/root/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95" \
#       -b 32 -n 2048 -k 8 -s 50 --warmup 10 --dtype auto --flash
#
#   # vLLM on CUDA (prefix-caching enabled):
#   python bench_verify.py vllm \
#       --model-id "/root/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95" \
#       -b 32 -n 2048 -k 8 -s 50 --warmup 10 --dtype auto --gpu-mem 0.95 --enable-prefix-caching
#
#   # Run all three:
#   python bench_verify.py all \
#       --mlx-model-path "$(ls -d ~/.cache/huggingface/hub/models--mlx-community--Llama-3.2-3B-Instruct/snapshots/* | head -n1)" \
#       --hf-model-id "/root/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95" \
#       --vllm-model-id "/root/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95" \
#       -b 32 -n 2048 -k 8 -s 50 --warmup 10 --dtype auto --flash --gpu-mem 0.95 --enable-prefix-caching

import argparse
import math
import time
import os
import numpy as np

def _pad_or_repeat_to_len(ids, target_len):
    if len(ids) == 0:
        ids = [1]
    if len(ids) >= target_len:
        return ids[:target_len]
    reps = math.ceil(target_len / len(ids))
    return (ids * reps)[:target_len]

def _build_prefix_and_verify_ids(tokenizer, b, n, k, prompt="Why is the sky blue? "):
    try:
        pref = tokenizer.encode(prompt, add_special_tokens=False)
    except Exception:
        pref = tokenizer.encode(" ", add_special_tokens=False) or [1]
    pref_ids = _pad_or_repeat_to_len(pref, n)
    prefix = np.tile(np.array(pref_ids, dtype=np.int32)[None, :], (b, 1))  # (B,N)
    try:
        sp = tokenizer.encode(" ", add_special_tokens=False)
    except Exception:
        sp = [1]
    vtok = sp[0] if len(sp) else 1
    verify = np.full((b, k), vtok, dtype=np.int32)  # (B,K)
    return prefix, verify

# ----------------------------- MLX backend -----------------------------
def run_mlx(model_path, b, n, k, steps, warmup, prompt):
    import mlx.core as mx
    from mlx_lm.utils import load_model
    from mlx_lm.tokenizer_utils import load_tokenizer
    from mlx_lm.models.cache import make_prompt_cache

    print(f"\n[MLX] model: {model_path}")
    model, _cfg = load_model(model_path)
    tok = load_tokenizer(model_path)

    prefix, verify = _build_prefix_and_verify_ids(tok, b, n, k, prompt)

    cache = make_prompt_cache(model)
    prefix_mx = mx.array(prefix, dtype=mx.int32)
    t0 = time.perf_counter()
    logits = model(prefix_mx, cache=cache)  # (B,N,V)
    mx.eval(logits)
    t_prefill = time.perf_counter() - t0

    verify_mx = mx.array(verify, dtype=mx.int32)
    times = []
    for i in range(warmup + steps):
        t1 = time.perf_counter()
        out = model(verify_mx, cache=cache)   # (B,K,V)
        mx.eval(out)
        dt = time.perf_counter() - t1
        if i >= warmup:
            times.append(dt)
        # Keep effective KV length ≈ N
        for c in cache:
            c.trim(k)

    avg = float(np.mean(times)) if times else 0.0
    tps = (b * k) / avg if avg > 0 else float("inf")
    print(f"[MLX] prefill: N={n}, B={b}   time={t_prefill*1000:.1f} ms")
    print(f"[MLX] verify:  K={k}, B={b}, steps={steps}, warmup={warmup}")
    print(f"[MLX] avg step: {avg*1000:.2f} ms   → {tps:.1f} tok/s (B×K per step)")
    return {"backend": "mlx", "avg_step_s": avg, "tok_per_sec": tps, "prefill_s": t_prefill}

# -------------------------- HF/Transformers backend --------------------------
def _pick_torch_dtype(arg):
    import torch
    if arg == "fp16": return torch.float16
    if arg == "bf16": return torch.bfloat16
    return None  # auto

def run_hf(model_id, b, n, k, steps, warmup, prompt, dtype_arg="auto", flash=False, device="cuda"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    assert torch.cuda.is_available(), "HF backend expects a CUDA device."
    print(f"\n[HF ] model: {model_id}")
    print(f"[HF ] dtype: {dtype_arg}   flash={flash}")

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    torch.backends.cuda.matmul.allow_tf32 = True

    model_kwargs = {"low_cpu_mem_usage": True}
    if flash:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=_pick_torch_dtype(dtype_arg) or (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16),
        **model_kwargs,
    ).to(device)
    model.eval()


    attn_impl = getattr(model.config, "_attn_implementation", None)
    print("attn_impl =", attn_impl)  # should be 'flash_attention_2'
    from transformers.models.llama.modeling_llama import LlamaFlashAttention2
    print(isinstance(model.model.layers[0].self_attn, LlamaFlashAttention2))


    prefix, verify = _build_prefix_and_verify_ids(tok, b, n, k, prompt)
    prefix_t = torch.tensor(prefix, dtype=torch.long, device=device)  # (B,N)
    verify_t = torch.tensor(verify, dtype=torch.long, device=device)  # (B,K)

    with torch.inference_mode():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(prefix_t, use_cache=True)
        torch.cuda.synchronize()
        t_prefill = time.perf_counter() - t0
        past0 = out.past_key_values

    attn = torch.ones((b, n + k), dtype=torch.long, device=device)
    times = []
    for i in range(warmup + steps):
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        with torch.inference_mode():
            _ = model(verify_t, use_cache=True, past_key_values=past0, attention_mask=attn)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t1
        if i >= warmup:
            times.append(dt)

    avg = float(np.mean(times)) if times else 0.0
    tps = (b * k) / avg if avg > 0 else float("inf")
    print(f"[HF ] prefill: N={n}, B={b}   time={t_prefill*1000:.1f} ms")
    print(f"[HF ] verify:  K={k}, B={b}, steps={steps}, warmup={warmup}")
    print(f"[HF ] avg step: {avg*1000:.2f} ms   → {tps:.1f} tok/s (B×K per step)")
    return {"backend": "hf", "avg_step_s": avg, "tok_per_sec": tps, "prefill_s": t_prefill}

# ------------------------------- vLLM backend -------------------------------
def _vllm_dtype(arg: str) -> str:
    if arg in ("auto", "fp16", "bfloat16", "float16", "float32"): return arg
    return "auto"

def run_vllm(model_id, b, n, k, steps, warmup, prompt,
             dtype="auto", gpu_mem=0.95, enable_prefix_caching=True,
             tp_size=1, enforce_eager=False, disable_cudagraph=False, max_model_len=None):
    # vLLM does prefill on the entire prompt. We rely on Automatic Prefix Caching:
    # 1) prefill once with (B,N) and generate 1 token to ensure the KV gets captured.
    # 2) for each step, call generate with (B,N+K) and max_tokens=0 (if supported),
    #    so only the K-token suffix is computed thanks to prefix cache reuse.
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    import torch

    # More verbose logs from child processes
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "DEBUG")
    os.environ.setdefault("VLLM_ENGINE_LOGGING_LEVEL", "DEBUG")

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<not-set>")
    print(f"\n[vLLM] model: {model_id}")
    print(f"[vLLM] dtype={dtype}  gpu_mem={gpu_mem:.2f}  prefix_cache={enable_prefix_caching}  tp={tp_size}")
    print(f"[vLLM] CUDA_VISIBLE_DEVICES={visible}  torch.cuda.device_count()={torch.cuda.device_count()}")

    try:
        llm = LLM(
            model=model_id,
            trust_remote_code=True,
            dtype=_vllm_dtype(dtype),
            gpu_memory_utilization=float(gpu_mem),
            enable_prefix_caching=bool(enable_prefix_caching),
            tensor_parallel_size=int(tp_size),
            # Make startup more robust:
            enforce_eager=bool(enforce_eager),
        )
    except Exception as e:
        print("[vLLM] Engine init failed. Hints:")
        print("  • Try: CUDA_VISIBLE_DEVICES=0  (or set --tp-size 1)")
        print("  • Add: --enforce-eager  --no-cudagraph")
        print("  • If in Docker: use --shm-size 8g  --ipc=host")
        print("  • Ensure flash-attn matches your GPU arch, or just run without it.")
        raise
    tok = llm.get_tokenizer()

    prefix, verify = _build_prefix_and_verify_ids(tok, b, n, k, prompt)

    # Build B prompts for prefill warmup (prefix only)
    prefill_prompts = [TokensPrompt(prompt_token_ids=prefix[i].tolist()) for i in range(b)]
    sp_prefill = SamplingParams(max_tokens=1, detokenize=False, temperature=0.0)

    t0 = time.perf_counter()
    _ = llm.generate(prefill_prompts, sp_prefill)  # ensures prefix KV is cached
    t_prefill = time.perf_counter() - t0

    # Verify-step prompts = prefix + K tokens (identical prefix to trigger caching)
    times = []
    # Prefer no decode work: max_tokens=0 (supported in newer vLLM); fall back to 1 if needed.
    try_zero = True
    for i in range(warmup + steps):
        prompts = [
            TokensPrompt(prompt_token_ids=np.concatenate([prefix[j], verify[j]]).astype(int).tolist())
            for j in range(b)
        ]
        t1 = time.perf_counter()
        try:
            if try_zero:
                sp = SamplingParams(max_tokens=0, detokenize=False)  # "prompt-only" compute
                _ = llm.generate(prompts, sp)
            else:
                sp = SamplingParams(max_tokens=1, detokenize=False, temperature=0.0)
                _ = llm.generate(prompts, sp)
        except Exception:
            # Older vLLM may not accept max_tokens=0; switch to +1 decode fallback
            try_zero = False
            sp = SamplingParams(max_tokens=1, detokenize=False, temperature=0.0)
            _ = llm.generate(prompts, sp)
        dt = time.perf_counter() - t1
        if i >= warmup:
            times.append(dt)

    avg = float(np.mean(times)) if times else 0.0
    tps = (b * k) / avg if avg > 0 else float("inf")
    extra = "" if try_zero else "  (+1 decode included)"
    print(f"[vLLM] prefill: N={n}, B={b}   time={t_prefill*1000:.1f} ms")
    print(f"[vLLM] verify:  K={k}, B={b}, steps={steps}, warmup={warmup}{extra}")
    print(f"[vLLM] avg step: {avg*1000:.2f} ms   → {tps:.1f} tok/s (B×K per step)")
    return {"backend": "vllm", "avg_step_s": avg, "tok_per_sec": tps, "prefill_s": t_prefill, "includes_decode": (not try_zero)}

# --------------------------------- CLI ---------------------------------
def main():
    p = argparse.ArgumentParser(description="Benchmark verify-forward (B×K tokens with KV length N)")
    sub = p.add_subparsers(dest="mode", required=True)

    # MLX
    mlx = sub.add_parser("mlx", help="Run MLX benchmark")
    mlx.add_argument("--model-path", type=str, required=True)
    mlx.add_argument("-b", "--batch", type=int, default=8)
    mlx.add_argument("-n", "--n-prefix", type=int, default=2048)
    mlx.add_argument("-k", "--k-verify", type=int, default=8)
    mlx.add_argument("-s", "--steps", type=int, default=50)
    mlx.add_argument("--warmup", type=int, default=10)
    mlx.add_argument("--prompt", type=str, default="Why is the sky blue? ")

    # HF
    hf = sub.add_parser("hf", help="Run HF/Transformers (CUDA) benchmark")
    hf.add_argument("--model-id", type=str, required=True)
    hf.add_argument("-b", "--batch", type=int, default=8)
    hf.add_argument("-n", "--n-prefix", type=int, default=2048)
    hf.add_argument("-k", "--k-verify", type=int, default=8)
    hf.add_argument("-s", "--steps", type=int, default=50)
    hf.add_argument("--warmup", type=int, default=10)
    hf.add_argument("--prompt", type=str, default="Why is the sky blue? ")
    hf.add_argument("--dtype", type=str, default="auto", choices=["auto", "fp16", "bf16"])
    hf.add_argument("--flash", action="store_true")
    hf.add_argument("--device", type=str, default="cuda")

    # vLLM
    v = sub.add_parser("vllm", help="Run vLLM (CUDA) benchmark with prefix caching")
    v.add_argument("--model-id", type=str, required=True)
    v.add_argument("-b", "--batch", type=int, default=8)
    v.add_argument("-n", "--n-prefix", type=int, default=2048)
    v.add_argument("-k", "--k-verify", type=int, default=8)
    v.add_argument("-s", "--steps", type=int, default=50)
    v.add_argument("--warmup", type=int, default=10)
    v.add_argument("--prompt", type=str, default="Why is the sky blue? ")
    v.add_argument("--dtype", type=str, default="auto", choices=["auto", "fp16", "bfloat16", "float16", "float32"])
    v.add_argument("--gpu-mem", type=float, default=0.95)
    v.add_argument("--enable-prefix-caching", action="store_true")
    v.add_argument("--tp-size", type=int, default=1, help="tensor_parallel_size (set 1 first)")
    v.add_argument("--enforce-eager", action="store_true", help="Force eager (avoid cudagraph pitfalls)")
    v.add_argument("--no-cudagraph", action="store_true", help="Set max_seq_len_to_capture=0")
    v.add_argument("--max-model-len", type=int, default=None, help="Cap model context; try N+K")

    # both (mlx+hf)
    both = sub.add_parser("both", help="Run MLX + HF")
    both.add_argument("--mlx-model-path", type=str, required=True)
    both.add_argument("--hf-model-id", type=str, required=True)
    both.add_argument("-b", "--batch", type=int, default=8)
    both.add_argument("-n", "--n-prefix", type=int, default=2048)
    both.add_argument("-k", "--k-verify", type=int, default=8)
    both.add_argument("-s", "--steps", type=int, default=50)
    both.add_argument("--warmup", type=int, default=10)
    both.add_argument("--prompt", type=str, default="Why is the sky blue? ")
    both.add_argument("--dtype", type=str, default="auto", choices=["auto", "fp16", "bf16"])
    both.add_argument("--flash", action="store_true")
    both.add_argument("--device", type=str, default="cuda")

    # all (mlx+hf+vllm)
    allp = sub.add_parser("all", help="Run MLX + HF + vLLM")
    allp.add_argument("--mlx-model-path", type=str, required=True)
    allp.add_argument("--hf-model-id", type=str, required=True)
    allp.add_argument("--vllm-model-id", type=str, required=True)
    allp.add_argument("-b", "--batch", type=int, default=8)
    allp.add_argument("-n", "--n-prefix", type=int, default=2048)
    allp.add_argument("-k", "--k-verify", type=int, default=8)
    allp.add_argument("-s", "--steps", type=int, default=50)
    allp.add_argument("--warmup", type=int, default=10)
    allp.add_argument("--prompt", type=str, default="Why is the sky blue? ")
    allp.add_argument("--dtype", type=str, default="auto", choices=["auto", "fp16", "bf16"])
    allp.add_argument("--flash", action="store_true")
    allp.add_argument("--device", type=str, default="cuda")
    allp.add_argument("--gpu-mem", type=float, default=0.95)
    allp.add_argument("--enable-prefix-caching", action="store_true")
    allp.add_argument("--tp-size", type=int, default=1)
    allp.add_argument("--enforce-eager", action="store_true")
    allp.add_argument("--no-cudagraph", action="store_true")
    allp.add_argument("--max-model-len", type=int, default=None)

    args = p.parse_args()

    if args.mode == "mlx":
        run_mlx(args.model_path, args.batch, args.n_prefix, args.k_verify, args.steps, args.warmup, args.prompt)

    elif args.mode == "hf":
        run_hf(args.model_id, args.batch, args.n_prefix, args.k_verify, args.steps, args.warmup,
               args.prompt, args.dtype, args.flash, args.device)

    elif args.mode == "vllm":
        run_vllm(args.model_id, args.batch, args.n_prefix, args.k_verify, args.steps, args.warmup,
                 args.prompt, args.dtype, args.gpu_mem, args.enable_prefix_caching,
                 args.tp_size, args.enforce_eager, args.no_cudagraph, args.max_model_len)

    elif args.mode == "both":
        r1 = run_mlx(args.mlx_model_path, args.batch, args.n_prefix, args.k_verify, args.steps, args.warmup, args.prompt)
        r2 = run_hf(args.hf_model_id, args.batch, args.n_prefix, args.k_verify, args.steps, args.warmup,
                    args.prompt, args.dtype, args.flash, args.device)
        print("\n==== Summary (avg per step) ====")
        print(f"MLX : {r1['avg_step_s']*1000:.2f} ms | {r1['tok_per_sec']:.1f} tok/s")
        print(f"HF  : {r2['avg_step_s']*1000:.2f} ms | {r2['tok_per_sec']:.1f} tok/s")

    else:  # all
        r1 = run_mlx(args.mlx_model_path, args.batch, args.n_prefix, args.k_verify, args.steps, args.warmup, args.prompt)
        r2 = run_hf(args.hf_model_id, args.batch, args.n_prefix, args.k_verify, args.steps, args.warmup,
                    args.prompt, args.dtype, args.flash, args.device)
        r3 = run_vllm(args.vllm_model_id, args.batch, args.n_prefix, args.k_verify, args.steps, args.warmup,
                      args.prompt, args.dtype, args.gpu_mem, args.enable_prefix_caching,
                      args.tp_size, args.enforce_eager, args.no_cudagraph, args.max_model_len)
        print("\n==== Summary (avg per step) ====")
        print(f"MLX : {r1['avg_step_s']*1000:.2f} ms | {r1['tok_per_sec']:.1f} tok/s")
        print(f"HF  : {r2['avg_step_s']*1000:.2f} ms | {r2['tok_per_sec']:.1f} tok/s")
        dec = " (+1 decode)" if r3.get("includes_decode") else ""
        print(f"vLLM: {r3['avg_step_s']*1000:.2f} ms | {r3['tok_per_sec']:.1f} tok/s{dec}")

if __name__ == "__main__":
    main()
