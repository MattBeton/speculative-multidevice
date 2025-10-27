import argparse, time, math, os
import numpy as np
import torch, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from gemlite.helper import A8W8_INT8_dynamic

def _pad_or_repeat(ids, target_len, pad_id=1):
    if not ids: ids = [pad_id]
    if len(ids) >= target_len: return ids[:target_len]
    reps = math.ceil(target_len / len(ids))
    return (ids * reps)[:target_len]

def make_prefix_and_verify(tok, B, N, K, prompt):
    try: pref = tok.encode(prompt, add_special_tokens=False)
    except Exception: pref = tok.encode(" ", add_special_tokens=False) or [tok.eos_token_id or 1]
    pref_ids = _pad_or_repeat(pref, N, pad_id=(tok.eos_token_id or 1))
    prefix = np.tile(np.array(pref_ids, dtype=np.int32)[None, :], (B, 1))
    try: sp = tok.encode(" ", add_special_tokens=False)
    except Exception: sp = [tok.eos_token_id or 1]
    vtok = sp[0] if len(sp) else (tok.eos_token_id or 1)
    verify = np.full((B, K), vtok, dtype=np.int32)
    return prefix, verify

def load_model(model_id: str, dtype: torch.dtype, device: str, attn: str):
    kwargs = dict(low_cpu_mem_usage=True)
    if attn in ("sdpa", "flash_attention_2"):
        kwargs["attn_implementation"] = attn
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, **kwargs).to(device)
    model.eval()
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    return model, tok

def patch_linears_int8(model: nn.Module, device: str):
    for name, mod in model.named_modules():
        setattr(mod, "name", name)
    proc = A8W8_INT8_dynamic(device=device)
    def convert(m: nn.Module):
        for name, child in list(m.named_children()):
            if isinstance(child, nn.Linear):
                setattr(m, name, proc.from_linear(child))
            else:
                convert(child)
    convert(model)
    torch.cuda.empty_cache()
    return model

@torch.inference_mode()
def bench_verify_step(model, prefix_ids, verify_ids, device, warmup, steps):
    B, N = prefix_ids.shape
    _, K = verify_ids.shape
    prefix_t = torch.tensor(prefix_ids, device=device, dtype=torch.long)
    verify_t = torch.tensor(verify_ids, device=device, dtype=torch.long)
    attn = torch.ones((B, N + K), dtype=torch.long, device=device)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model(prefix_t, use_cache=True)  # build past of length N
    torch.cuda.synchronize()
    t_prefill = time.perf_counter() - t0
    past = out.past_key_values

    times = []
    for i in range(warmup + steps):
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        _ = model(verify_t, use_cache=True, past_key_values=past, attention_mask=attn)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t1
        if i >= warmup: times.append(dt)

    avg = float(np.mean(times)) if times else 0.0
    tps = (B * K) / avg if avg > 0 else float("inf")
    return t_prefill, avg, tps

def run_once(model_id, device, dtype, attn, B, N, K, steps, warmup, prompt):
    # FP16
    m_fp16, tok = load_model(model_id, dtype, device, attn)
    prefix, verify = make_prefix_and_verify(tok, B, N, K, prompt)
    tpf16, avg16, tps16 = bench_verify_step(m_fp16, prefix, verify, device, warmup, steps)

    # INT8
    m_i8, _ = load_model(model_id, dtype, device, attn)
    m_i8 = patch_linears_int8(m_i8, device)
    # prime Triton JIT/pack (exclude from timing)
    dummy = torch.tensor(prefix[:1, :16], device=device, dtype=torch.long)
    _ = m_i8(dummy, use_cache=True); torch.cuda.synchronize()

    tpi8, avgi8, tpsi8 = bench_verify_step(m_i8, prefix, verify, device, warmup, steps)
    return (tpf16, avg16, tps16), (tpi8, avgi8, tpsi8)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16","bf16"])
    ap.add_argument("--attn", type=str, default="sdpa", choices=["sdpa","flash_attention_2"])
    ap.add_argument("-b", "--batch", type=int, nargs="+", default=[8])
    ap.add_argument("-n", "--n-prefix", type=int, default=2048)
    ap.add_argument("-k", "--k-verify", type=int, nargs="+", default=[8])
    ap.add_argument("-s", "--steps", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--prompt", type=str, default="Why is the sky blue? ")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    torch.backends.cuda.matmul.allow_tf32 = False
    dt = torch.bfloat16 if (args.dtype=="bf16" and torch.cuda.is_bf16_supported()) else torch.float16

    print(f"Device: {torch.cuda.get_device_name(0)}  CC={torch.cuda.get_device_capability(0)}  torch={torch.__version__}")
    print(f"Model : {args.model}   DType: {dt}   Attn: {args.attn}")
    print(f"N={args.n_prefix}   steps={args.steps} warmup={args.warmup}   B={args.batch}   K={args.k_verify}")

    rows = []
    for B in args.batch:
        for K in args.k_verify:
            (tpf16, avg16, tps16), (tpi8, avgi8, tpsi8) = run_once(
                args.model, args.device, dt, args.attn, B, args.n_prefix, K, args.steps, args.warmup, args.prompt
            )
            speed = (avg16 / avgi8) if (avg16>0 and avgi8>0) else float("nan")
            rows.append((B, K, tpf16, avg16, tps16, tpi8, avgi8, tpsi8, speed))
            print(f"\n[B={B} K={K}]")
            print(f"  FP16 verify: {avg16*1e3:.2f} ms → {tps16:.1f} tok/s | prefill {tpf16*1e3:.1f} ms")
            print(f"  INT8 verify: {avgi8*1e3:.2f} ms → {tpsi8:.1f} tok/s | prefill {tpi8*1e3:.1f} ms (post-JIT) ")
            print(f"  Speedup INT8/FP16: {speed:.2f}×")

    # Pretty summary table
    print("\n==== Summary (verify step) ====")
    print("   B    K |  FP16 ms   tok/s  ||  INT8 ms   tok/s  ||  Speedup")
    for (B,K,tpf16,avg16,tps16,tpi8,avgi8,tpsi8,speed) in rows:
        print(f"{B:4d} {K:4d} | {avg16*1e3:7.2f}  {tps16:6.1f}  || {avgi8*1e3:7.2f}  {tpsi8:6.1f}  ||  {speed:6.2f}×")

if __name__ == "__main__":
    main()
