#!/usr/bin/env python3
"""
Benchmark Llama-3.2-3B-Instruct verify step for speculative decoding (MLX).

This script measures the time to do a forward pass of k tokens after n have
been prefilled, using Apple's MLX runtime on Apple silicon and the
mlx-community conversion of the model.

Example:
  pip install --upgrade mlx-lm
  python mlx_verify_bench.py -b 8 -k 8 -n 1024 \
      --model mlx-community/Llama-3.2-3B-Instruct

To try a quantized variant (often faster on laptops):
  --model mlx-community/Llama-3.2-3B-Instruct-4bit
"""

import time
import argparse
import numpy as np
from typing import Dict, List

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache


def _to_fixed_length_tokens(tokenizer, total_length: int) -> List[int]:
    """
    Create exactly `total_length` token ids by repeating a base text.
    We avoid return_tensors to keep it framework-agnostic.
    """
    base_text = "The quick brown fox jumps over the lazy dog. " * 300
    toks = tokenizer.encode(base_text, add_special_tokens=False)
    if len(toks) >= total_length:
        return toks[:total_length]
    # Repeat/crop
    reps = (total_length + len(toks) - 1) // len(toks)
    toks = (toks * reps)[:total_length]
    return toks


def _as_batched(tokens_1d: mx.array, batch_size: int) -> mx.array:
    """
    Make a (batch, length) array by broadcasting a 1D token array.
    """
    tokens_1d = tokens_1d.reshape((-1,))
    length = int(tokens_1d.shape[0])
    return mx.broadcast_to(tokens_1d[None, :], (batch_size, length))


def _create_cache(model, max_kv_size: int | None):
    """
    Build a prompt cache; if the installed mlx-lm supports max_kv_size, pass it.
    """
    try:
        if max_kv_size is not None:
            return make_prompt_cache(model, max_kv_size=max_kv_size)
    except TypeError:
        pass
    return make_prompt_cache(model)


def benchmark_verify_step(
    model,
    prefill_ids: mx.array,     # (B, n)
    verify_ids: mx.array,      # (B, k)
    batch_size: int,
    num_warmup: int = 3,
    num_iterations: int = 10,
    max_kv_size: int | None = None,
) -> Dict:
    """
    Benchmark the verify step: forward pass of k tokens after n prefilled tokens.

    Returns a dict with latency stats and throughput.
    """
    B, n = prefill_ids.shape
    _, k = verify_ids.shape
    assert B == batch_size, "Batch mismatch"

    print("\nRunning verify benchmark:")
    print(f"  Backend: MLX / Metal available: {mx.metal.is_available()}")
    try:
        info = mx.metal.device_info()
        print(f"  Metal device info: arch={info.get('architecture')} "
              f"mem={info.get('memory_size')} bytes "
              f"max_ws={info.get('max_recommended_working_set_size')} bytes")
    except Exception:
        pass
    print(f"  Batch size: {batch_size}")
    print(f"  Prefill length: {n} tokens")
    print(f"  Verify tokens: {k}")
    print(f"  Input shapes - Prefill: {tuple(prefill_ids.shape)}, Verify: {tuple(verify_ids.shape)}")

    # Build cache and prefill once
    cache = _create_cache(model, max_kv_size=max_kv_size)
    print("\n  Performing initial prefill...")
    logits = model(prefill_ids, cache=cache)     # updates cache in-place
    mx.eval(logits)                               # force compute

    # Warm-up verify passes
    print(f"  Warming up verify step with {num_warmup} iterations...")
    for _ in range(num_warmup):
        _ = model(verify_ids, cache=cache)
        mx.eval(_)
        # try to roll back the cache by k tokens; if not possible, rebuild+prefill
        try:
            trim_prompt_cache(cache, int(k))
        except Exception:
            cache = _create_cache(model, max_kv_size=max_kv_size)
            _ = model(prefill_ids, cache=cache)
            mx.eval(_)

    # Timed verify passes
    print(f"  Running {num_iterations} benchmark iterations for verify step...")
    latencies_ms = []
    first_logits_shape = None

    for i in range(num_iterations):
        # Ensure we are at the post-prefill state (cache trimmed in warmup/last loop)
        start = time.perf_counter()
        out = model(verify_ids, cache=cache)  # process k tokens reusing KV
        mx.eval(out)                          # synchronize/force compute
        end = time.perf_counter()

        if first_logits_shape is None:
            first_logits_shape = tuple(out.shape)
            print(f"  Verify output logits shape: {first_logits_shape}")

        latencies_ms.append((end - start) * 1000.0)

        # reset cache to post-prefill for next iteration
        try:
            trim_prompt_cache(cache, int(k))
        except Exception:
            cache = _create_cache(model, max_kv_size=max_kv_size)
            _ = model(prefill_ids, cache=cache)
            mx.eval(_)

    lat = np.array(latencies_ms, dtype=np.float64)
    mean_ms = float(np.mean(lat))
    std_ms = float(np.std(lat))
    total_tokens = int(batch_size * k)
    tokens_per_second = (total_tokens / mean_ms) * 1000.0  # ms -> s
    results = {
        "batch_size": batch_size,
        "prefill_length": int(n),
        "verify_tokens": int(k),
        "total_tokens": total_tokens,
        "mean_latency_ms": mean_ms,
        "std_latency_ms": std_ms,
        "min_latency_ms": float(np.min(lat)),
        "max_latency_ms": float(np.max(lat)),
        "p50_latency_ms": float(np.percentile(lat, 50)),
        "p90_latency_ms": float(np.percentile(lat, 90)),
        "p99_latency_ms": float(np.percentile(lat, 99)),
        "tokens_per_second": float(tokens_per_second),
        "time_per_token_ms": float(mean_ms / total_tokens),
        "all_latencies": latencies_ms,
        "logits_shape": first_logits_shape,
    }
    return results


def grid_search(
    model,
    tokenizer,
    prefill_length: int = 1024,
    batch_sizes: List[int] = [4, 8, 16, 32],
    verify_tokens_list: List[int] = [4, 8, 12],
    num_warmup: int = 3,
    num_iterations: int = 10,
    max_kv_size: int | None = None,
) -> List[Dict]:
    """
    Perform grid search over batch sizes and verify token counts.
    """
    max_verify = max(verify_tokens_list)
    total_len = prefill_length + max_verify
    base_tokens = _to_fixed_length_tokens(tokenizer, total_len)
    base_tokens = mx.array(base_tokens, dtype=mx.int32)

    print(f"\nCreated input tensor with shape: {(1, total_len)}")

    results = []

    print("\n" + "=" * 70)
    print("Starting Grid Search")
    print("=" * 70)
    print(f"Batch sizes: {batch_sizes}")
    print(f"Verify token counts: {verify_tokens_list}")
    print(f"Prefill length: {prefill_length}")
    print(f"Total configurations: {len(batch_sizes) * len(verify_tokens_list)}")

    for bsz in batch_sizes:
        for k in verify_tokens_list:
            print("\n" + "-" * 70)
            print(f"Configuration: batch_size={bsz}, verify_tokens={k}")
            print("-" * 70)
            try:
                tokens = base_tokens[: prefill_length + k]
                batched = _as_batched(tokens, bsz)
                prefill_ids = batched[:, :prefill_length]
                verify_ids = batched[:, prefill_length: prefill_length + k]

                # Ensure the cache window is large enough to avoid rotation during verify
                kv_size = max_kv_size
                if kv_size is None:
                    kv_size = prefill_length + k + 8  # tiny headroom

                result = benchmark_verify_step(
                    model=model,
                    prefill_ids=prefill_ids,
                    verify_ids=verify_ids,
                    batch_size=bsz,
                    num_warmup=num_warmup,
                    num_iterations=num_iterations,
                    max_kv_size=kv_size,
                )
                results.append(result)
                print(f"\n  Results:")
                print(f"    Mean latency: {result['mean_latency_ms']:.2f} ms")
                print(f"    Throughput: {result['tokens_per_second']:.2f} tokens/second")
                print(f"    Time per token: {result['time_per_token_ms']:.3f} ms")
            except Exception as e:
                print(f"  ERROR: Failed with {e}")
                results.append({
                    "batch_size": bsz,
                    "verify_tokens": k,
                    "error": str(e),
                    "tokens_per_second": 0.0,
                })

    return results


def print_summary(results: List[Dict]):
    """Print a summary table of all results."""
    print("\n" + "=" * 70)
    print("GRID SEARCH SUMMARY")
    print("=" * 70)

    print(f"\n{'Batch Size':<12} {'Verify Tokens':<15} {'Latency (ms)':<15} "
          f"{'Throughput (tok/s)':<20} {'Time/Token (ms)':<15}")
    print("-" * 92)

    sorted_results = sorted(results, key=lambda x: x.get('tokens_per_second', 0), reverse=True)
    for r in sorted_results:
        if 'error' in r:
            print(f"{r['batch_size']:<12} {r.get('verify_tokens','N/A'):<15} "
                  f"{'ERROR':<15} {'ERROR':<20} {'ERROR':<15}")
        else:
            print(f"{r['batch_size']:<12} {r['verify_tokens']:<15} "
                  f"{r['mean_latency_ms']:<15.2f} {r['tokens_per_second']:<20.2f} "
                  f"{r['time_per_token_ms']:<15.3f}")

    best = max(sorted_results, key=lambda x: x.get('tokens_per_second', 0))
    if best.get('tokens_per_second', 0) > 0:
        print(f"\nBest configuration:")
        print(f"  Batch size: {best['batch_size']}")
        print(f"  Verify tokens: {best['verify_tokens']}")
        print(f"  Throughput: {best['tokens_per_second']:.2f} tokens/second")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Llama verify step for speculative decoding (MLX)")
    parser.add_argument('--batch-size', '-b', type=int, default=8,
                        help='Default batch size for single benchmark (default: 8)')
    parser.add_argument('--verify-tokens', '-k', type=int, default=8,
                        help='Number of tokens to verify (default: 8)')
    parser.add_argument('--prefill-length', '-n', type=int, default=1024,
                        help='Number of tokens to prefill (default: 1024)')
    parser.add_argument('--grid-search', action='store_true',
                        help='Run grid search over batch sizes and verify tokens')
    parser.add_argument('--warmup', type=int, default=3,
                        help='Number of warmup iterations (default: 3)')
    parser.add_argument('--iterations', type=int, default=10,
                        help='Number of benchmark iterations (default: 10)')
    parser.add_argument('--model', type=str, default='mlx-community/Llama-3.2-3B-Instruct',
                        help='MLX model repo (default: mlx-community/Llama-3.2-3B-Instruct)')
    parser.add_argument('--max-kv-size', type=int, default=None,
                        help='Fixed KV cache window; if unset we size to n+k+headroom')
    args = parser.parse_args()

    print("=" * 70)
    print("Llama-3.2-3B-Instruct Verify Step Benchmark (MLX)")
    print("=" * 70)

    # Load model + tokenizer (downloads from HF if needed)
    print(f"\nLoading model from {args.model} ...")
    model, tokenizer = load(args.model)
    print("Model loaded successfully!")

    # Print minimal model info (attributes vary by model)
    try:
        cfg = getattr(model, "config", None) or getattr(model, "args", None)
        if cfg is not None:
            hidden = getattr(cfg, "hidden_size", None)
            num_layers = getattr(cfg, "num_hidden_layers", None)
            num_heads = getattr(cfg, "num_attention_heads", None)
            kv_heads = getattr(cfg, "num_key_value_heads", None)
            vocab = getattr(cfg, "vocab_size", None)
            print("\nModel Configuration:")
            if hidden is not None: print(f"  Hidden size: {hidden}")
            if num_layers is not None: print(f"  Num layers: {num_layers}")
            if num_heads is not None: print(f"  Num attention heads: {num_heads}")
            if kv_heads is not None: print(f"  Num KV heads: {kv_heads}")
            if vocab is not None: print(f"  Vocab size: {vocab}")
    except Exception:
        pass

    if args.grid_search:
        results = grid_search(
            model=model,
            tokenizer=tokenizer,
            prefill_length=args.prefill_length,
            batch_sizes=[4, 8, 16, 32],
            verify_tokens_list=[4, 8, 12],
            num_warmup=args.warmup,
            num_iterations=args.iterations,
            max_kv_size=args.max_kv_size,
        )
        print_summary(results)
        return results

    # Single-run pathway
    total_len = args.prefill_length + args.verify_tokens
    one_d_tokens = mx.array(
        _to_fixed_length_tokens(tokenizer, total_len),
        dtype=mx.int32,
    )
    batched = _as_batched(one_d_tokens, args.batch_size)
    prefill_ids = batched[:, :args.prefill_length]
    verify_ids = batched[:, args.prefill_length: total_len]

    kv_size = args.max_kv_size
    if kv_size is None:
        kv_size = args.prefill_length + args.verify_tokens + 8  # small headroom

    results = benchmark_verify_step(
        model=model,
        prefill_ids=prefill_ids,
        verify_ids=verify_ids,
        batch_size=args.batch_size,
        num_warmup=args.warmup,
        num_iterations=args.iterations,
        max_kv_size=kv_size,
    )

    # Print results
    print("\n" + "=" * 70)
    print("Benchmark Results")
    print("=" * 70)
    print(f"Batch Size: {results['batch_size']}")
    print(f"Prefill Length: {results['prefill_length']} tokens")
    print(f"Verify Tokens: {results['verify_tokens']}")
    print(f"Total Tokens Processed: {results['total_tokens']}")

    print(f"\nLatency Statistics (ms):")
    print(f"  Mean:   {results['mean_latency_ms']:.2f}")
    print(f"  Std:    {results['std_latency_ms']:.2f}")
    print(f"  Min:    {results['min_latency_ms']:.2f}")
    print(f"  Max:    {results['max_latency_ms']:.2f}")
    print(f"  P50:    {results['p50_latency_ms']:.2f}")
    print(f"  P90:    {results['p90_latency_ms']:.2f}")
    print(f"  P99:    {results['p99_latency_ms']:.2f}")

    print(f"\nThroughput: {results['tokens_per_second']:.2f} tokens/second")
    print(f"Time per token: {results['time_per_token_ms']:.3f} ms")

    # MLX doesn't expose live per-process memory like torch.cuda.*; print device info instead
    try:
        info = mx.metal.device_info()
        print("\nMetal Device Info:")
        for k, v in info.items():
            print(f"  {k}: {v}")
    except Exception:
        pass

    print("\nBenchmark complete!")
    return results


if __name__ == "__main__":
    _ = main()

