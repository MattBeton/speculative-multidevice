#!/usr/bin/env python3
"""
Benchmark Llama-3.2-3B-Instruct verify step for speculative decoding (MLX).
This script measures the time to do a forward pass of k tokens after n have been prefilled.
"""

import time
import numpy as np
import argparse
from typing import Dict, List

import mlx.core as mx
from mlx_lm.utils import load_model
from mlx_lm.tokenizer_utils import load_tokenizer
from mlx_lm.models.cache import make_prompt_cache


def benchmark_verify_step(
    model,
    tokenizer,
    input_ids: np.ndarray,
    prefill_length: int,
    verify_tokens: int,
    batch_size: int,
    num_warmup: int = 3,
    num_iterations: int = 10
) -> Dict:
    """
    Benchmark the verify step: forward pass of k tokens after n prefilled tokens.

    Args:
        model: The MLX model to benchmark
        tokenizer: MLX tokenizer
        input_ids: Input array of token IDs (shape: [total_length])
        prefill_length: Number of tokens to prefill (n)
        verify_tokens: Number of tokens to verify (k)
        batch_size: Batch size for parallel verification
        num_warmup: Number of warmup iterations
        num_iterations: Number of benchmark iterations

    Returns:
        Dictionary with benchmark results
    """
    # Prepare batch input - replicate input for batch size
    batched_input_ids = np.tile(input_ids[:prefill_length + verify_tokens][None, :], (batch_size, 1))

    # Split into prefill and verify portions
    prefill_ids = batched_input_ids[:, :prefill_length]
    verify_ids = batched_input_ids[:, prefill_length:prefill_length + verify_tokens]

    print(f"\nRunning verify benchmark:")
    print(f"  Backend: MLX")
    print(f"  Batch size: {batch_size}")
    print(f"  Prefill length: {prefill_length} tokens")
    print(f"  Verify tokens: {verify_tokens}")
    print(f"  Input shapes - Prefill: {prefill_ids.shape}, Verify: {verify_ids.shape}")

    # Step 1: Prefill to get KV cache
    print(f"\n  Performing initial prefill...")
    cache = make_prompt_cache(model)  # Reset cache
    prefill_mx = mx.array(prefill_ids, dtype=mx.int32)
    
    t0 = time.perf_counter()
    _ = model(prefill_mx, cache=cache)
    mx.eval()
    t_prefill = time.perf_counter() - t0

    # Warmup for verify step
    print(f"  Warming up verify step with {num_warmup} iterations...")
    verify_mx = mx.array(verify_ids, dtype=mx.int32)
    
    for _ in range(num_warmup):
        _ = model(verify_mx, cache=cache)
        mx.eval()
        # Keep effective KV length ≈ N by trimming K tokens
        for c in cache:
            c.trim(verify_tokens)

    # Benchmark verify step
    print(f"  Running {num_iterations} benchmark iterations for verify step...")
    latencies = []

    for i in range(num_iterations):
        start_time = time.perf_counter()

        verify_outputs = model(verify_mx, cache=cache)
        mx.eval()

        end_time = time.perf_counter()

        latency = (end_time - start_time) * 1000  # Convert to milliseconds
        latencies.append(latency)

        # Verify output shape on first iteration
        if i == 0:
            logits_shape = verify_outputs.shape
            print(f"  Verify output logits shape: {logits_shape}")

        # Keep effective KV length ≈ N by trimming K tokens
        for c in cache:
            c.trim(verify_tokens)

    # Calculate statistics
    latencies = np.array(latencies)
    mean_latency = np.mean(latencies)
    std_latency = np.std(latencies)

    # Calculate throughput - total tokens processed per second across all batches
    total_tokens = batch_size * verify_tokens
    tokens_per_second = (total_tokens / mean_latency) * 1000  # latency is in ms

    results = {
        'batch_size': batch_size,
        'prefill_length': prefill_length,
        'verify_tokens': verify_tokens,
        'total_tokens': total_tokens,
        'mean_latency_ms': mean_latency,
        'std_latency_ms': std_latency,
        'min_latency_ms': np.min(latencies),
        'max_latency_ms': np.max(latencies),
        'p50_latency_ms': np.percentile(latencies, 50),
        'p90_latency_ms': np.percentile(latencies, 90),
        'p99_latency_ms': np.percentile(latencies, 99),
        'tokens_per_second': tokens_per_second,
        'time_per_token_ms': mean_latency / total_tokens,
        'all_latencies': latencies.tolist()
    }

    return results


def grid_search(
    model,
    tokenizer,
    prefill_length: int = 1024,
    batch_sizes: List[int] = [4, 8, 16, 32],
    verify_tokens_list: List[int] = [4, 8, 12],
    num_warmup: int = 3,
    num_iterations: int = 10
) -> List[Dict]:
    """
    Perform grid search over batch sizes and verify token counts.

    Args:
        model: The MLX model to benchmark
        tokenizer: Tokenizer for creating input
        prefill_length: Number of tokens to prefill
        batch_sizes: List of batch sizes to test
        verify_tokens_list: List of verify token counts to test
        num_warmup: Number of warmup iterations
        num_iterations: Number of benchmark iterations per configuration

    Returns:
        List of result dictionaries for each configuration
    """
    # Create input tokens (prefill + max verify tokens)
    max_verify = max(verify_tokens_list)
    total_length = prefill_length + max_verify

    # Create a text with enough tokens
    base_text = "The quick brown fox jumps over the lazy dog. " * 300
    try:
        initial_tokens = tokenizer.encode(base_text, add_special_tokens=False)
    except Exception:
        initial_tokens = tokenizer.encode(" ", add_special_tokens=False) or [1]

    # Ensure we have exactly total_length tokens
    if len(initial_tokens) >= total_length:
        tokens = np.array(initial_tokens[:total_length], dtype=np.int32)
    else:
        # Repeat tokens to fill
        tokens_needed = total_length
        tokens_list = []
        while tokens_needed > 0:
            chunk_size = min(tokens_needed, len(initial_tokens))
            tokens_list.extend(initial_tokens[:chunk_size])
            tokens_needed -= chunk_size
        tokens = np.array(tokens_list, dtype=np.int32)

    print(f"\nCreated input array with shape: {tokens.shape}")

    results = []

    print("\n" + "="*70)
    print("Starting Grid Search")
    print("="*70)
    print(f"Batch sizes: {batch_sizes}")
    print(f"Verify token counts: {verify_tokens_list}")
    print(f"Prefill length: {prefill_length}")
    print(f"Total configurations: {len(batch_sizes) * len(verify_tokens_list)}")

    for batch_size in batch_sizes:
        for verify_tokens in verify_tokens_list:
            print("\n" + "-"*70)
            print(f"Configuration: batch_size={batch_size}, verify_tokens={verify_tokens}")
            print("-"*70)

            try:
                result = benchmark_verify_step(
                    model=model,
                    tokenizer=tokenizer,
                    input_ids=tokens[:prefill_length + verify_tokens],
                    prefill_length=prefill_length,
                    verify_tokens=verify_tokens,
                    batch_size=batch_size,
                    num_warmup=num_warmup,
                    num_iterations=num_iterations
                )
                results.append(result)

                # Print immediate results
                print(f"\n  Results:")
                print(f"    Mean latency: {result['mean_latency_ms']:.2f} ms")
                print(f"    Throughput: {result['tokens_per_second']:.2f} tokens/second")
                print(f"    Time per token: {result['time_per_token_ms']:.3f} ms")

            except Exception as e:
                print(f"  ERROR: Failed with {e}")
                import traceback
                traceback.print_exc()
                # Add failed result
                results.append({
                    'batch_size': batch_size,
                    'verify_tokens': verify_tokens,
                    'error': str(e),
                    'tokens_per_second': 0
                })

    return results


def print_summary(results: List[Dict]):
    """Print a summary table of all results."""
    print("\n" + "="*70)
    print("GRID SEARCH SUMMARY")
    print("="*70)

    # Print header
    print(f"\n{'Batch Size':<12} {'Verify Tokens':<15} {'Latency (ms)':<15} {'Throughput (tok/s)':<20} {'Time/Token (ms)':<15}")
    print("-"*92)

    # Sort results by throughput for easier comparison
    sorted_results = sorted(results, key=lambda x: x.get('tokens_per_second', 0), reverse=True)

    for result in sorted_results:
        if 'error' in result:
            print(f"{result['batch_size']:<12} {result.get('verify_tokens', 'N/A'):<15} {'ERROR':<15} {'ERROR':<20} {'ERROR':<15}")
        else:
            print(f"{result['batch_size']:<12} {result['verify_tokens']:<15} "
                  f"{result['mean_latency_ms']:<15.2f} {result['tokens_per_second']:<20.2f} "
                  f"{result['time_per_token_ms']:<15.3f}")

    # Find best configuration
    best_result = max(sorted_results, key=lambda x: x.get('tokens_per_second', 0))
    if best_result.get('tokens_per_second', 0) > 0:
        print(f"\nBest configuration:")
        print(f"  Batch size: {best_result['batch_size']}")
        print(f"  Verify tokens: {best_result['verify_tokens']}")
        print(f"  Throughput: {best_result['tokens_per_second']:.2f} tokens/second")


def main():
    parser = argparse.ArgumentParser(description='Benchmark Llama verify step for speculative decoding (MLX)')
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

    args = parser.parse_args()

    # Model configuration
    model_name = "meta-llama/Llama-3.2-3B-Instruct"

    print("="*70)
    print("Llama-3.2-3B-Instruct Verify Step Benchmark (MLX)")
    print("="*70)

    # Load tokenizer
    print(f"\nLoading tokenizer from {model_name}...")
    tokenizer = load_tokenizer(model_name)

    # Load model
    print(f"\nLoading model {model_name}...")
    print("This may take a while as the model downloads...")
    model, config = load_model(model_name)

    print(f"Model loaded successfully!")

    # Get model configuration details
    print(f"\nModel Configuration:")
    if hasattr(config, 'hidden_size'):
        print(f"  Hidden size: {config.hidden_size}")
    if hasattr(config, 'num_hidden_layers'):
        print(f"  Num layers: {config.num_hidden_layers}")
    if hasattr(config, 'num_attention_heads'):
        print(f"  Num attention heads: {config.num_attention_heads}")
    if hasattr(config, 'num_key_value_heads'):
        print(f"  Num KV heads: {config.num_key_value_heads}")
    if hasattr(config, 'vocab_size'):
        print(f"  Vocab size: {config.vocab_size}")

    if args.grid_search:
        # Run grid search
        batch_sizes = [4, 8, 16, 32]
        verify_tokens_list = [4, 8, 12]

        results = grid_search(
            model=model,
            tokenizer=tokenizer,
            prefill_length=args.prefill_length,
            batch_sizes=batch_sizes,
            verify_tokens_list=verify_tokens_list,
            num_warmup=args.warmup,
            num_iterations=args.iterations
        )

        # Print summary
        print_summary(results)

    else:
        # Run single benchmark with specified parameters
        # Create input tokens
        total_length = args.prefill_length + args.verify_tokens
        base_text = "The quick brown fox jumps over the lazy dog. " * 300
        try:
            initial_tokens = tokenizer.encode(base_text, add_special_tokens=False)
        except Exception:
            initial_tokens = tokenizer.encode(" ", add_special_tokens=False) or [1]

        if len(initial_tokens) >= total_length:
            tokens = np.array(initial_tokens[:total_length], dtype=np.int32)
        else:
            tokens_needed = total_length
            tokens_list = []
            while tokens_needed > 0:
                chunk_size = min(tokens_needed, len(initial_tokens))
                tokens_list.extend(initial_tokens[:chunk_size])
                tokens_needed -= chunk_size
            tokens = np.array(tokens_list, dtype=np.int32)

        results = benchmark_verify_step(
            model=model,
            tokenizer=tokenizer,
            input_ids=tokens,
            prefill_length=args.prefill_length,
            verify_tokens=args.verify_tokens,
            batch_size=args.batch_size,
            num_warmup=args.warmup,
            num_iterations=args.iterations
        )

        # Print results
        print("\n" + "="*70)
        print("Benchmark Results")
        print("="*70)
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

    print("\nBenchmark complete!")

    return results


if __name__ == "__main__":
    results = main()

