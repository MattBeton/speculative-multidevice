#!/usr/bin/env python3
"""
Benchmark Llama-3.2-3B-Instruct verify step for speculative decoding.
This script measures the time to do a forward pass of k tokens after n have been prefilled.
"""

import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import argparse
from typing import Dict, List, Tuple

def benchmark_verify_step(
    model,
    input_ids: torch.Tensor,
    prefill_length: int,
    verify_tokens: int,
    batch_size: int,
    num_warmup: int = 3,
    num_iterations: int = 10
) -> Dict:
    """
    Benchmark the verify step: forward pass of k tokens after n prefilled tokens.

    Args:
        model: The model to benchmark
        input_ids: Input tensor of token IDs (shape: [1, total_length])
        prefill_length: Number of tokens to prefill (n)
        verify_tokens: Number of tokens to verify (k)
        batch_size: Batch size for parallel verification
        num_warmup: Number of warmup iterations
        num_iterations: Number of benchmark iterations

    Returns:
        Dictionary with benchmark results
    """
    device = next(model.parameters()).device

    # Prepare batch input - replicate input for batch size
    batched_input_ids = input_ids.repeat(batch_size, 1).to(device)

    # Split into prefill and verify portions
    prefill_ids = batched_input_ids[:, :prefill_length]
    verify_ids = batched_input_ids[:, prefill_length:prefill_length + verify_tokens]

    # Create attention masks
    prefill_mask = torch.ones_like(prefill_ids)

    print(f"\nRunning verify benchmark:")
    print(f"  Device: {device}")
    print(f"  Batch size: {batch_size}")
    print(f"  Prefill length: {prefill_length} tokens")
    print(f"  Verify tokens: {verify_tokens}")
    print(f"  Input shapes - Prefill: {prefill_ids.shape}, Verify: {verify_ids.shape}")

    # Step 1: Prefill to get KV cache
    print(f"\n  Performing initial prefill...")
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
            prefill_outputs = model(
                input_ids=prefill_ids,
                attention_mask=prefill_mask,
                use_cache=True,
                return_dict=True
            )

    past_key_values = prefill_outputs.past_key_values
    torch.cuda.synchronize() if torch.cuda.is_available() else None

    # Prepare full attention mask for verify step
    full_attention_mask = torch.ones((batch_size, prefill_length + verify_tokens), device=device)

    # Warmup for verify step
    print(f"  Warming up verify step with {num_warmup} iterations...")
    for _ in range(num_warmup):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                _ = model(
                    input_ids=verify_ids,
                    attention_mask=full_attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True
                )
        torch.cuda.synchronize() if torch.cuda.is_available() else None

    # Benchmark verify step
    print(f"  Running {num_iterations} benchmark iterations for verify step...")
    latencies = []

    for i in range(num_iterations):
        # Clear any pending operations
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                verify_outputs = model(
                    input_ids=verify_ids,
                    attention_mask=full_attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True
                )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.perf_counter()

        latency = (end_time - start_time) * 1000  # Convert to milliseconds
        latencies.append(latency)

        # Verify output shape on first iteration
        if i == 0:
            logits_shape = verify_outputs.logits.shape
            print(f"  Verify output logits shape: {logits_shape}")

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
        model: The model to benchmark
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
    initial_tokens = tokenizer.encode(base_text, return_tensors="pt", add_special_tokens=False)

    # Ensure we have exactly total_length tokens
    if initial_tokens.shape[1] >= total_length:
        tokens = initial_tokens[:, :total_length]
    else:
        # Repeat tokens to fill
        tokens_needed = total_length
        tokens_list = []
        while tokens_needed > 0:
            chunk_size = min(tokens_needed, initial_tokens.shape[1])
            tokens_list.append(initial_tokens[:, :chunk_size])
            tokens_needed -= chunk_size
        tokens = torch.cat(tokens_list, dim=1)

    print(f"\nCreated input tensor with shape: {tokens.shape}")

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
                    input_ids=tokens[:, :prefill_length + verify_tokens],
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
    parser = argparse.ArgumentParser(description='Benchmark Llama verify step for speculative decoding')
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
    print("Llama-3.2-3B-Instruct Verify Step Benchmark")
    print("="*70)

    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("WARNING: CUDA not available, running on CPU")

    # Load tokenizer
    print(f"\nLoading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load model
    print(f"\nLoading model {model_name}...")
    print("This may take a while as the model downloads...")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Use FP16 for efficiency
        device_map="auto",  # Automatically place on GPU
        use_cache=True  # Enable KV cache (required for verify step)
    )

    # Ensure model is in eval mode
    model.eval()

    print(f"Model loaded successfully!")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    print(f"Model device: {next(model.parameters()).device}")

    # Get model configuration details
    print(f"\nModel Configuration:")
    print(f"  Hidden size: {model.config.hidden_size}")
    print(f"  Num layers: {model.config.num_hidden_layers}")
    print(f"  Num attention heads: {model.config.num_attention_heads}")
    if hasattr(model.config, 'num_key_value_heads'):
        print(f"  Num KV heads: {model.config.num_key_value_heads}")
    print(f"  Vocab size: {model.config.vocab_size}")

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
        initial_tokens = tokenizer.encode(base_text, return_tensors="pt", add_special_tokens=False)

        if initial_tokens.shape[1] >= total_length:
            tokens = initial_tokens[:, :total_length]
        else:
            tokens_needed = total_length
            tokens_list = []
            while tokens_needed > 0:
                chunk_size = min(tokens_needed, initial_tokens.shape[1])
                tokens_list.append(initial_tokens[:, :chunk_size])
                tokens_needed -= chunk_size
            tokens = torch.cat(tokens_list, dim=1)

        results = benchmark_verify_step(
            model=model,
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
        print(f"Time per token:{results['time_per_token_ms']:.3f} ms")

    # Memory usage
    if torch.cuda.is_available():
        allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        reserved_mb = torch.cuda.memory_reserved() / (1024 * 1024)
        print(f"\nGPU Memory Usage:")
        print(f"  Allocated: {allocated_mb:.2f} MB")
        print(f"  Reserved:  {reserved_mb:.2f} MB")

    print("\nBenchmark complete!")

    return results


if __name__ == "__main__":
    results = main()
