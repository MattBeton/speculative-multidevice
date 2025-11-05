#!/usr/bin/env python3
"""
bench_roundtrip.py

Measures roundtrip latency between client and server by sending ResetRequest
and measuring the time until ResetResponse is received.
"""

import argparse
import asyncio
import statistics
import time

from shared import (
    MessageChannel,
    ResetRequest,
    ResetResponse,
)


async def measure_roundtrip(
    host: str,
    port: int,
    iterations: int,
    warmup: int,
) -> None:
    """Measure roundtrip latency by sending ResetRequest and measuring ResetResponse time."""
    reader, writer = await asyncio.open_connection(host, port)
    channel = MessageChannel(reader, writer)

    latencies = []
    
    print(f"Measuring roundtrip latency ({iterations} iterations, {warmup} warmup)...")
    
    for i in range(iterations + warmup):
        # Send ResetRequest and measure roundtrip time
        t0 = time.perf_counter()
        await channel.send(ResetRequest())
        resp = await channel.recv()
        t1 = time.perf_counter()
        
        if not isinstance(resp, ResetResponse):
            raise RuntimeError(f"Expected ResetResponse, got {type(resp)!r}")
        
        latency_ms = (t1 - t0) * 1000
        latencies.append(latency_ms)
        
        if i < warmup:
            print(f"  Warmup {i+1}/{warmup}: {latency_ms:.2f} ms")
        else:
            iter_num = i - warmup + 1
            print(f"  Iteration {iter_num}/{iterations}: {latency_ms:.2f} ms")

    await channel.close()

    # Calculate statistics (excluding warmup)
    measured_latencies = latencies[warmup:]
    
    if not measured_latencies:
        print("No measurements collected!")
        return

    mean_latency = statistics.mean(measured_latencies)
    median_latency = statistics.median(measured_latencies)
    min_latency = min(measured_latencies)
    max_latency = max(measured_latencies)
    
    if len(measured_latencies) > 1:
        stdev_latency = statistics.stdev(measured_latencies)
        p50 = statistics.median(measured_latencies)
        p95 = sorted(measured_latencies)[int(len(measured_latencies) * 0.95)]
        p99 = sorted(measured_latencies)[int(len(measured_latencies) * 0.99)]
    else:
        stdev_latency = 0.0
        p50 = median_latency
        p95 = max_latency
        p99 = max_latency

    print("\n==== Roundtrip Latency Results ====")
    print(f"Server            : {host}:{port}")
    print(f"Iterations        : {iterations}")
    print(f"Warmup            : {warmup}")
    print(f"\nLatency (ms):")
    print(f"  Mean            : {mean_latency:.2f}")
    print(f"  Median (p50)    : {median_latency:.2f}")
    print(f"  Min             : {min_latency:.2f}")
    print(f"  Max             : {max_latency:.2f}")
    print(f"  Std dev         : {stdev_latency:.2f}")
    print(f"  p95             : {p95:.2f}")
    print(f"  p99             : {p99:.2f}")


def main():
    ap = argparse.ArgumentParser(
        description="Benchmark roundtrip latency between client and server"
    )
    ap.add_argument("--host", default="100.116.69.48", help="Server host")
    ap.add_argument("--port", type=int, default=7070, help="Server port")
    ap.add_argument(
        "-i", "--iterations", type=int, default=100,
        help="Number of measurement iterations (default: 100)"
    )
    ap.add_argument(
        "-w", "--warmup", type=int, default=10,
        help="Number of warmup iterations (default: 10)"
    )
    args = ap.parse_args()

    asyncio.run(
        measure_roundtrip(
            host=args.host,
            port=args.port,
            iterations=args.iterations,
            warmup=args.warmup,
        )
    )


if __name__ == "__main__":
    main()

