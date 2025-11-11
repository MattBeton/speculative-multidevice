#!/usr/bin/env python3
"""
bench_server_tps.py (disjoint prompts)

- Prefill: one 8x batch with N=200 tokens that are pairwise prefix-disjoint.

- Verify: 5 rounds; each round sends K=8 draft tokens per stream (64 positions/round).

- Reports TPS = total verified positions / total wall time over verify rounds.

Works with the MessageChannel API defined in shared.py (PrefillRequest/VerifyRequest).
"""

import argparse
import asyncio
import time
from typing import List

import numpy as np
from transformers import AutoTokenizer  # only to read vocab size / special ids

from shared import (
    MessageChannel,
    PrefillRequest,
    PrefillResponse,
    VerifyRequest,
    VerifyResponse,
)

# ----------------- Token utilities -----------------

def _reserved_ids(tok) -> set:
    """Collect special/reserved token ids to avoid (bos/eos/pad/unk + all_special_ids)."""
    res = set()
    for name in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
        tid = getattr(tok, name, None)
        if tid is not None:
            try:
                res.add(int(tid))
            except Exception:
                pass
    try:
        for tid in (tok.all_special_ids or []):
            res.add(int(tid))
    except Exception:
        pass
    return res


def _allowed_pool(tok, low_floor: int = 32) -> List[int]:
    """Return a pool of 'normal' token ids to choose from (avoid control/specials)."""
    try:
        vocab_size = int(tok.vocab_size)
    except Exception:
        vocab_size = 32000
    res = _reserved_ids(tok)
    pool = [t for t in range(low_floor, vocab_size) if t not in res]
    if len(pool) < 1024:
        # Extremely defensive fallback; still ensure disjoint first tokens.
        pool = [t for t in range(1, vocab_size) if t not in res]
    return pool


def _build_disjoint_prompts(
    tok,
    batch_size: int,
    n_tokens: int,
    header_len: int = 8,
    seed: int = 12345,
) -> List[List[int]]:
    """
    Create B prompts (each length N) such that no two prompts share any prefix.
    Guarantee: the first token of every stream is distinct ⇒ pairwise LCP = 0.
    We also diversify the next (header_len-1) tokens per stream for extra safety.
    """
    assert header_len > 0 and header_len <= n_tokens
    pool = _allowed_pool(tok, low_floor=32)
    rng_global = np.random.default_rng(seed)

    # Pick B * header_len distinct header tokens with wide spacing
    # (wrap if pool smaller; uniqueness of position 0 is what enforces LCP=0).
    prompts: List[List[int]] = []
    for i in range(batch_size):
        # header[0] is unique per stream (this alone enforces LCP=0)
        h0 = pool[i % len(pool)]
        # next header_len-1 tokens are stream-distinct too (belt-and-suspenders)
        header = [h0]
        cursor = (i * 9973) % len(pool)
        for j in range(1, header_len):
            cursor = (cursor + 7919) % len(pool)
            header.append(pool[(cursor + j * 37) % len(pool)])

        # tail: random (repeats allowed). Doesn't affect prefix disjointness.
        tail_len = n_tokens - header_len
        tail = rng_global.choice(pool, size=tail_len, replace=True).astype(int).tolist()
        prompts.append(header + tail)

    # Sanity: compute max LCP across pairs (should be 0)
    def _lcp_len(a: List[int], b: List[int]) -> int:
        L = min(len(a), len(b))
        for t in range(L):
            if a[t] != b[t]:
                return t
        return L

    max_lcp = 0
    for x in range(batch_size):
        for y in range(x + 1, batch_size):
            max_lcp = max(max_lcp, _lcp_len(prompts[x], prompts[y]))
    if max_lcp != 0:
        raise AssertionError(f"Prompts not disjoint: max common prefix = {max_lcp} tokens")
    return prompts


def _per_stream_verify_tokens(tok, batch_size: int) -> List[int]:
    """Give each stream its own verify token id to avoid suffix coincidences."""
    pool = _allowed_pool(tok, low_floor=32)
    return [pool[(113 * i + 7) % len(pool)] for i in range(batch_size)]


# ----------------- Bench -----------------

async def run_bench(
    host: str,
    port: int,
    tokenizer_id: str,
    batch_size: int,
    prefill_len: int,
    k_verify: int,
    rounds: int,
    warmup: int,
    header_len: int,
) -> None:
    tok = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=True)
    prompts = _build_disjoint_prompts(tok, batch_size, prefill_len, header_len=header_len)
    vtok_per_stream = _per_stream_verify_tokens(tok, batch_size)

    reader, writer = await asyncio.open_connection(host, port)
    channel = MessageChannel(reader, writer)

    # Prefill batch (sets server state; some servers may not compute here)
    # New API: PrefillRequest takes prompts: list[list[int]] (positional, no stream IDs)
    t0 = time.perf_counter()
    await channel.send(PrefillRequest(prompts=prompts))
    resp = await channel.recv()
    t_prefill = time.perf_counter() - t0
    if not isinstance(resp, PrefillResponse):
        raise RuntimeError(f"Expected PrefillResponse, got {type(resp)!r}")

    # Verify rounds: each sends K tokens for each of B streams
    # New API: VerifyRequest takes positional lists (no stream IDs)
    latencies = []
    # Use a reasonable top-k size (server typically uses 20)
    TOP_K_SIZE = 20
    for r in range(rounds + warmup):
        # Build batch arrays: draft_toks[i] is the draft tokens for stream i
        draft_toks_batch: List[List[int]] = []
        draft_topk_idx_batch: List[List[List[int]]] = []
        draft_topk_vals_batch: List[List[List[float]]] = []

        for i in range(batch_size):
            vt = vtok_per_stream[i]
            # K positions per stream
            draft_toks_batch.append([vt] * k_verify)
            # Create dummy topk data: for each of K positions, provide a topk list
            # where the draft token is included (at position 0) so verification can proceed
            stream_topk_idx: List[List[int]] = []
            stream_topk_vals: List[List[float]] = []
            for _ in range(k_verify):
                # Include the draft token itself as the first element
                topk_idx = [vt] + [vt + j + 1 for j in range(TOP_K_SIZE - 1)]
                topk_vals = [0.5] + [0.01 / (j + 1) for j in range(TOP_K_SIZE - 1)]
                stream_topk_idx.append(topk_idx)
                stream_topk_vals.append(topk_vals)
            draft_topk_idx_batch.append(stream_topk_idx)
            draft_topk_vals_batch.append(stream_topk_vals)

        t1 = time.perf_counter()
        await channel.send(VerifyRequest(
            draft_toks=draft_toks_batch,
            draft_topk_idx=draft_topk_idx_batch,
            draft_topk_vals=draft_topk_vals_batch,
        ))
        out = await channel.recv()
        dt = time.perf_counter() - t1
        if not isinstance(out, VerifyResponse):
            raise RuntimeError(f"Expected VerifyResponse, got {type(out)!r}")
        latencies.append(dt)
        
        # ensure post_verify can complete
        await asyncio.sleep(0.1)

    await channel.close()

    total_positions = batch_size * k_verify * rounds
    total_verify_time = sum(latencies[warmup:])
    tps = total_positions / total_verify_time if total_verify_time > 0 else float("nan")

    print("\n==== bench_server_tps (disjoint prompts) ====")
    print(f"Server              : {host}:{port}")
    print(f"Tokenizer           : {tokenizer_id}")
    print(f"B={batch_size}  N={prefill_len}  K={k_verify}  rounds={rounds}  header_len={header_len}")
    print(f"Prefill batch time  : {t_prefill*1000:.1f} ms  (server may not compute here)")
    print(f"Round latencies (ms): {[round(x*1000, 2) for x in latencies]}")
    print(f"TOTAL positions     : {total_positions}")
    print(f"TOTAL verify time   : {total_verify_time:.4f} s")
    print(f"TPS (pos/sec)       : {tps:.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="192.168.200.2")
    ap.add_argument("--port", type=int, default=7070)
    ap.add_argument("--tokenizer", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("-b", "--batch", type=int, default=8)
    ap.add_argument("-n", "--prefill-len", type=int, default=16)
    ap.add_argument("-k", "--k-verify", type=int, default=8)
    ap.add_argument("-r", "--rounds", type=int, default=5)
    ap.add_argument("-w", "--warmup", type=int, default=3)
    ap.add_argument("--header-len", type=int, default=8, help="unique leading tokens per stream")
    args = ap.parse_args()

    asyncio.run(
        run_bench(
            host=args.host,
            port=args.port,
            tokenizer_id=args.tokenizer,
            batch_size=args.batch,
            prefill_len=args.prefill_len,
            k_verify=args.k_verify,
            rounds=args.rounds,
            warmup=args.warmup,
            header_len=args.header_len,
        )
    )


if __name__ == "__main__":
    main()
