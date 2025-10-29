#!/usr/bin/env python3
"""
bench_server_tps.py (disjoint prompts)
- Prefill: one 8x batch with N=200 tokens that are pairwise prefix-disjoint.
- Verify: 5 rounds; each round sends K=8 draft tokens per stream (64 positions/round).
- Reports TPS = total verified positions / total wall time over verify rounds.

Works with the MessageChannel batch API used by vllm_speculative_server.py.
"""

import argparse
import asyncio
import time
from typing import List, Tuple

import numpy as np
from transformers import AutoTokenizer  # only to read vocab size / special ids

from shared import (
    MessageChannel,
    PrefillBatchRequest,
    PrefillBatchResponse,
    VerifyBatchRequest,
    VerifyBatchResponse,
    VerifyItem,
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

        # tail: random (repeats allowed). Doesn’t affect prefix disjointness.
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
    header_len: int,
) -> None:
    tok = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=True)

    prompts = _build_disjoint_prompts(tok, batch_size, prefill_len, header_len=header_len)
    vtok_per_stream = _per_stream_verify_tokens(tok, batch_size)

    reader, writer = await asyncio.open_connection(host, port)
    channel = MessageChannel(reader, writer)

    # Prefill batch (sets server state; some servers may not compute here)
    items = [{"stream_id": f"s{i}", "prompt": p} for i, p in enumerate(prompts)]
    t0 = time.perf_counter()
    await channel.send(PrefillBatchRequest(items=items))
    resp = await channel.recv()
    t_prefill = time.perf_counter() - t0
    if not isinstance(resp, PrefillBatchResponse):
        raise RuntimeError(f"Expected PrefillBatchResponse, got {type(resp)!r}")

    # Verify rounds: each sends K tokens for each of B streams
    latencies = []
    for r in range(rounds):
        verify_items: List[VerifyItem] = []
        for i in range(batch_size):
            vt = vtok_per_stream[i]
            verify_items.append(
                VerifyItem(
                    stream_id=f"s{i}",
                    draft_toks=[vt] * k_verify,    # K positions per stream
                    # We leave topk rows empty; verifier will still teacher-force K rows.
                    draft_topk_idx=[],
                    draft_topk_vals=[],
                )
            )
        t1 = time.perf_counter()
        await channel.send(VerifyBatchRequest(items=verify_items))
        out = await channel.recv()
        dt = time.perf_counter() - t1
        if not isinstance(out, VerifyBatchResponse):
            raise RuntimeError(f"Expected VerifyBatchResponse, got {type(out)!r}")
        latencies.append(dt)

    await channel.close()

    total_positions = batch_size * k_verify * rounds
    total_verify_time = sum(latencies)
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
    ap.add_argument("-n", "--prefill-len", type=int, default=200)
    ap.add_argument("-k", "--k-verify", type=int, default=8)
    ap.add_argument("-r", "--rounds", type=int, default=5)
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
            header_len=args.header_len,
        )
    )


if __name__ == "__main__":
    main()




# #!/usr/bin/env python3
# """
# bench_server_tps.py
# - Prefill: one 8x batch of ~200 tokens.
# - Verify: 5 rounds; each round sends 8 draft tokens per stream (64 positions total).
# - Reports TPS = total verified positions / total wall time over verify rounds.
#
# Works with the MessageChannel batch API used by vllm_speculative_server.py.
# """
#
# import argparse
# import asyncio
# import math
# import time
# from typing import List
#
# from transformers import AutoTokenizer  # local-only; model weights not loaded
#
# from shared import (
#     MessageChannel,
#     PrefillBatchRequest,
#     PrefillBatchResponse,
#     VerifyBatchRequest,
#     VerifyBatchResponse,
#     VerifyItem,
# )
#
#
# def _pad_or_repeat(ids: List[int], target_len: int, pad_id: int) -> List[int]:
#     if not ids:
#         return [pad_id] * target_len
#     if len(ids) >= target_len:
#         return ids[:target_len]
#     reps = math.ceil(target_len / len(ids))
#     out = (ids * reps)[:target_len]
#     return out
#
#
# def _make_prompts(tok, batch_size: int, target_len: int) -> List[List[int]]:
#     """
#     Build B prompts of exactly target_len tokens using the *same tokenizer*
#     as the server's base model. We use the chat template if present.
#     We also make each stream slightly different to avoid accidental de-dupe.
#     """
#     base_text = "Why is the sky blue?"
#     try:
#         ids = tok.apply_chat_template(
#             [{"role": "user", "content": base_text}],
#             add_generation_prompt=True,
#             tokenize=True,
#         )
#     except Exception:
#         ids = tok.encode(base_text, add_special_tokens=True)
#
#     pad_id = getattr(tok, "pad_token_id", None)
#     if pad_id is None:
#         # Reasonable fallback; Llama models often have no pad -> use EOS or 1
#         pad_id = getattr(tok, "eos_token_id", 1) or 1
#
#     base = _pad_or_repeat([int(x) for x in ids], target_len, pad_id=pad_id)
#
#     # Make each stream unique by tweaking the last token (keeps length == target_len)
#     per_stream = []
#     for i in range(batch_size):
#         tweak = tok.encode(chr(ord('a') + (i % 26)), add_special_tokens=False) or [base[-1]]
#         curr = list(base)
#         curr[-1] = int(tweak[0])
#         per_stream.append(curr)
#     return per_stream
#
#
# async def run_bench(
#     host: str,
#     port: int,
#     model_id: str,
#     batch_size: int,
#     prefill_len: int,
#     k_verify: int,
#     rounds: int,
# ) -> None:
#     # 1) Tokenizer must match the server's base model
#     tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
#
#     # Draft token to repeat in verify rows (any valid token is fine for timing)
#     try:
#         sp = tok.encode(" ", add_special_tokens=False)
#     except Exception:
#         sp = []
#     vtok = int(sp[0]) if sp else (getattr(tok, "eos_token_id", 1) or 1)
#
#     # Build B prompts of exact length N
#     prompts = _make_prompts(tok, batch_size, prefill_len)
#
#     # 2) Connect to server
#     reader, writer = await asyncio.open_connection(host, port)
#     channel = MessageChannel(reader, writer)
#
#     # 3) Prefill all streams (this is just to set server state)
#     items = [{"stream_id": f"s{i}", "prompt": p} for i, p in enumerate(prompts)]
#     t0 = time.perf_counter()
#     await channel.send(PrefillBatchRequest(items=items))
#     resp = await channel.recv()
#     t_prefill = time.perf_counter() - t0
#
#     if not isinstance(resp, PrefillBatchResponse):
#         raise RuntimeError(f"Expected PrefillBatchResponse, got {type(resp)!r}")
#
#     # 4) Verify rounds: each round sends K draft tokens for each of B streams
#     latencies = []
#     total_positions = batch_size * k_verify * rounds  # what we time
#
#     for r in range(rounds):
#         verify_items: List[VerifyItem] = []
#         for i in range(batch_size):
#             verify_items.append(
#                 VerifyItem(
#                     stream_id=f"s{i}",
#                     draft_toks=[vtok] * k_verify,
#                     # Leave top-k rows empty; the server will still do all teacher-forced K positions,
#                     # which is exactly the forward cost we want to time.
#                     draft_topk_idx=[],
#                     draft_topk_vals=[],
#                 )
#             )
#
#         t1 = time.perf_counter()
#         await channel.send(VerifyBatchRequest(items=verify_items))
#         out = await channel.recv()
#         dt = time.perf_counter() - t1
#
#         if not isinstance(out, VerifyBatchResponse):
#             raise RuntimeError(f"Expected VerifyBatchResponse, got {type(out)!r}")
#
#         latencies.append(dt)
#
#     await channel.close()
#
#     total_verify_time = sum(latencies)
#     tps = total_positions / total_verify_time if total_verify_time > 0 else float("nan")
#
#     # 5) Report
#     print("\n==== bench_server_tps ====")
#     print(f"Server       : {host}:{port}")
#     print(f"Tokenizer    : {model_id}")
#     print(f"B={batch_size}  N={prefill_len}  K={k_verify}  rounds={rounds}")
#     print(f"Prefill (batch): {t_prefill*1000:.1f} ms  (server may not compute here)")
#     print(f"Verify rounds latencies (ms): {[round(x*1000, 1) for x in latencies]}")
#     print(f"\nTOTAL verified positions: {total_positions}")
#     print(f"TOTAL verify time       : {total_verify_time:.3f} s")
#     print(f"TPS (positions / second): {tps:.1f}")
#
#
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--host", default="192.168.200.2")
#     ap.add_argument("--port", type=int, default=7070)
#     # Must match the base model used by the server so token IDs line up:
#     ap.add_argument("--tokenizer", default="meta-llama/Llama-3.2-3B-Instruct")
#     ap.add_argument("-b", "--batch", type=int, default=8)
#     ap.add_argument("-n", "--prefill-len", type=int, default=200)
#     ap.add_argument("-k", "--k-verify", type=int, default=8)
#     ap.add_argument("-r", "--rounds", type=int, default=5)
#     args = ap.parse_args()
#
#     asyncio.run(
#         run_bench(
#             host=args.host,
#             port=args.port,
#             model_id=args.tokenizer,
#             batch_size=args.batch,
#             prefill_len=args.prefill_len,
#             k_verify=args.k_verify,
#             rounds=args.rounds,
#         )
#     )
#
#
# if __name__ == "__main__":
#     main()
# #
# #
# #
# #
# # import asyncio
# # from pathlib import Path
# # from typing import Dict, List, Optional, Tuple
# #
# # import numpy as np
# #
# # from model import MLXGenerationModel
# # from timing import TokenTimer
# # from shared import (
# #     MessageChannel,
# #     PrefillBatchRequest,
# #     PrefillBatchResponse,
# #     VerifyBatchRequest,
# #     VerifyBatchResponse,
# #     VerifyItem,
# #     VerifyResponseItem,
# #     run_mlx,
# # )
# #
# # async def main() -> None:
# #     # Connect to server
# #     # IMPORTANT: change IP/port to the verifier host if remote
# #     ip = '192.168.200.2'
# #     reader, writer = await asyncio.open_connection(ip, 7070)
# #     channel = MessageChannel(reader, writer)
# #
# #     timer = TokenTimer()
# #
# #     await self._channel.send(PrefillBatchRequest(items=items))
# #     msg = await self._channel.recv()
# #
# #     print(msg)
# #
# #     with timer.measure('decode', 8):
# #         verify_items: list[VerifyItem] = []
# #
# #
# #
# #
# #
# # if __name__ == "__main__":
# #     asyncio.run(main())
