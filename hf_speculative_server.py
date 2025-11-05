# hf_speculative_server.py
# Hugging Face Transformers-based speculative decoding verifier (batched).
# - Single forward per K-group across streams using left-padded KV caches.
# - Commits by slicing returned KV and storing as DynamicCache (no extra forward).
# - Matches MLX logits-based acceptance math.
#
# Requirements:
#   pip install "transformers>=4.40" torch --upgrade
#   (plus a CUDA build of torch if you want GPU; optional FlashAttention 2)
#
# Optional env:
#   HF_BASE_MODEL=meta-llama/Llama-3.2-3B-Instruct
#   HF_SPEC_HOST=0.0.0.0
#   HF_SPEC_PORT=7070
#   HF_TOP_K=20
#   HF_ATTN_IMPL=flash_attention_2   # if installed
#   HF_TOKEN=hf_xxx                  # if the model is gated

import asyncio
import os
import time
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from shared import (
    MessageChannel,
    PrefillRequest,
    PrefillResponse,
    ResetRequest,
    VerifyRequest,
    VerifyResponse,
)

from const import (
    BASE_MODEL,
    HOST,
    PORT,
    SEED,
    TOP_K,
    ATTN_IMPL_ENV,
    DEVICE,
    DTYPE,
    GENERATOR,
    PY_RNG,
    HF_TOKEN,
)

from utils_hf import *

class BatchVerifier:
    """
    Batched verifier using HF Transformers with left-padded KV caches.
    - Prefill with prefix-only (exclude the stream's last token).
    - Verify on [last] + K drafted tokens so logits rows 0..K align with d0..dK-1 and fallback.
    - Commit by slicing the returned KV to the committed tail; no extra forward needed.
    """
    def __init__(self, model: AutoModelForCausalLM, tok: AutoTokenizer):
        self.model = model
        self.tok = tok
        self.cache = DynamicCache()
        self.tokens: List[List[int]] = []   # per-stream committed tokens (prefix-only at prefill, then grow)
        self.last:   List[int] = []         # per-stream "last token" to start each verify step

        # timing
        self.verify_iterations = 0
        self.total_forward_time = 0.0
        self.total_e2e_time = 0.0
        self.total_tokens_processed = 0

    async def reset(self) -> None:
        self.cache = DynamicCache()
        self.tokens = []
        self.last = []
        self.verify_iterations = 0
        self.total_forward_time = 0.0
        self.total_e2e_time = 0.0
        self.total_tokens_processed = 0

    async def prefill_batch(self, prompts: list[list[int]]) -> None:
        self.tokens = [p[:-1] for p in prompts]
        self.last   = [int(p[-1]) for p in prompts]
        
        # We prefill on all but the last tokens.
        self.cache = prefill(self.model, self.tokens, tokenizer=self.tok)

        # Zero out pad area for numerical cleanliness
        self.cache = zero_cache(self.cache, [len(x) for x in self.tokens])

    @torch.inference_mode()
    async def verify_batch(
        self,
        draft_toks: list[list[int]],
        draft_topk_vals: list[list[list[float]]],
        draft_topk_idx: list[list[list[int]]],
    ) -> VerifyResponse:
        B = len(draft_toks)

        # All streams must share the same K in this simple batch path
        K = max(len(x) for x in draft_toks)
        # Pad empty streams with 0-length rows; we will skip them below
        assert all((len(x) == K) or (len(x) == 0) for x in draft_toks)

        # Build [last] + drafts per stream; empty -> just [last] to get a fallback row
        rows = []
        for i in range(B):
            if len(draft_toks[i]) == 0:
                rows.append([self.last[i]]) # TODO: Shouldn't this be padded to make it length K+1?
            else:
                rows.append([self.last[i]] + [int(t) for t in draft_toks[i]])
        toks_verify = torch.tensor(rows, dtype=torch.long, device=DEVICE)  # (B, K+1)

        # Forward once with the existing batched cache
        outputs = self.model(
            toks_verify,
            use_cache=True,
            past_key_values=self.cache
        )
        logits = outputs.logits  # (B, K+1, V)
        base_topk_vals, base_topk_idx = torch.topk(logits, TOP_K, dim=-1)

        accepted = []
        base_tok = []
        hit_eos  = []

        # Per-stream acceptance using the same "logits-based" rule as the MLX client
        for i in range(B):
            stream_K = len(draft_toks[i])
            if stream_K == 0:
                # No drafts: take fallback row 0
                row_logits = logits[i, 0]
                tk_vals, tk_idx = torch.topk(row_logits, TOP_K, dim=-1)
                probs = torch.softmax(tk_vals, dim=-1)
                choice = torch.multinomial(probs, 1, generator=GENERATOR[DEVICE.type]).item()
                tok = int(tk_idx[choice].item())
                accepted.append(0)
                base_tok.append(tok)
                hit_eos.append(False)
                continue

            m = 0
            eos_hit = False
            for j in range(stream_K):
                tok = int(draft_toks[i][j])

                # draft row j
                d_idx = torch.tensor(draft_topk_idx[i][j], device=DEVICE, dtype=torch.long)
                d_val = torch.tensor(draft_topk_vals[i][j], device=DEVICE, dtype=logits.dtype)
                pos = (d_idx == tok).nonzero(as_tuple=False)
                if pos.numel() != 1:
                    # sample not in draft top-k (shouldn't happen if sampled from it)
                    break
                draft_logit = float(d_val[pos[0, 0]].item())

                # base row j (aligned because we fed [last] + drafts)
                b_idx = base_topk_idx[i, j]
                b_val = base_topk_vals[i, j]
                posb = (b_idx == tok).nonzero(as_tuple=False)
                if posb.numel() == 0:
                    # token not in base top-k -> reject
                    break
                base_logit = float(b_val[posb[0, 0]].item())

                if draft_logit <= base_logit:
                    m += 1
                else:
                    u = PY_RNG.uniform(0.0, 1.0)
                    # keep the same (approximate) rule as the MLX client
                    # if you want the exact rule, use math.exp(base_logit - draft_logit)
                    accept = (u <= (base_logit / draft_logit))
                    if not accept:
                        break
                    m += 1

                if (self.tok.eos_token_id is not None) and (tok == self.tok.eos_token_id):
                    eos_hit = True
                    break

            # Base fallback token from row m if EOS not hit
            if not eos_hit:
                row_logits = logits[i, m]   # row m is the fallback distribution
                tk_vals, tk_idx = torch.topk(row_logits, TOP_K, dim=-1)
                probs = torch.softmax(tk_vals, dim=-1)
                choice = torch.multinomial(probs, 1, generator=GENERATOR[DEVICE.type]).item()
                tok = int(tk_idx[choice].item())
                base_tok.append(tok)
            else:
                base_tok.append(-1)  # use -1 to signal None on the wire

            accepted.append(m)
            hit_eos.append(eos_hit)

        # ---- Commit: slice returned KV and advance tokens/last ----
        # outputs.past_key_values has past_len + (1 + stream_K) new steps per stream
        new_cache = outputs.past_key_values  # DynamicCache
        # Build per-row rollback = (1 + stream_K) - (accepted + base_appended)
        r = []
        new_tokens = []
        for i in range(B):
            stream_K = len(draft_toks[i])
            base_appended = 0 if hit_eos[i] else 1
            r.append((1 + stream_K) - (accepted[i] + base_appended))

            # tokens extended by [last] + drafts for this verify step
            ext = self.tokens[i] + [self.last[i]] + [int(t) for t in draft_toks[i]]
            # trim uncommitted steps
            keep = (1 + stream_K) - r[-1]
            new_tokens.append(ext + ([] if keep == 0 else []) )
        # Rollback uncommitted positions in KV and tokens
        self.cache, self.tokens = rollback_dynamic_per_row_simple(new_cache, [t + [self.last[i]] + draft_toks[i] for i, t in enumerate(self.tokens)], r)

        # Advance "last" per stream
        for i in range(B):
            if not hit_eos[i]:
                if accepted[i] == 0:
                    self.last[i] = base_tok[i]
                else:
                    # If we accepted something and also appended base fallback, last becomes base.
                    # If we accepted all K and appended base, last is base.
                    # If EOS was hit, last is the final accepted draft token (handled in else below).
                    self.last[i] = base_tok[i] if base_tok[i] != -1 else int(draft_toks[i][accepted[i]-1])
            else:
                # EOS hit: set last to final accepted draft token
                self.last[i] = int(draft_toks[i][accepted[i]-1]) if accepted[i] > 0 else self.last[i]

        return VerifyResponse(
            accepted_len=[int(x) for x in accepted],
            base_token=[int(x) for x in base_tok],
            hit_eos=[bool(x) for x in hit_eos],
        )


async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, verifier: BatchVerifier) -> None:
    peer = writer.get_extra_info("peername")
    print(f"client connected: {peer}")
    channel = MessageChannel(reader, writer)
    try:
        while True:
            msg = await channel.recv()
            if msg is None:
                print(f"client disconnected: {peer}")
                break

            if isinstance(msg, ResetRequest):
                await verifier.reset()

            elif isinstance(msg, PrefillRequest):
                await verifier.prefill_batch(msg.prompts)
                await channel.send(PrefillResponse(ok=True))

            elif isinstance(msg, VerifyRequest):
                res = await verifier.verify_batch(
                    msg.draft_toks, msg.draft_topk_vals, msg.draft_topk_idx,
                )
                await channel.send(res)

            else:
                raise RuntimeError(f"unhandled message type: {type(msg)!r}")
    finally:
        # Print timing summary
        if verifier.verify_iterations > 0:
            avg_fwd = verifier.total_forward_time / verifier.verify_iterations if verifier.verify_iterations else 0.0
            avg_e2e = verifier.total_e2e_time / verifier.verify_iterations if verifier.verify_iterations else 0.0
            avg_tps_fwd = verifier.total_tokens_processed / verifier.total_forward_time if verifier.total_forward_time > 0 else 0.0
            avg_tps_e2e = verifier.total_tokens_processed / verifier.total_e2e_time if verifier.total_e2e_time > 0 else 0.0
            print(f"\n[BATCH SERVER TIMING SUMMARY]")
            print(f"  Total verify iterations: {verifier.verify_iterations}")
            print(f"  Total tokens processed:  {verifier.total_tokens_processed}")
            print(f"  Total forward time:      {verifier.total_forward_time:.4f}s")
            print(f"  Total end-to-end time:   {verifier.total_e2e_time:.4f}s")
            print(f"  Avg time/verify (fwd):   {avg_fwd:.4f}s")
            print(f"  Avg time/verify (e2e):   {avg_e2e:.4f}s")
            print(f"  Avg tokens/sec (fwd):    {avg_tps_fwd:.2f}")
            print(f"  Avg tokens/sec (e2e):    {avg_tps_e2e:.2f}")
        await channel.close()


async def main() -> None:
    print(f"Loading base model with Transformers: {BASE_MODEL}")

    model, tok = load_model(BASE_MODEL)
    model.eval()

    torch.manual_seed(SEED)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    verifier = BatchVerifier(model, tok)

    server = await asyncio.start_server(
        lambda r, w: handle_client(r, w, verifier), HOST, PORT
    )
    addrs = ", ".join(str(sock.getsockname()) for sock in server.sockets)
    print(f"HF Verifier listening on {addrs} (dtype={DTYPE}, device={DEVICE})")
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
