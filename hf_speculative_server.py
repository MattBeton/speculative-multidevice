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
    Batched verifier using HF Transformers.
    - Prefills b batches in parallel
    - Verifies batches of (b,k) tokens in parallel
    - Reshuffle of KV should be performed async
    Note that the batch size b must always stay constant. 
    Currently not robust to dynamic streams.
    """
    def __init__(self, model: AutoModelForCausalLM, tok: AutoTokenizer):
        self.model = model
        self.tok = tok
        self.cache = DynamicCache()

        # Timing statistics
        self.verify_iterations = 0
        self.total_forward_time = 0.0
        self.total_e2e_time = 0.0
        self.total_tokens_processed = 0

    async def reset(self) -> None:
        self.cache = DynamicCache()        

        self.verify_iterations = 0
        self.total_forward_time = 0.0
        self.total_e2e_time = 0.0
        self.total_tokens_processed = 0

    async def prefill_batch(self, prompts: list[list[int]]) -> None:
        self.tokens = prompts

        self.cache = prefill(self.model, prompts)
        self.cache = zero_cache(self.cache, [len(x) for x in prompts])

        print_cache(self.cache, 2, 10)

    @torch.inference_mode()
    async def verify_batch(
        self,
        draft_toks: list[list[int]],
        draft_topk_vals: list[list[list[float]]],
        draft_topk_idx: list[list[list[int]]],
    ) -> VerifyResponse:
        print(draft_toks)

        # Handle empty drafts
        orig_draft_toks = draft_toks
        draft_toks = [
            x if len(x) != 0 else [PAD_ID] * 9
            for x in draft_toks
        ]
        assert all([len(x) == len(draft_toks[0]) for x in draft_toks])
        x = torch.tensor(draft_toks, dtype=torch.long, device=DEVICE)

        outputs = self.model(
            x,
            use_cache=True,
            past_key_values=self.cache
        )

        self.tokens = [x + y for x, y in zip(self.tokens, draft_toks)]

        # Get logits and compute top-k for base model
        logits = outputs.logits  # (batch_size, seq_len, vocab_size)
        base_topk_vals, base_topk_idx = torch.topk(logits, TOP_K, dim=-1)

        accept_values = []
        base_tokens = []
        hit_eos = []

        # Process each stream
        for i in range(len(draft_toks)):
            # Handle empty draft (finished stream)
            if len(orig_draft_toks[i]) == 0:
                accept_values.append(0)
                base_tokens.append(-1)
                hit_eos.append(True)
                continue

            accepted = 0
            hit_eos_flag = False

            # Verify each draft token
            for j in range(len(orig_draft_toks[i])):
                tok = draft_toks[i][j]

                # Get draft logit (raw)
                d_idx = torch.tensor(draft_topk_idx[i][j], device=DEVICE)
                d_vals = torch.tensor(draft_topk_vals[i][j], device=DEVICE)
                d_mask = (d_idx == tok)
                if d_mask.sum() != 1:
                    raise RuntimeError(f"draft top-k must contain sampled token (stream {i}, pos {j})")
                draft_logit = d_vals[d_mask].item()

                # Get base logit (raw)
                b_idx = base_topk_idx[i, j]
                b_vals = base_topk_vals[i, j]
                b_mask = (b_idx == tok)
                in_base_topk = b_mask.any()
                base_logit = b_vals[b_mask].item() if in_base_topk else float("-inf")

                # Acceptance logic
                if base_logit == float("-inf"):
                    break
                elif draft_logit <= base_logit:
                    accepted += 1
                else:
                    # Convert to probabilities for ratio calculation
                    draft_prob = torch.softmax(d_vals, dim=-1)[d_mask].item()
                    base_prob = torch.softmax(b_vals, dim=-1)[b_mask].item() if in_base_topk else 0.0
                    u = PY_RNG.uniform(0.0, 1.0)
                    if u <= (base_prob / draft_prob):
                        accepted += 1
                    else:
                        break

                # Check EOS
                if self.tok.eos_token_id and tok == self.tok.eos_token_id:
                    hit_eos_flag = True
                    break

            # Sample base fallback token if not EOS
            if not hit_eos_flag:
                accepted_logits = logits[i, accepted]
                top_k_logits, top_k_indices = torch.topk(accepted_logits, TOP_K)
                probs = torch.softmax(top_k_logits, dim=-1)
                sampled_idx = torch.multinomial(probs, 1, generator=GENERATOR[DEVICE.type])
                base_token = top_k_indices[sampled_idx].item()
            else:
                base_token = -1

            accept_values.append(accepted)
            base_tokens.append(base_token)
            hit_eos.append(hit_eos_flag)

        print(accept_values)

        rollback_values = [len(draft_toks[0]) - y for y in accept_values]
        rollback_dynamic_per_row_simple(self.cache, self.tokens, rollback_values)

        return VerifyResponse(
            accepted_len=accept_values,
            base_token=base_tokens,
            hit_eos=hit_eos,
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
