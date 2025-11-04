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
        assert all([len(x) == len(draft_toks[0]) for x in draft_toks])
        x = torch.tensor(draft_toks, dtype=torch.long, device=DEVICE)

        outputs = self.model(
            x,
            use_cache=True,
            past_key_values=self.cache
        )

        self.tokens = [x + y for x, y in zip(self.tokens, draft_toks)]

        # Get logits from model output
        logits = outputs.logits  # Shape: (batch_size, seq_len, vocab_size)

        # TODO: Logic to use logits for acceptance.
        # For now, using random acceptance for testing
        accept_values = []
        import random
        for _ in range(len(draft_toks)):
            accept_values.append(random.randint(1, len(draft_toks[0])))

        # Sample base tokens from the accepted position's logits
        base_tokens = []
        for i, accepted in enumerate(accept_values):
            accepted_logits = logits[i, accepted-1, :]  # (vocab_size,)

            top_k_logits, top_k_indices = torch.topk(accepted_logits, TOP_K)
            probs = torch.softmax(top_k_logits, dim=-1)

            # Sample from the top-k distribution
            sampled_idx = torch.multinomial(probs, num_samples=1, generator=GENERATOR["cuda" if DEVICE.type == "cuda" else "cpu"])
            sampled_token = top_k_indices[sampled_idx].item()

            base_tokens.append(sampled_token)

        rollback_values = [len(draft_toks[0]) - y for y in accept_values]
        rollback_dynamic_per_row_simple(self.cache, self.tokens, rollback_values)

        hit_eos = [False for _ in range(len(draft_toks))]

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
