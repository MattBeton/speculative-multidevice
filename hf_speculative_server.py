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
from typing import Any, Dict, List, Optional, Tuple

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
        self.post_verify_task: Optional[asyncio.Task[None]] = None

        # timing
        self.verify_iterations = 0
        self.total_forward_time = 0.0
        self.total_post_model_time = 0.0
        self.total_e2e_time = 0.0
        self.total_tokens_processed = 0
        self.total_positions_verified = 0
        self.total_positions_verified = 0

    async def reset(self) -> None:
        if self.post_verify_task is not None and not self.post_verify_task.done():
            await self.post_verify_task
        self.cache = DynamicCache()
        self.tokens = []
        self.last = []
        self.post_verify_task = None
        self.verify_iterations = 0
        self.total_forward_time = 0.0
        self.total_post_model_time = 0.0
        self.total_e2e_time = 0.0
        self.total_tokens_processed = 0
        self.total_positions_verified = 0

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
        # Check that post_verify coroutine is not running - if it's running then await it
        if self.post_verify_task is not None and not self.post_verify_task.done():
            await self.post_verify_task
        
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
        model_start = time.perf_counter()
        outputs = self.model(
            toks_verify,
            use_cache=True,
            past_key_values=self.cache
        )
        model_end = time.perf_counter()
        model_time = model_end - model_start
        
        post_model_start = time.perf_counter()
        logits = outputs.logits  # (B, K+1, V)
        base_topk_vals, base_topk_idx = torch.topk(logits, TOP_K, dim=-1)

        # ------- Vectorized accept + fallback sampling -------
        # base_topk_* already computed on GPU: (B, K+1, TOP_K)
        # We will:
        #   1) compute accept mask for every (stream, position) in one shot
        #   2) turn that mask into "m" (accepted prefix length) via cumprod/sum
        #   3) sample fallback token from row m per stream in one multinomial()
        B, KP1, k = base_topk_idx.shape
        K = KP1 - 1
        device = logits.device

        # Streams with drafts (either K or 0 by earlier assert)
        active_mask = torch.tensor([len(x) > 0 for x in draft_toks], device=device)
        all_rows = torch.arange(B, device=device)

        # Defaults for every stream (including empty)
        accepted = torch.zeros(B, dtype=torch.long, device=device)
        hit_eos  = torch.zeros(B, dtype=torch.bool, device=device)

        if active_mask.any():
            act_idx = active_mask.nonzero(as_tuple=False).squeeze(-1)  # (Ba,)
            Ba = act_idx.numel()

            # Pack draft tensors for active streams only
            # shapes: (Ba, K), (Ba, K, k), (Ba, K, k)
            draft_tokens_t = torch.tensor([draft_toks[i]       for i in act_idx], dtype=torch.long,       device=device)
            d_idx_t        = torch.tensor([draft_topk_idx[i]   for i in act_idx], dtype=torch.long,       device=device)
            d_val_t        = torch.tensor([draft_topk_vals[i]  for i in act_idx], dtype=logits.dtype,     device=device)

            # Base top‑k rows aligned to verification positions 0..K‑1
            base_idx_v = base_topk_idx[act_idx, :K, :]   # (Ba, K, k)
            base_val_v = base_topk_vals[act_idx, :K, :]  # (Ba, K, k)

            # Locate the sampled draft token in the base/draft top‑k rows
            # -> base_logit & draft_logit tensors, both (Ba, K)

            # Base: where is tok in base top‑k?
            eq_b      = (base_idx_v == draft_tokens_t.unsqueeze(-1))   # (Ba, K, k)
            present_b = eq_b.any(-1)                                   # (Ba, K)
            pos_b     = eq_b.float().argmax(-1)                        # (Ba, K) — first True index
            base_logit = base_val_v.gather(-1, pos_b.unsqueeze(-1)).squeeze(-1)  # (Ba, K)

            # Draft: where is tok in draft top‑k?
            eq_d      = (d_idx_t == draft_tokens_t.unsqueeze(-1))      # (Ba, K, k)
            present_d = eq_d.any(-1)                                   # (Ba, K)
            pos_d     = eq_d.float().argmax(-1)                        # (Ba, K)
            draft_logit = d_val_t.gather(-1, pos_d.unsqueeze(-1)).squeeze(-1)    # (Ba, K)

            # MLX-style approximate accept rule: accept if base>=draft,
            # otherwise with prob ~ (base_logit / draft_logit).
            base_ge = base_logit >= draft_logit
            ratio   = base_logit / (draft_logit + 1e-9)                # same semantics as before
            U       = torch.rand_like(ratio)
            accept  = present_b & present_d & (base_ge | (U <= ratio)) # (Ba, K)

            # Stop on accepted EOS (include that position in 'm' then stop)
            eos_id = getattr(self.tok, "eos_token_id", None)
            if eos_id is not None:
                eos_mask      = (draft_tokens_t == int(eos_id))        # (Ba, K)
                eos_accepted  = eos_mask & accept                      # (Ba, K)
                idxs          = torch.arange(K, device=device).unsqueeze(0).expand_as(eos_accepted)
                eos_pos       = torch.where(eos_accepted, idxs, torch.full_like(idxs, K))
                first_eos     = eos_pos.min(dim=1).values              # (Ba,)
                eos_hit_act   = first_eos < K
                m_eos_limit   = torch.where(eos_hit_act, first_eos + 1, torch.full_like(first_eos, K))
            else:
                eos_hit_act = torch.zeros(accept.shape[0], dtype=torch.bool, device=device)
                m_eos_limit = torch.full((accept.shape[0],), K, dtype=torch.long, device=device)

            # Stop at first reject: cumprod gives 1 while all previous accepts are True
            accepted_prefix = accept.to(torch.int32).cumprod(dim=1)    # (Ba, K)
            m_reject_limit  = accepted_prefix.sum(dim=1)               # (Ba,)

            # Final accepted length m for each active stream
            m_act = torch.minimum(m_reject_limit, m_eos_limit)         # (Ba,)

            accepted[act_idx] = m_act
            hit_eos[act_idx]  = eos_hit_act

        # Sample fallback/base token row-wise from row 'm' (row 0 for empty streams)
        row_idx   = accepted.clamp(max=K)                              # (B,)
        sel_vals  = base_topk_vals[all_rows, row_idx, :]               # (B, k)
        sel_idx   = base_topk_idx [all_rows, row_idx, :]               # (B, k)
        probs     = torch.softmax(sel_vals, dim=-1)
        choice    = torch.multinomial(probs, 1, generator=GENERATOR[DEVICE.type])  # (B, 1)
        base_tok  = sel_idx.gather(-1, choice).squeeze(-1)             # (B,)
        base_tok  = base_tok.masked_fill(hit_eos, -1)                  # -1 signals None on the wire

        accepted = accepted.tolist()
        base_tok = base_tok.tolist()
        hit_eos  = hit_eos.tolist()
        # ------- end vectorized block -------

        # Spawn post_verify ready for the next request
        self.post_verify_task = asyncio.create_task(
            self.post_verify(outputs, draft_toks, accepted, hit_eos, base_tok)
        )

        post_model_end = time.perf_counter()
        post_model_time = post_model_end - post_model_start
        
        # Update timing statistics
        self.verify_iterations += 1
        self.total_forward_time += model_time
        self.total_post_model_time += post_model_time
        # Count tokens processed (sum of accepted + base tokens appended)
        tokens_this_batch = sum(accepted) + sum(1 for eos in hit_eos if not eos)
        self.total_tokens_processed += tokens_this_batch
        # Count positions verified (B * K) to match benchmark metric
        positions_this_batch = B * K
        self.total_positions_verified += positions_this_batch

        return VerifyResponse(
            accepted_len=[int(x) for x in accepted],
            base_token=[int(x) for x in base_tok],
            hit_eos=[bool(x) for x in hit_eos],
        )

    async def post_verify(
        self,
        outputs: Any,
        draft_toks: list[list[int]],
        accepted: list[int],
        hit_eos: list[bool],
        base_tok: list[int],
    ) -> None:
        B = len(accepted)

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
            total_time = verifier.total_forward_time + verifier.total_post_model_time
            avg_fwd = verifier.total_forward_time / verifier.verify_iterations if verifier.verify_iterations else 0.0
            avg_post = verifier.total_post_model_time / verifier.verify_iterations if verifier.verify_iterations else 0.0
            avg_total = total_time / verifier.verify_iterations if verifier.verify_iterations else 0.0
            combined_tps = verifier.total_positions_verified / total_time if total_time > 0 else 0.0
            print(f"\n[BATCH SERVER TIMING SUMMARY]")
            print(f"  Total verify iterations: {verifier.verify_iterations}")
            print(f"  Total positions verified: {verifier.total_positions_verified}")
            print(f"  Total tokens processed:  {verifier.total_tokens_processed}")
            print(f"  Total model call time:   {verifier.total_forward_time:.4f}s")
            print(f"  Total post-model time:   {verifier.total_post_model_time:.4f}s")
            print(f"  Total time:              {total_time:.4f}s")
            print(f"  Avg time/verify (model): {avg_fwd:.4f}s")
            print(f"  Avg time/verify (post):  {avg_post:.4f}s")
            print(f"  Avg time/verify (total): {avg_total:.4f}s")
            print(f"  TPS (positions/sec):    {combined_tps:.1f}")
            # Reset timing statistics for the next client connection
            verifier.verify_iterations = 0
            verifier.total_forward_time = 0.0
            verifier.total_post_model_time = 0.0
            verifier.total_e2e_time = 0.0
            verifier.total_tokens_processed = 0
            verifier.total_positions_verified = 0
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
