# hf_speculative_server.py
import asyncio
import time
from typing import List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from shared import (
    MessageChannel,
    PrefillRequest,
    PrefillResponse,
    ResetRequest,
    ResetResponse,
    VerifyRequest,
    VerifyResponse,
)

from const import (
    BASE_MODEL,
    HOST,
    PORT,
    SEED,
    DEVICE,
    DTYPE,
    TOP_K,
    ATTN_IMPL_ENV,
    GENERATOR,
    PY_RNG,
    HF_TOKEN,
)

from utils_hf import prefill as hf_prefill, zero_cache as hf_zero_cache


# ------------------------------------------------------------
# Verifier (batched) — clean, three steps per verify:
#   1) forward()         2) accept + choose fallback    3) spawn commit()
# ------------------------------------------------------------
class BatchVerifier:
    def __init__(self, model: AutoModelForCausalLM, tok: AutoTokenizer):
        self.model = model
        self.tok = tok

        self.cache: DynamicCache = DynamicCache()
        self.last: List[int] = []
        self.lens: List[int] = []  # committed per-stream lengths (prefix-only at prefill)
        self.post_verify_task: Optional[asyncio.Task[None]] = None

        self.eos: Optional[int] = getattr(tok, "eos_token_id", None)

        # timing
        self.verify_iterations = 0
        self.total_forward_time = 0.0
        self.total_post_model_time = 0.0
        self.total_tokens_processed = 0
        self.total_positions_verified = 0

    async def reset(self) -> None:
        if self.post_verify_task is not None and not self.post_verify_task.done():
            await self.post_verify_task
        self.cache = DynamicCache()
        self.last = []
        self.lens = []
        self.post_verify_task = None
        self.verify_iterations = 0
        self.total_forward_time = 0.0
        self.total_post_model_time = 0.0
        self.total_tokens_processed = 0
        self.total_positions_verified = 0

    async def prefill_batch(self, prompts: List[List[int]]) -> None:
        # Store prefix lengths and last token per stream.
        self.lens = [len(p) - 1 for p in prompts]
        self.last = [int(p[-1]) for p in prompts]

        # Prefill on all but the last token.
        self.cache = hf_prefill(self.model, [p[:-1] for p in prompts], tokenizer=self.tok)
        self.cache = hf_zero_cache(self.cache, [len(p) - 1 for p in prompts])

    @torch.inference_mode()
    async def verify_batch(
        self,
        draft_toks: List[List[int]],
        draft_topk_vals: List[List[List[float]]],
        draft_topk_idx: List[List[List[int]]],
    ) -> VerifyResponse:
        # Ensure previous commit finished before using self.cache again.
        if self.post_verify_task is not None and not self.post_verify_task.done():
            await self.post_verify_task

        B = len(draft_toks)
        K = max(len(x) for x in draft_toks) if B else 0
        assert all((len(x) == K) or (len(x) == 0) for x in draft_toks)

        # Build [last] + drafts (empty -> [last] only).
        rows = []
        for i in range(B):
            if len(draft_toks[i]) == 0:
                rows.append([self.last[i]])
            else:
                rows.append([self.last[i]] + [int(t) for t in draft_toks[i]])
        x = torch.tensor(rows, dtype=torch.long, device=DEVICE)  # (B, K+1)

        # 1) forward
        t0 = time.perf_counter()
        outputs = self.model(input_ids=x, use_cache=True, past_key_values=self.cache)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        fwd = t1 - t0

        # 2) accept + choose fallback (no global top-k)
        t2 = time.perf_counter()
        logits = outputs.logits  # (B, K+1, V)
        accepted, base_tok, hit_eos = self._accept_and_choose(
            logits, draft_toks, draft_topk_idx, draft_topk_vals
        )
        t3 = time.perf_counter()
        post = t3 - t2

        # 3) spawn commit
        self.post_verify_task = asyncio.create_task(
            self._commit(outputs.past_key_values, draft_toks, accepted, hit_eos, base_tok)
        )

        # metrics
        self.verify_iterations += 1
        self.total_forward_time += fwd
        self.total_post_model_time += post
        tokens_this = sum(accepted) + sum(1 for h in hit_eos if not h)
        self.total_tokens_processed += tokens_this
        self.total_positions_verified += (B * K)

        # response
        base_out = [int(t) for t in base_tok]  # -1 means None (EOS hit)
        return VerifyResponse(
            accepted_len=[int(m) for m in accepted],
            base_token=base_out,
            hit_eos=[bool(h) for h in hit_eos],
        )

    # ------------------------------------------------------------
    # Acceptance + fallback (vectorized, no global top-k)
    # ------------------------------------------------------------
    def _accept_and_choose(
        self,
        logits: torch.Tensor,                            # (B, K+1, V)
        draft_toks: List[List[int]],
        d_idx: List[List[List[int]]],                    # (B, K, k) as lists
        d_val: List[List[List[float]]],                  # (B, K, k) as lists
        sample_mode: str = "argmax",                     # "argmax" (fast) or "topk"
        sample_topk: int = 8,                            # used only if sample_mode == "topk"
    ) -> Tuple[List[int], List[int], List[bool]]:
        B, KP1, V = logits.shape
        K = KP1 - 1
        device = logits.device

        active_mask = torch.tensor([len(t) > 0 for t in draft_toks], device=device)
        all_rows = torch.arange(B, device=device)

        accepted = torch.zeros(B, dtype=torch.long, device=device)
        hit_eos = torch.zeros(B, dtype=torch.bool, device=device)

        if active_mask.any():
            act = active_mask.nonzero(as_tuple=False).squeeze(-1)  # (Ba,)
            Ba = act.numel()

            # Draft tokens (Ba, K)
            d_tokens = torch.tensor(
                [draft_toks[i] for i in act], dtype=torch.long, device=device
            )

            # Base logits at drafted tokens: gather one logit per position.
            base_rows = logits[act, :K, :]                          # (Ba, K, V)
            base_logit = base_rows.gather(-1, d_tokens.unsqueeze(-1)).squeeze(-1)  # (Ba, K)

            # Draft logits: find the sampled token in the draft top-k row.
            d_idx_t = torch.tensor([d_idx[i] for i in act], dtype=torch.long, device=device)     # (Ba, K, k)
            d_val_t = torch.tensor([d_val[i] for i in act], dtype=logits.dtype, device=device)   # (Ba, K, k)
            eq = (d_idx_t == d_tokens.unsqueeze(-1))                 # (Ba, K, k)
            present = eq.any(-1)                                     # (Ba, K)
            pos = eq.float().argmax(-1)                              # (Ba, K)
            draft_logit = d_val_t.gather(-1, pos.unsqueeze(-1)).squeeze(-1)  # (Ba, K)

            # Accept rule (logit-based): base>=draft else accept with prob base/draft.
            base_ge = base_logit >= draft_logit
            ratio = base_logit / (draft_logit + 1e-9)
            U = torch.rand_like(ratio)
            accept = present & (base_ge | (U <= ratio))              # (Ba, K)

            # Stop at accepted EOS.
            if self.eos is not None:
                eos_mask = (d_tokens == int(self.eos))               # (Ba, K)
                eos_accepted = eos_mask & accept
                idxs = torch.arange(K, device=device)[None, :].expand_as(eos_accepted)
                eos_pos = torch.where(eos_accepted, idxs, torch.full_like(idxs, K))
                first_eos = eos_pos.min(dim=1).values                # (Ba,)
                eos_hit_act = first_eos < K
                m_eos_limit = torch.where(eos_hit_act, first_eos + 1, torch.full_like(first_eos, K))
            else:
                eos_hit_act = torch.zeros(Ba, dtype=torch.bool, device=device)
                m_eos_limit = torch.full((Ba,), K, dtype=torch.long, device=device)

            accepted_prefix = accept.to(torch.int32).cumprod(dim=1)
            m_reject_limit = accepted_prefix.sum(dim=1)              # (Ba,)

            m_act = torch.minimum(m_reject_limit, m_eos_limit)       # (Ba,)
            accepted[act] = m_act
            hit_eos[act] = eos_hit_act

        # Row m per stream; clamp at K for safety.
        row_idx = accepted.clamp(max=K)

        # Fallback choice for streams w/o EOS: from base row m.
        row_logits = logits[all_rows, row_idx, :]                    # (B, V)
        if sample_mode == "topk":
            k = min(int(sample_topk), V)
            vals, idx = torch.topk(row_logits, k=k, dim=-1)          # (B, k)
            probs = torch.softmax(vals, dim=-1)
            choice = torch.multinomial(probs, 1, generator=GENERATOR[DEVICE.type]).squeeze(-1)
            base_choice = idx.gather(-1, choice.unsqueeze(-1)).squeeze(-1)   # (B,)
        else:
            base_choice = row_logits.argmax(dim=-1)                  # (B,)

        # Don’t append base when EOS was accepted at m-1.
        base_choice = base_choice.masked_fill(hit_eos, -1)

        return accepted.tolist(), base_choice.tolist(), hit_eos.tolist()

    # ------------------------------------------------------------
    # Commit: slice per-row keep and repack to new uniform length.
    # Keeps shape small and avoids growing S across rounds.
    # ------------------------------------------------------------

    async def _commit(
        self,
        new_cache: DynamicCache,
        draft_toks: List[List[int]],
        accepted: List[int],
        hit_eos: List[bool],
        base_tok: List[int],
    ) -> None:
        B = len(accepted)
        if B == 0:
            return

        # Per-stream counts
        keep   = [accepted[i] + (0 if hit_eos[i] else 1) for i in range(B)]
        L_prev = self.lens
        L_new  = [L_prev[i] + keep[i] for i in range(B)]

        # Old unified length (what's currently in self.cache)
        old_legacy = self.cache.to_legacy_cache()
        if len(old_legacy) == 0 or len(old_legacy[0]) == 0:
            raise RuntimeError("Cannot commit with empty cache - prefill must be called first")
        S_prev = old_legacy[0][0].shape[2]  # (B, H, S_prev, D)

        # New unified length target after commit
        S_target = max(L_new) if L_new else 0

        # New cache returned by the forward
        new_legacy = new_cache.to_legacy_cache()  # (B, H, S_src, D) with S_src = S_prev + (K+1) for non-empty rows

        dst = DynamicCache()
        for layer, ((k_old, v_old), (k_b, v_b)) in enumerate(zip(old_legacy, new_legacy)):
            # Shapes: k_old: (B, H, S_prev, D), k_b: (B, H, S_src, D)
            B0, H, S_prev_chk, D = k_old.shape
            assert B0 == B and S_prev_chk == S_prev

            # Note: S_src can differ per layer but is consistent across rows in practice.
            _, _, S_src, _ = k_b.shape

            k_out = k_b.new_zeros((B, H, S_target, D))
            v_out = v_b.new_zeros((B, H, S_target, D))

            for i in range(B):
                Lp = int(L_prev[i])
                ki = int(keep[i])
                Ln = int(L_new[i])
                dst_start = S_target - Ln

                # 1) Copy the previous committed prefix from OLD cache
                if Lp > 0:
                    src_prev_start = S_prev - Lp
                    k_prefix = k_old[i:i+1, :, src_prev_start:S_prev, :].contiguous()  # (1,H,Lp,D)
                    v_prefix = v_old[i:i+1, :, src_prev_start:S_prev, :].contiguous()
                    k_out[i:i+1, :, dst_start:dst_start+Lp, :] = k_prefix
                    v_out[i:i+1, :, dst_start:dst_start+Lp, :] = v_prefix

                # 2) Append the new committed tail from the END of NEW cache
                if ki > 0:
                    k_tail = k_b[i:i+1, :, S_src-ki:S_src, :].contiguous()  # (1,H,ki,D)
                    v_tail = v_b[i:i+1, :, S_src-ki:S_src, :].contiguous()
                    k_out[i:i+1, :, dst_start+Lp:dst_start+Lp+ki, :] = k_tail
                    v_out[i:i+1, :, dst_start+Lp:dst_start+Lp+ki, :] = v_tail

            dst.update(k_out, v_out, layer)

        # Swap in compacted cache
        self.cache = dst

        # Advance "last" and lengths
        for i in range(B):
            if not hit_eos[i]:
                self.last[i] = int(base_tok[i])
            elif accepted[i] > 0:
                self.last[i] = int(draft_toks[i][accepted[i] - 1])
            self.lens[i] = int(L_new[i])


# ------------------------------------------------------------
# Server plumbing (unchanged except for new class)
# ------------------------------------------------------------
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
                await channel.send(ResetResponse(ok=True))

            elif isinstance(msg, PrefillRequest):
                await verifier.prefill_batch(msg.prompts)
                await channel.send(PrefillResponse(ok=True))

            elif isinstance(msg, VerifyRequest):
                res = await verifier.verify_batch(msg.draft_toks, msg.draft_topk_vals, msg.draft_topk_idx)
                await channel.send(res)

            else:
                raise RuntimeError(f"unhandled message type: {type(msg)!r}")
    finally:
        # timing summary
        if verifier.verify_iterations > 0:
            total_time = verifier.total_forward_time + verifier.total_post_model_time
            avg_fwd = verifier.total_forward_time / verifier.verify_iterations if verifier.verify_iterations else 0.0
            avg_post = verifier.total_post_model_time / verifier.verify_iterations if verifier.verify_iterations else 0.0
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
            print(f"  TPS (positions/sec):    {combined_tps:.1f}")
        await channel.close()


def load_model(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, token=HF_TOKEN)
    from_kwargs = {
        "torch_dtype": DTYPE,
        "device_map": None,
        "low_cpu_mem_usage": True,
        "token": HF_TOKEN,
    }
    if ATTN_IMPL_ENV:
        from_kwargs["attn_implementation"] = ATTN_IMPL_ENV
    model = AutoModelForCausalLM.from_pretrained(model_id, **from_kwargs).to(DEVICE)
    model.eval()
    return model, tok


async def main() -> None:
    print(f"Loading base model with Transformers: {BASE_MODEL}")
    model, tok = load_model(BASE_MODEL)

    torch.manual_seed(SEED)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    verifier = BatchVerifier(model, tok)
    server = await asyncio.start_server(lambda r, w: handle_client(r, w, verifier), HOST, PORT)
    addrs = ", ".join(str(sock.getsockname()) for sock in server.sockets)
    print(f"HF Verifier listening on {addrs} (dtype={DTYPE}, device={DEVICE})")
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
