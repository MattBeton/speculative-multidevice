# mlx_speculative_server.py
import asyncio
from pathlib import Path

import numpy as np

from mlx_model import MLXGenerationModel
from shared import (
    MessageChannel,
    PrefillRequest,
    PrefillResponse,
    ResetRequest,
    ResetResponse,
    VerifyRequest,
    VerifyResponse,
    run_mlx,
)

# ---- Configure the base (verifier) model ----
BASE_MODEL_PATH = next(Path(
    "~/.cache/huggingface/hub/models--mlx-community--Llama-3.2-3B-Instruct/snapshots/"
    # "~/.cache/huggingface/hub/models--mlx-community--Llama-3.2-1B-Instruct-bf16/snapshots/"
).expanduser().glob("*"))


class MLXVerifierSession:
    """Batched session state for the base model verifier."""
    def __init__(self) -> None:
        self.model: MLXGenerationModel = MLXGenerationModel(BASE_MODEL_PATH)
        self.last: list[int | None] = []
        self._eos: int | None = None
        self.batch_size: int | None = None

    async def reset(self) -> None:
        """Reset all model states."""
        await run_mlx(self.model.reset)
        self.last = []
        self.batch_size = None

    async def prefill(self, prompts: list[list[int]]) -> None:
        """Prefill batch of prompts."""
        self.batch_size = len(prompts)

        tokens = [prompt[:-1] for prompt in prompts]
        await run_mlx(self.model.prefill, tokens)
        self.last = [prompt[-1] for prompt in prompts]

    async def verify(
        self, 
        draft_toks: list[list[int]],
        draft_topk_idx: list[list[list[int]]],
        draft_topk_vals: list[list[list[float]]],
    ) -> VerifyResponse:
        """Verify batch of draft tokens."""

        K = max(len(x) for x in draft_toks)
        assert all((len(x) == K) or (len(x) == 0) for x in draft_toks)

        # Build [last] + drafts (empty -> [last] only).
        rows: list[list[int]] = []
        for i in range(self.batch_size):
            assert len(draft_toks[i]) != 0, 'this shouldnt happen'
            rows.append([self.last[i]] + [int(t) for t in draft_toks[i]])
        x = np.array(rows, dtype=np.int32)  # (B, K+1)

        import time
        t1 = time.perf_counter()
        tok, topk_idx, topk_vals = await run_mlx(self.model.forward, x, only_final=False)
        print(f'time in forward: {time.perf_counter() - t1}')

        accepted, base_choice, hit_eos = self._accept_and_choose(
            topk_idx, topk_vals, draft_toks, draft_topk_idx, draft_topk_vals
        )

        self.last = base_choice

        # TODO: This can go into a separate coroutine.
        await run_mlx(self.model.rollback_tokens, [K - x for x in accepted])

        return VerifyResponse(
            accepted_len=accepted,
            base_token=base_choice,
            hit_eos=hit_eos,
        )

    # def _accept_and_choose(
    #     self,
    #     topk_idx: np.ndarray,                             # (B, K+1, k)  base model top-k token ids
    #     topk_vals: np.ndarray,                            # (B, K+1, k)  base model top-k "logits"/scores
    #     draft_toks: list[list[int]],                      # (B, K)       chosen tokens from the draft model
    #     draft_idx: list[list[list[int]]],                 # (B, K, k)    draft model top-k token ids used when sampling
    #     draft_val: list[list[list[float]]],               # (B, K, k)    draft model top-k values (same scale as topk_vals)
    #     sample_mode: str = "argmax",                      # "argmax" or "topk"
    #     sample_topk: int = 20,                            # used only if sample_mode == "topk"
    # ) -> tuple[list[int], list[int], list[bool]]:
    #     """
    #     Vectorized accept-and-choose in NumPy space.
    #
    #     Semantics:
    #     - Acceptance at position j compares the base value for the drafted token vs the draft value
    #         (base >= draft -> accept; else accept with probability base/draft). If the drafted token
    #         is NOT present in the base top-k for that row, it is considered rejected.
    #     - Acceptance stops at the first rejected position; we also stop (and mark hit_eos=True)
    #         if an accepted EOS is encountered.
    #     - Fallback choice is taken from the base row at index m, where m is the number of
    #         accepted tokens (clamped to K). If EOS was accepted at m-1, we return base_choice=-1.
    #     - topk_idx/topk_vals are for rows 0..K (K used for fallback).
    #     - draft_* arrays are used only for rows 0..K-1 (to evaluate acceptance).
    #     """
    #     B, KP1, k_total = topk_idx.shape
    #     K = KP1 - 1
    #     assert KP1 == topk_vals.shape[1] and k_total == topk_vals.shape[2]
    #
    #     print('starting accept and choose loop')
    #
    #     # Active streams are those with a non-empty draft.
    #     active_mask = np.array([len(t) > 0 for t in draft_toks], dtype=bool)
    #     accepted = np.zeros(B, dtype=np.int64)
    #     hit_eos = np.zeros(B, dtype=bool)
    #
    #     if active_mask.any():
    #         act = np.nonzero(active_mask)[0]
    #         Ba = act.shape[0]
    #
    #         # Gather draft data for active rows
    #         d_tokens = np.array([draft_toks[i] for i in act], dtype=np.int64)           # (Ba, K)
    #         d_idx_t  = np.array([draft_idx[i]  for i in act], dtype=np.int64)           # (Ba, K, k)
    #         d_val_t  = np.array([draft_val[i]  for i in act], dtype=topk_vals.dtype)    # (Ba, K, k)
    #
    #         # Base top-k for the first K rows (positions being validated)
    #         base_idx  = topk_idx[act, :K, :]                                            # (Ba, K, k)
    #         base_vals = topk_vals[act, :K, :]                                           # (Ba, K, k)
    #
    #         # Locate the drafted token within the base top-k (required for acceptance)
    #         eq_base   = (base_idx == d_tokens[..., None])                                # (Ba, K, k)
    #         in_base   = eq_base.any(axis=-1)                                             # (Ba, K)
    #         pos_base  = eq_base.argmax(axis=-1)                                          # (Ba, K)
    #
    #         # Locate the drafted token within the draft top-k (should be present, but check anyway)
    #         eq_draft  = (d_idx_t == d_tokens[..., None])                                 # (Ba, K, k)
    #         in_draft  = eq_draft.any(axis=-1)                                            # (Ba, K)
    #         pos_draft = eq_draft.argmax(axis=-1)                                         # (Ba, K)
    #
    #         rows = np.arange(Ba)[:, None]
    #         cols = np.arange(K)[None, :]
    #
    #         # Gather values for the drafted token from base and draft top-k
    #         base_val_for_tok  = base_vals[rows, cols, pos_base]                           # (Ba, K)
    #         draft_val_for_tok = d_val_t[rows, cols, pos_draft]                            # (Ba, K)
    #
    #         # Acceptance rule; only evaluate where token is present in BOTH base & draft top-k
    #         present = in_base & in_draft
    #         eps = 1e-9
    #
    #         base_ge = base_val_for_tok >= draft_val_for_tok
    #         ratio   = base_val_for_tok / (draft_val_for_tok + eps)
    #         U       = np.random.random(size=ratio.shape)
    #
    #         accept = present & (base_ge | (U <= ratio))                                   # (Ba, K)
    #
    #         # Stop at accepted EOS (if configured)
    #         eos = getattr(self, "eos", None)
    #         if eos is not None:
    #             eos = int(eos)
    #             eos_mask      = (d_tokens == eos)                                         # (Ba, K)
    #             eos_accepted  = eos_mask & accept
    #             idxs          = np.broadcast_to(np.arange(K, dtype=np.int64), eos_accepted.shape)
    #             whereK        = np.where(eos_accepted, idxs, K)                           # positions of accepted EOS, else K
    #             first_eos     = whereK.min(axis=1)                                        # (Ba,)
    #             eos_hit_act   = first_eos < K
    #             m_eos_limit   = np.where(eos_hit_act, first_eos + 1, K).astype(np.int64)  # (Ba,)
    #         else:
    #             eos_hit_act = np.zeros(Ba, dtype=bool)
    #             m_eos_limit = np.full(Ba, K, dtype=np.int64)
    #
    #         # Contiguous accept-prefix length (stop at first reject)
    #         accepted_prefix = accept.astype(np.int32).cumprod(axis=1)                     # (Ba, K)
    #         m_reject_limit  = accepted_prefix.sum(axis=1)                                 # (Ba,)
    #
    #         m_act = np.minimum(m_reject_limit, m_eos_limit)                               # (Ba,)
    #         accepted[act] = m_act
    #         hit_eos[act]  = eos_hit_act
    #
    #     # Fallback choice from the base row at index m (clamped to K)
    #     row_idx = np.minimum(accepted, K)                                                 # (B,)
    #     rowsB = np.arange(B)
    #
    #     row_topk_idx  = topk_idx[rowsB, row_idx, :]                                       # (B, k)
    #     row_topk_vals = topk_vals[rowsB, row_idx, :]                                       # (B, k)
    #
    #     if sample_mode == "topk":
    #         k_use = int(min(sample_topk, row_topk_idx.shape[1]))
    #         idx_subset  = row_topk_idx[:, :k_use]                                         # (B, k_use)
    #         vals_subset = row_topk_vals[:, :k_use].astype(np.float64)                     # (B, k_use)
    #
    #         # Softmax over the subset
    #         vals_shift = vals_subset - vals_subset.max(axis=1, keepdims=True)
    #         np.exp(vals_shift, out=vals_shift)
    #         probs = vals_shift / vals_shift.sum(axis=1, keepdims=True)
    #
    #         # Sample one index per row via inverse CDF
    #         u   = np.random.random(size=B)
    #         cdf = probs.cumsum(axis=1)
    #         chosen_pos = (cdf >= u[:, None]).argmax(axis=1)                               # (B,)
    #         base_choice = idx_subset[rowsB, chosen_pos]
    #     else:
    #         # Argmax over the available top-k values
    #         chosen_pos  = row_topk_vals.argmax(axis=1)
    #         base_choice = row_topk_idx[rowsB, chosen_pos]
    #
    #     # Suppress base choice when EOS was accepted at m-1
    #     base_choice = base_choice.astype(np.int64)
    #     base_choice[hit_eos] = -1
    #
    #     return accepted.tolist(), base_choice.tolist(), hit_eos.tolist()



    def _accept_and_choose(
        self,
        topk_idx: np.ndarray,                            # (B, K+1, k)
        topk_vals: np.ndarray,                            # (B, K+1, k)
        draft_toks: list[list[int]],
        draft_idx: list[list[list[int]]],           # (B, K, k) as lists
        draft_val: list[list[list[float]]],                  # (B, K, k) as lists
        sample_mode: str = "argmax",                     # "argmax" (fast) or "topk"
        sample_topk: int = 20,                            # used only if sample_mode == "topk"
    ) -> tuple[list[int], list[int], list[bool]]:
        K = len(draft_idx[0])

        import random
        accepted = [random.randint(0, K) for _ in range(self.batch_size)]

        base_choice: list[int] = []
        for i in range(self.batch_size):
            base_choice.append(int(topk_idx[i, accepted[i], np.argmax(topk_vals[i, accepted[i], :])]))

        hit_eos = [False for _ in range(self.batch_size)]

        return accepted, base_choice, hit_eos

async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Handle a client connection with batched requests."""
    peer = writer.get_extra_info("peername")
    print(f"Client connected: {peer}")
    channel = MessageChannel(reader, writer)
    session = MLXVerifierSession()

    try:
        while True:
            msg = await channel.recv()
            if msg is None:
                print(f"Client disconnected: {peer}")
                break

            if isinstance(msg, ResetRequest):
                await session.reset()
                await channel.send(ResetResponse(ok=True))
            elif isinstance(msg, PrefillRequest):
                await session.prefill(msg.prompts)
                await channel.send(PrefillResponse(ok=True))
            elif isinstance(msg, VerifyRequest):
                resp = await session.verify(
                    msg.draft_toks,
                    msg.draft_topk_vals,
                    msg.draft_topk_idx,
                )
                await channel.send(resp)
            else:
                raise RuntimeError(f"Unhandled message type: {type(msg)!r}")
    finally:
        await channel.close()


async def main() -> None:
    import os
    PORT = int(os.environ.get("MLX_SPEC_PORT", "7070"))  # Default to 7071 to avoid conflict with HF server
    server = await asyncio.start_server(handle_client, "127.0.0.1", PORT)
    addrs = ", ".join(str(sock.getsockname()) for sock in server.sockets)
    print(f"MLX Verifier listening on {addrs}")
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
