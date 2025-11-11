# mlx_speculative_server.py
import asyncio
from pathlib import Path
from typing import List, Optional, Tuple

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
    # "/Users/frank/.cache/huggingface/hub/models--mlx-community--Llama-3.2-3B-Instruct/snapshots/"
    "~/.cache/huggingface/hub/models--mlx-community--Llama-3.2-1B-Instruct-bf16/snapshots/"
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
        await run_mlx(model.reset)
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

        tok, topk_idx, topk_vals = await run_mlx(self.model.forward, x, only_final=False)

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

    def _accept_and_choose(
        self,
        topk_idx: np.ndarray,                            # (B, K+1, k)
        topk_vals: np.ndarray,                            # (B, K+1, k)
        draft_toks: List[List[int]],
        draft_idx: List[List[List[int]]],           # (B, K, k) as lists
        draft_val: List[List[List[float]]],                  # (B, K, k) as lists
        sample_mode: str = "argmax",                     # "argmax" (fast) or "topk"
        sample_topk: int = 20,                            # used only if sample_mode == "topk"
    ) -> Tuple[List[int], List[int], List[bool]]:
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
