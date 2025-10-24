# speculative_server.py
import asyncio
from pathlib import Path
from typing import Optional

import numpy as np

from model import MLXGenerationModel
from shared import (
    MessageChannel,
    PrefillRequest,
    PrefillResponse,
    ResetRequest,
    VerifyRequest,
    VerifyResponse,
    run_mlx,
)

# ---- Configure the base (verifier) model ----
BASE_MODEL_PATH = next(Path(
    "/Users/frank/.cache/huggingface/hub/models--mlx-community--Llama-3.2-3B-Instruct/snapshots/"
).glob("*"))


class VerifierSession:
    """Minimal session state for the base model verifier."""
    def __init__(self) -> None:
        self.model = MLXGenerationModel(BASE_MODEL_PATH)
        self._last: Optional[int] = None
        self._eos: Optional[int] = self.model.eos_token_id()

    async def reset(self) -> None:
        await run_mlx(self.model.reset)
        self._last = None

    async def prefill(self, prompt: list[int]) -> None:
        # Prefill with all but the final token; save the last to start decode.
        if not prompt or len(prompt) < 1:
            raise ValueError("empty prompt")
        prefix = np.array(prompt[:-1], dtype=np.int32)
        await run_mlx(self.model.forward, prefix, False)  # fills cache
        self._last = int(prompt[-1])

    async def verify(self, req: VerifyRequest) -> VerifyResponse:
        if self._last is None:
            raise RuntimeError("verify called before prefill")

        draft_toks = np.array(req.draft_toks, dtype=np.int32)  # (K,)
        d_topk_idx = np.array(req.draft_topk_idx, dtype=np.int32)  # (K, top_k)
        d_topk_vals = np.array(req.draft_topk_vals, dtype=np.float32)  # (K, top_k)

        # Base verifies positions [0..K], feeding current last + draft tokens
        toks_verify = np.concatenate((np.array([self._last], dtype=np.int32), draft_toks))
        base_toks, b_topk_idx, b_topk_vals = await run_mlx(self.model.forward, toks_verify, False)
        # Shapes: base_toks -> (K+1,), b_topk_idx/vals -> (K+1, top_k)

        accepted = 0
        hit_eos = False

        for i in range(len(draft_toks)):
            tok = int(draft_toks[i])

            d_idx_row = d_topk_idx[i]
            d_val_row = d_topk_vals[i]
            d_mask = (d_idx_row == tok)
            if d_mask.sum() != 1:
                raise RuntimeError("draft top-k must contain the sampled token exactly once")
            draft_logit = float(d_val_row[d_mask][0])

            b_idx_row = b_topk_idx[i]
            b_val_row = b_topk_vals[i]
            b_mask = (b_idx_row == tok)
            in_base_topk = bool(b_mask.any())
            base_logit = float(b_val_row[b_mask][0]) if in_base_topk else float("-inf")

            if base_logit == float("-inf"):
                break
            elif draft_logit <= base_logit:
                accepted += 1
            else:
                u = np.random.uniform(0.0, 1.0)
                if u <= (base_logit / draft_logit):
                    accepted += 1
                else:
                    break

            if self._eos is not None and tok == self._eos:
                hit_eos = True
                break

        # If EOS not hit, append base fallback token (position m)
        base_token = None
        base_appended = 0
        if not hit_eos:
            base_token = int(base_toks[accepted])
            base_appended = 1

        # Roll back only the uncommitted verifier steps
        spec_k = len(draft_toks)
        base_trim = (spec_k + 1) - (accepted + base_appended)
        if base_trim > 0:
            await run_mlx(self.model.trim_cache, base_trim)

        # Advance "last" to the committed tail
        if base_appended == 1:
            self._last = base_token
        elif accepted > 0:
            self._last = int(draft_toks[accepted - 1])

        return VerifyResponse(accepted_len=accepted, base_token=base_token, hit_eos=hit_eos)


async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    peer = writer.get_extra_info("peername")
    print(f"client connected: {peer}")
    channel = MessageChannel(reader, writer)
    session = VerifierSession()
    try:
        while True:
            msg = await channel.recv()
            if msg is None:
                print(f"client disconnected: {peer}")
                break

            if isinstance(msg, ResetRequest):
                await session.reset()
            elif isinstance(msg, PrefillRequest):
                await session.prefill(msg.prompt)
                await channel.send(PrefillResponse(ok=True))
            elif isinstance(msg, VerifyRequest):
                resp = await session.verify(msg)
                await channel.send(resp)
            else:
                raise RuntimeError(f"unhandled message type: {type(msg)!r}")
    finally:
        await channel.close()


async def main() -> None:
    server = await asyncio.start_server(handle_client, "127.0.0.1", 7070)
    addrs = ", ".join(str(sock.getsockname()) for sock in server.sockets)
    print(f"Verifier listening on {addrs}")
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
