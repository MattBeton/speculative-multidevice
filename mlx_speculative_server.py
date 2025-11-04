# mlx_speculative_server.py
import asyncio
from pathlib import Path
from typing import List, Optional

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


class BatchVerifierSession:
    """Batched session state for the base model verifier."""
    def __init__(self) -> None:
        self.models: List[MLXGenerationModel] = []
        self._last_tokens: List[Optional[int]] = []
        self._eos: Optional[int] = None

    def _ensure_models(self, batch_size: int) -> None:
        """Ensure we have enough models for the batch size."""
        while len(self.models) < batch_size:
            model = MLXGenerationModel(BASE_MODEL_PATH)
            self.models.append(model)
            self._last_tokens.append(None)
            if self._eos is None:
                self._eos = model.eos_token_id()

    async def reset(self) -> None:
        """Reset all model states."""
        for model in self.models:
            await run_mlx(model.reset)
        self._last_tokens = [None] * len(self.models)

    async def prefill(self, prompts: List[List[int]]) -> None:
        """Prefill batch of prompts."""
        batch_size = len(prompts)
        self._ensure_models(batch_size)

        # Process each prompt in the batch
        for i, prompt in enumerate(prompts):
            if not prompt or len(prompt) < 1:
                raise ValueError(f"Empty prompt at index {i}")

            # Prefill with all but the final token; save the last to start decode
            prefix = np.array(prompt[:-1], dtype=np.int32)
            await run_mlx(self.models[i].forward, prefix, False)  # fills cache
            self._last_tokens[i] = int(prompt[-1])

    async def verify(self, req: VerifyRequest) -> VerifyResponse:
        """Verify batch of draft tokens."""
        batch_size = len(req.draft_toks)

        # Check that we have prefilled all streams
        for i in range(batch_size):
            if i >= len(self._last_tokens) or self._last_tokens[i] is None:
                raise RuntimeError(f"verify called before prefill for stream {i}")

        accepted_lens: List[int] = []
        base_tokens: List[int] = []
        hit_eos_flags: List[bool] = []

        # Process each stream in the batch
        for i in range(batch_size):
            # Handle empty draft (finished stream)
            if len(req.draft_toks[i]) == 0:
                accepted_lens.append(0)
                base_tokens.append(-1)
                hit_eos_flags.append(True)
                continue

            # Convert lists to numpy arrays for this stream
            draft_toks = np.array(req.draft_toks[i], dtype=np.int32)  # (K,)
            d_topk_idx = np.array(req.draft_topk_idx[i], dtype=np.int32)  # (K, top_k)
            d_topk_vals = np.array(req.draft_topk_vals[i], dtype=np.float32)  # (K, top_k)

            # Base verifies positions [0..K], feeding current last + draft tokens
            toks_verify = np.concatenate((np.array([self._last_tokens[i]], dtype=np.int32), draft_toks))
            base_toks, b_topk_idx, b_topk_vals = await run_mlx(
                self.models[i].forward, toks_verify, False
            )
            # Shapes: base_toks -> (K+1,), b_topk_idx/vals -> (K+1, top_k)

            accepted = 0
            hit_eos = False

            # Verify each draft token
            for j in range(len(draft_toks)):
                tok = int(draft_toks[j])

                # Get draft probabilities
                d_idx_row = d_topk_idx[j]
                d_val_row = d_topk_vals[j]
                d_mask = (d_idx_row == tok)
                if d_mask.sum() != 1:
                    raise RuntimeError(f"draft top-k must contain the sampled token exactly once (stream {i}, pos {j})")
                draft_logit = float(d_val_row[d_mask][0])

                # Get base probabilities
                b_idx_row = b_topk_idx[j]
                b_val_row = b_topk_vals[j]
                b_mask = (b_idx_row == tok)
                in_base_topk = bool(b_mask.any())
                base_logit = float(b_val_row[b_mask][0]) if in_base_topk else float("-inf")

                # Acceptance logic
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

                # Check for EOS
                if self._eos is not None and tok == self._eos:
                    hit_eos = True
                    break

            # If EOS not hit, get base fallback token
            base_token = None
            base_appended = 0
            if not hit_eos:
                base_token = int(base_toks[accepted])
                base_appended = 1

            # Roll back only the uncommitted verifier steps
            spec_k = len(draft_toks)
            base_trim = (spec_k + 1) - (accepted + base_appended)
            if base_trim > 0:
                await run_mlx(self.models[i].trim_cache, base_trim)

            # Advance "last" to the committed tail
            if base_appended == 1:
                self._last_tokens[i] = base_token
            elif accepted > 0:
                self._last_tokens[i] = int(draft_toks[accepted - 1])

            # Collect results for this stream
            accepted_lens.append(accepted)
            base_tokens.append(base_token if base_token is not None else -1)  # Use -1 for None
            hit_eos_flags.append(hit_eos)

        return VerifyResponse(
            accepted_len=accepted_lens,
            base_token=base_tokens,
            hit_eos=hit_eos_flags
        )


async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Handle a client connection with batched requests."""
    peer = writer.get_extra_info("peername")
    print(f"Client connected: {peer}")
    channel = MessageChannel(reader, writer)
    session = BatchVerifierSession()

    try:
        while True:
            msg = await channel.recv()
            if msg is None:
                print(f"Client disconnected: {peer}")
                break

            if isinstance(msg, ResetRequest):
                await session.reset()
            elif isinstance(msg, PrefillRequest):
                await session.prefill(msg.prompts)
                await channel.send(PrefillResponse(ok=True))
            elif isinstance(msg, VerifyRequest):
                resp = await session.verify(msg)
                await channel.send(resp)
            else:
                raise RuntimeError(f"Unhandled message type: {type(msg)!r}")
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