# speculative_client.py
import asyncio
from pathlib import Path
from typing import List

import numpy as np

from model import MLXGenerationModel
from timing import TokenTimer
from shared import (
    MessageChannel,
    PrefillRequest,
    PrefillResponse,
    ResetRequest,
    VerifyRequest,
    VerifyResponse,
    run_mlx,
)

# ---- Configure the draft (client) model ----
DRAFT_MODEL_PATH = next(Path(
    "/Users/frank/.cache/huggingface/hub/models--mlx-community--Llama-3.2-1B-Instruct-bf16/snapshots/"
).glob("*"))

PROMPT = "Why is the sky blue?"
SPEC_K = 8
MAX_NEW_TOKENS = 64


def _print_phase_summary(name: str, tokens: int, seconds: float) -> None:
    tokps = (tokens / seconds) if seconds > 0 else float("inf")
    print(f"[{name:7}] {tokens} toks in {seconds:.3f}s  → {tokps:.1f} tok/s")


class DraftClient:
    def __init__(self, channel: MessageChannel):
        self._channel = channel
        self.model = MLXGenerationModel(DRAFT_MODEL_PATH)
        self.eos = self.model.eos_token_id()

    async def reset_remote(self) -> None:
        await self._channel.send(ResetRequest())

    async def prefill_both(self, ids: np.ndarray, timer: TokenTimer) -> None:
        """Run prefill concurrently on both nodes."""
        async def _prefill_local():
            # Local: prefill with all but the last prompt token.
            prefix = ids[:-1]
            await run_mlx(self.model.forward, prefix, False)

        async def _prefill_remote():
            await self._channel.send(PrefillRequest(prompt=ids.astype(int).tolist()))
            msg = await self._channel.recv()
            if not isinstance(msg, PrefillResponse):
                raise RuntimeError(f"expected PrefillResponse, got {type(msg)!r}")

        with timer.measure("prefill", lambda: len(ids) - 1):
            await asyncio.gather(_prefill_local(), _prefill_remote())

    async def decode_once(self) -> str:
        """Run one full speculative generation and print stats; return the text."""
        ids = await run_mlx(self.model.tokenize, PROMPT)
        ids = np.array(ids, dtype=np.int32)

        await self.reset_remote()

        timer = TokenTimer()
        await self.prefill_both(ids, timer)

        prompt_last = int(ids[-1])
        generated: List[int] = []
        last = prompt_last

        # Speculative loop
        def _committed_len() -> int:
            return len(generated)

        with timer.measure("decode", _committed_len):
            while len(generated) < MAX_NEW_TOKENS:
                # 1) Draft K tokens autoregressively
                draft_toks = []
                draft_topk_idx = []
                draft_topk_vals = []

                cur = last
                for _ in range(SPEC_K):
                    y = np.array([[cur]], dtype=np.int32)  # (1, 1)
                    tok, topk_idx, topk_vals = await run_mlx(self.model.forward, y)
                    t = int(tok[-1])  # scalar
                    draft_toks.append(t)
                    draft_topk_idx.append(topk_idx[-1].tolist())
                    draft_topk_vals.append([float(v) for v in topk_vals[-1]])
                    cur = t  # advance locally

                # 2) Verify on base (remote)
                req = VerifyRequest(
                    draft_toks=draft_toks,
                    draft_topk_idx=draft_topk_idx,
                    draft_topk_vals=draft_topk_vals,
                )
                print(f'sending request {req}')
                await self._channel.send(req)
                resp = await self._channel.recv()
                print(f'received response {resp}')
                if not isinstance(resp, VerifyResponse):
                    raise RuntimeError(f"expected VerifyResponse, got {type(resp)!r}")

                m = int(resp.accepted_len)
                base_tok = resp.base_token
                hit_eos = bool(resp.hit_eos)

                accepted_tokens = draft_toks[:m]
                base_appended = 0 if base_tok is None else 1

                # 3) Commit tokens locally (append accepted + optional base)
                generated.extend(accepted_tokens)
                if base_tok is not None:
                    generated.append(int(base_tok))

                # 4) Align draft KV cache
                draft_trim = SPEC_K - (m + base_appended)
                if draft_trim > 0:
                    await run_mlx(self.model.trim_cache, draft_trim)
                elif draft_trim < 0:
                    # Only when m==K and base appended -> catch up one step
                    await run_mlx(
                        self.model.forward,
                        np.array([[int(draft_toks[m - 1])]], dtype=np.int32),
                    )

                # 5) Prepare next step
                if hit_eos or len(generated) >= MAX_NEW_TOKENS:
                    break
                last = int(base_tok) if base_tok is not None else int(accepted_tokens[-1])

        # Report + decode
        prefill = timer.get("prefill")
        decode = timer.get("decode")
        if prefill:
            _print_phase_summary("prefill", prefill.tokens, prefill.seconds)
        if decode:
            _print_phase_summary("decode", decode.tokens, decode.seconds)

        text = await run_mlx(self.model.decode, generated)
        return text


async def main() -> None:
    ip = '192.168.200.2'
    reader, writer = await asyncio.open_connection(ip, 7070)
    channel = MessageChannel(reader, writer)
    client = DraftClient(channel)
    # client = DraftClient(None)

    try:
        text = await client.decode_once()
        print("\n" + text)
    finally:
        await channel.close()


if __name__ == "__main__":
    asyncio.run(main())
