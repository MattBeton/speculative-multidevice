# -*- coding: utf-8 -*-
# pure_client.py
# Pure remote decoding: generate 1 token at a time via the server (no local model).
# Requires: speculative_server.py running on 127.0.0.1:7070

import asyncio
from pathlib import Path
from typing import List

from mlx_lm.tokenizer_utils import load_tokenizer

from shared import (
    MessageChannel,
    PrefillRequest,
    PrefillResponse,
    ResetRequest,
    VerifyRequest,
    VerifyResponse,
)
from timing import TokenTimer


# Use the SAME base model path the server is using (only for tokenizer).
BASE_MODEL_PATH = next(Path(
    "/Users/frank/.cache/huggingface/hub/models--mlx-community--Llama-3.2-3B-Instruct/snapshots/"
).glob("*"))

PROMPT = "Why is the sky blue?"
MAX_NEW_TOKENS = 64


def _print_phase_summary(name: str, tokens: int, seconds: float) -> None:
    tokps = (tokens / seconds) if seconds > 0 else float("inf")
    print(f"[{name:7}] {tokens} toks in {seconds:.3f}s  → {tokps:.1f} tok/s")


async def main() -> None:
    # ---- Tokenizer only (no local model compute) ----
    tok = load_tokenizer(BASE_MODEL_PATH)
    ids: List[int] = tok.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        add_generation_prompt=True,
        tokenize=True,
    )
    ids = [int(x) for x in ids]
    if not ids:
        raise RuntimeError("empty tokenized prompt")
    eos_id = getattr(tok, "eos_token_id", None)

    # ---- Wire up the channel ----
    reader, writer = await asyncio.open_connection("127.0.0.1", 7070)
    channel = MessageChannel(reader, writer)

    timer = TokenTimer()
    generated: List[int] = []

    try:
        # Reset remote KV
        await channel.send(ResetRequest())

        # Prefill on server (we measure client-side e2e time)
        with timer.measure("prefill", lambda: max(len(ids) - 1, 0)):
            await channel.send(PrefillRequest(prompt=ids))
            msg = await channel.recv()
            if not isinstance(msg, PrefillResponse):
                raise RuntimeError(f"expected PrefillResponse, got {type(msg)!r}")

        # Decode: K=0 speculative step ⇒ ask server for "next token" each round
        def _gen_len() -> int:
            return len(generated)

        with timer.measure("decode", _gen_len):
            for _ in range(MAX_NEW_TOKENS):
                await channel.send(
                    VerifyRequest(
                        draft_toks=[],          # K = 0
                        draft_topk_idx=[],      # empty is fine
                        draft_topk_vals=[],     # empty is fine
                    )
                )
                resp = await channel.recv()
                if not isinstance(resp, VerifyResponse):
                    raise RuntimeError(f"expected VerifyResponse, got {type(resp)!r}")

                base_tok = resp.base_token
                if base_tok is None:
                    break  # server decided there's nothing to append (should be rare here)

                t = int(base_tok)
                generated.append(t)

                # Stop on EOS (if tokenizer has one)
                if eos_id is not None and t == eos_id:
                    break

        # Stats
        prefill = timer.get("prefill")
        decode = timer.get("decode")
        if prefill:
            _print_phase_summary("prefill", prefill.tokens, prefill.seconds)
        if decode:
            _print_phase_summary("decode", decode.tokens, decode.seconds)

        # Detokenize only the generated continuation
        text = tok.decode(generated)
        print("\n" + text)

    finally:
        await channel.close()


if __name__ == "__main__":
    asyncio.run(main())

