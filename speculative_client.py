# speculative_client.py
import argparse
import asyncio
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from mlx_model import MLXGenerationModel
from timing import TokenTimer
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

from const import SPEC_K, MAX_NEW_TOKENS

# ---- Configure the draft (client) model ----
DRAFT_MODEL_PATH = next(Path(
    "~/.cache/huggingface/hub/models--mlx-community--Llama-3.2-1B-Instruct-bf16/snapshots/"
).expanduser().glob("*"))

# Fixed prompts list
PROMPTS: list[str] = [
    "Why is the sky blue?",
    "Explain speculative decoding in simple terms.",
    "Write a sonnet about the iPhone.",
    "What are the benefits of renewable energy?",
    "Describe the process of photosynthesis.",
    "How does machine learning work?",
    "What is the difference between AI and AGI?",
    "Explain the theory of relativity.",
]


def _print_phase_summary(name: str, tokens: int, seconds: float) -> None:
    tokps = (tokens / seconds) if seconds > 0 else float("inf")
    print(f"[{name:7}] {tokens} toks in {seconds:.3f}s  → {tokps:.1f} tok/s")


class DraftClient:
    """Client for batched speculative decoding with fixed batch size."""
    def __init__(self, channel: MessageChannel, prompts: list[str]):
        self._channel = channel
        self._prompts = prompts
        self.model = MLXGenerationModel(DRAFT_MODEL_PATH)
        self.last: list[int] = None

    async def reset_remote(self) -> None:
        """Send reset request to server and wait for acknowledgment."""
        await self._channel.send(ResetRequest())
        resp = await self._channel.recv()
        assert isinstance(resp, ResetResponse)

    async def prefill_both(self, timer: TokenTimer) -> None:
        """Tokenize + prefill locally; send prefill request remotely."""
        prompts: list[list[int]] = [
            await run_mlx(self.model.tokenize, prompt)
            for prompt in self._prompts
        ]

        tokens = [prompt[:-1] for prompt in prompts]
        self.last = [prompt[-1] for prompt in prompts]

        with timer.measure("prefill", lambda: sum(len(prompt) - 1 for prompt in prompts)):
            prefill_local = asyncio.create_task(
                run_mlx(self.model.prefill, tokens)
            )

            async def send_and_recv():
                await self._channel.send(PrefillRequest(prompts=tokens))
                return await self._channel.recv()

            prefill_remote = asyncio.create_task(send_and_recv())

            await asyncio.gather(prefill_local, prefill_remote)

        prefill = timer.get('prefill')
        _print_phase_summary("prefill", prefill.tokens, prefill.seconds)


    async def decode_batch(self, spec_k: int, max_new_tokens: int, timer: TokenTimer) -> List[str]:
        """Drive speculative decode for all streams until each finishes or budget hit."""
        def _total_committed() -> int:
            return 100
            return sum(len(st.generated) for st in self._streams)

        # Timing accumulators for decode phase
        total_client_time = 0.0
        total_server_wait_time = 0.0

        with timer.measure("decode", _total_committed):
            round_count = 0
            max_rounds = 5
            while True:
                # Limit to 5 rounds of drafting
                if round_count >= max_rounds:
                    break
                
                round_count += 1

                # Prepare batch verify request
                draft_toks_batch: List[List[int]] = []
                draft_topk_idx_batch: List[List[List[int]]] = []
                draft_topk_vals_batch: List[List[List[float]]] = []

                # 1) Generate draft tokens for each stream (client work)
                client_start = time.perf_counter()

                current = self.last
                for _ in range(spec_k):
                    y = np.array([current], dtype=np.int32).reshape(-1, 1)
                    print(y.shape)
                    tok, s_topk_idx, s_topk_vals = await run_mlx(self.model.forward, y, only_final=False)

                    draft_toks_batch.append(tok.tolist())
                    draft_topk_idx_batch.append(s_topk_idx.tolist())
                    draft_topk_vals_batch.append(s_topk_vals.tolist())

                    current = tok

                    ## TODO: These are currently getting appended as (S, B, K) not (B, S, K)


                ### TMP
                for b in range(3):
                    print(self.model.decode(self.model.tokens[b, :]))
                raise Exception('stop')

                # Pause client timing before server wait
                client_time_before_server = time.perf_counter() - client_start
                total_client_time += client_time_before_server

                # 2) Send batch verify request and wait for response (server wait)
                verify_req = VerifyRequest(
                    draft_toks=draft_toks_batch,
                    draft_topk_vals=draft_topk_vals_batch,
                    draft_topk_idx=draft_topk_idx_batch
                )
                server_wait_start = time.perf_counter()
                await self._channel.send(verify_req)

                resp = await self._channel.recv()
                server_wait_end = time.perf_counter()
                server_wait_time = server_wait_end - server_wait_start
                total_server_wait_time += server_wait_time

                assert isinstance(resp, VerifyResponse)

                client_start = time.perf_counter()





                for i, stream in enumerate(self._streams):
                    # Skip finished streams
                    if stream.finished:
                        continue

                    # Extract results for this stream
                    accepted = resp.accepted_len[i]
                    base_tok = resp.base_token[i]
                    hit_eos = resp.hit_eos[i]

                    # Handle -1 as None for base_token
                    if base_tok == -1:
                        base_tok = None

                    # Get accepted tokens from draft
                    accepted_tokens = draft_toks_batch[i][:accepted]
                    base_appended = 0 if base_tok is None else 1

                    # Commit locally: accepted + optional base
                    stream.generated.extend(accepted_tokens)
                    if base_tok is not None:
                        stream.generated.append(base_tok)

                    # Align local draft KV cache
                    draft_trim = spec_k - (accepted + base_appended)
                    if draft_trim > 0:
                        await run_mlx(stream.model.trim_cache, draft_trim)
                    elif draft_trim < 0:
                        # Only possible when m==K and base appended -> catch up one step
                        last_accepted = draft_toks_batch[i][accepted - 1]
                        await run_mlx(
                            stream.model.forward,
                            np.array([[last_accepted]], dtype=np.int32)
                        )

                    # Prepare next iteration's "last"
                    if hit_eos or len(stream.generated) >= max_new_tokens:
                        stream.finished = True
                    else:
                        stream.last_token = base_tok if base_tok is not None else (
                            accepted_tokens[-1] if accepted_tokens else stream.last_token
                        )
                
                # Accumulate remaining client time for this round
                client_end = time.perf_counter()
                total_client_time += client_end - client_start

        # Store timing values for reporting
        self._decode_client_time = total_client_time
        self._decode_server_wait_time = total_server_wait_time

        # Decode final texts
        return [stream.decoded_text() for stream in self._streams]


async def main(host: str = 'localhost') -> None:
    # Connect to server
    reader, writer = await asyncio.open_connection(host, 7070)
    channel = MessageChannel(reader, writer)
    client = DraftClient(channel, PROMPTS)

    timer = TokenTimer()
    try:
        await client.reset_remote()
        await client.prefill_both(timer)
        texts = await client.decode_batch(SPEC_K, MAX_NEW_TOKENS, timer)

        # Stats
        prefill = timer.get("prefill")
        decode = timer.get("decode")
        total_tokens = 0
        total_seconds = 0

        if prefill:
            _print_phase_summary("prefill", prefill.tokens, prefill.seconds)
            total_tokens += prefill.tokens
            total_seconds += prefill.seconds

        if decode:
            _print_phase_summary("decode", decode.tokens, decode.seconds)
            # Print client vs server timing breakdown
            if hasattr(client, '_decode_client_time') and hasattr(client, '_decode_server_wait_time'):
                client_time = client._decode_client_time
                server_wait_time = client._decode_server_wait_time
                client_tokps = decode.tokens / client_time if client_time > 0 else float("inf")
                server_wait_pct = (server_wait_time / decode.seconds * 100) if decode.seconds > 0 else 0.0
                print(f"  └─ Client work: {client_time:.3f}s ({client_tokps:.1f} tok/s)")
                print(f"  └─ Server wait: {server_wait_time:.3f}s ({server_wait_pct:.1f}% of decode)")
            total_tokens += decode.tokens
            total_seconds += decode.seconds

        # Print total TPS
        if total_seconds > 0:
            total_tps = total_tokens / total_seconds
            print(f"[TOTAL  ] {total_tokens} toks in {total_seconds:.3f}s  → {total_tps:.1f} tok/s")

        # Output per stream
        print()
        for i, txt in enumerate(texts):
            print(f"--- [Stream {i}] ---")
            print(txt)
            print()
    finally:
        await channel.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speculative decoding client")
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host address to connect to (default: localhost)"
    )
    args = parser.parse_args()
    asyncio.run(main(host=args.host))
