# vllm_speculative_server.py
# pip install vllm transformers
from __future__ import annotations

import asyncio
from typing import Dict, List, Optional, Tuple

import numpy as np
from transformers import PreTrainedTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

from shared import (
    MessageChannel,
    PrefillRequest,
    PrefillResponse,
    ResetRequest,
    VerifyRequest,
    VerifyResponse,
)

# ---------------- Configuration ----------------
# Use your local snapshot path:
BASE_MODEL = "/root/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95/"
HOST = "0.0.0.0"
PORT = 7070

SEED = 90
BASE_TEMP = 1.0
DEFAULT_TOPK = 40  # used only if we can't infer top-k from the draft row


# ----------------- Utilities -------------------
def _normalize_from_logprobs(logps: np.ndarray) -> np.ndarray:
    """Renormalize a (subset) of logprobs to probabilities."""
    m = np.max(logps)
    probs = np.exp(logps - m)
    s = probs.sum()
    return probs / s if s > 0 else np.full_like(probs, 1.0 / probs.size)


def _draft_lp_from_row(
    d_ids: List[int],
    d_vals: List[float],
    tok_id: int,
) -> Tuple[float, int]:
    """
    Convert draft top-k 'values' into a normalized log-prob over that row,
    then return (logprob_of_tok, index).
    NOTE: Treats incoming values as logits; applies log-softmax over the row.
    """
    if not d_ids:
        return float("-inf"), -1
    try:
        j = d_ids.index(tok_id)
    except ValueError:
        return float("-inf"), -1

    x = np.array(d_vals, dtype=np.float64)
    m = float(np.max(x))
    z = float(np.exp(x - m).sum())
    lp_row = x - m - np.log(z)
    return float(lp_row[j]), j


def _sample_from_lp_dict(lp_dict: Dict[int, object], rng: np.random.Generator) -> int:
    """
    Sample a token id from vLLM logprob dict (token_id -> Logprob dataclass with .logprob).
    We renormalize over the subset vLLM provides (top-k + chosen).
    """
    items = [(int(t), float(o.logprob)) for t, o in lp_dict.items()]
    items.sort(key=lambda kv: kv[1], reverse=True)
    ids = np.array([t for t, _ in items], dtype=np.int64)
    lps = np.array([lp for _, lp in items], dtype=np.float64)
    probs = _normalize_from_logprobs(lps)
    idx = int(rng.choice(len(ids), p=probs))
    return int(ids[idx])


def _extract_prompt_rows(req_out, k: int) -> List[Optional[Dict[int, object]]]:
    """
    Extract vLLM's prompt_logprobs (teacher-forced) for each prompt token.
    Returns the last K rows (corresponding to the drafted tokens we appended).
    """
    prompt_rows = getattr(req_out, "prompt_logprobs", None)
    if prompt_rows is None:
        # Older variants sometimes attached to first sequence; try that:
        if req_out.outputs:
            prompt_rows = getattr(req_out.outputs[0], "prompt_logprobs", None)
    if prompt_rows is None:
        raise RuntimeError("vLLM did not return prompt_logprobs; ensure prompt_logprobs > 0")
    return prompt_rows[-k:] if k > 0 else []


# ----------------- Verifier Session -------------------
class VerifierSession:
    """
    vLLM-based verifier with MLX-identical commit/rollback semantics:
      - Keep state as _prefix (all committed tokens EXCEPT last) and _last (the trailing token).
      - On verify, build prompt = _prefix + [_last] + draft_toks and call vLLM ONCE with:
          prompt_logprobs=K, logprobs=K, max_tokens=1, detokenize=False
      - Acceptance at position i uses teacher-forced base logprob at row i vs. draft row.
      - Fallback token:
          * If reject at i: sample from prompt_logprobs row i (base distribution at that position).
          * If accept all: sample from generated step's logprobs[0].
      - Commit state exactly like the MLX server, moving tokens from 'last' into 'prefix'
        and setting a new 'last'.
    """

    def __init__(self, llm: LLM, tok: PreTrainedTokenizer):
        self.llm = llm
        self.tok = tok
        self._prefix: List[int] = []
        self._last: Optional[int] = None
        self._eos: Optional[int] = getattr(tok, "eos_token_id", None)
        self._rng = np.random.default_rng(SEED)

    async def reset(self) -> None:
        self._prefix = []
        self._last = None

    async def prefill(self, prompt: List[int]) -> None:
        if not prompt:
            raise ValueError("empty prompt")
        # Mirror MLX prefill: cache/prefix = all but last; 'last' = final prompt token
        self._prefix = [int(x) for x in prompt[:-1]]
        self._last = int(prompt[-1])

    async def verify(self, req: VerifyRequest) -> VerifyResponse:
        if self._last is None:
            raise RuntimeError("verify called before prefill")

        draft_toks = [int(t) for t in req.draft_toks]
        K = len(draft_toks)
        if K == 0:
            # Pure base step: ask base for one token after 'last'
            prompt_ids = self._prefix + [self._last]
            sp = SamplingParams(
                max_tokens=1, logprobs=DEFAULT_TOPK, temperature=BASE_TEMP, top_p=1.0, detokenize=False, seed=SEED
            )
            loop = asyncio.get_running_loop()
            out = (await loop.run_in_executor(
                None, lambda: self.llm.generate([TokensPrompt(prompt_token_ids=prompt_ids)], sp)
            ))[0]
            seq = out.outputs[0]
            gen_row = seq.logprobs[0] if seq.logprobs else None
            base_token = _sample_from_lp_dict(gen_row, self._rng) if gen_row else int(seq.token_ids[0])

            # Commit MLX-style: prefix += [last], last = base_token
            self._prefix.append(self._last)
            self._last = int(base_token)
            return VerifyResponse(accepted_len=0, base_token=int(base_token), hit_eos=False)

        # Infer base top-k from draft row length
        base_topk = len(req.draft_topk_idx[0]) if req.draft_topk_idx and len(req.draft_topk_idx[0]) > 0 else DEFAULT_TOPK

        # Single vLLM call for the whole K-chunk: teacher-force the draft tokens and get the next-token row.
        verify_prompt = self._prefix + [self._last] + draft_toks
        sp = SamplingParams(
            max_tokens=1,
            prompt_logprobs=int(base_topk),
            logprobs=int(base_topk),
            temperature=BASE_TEMP,
            top_p=1.0,
            detokenize=False,
            seed=SEED,
        )
        loop = asyncio.get_running_loop()
        req_out = (await loop.run_in_executor(
            None, lambda: self.llm.generate([TokensPrompt(prompt_token_ids=verify_prompt)], sp)
        ))[0]

        prompt_rows = _extract_prompt_rows(req_out, K)  # last K rows correspond to drafted tokens
        seq = req_out.outputs[0]
        gen_row = seq.logprobs[0] if seq.logprobs else None

        accepted = 0
        hit_eos = False
        base_token: Optional[int] = None

        # Walk the drafted tokens left-to-right
        for i, tok in enumerate(draft_toks):
            row = prompt_rows[i] or {}
            # Base logprob for drafted token at position i (teacher-forced)
            base_lp = float(row[tok].logprob) if tok in row else float("-inf")

            # Draft "logprob" over its own row (normalize draft's top-k values as logits)
            d_ids = [int(x) for x in req.draft_topk_idx[i]]
            d_vals = [float(x) for x in req.draft_topk_vals[i]]
            draft_lp, _ = _draft_lp_from_row(d_ids, d_vals, tok)

            # Accept if base_lp >= draft_lp; otherwise accept w.p. exp(base_lp - draft_lp)
            if base_lp == float("-inf"):
                # Reject immediately; fallback from THIS position's base distribution
                base_token = _sample_from_lp_dict(row, self._rng) if row else None
                break

            if base_lp >= draft_lp:
                accept = True
            else:
                u = float(self._rng.uniform(0.0, 1.0))
                accept = (u <= np.exp(base_lp - draft_lp))

            if not accept:
                base_token = _sample_from_lp_dict(row, self._rng)
                break

            # Accepted this drafted token
            accepted += 1

            # EOS inside accepted draft
            if self._eos is not None and tok == self._eos:
                hit_eos = True
                base_token = None  # No fallback when EOS is in accepted draft
                break

        # If we accepted all K and didn't hit EOS, fallback comes from the generated step
        if (accepted == K) and (not hit_eos):
            if gen_row:
                base_token = _sample_from_lp_dict(gen_row, self._rng)
            else:
                # If logprobs missing (unlikely), fall back to vLLM's chosen token
                base_token = int(seq.token_ids[0])

        # ---------------- Commit MLX-style ----------------
        if hit_eos:
            # Accepted up to and including EOS among drafted tokens:
            # prefix += [last] + accepted[:-1], last = accepted[-1] (EOS)
            self._prefix.extend([self._last] + draft_toks[:accepted - 1])
            self._last = draft_toks[accepted - 1]
            return VerifyResponse(accepted_len=accepted, base_token=None, hit_eos=True)

        # Not EOS:
        if base_token is not None:
            # Fallback appended.
            # prefix += [last] + accepted, last = base_token
            self._prefix.extend([self._last] + draft_toks[:accepted])
            self._last = int(base_token)
        else:
            # No fallback (should only happen if K>0 and accepted>0 but hit_eos=True handled above).
            # For completeness, mirror MLX: prefix += [last] + accepted[:-1], last = accepted[-1]
            if accepted > 0:
                self._prefix.extend([self._last] + draft_toks[:accepted - 1])
                self._last = draft_toks[accepted - 1]
            # If accepted == 0 and no base_token, do nothing.

        return VerifyResponse(accepted_len=accepted, base_token=base_token, hit_eos=False)


# -------------------- Server plumbing --------------------
async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, llm: LLM, tok: PreTrainedTokenizer) -> None:
    peer = writer.get_extra_info("peername")
    print(f"client connected: {peer}")
    channel = MessageChannel(reader, writer)
    session = VerifierSession(llm, tok)
    try:
        while True:
            msg = await channel.recv()
            if msg is None:
                print(f"client disconnected: {peer}")
                break

            if isinstance(msg, ResetRequest):
                await session.reset()
            elif isinstance(msg, PrefillRequest):
                await session.prefill([int(x) for x in msg.prompt])
                await channel.send(PrefillResponse(ok=True))
            elif isinstance(msg, VerifyRequest):
                resp = await session.verify(msg)
                await channel.send(resp)
            else:
                raise RuntimeError(f"unhandled message type: {type(msg)!r}")
    finally:
        await channel.close()


async def main() -> None:
    print(f"Loading base model with vLLM: {BASE_MODEL}")
    llm = LLM(model=BASE_MODEL, trust_remote_code=True, enforce_eager=True, max_seq_len_to_capture=0)
    tok: PreTrainedTokenizer = llm.get_tokenizer()

    server = await asyncio.start_server(lambda r, w: handle_client(r, w, llm, tok), HOST, PORT)
    addrs = ", ".join(str(sock.getsockname()) for sock in server.sockets)
    print(f"Verifier (vLLM) listening on {addrs}")
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
