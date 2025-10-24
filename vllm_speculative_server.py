# vllm_speculative_server.py
# pip install vllm transformers
import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional

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
# Use the same model your MLX base verifier used (3B Instruct),
# or point to any HF path or local snapshot.
# BASE_MODEL = "meta-llama/Meta-Llama-3.2-3B-Instruct"
BASE_MODEL = "/root/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95/"
HOST = "0.0.0.0"
PORT = 7070

SEED = 90               # deterministic sampling on the base
BASE_TEMP = 1.0         # base sampling temperature
DEFAULT_TOPK = 20       # base top-k if client sends K=0 or we can't infer it


# ------------- Utility: stable log-softmax -------------
def _log_softmax_topk_row(logits_row: List[float]) -> np.ndarray:
    """Approximate draft log-probs over its top-k subset (for acceptance math)."""
    x = np.array(logits_row, dtype=np.float64)
    m = np.max(x)
    z = np.exp(x - m).sum()
    return x - m - np.log(z)


@dataclass
class _BaseStep:
    """Holds base's per-step result for convenience."""
    next_token: int
    topk: Dict[int, object]  # vLLM Logprob objects: have .logprob and .decoded_token (optional)


class VerifierSession:
    """
    Minimal session state for the vLLM base verifier.
    Matches the behavior of your MLX verifier, but we compute verification
    step-by-step using vLLM's one-token generation with logprobs.
    """

    def __init__(self, llm: LLM, tok: PreTrainedTokenizer):
        self.llm = llm
        self.tok = tok
        self.eos: Optional[int] = getattr(tok, "eos_token_id", None)
        self._context: List[int] = []      # committed context: full prompt + accepted/fallback tokens

    async def reset(self) -> None:
        self._context = []

    async def prefill(self, prompt: List[int]) -> None:
        # We simply store the full prompt. vLLM's prefix caching will help later.
        if not prompt:
            raise ValueError("empty prompt")
        self._context = [int(t) for t in prompt]

    # ----------- vLLM one-step helper -----------
    async def _base_step(self, k: int) -> _BaseStep:
        """
        Query base distribution for the *next* token given current context.
        Returns base-chosen token (per vLLM sampler) and a dict of top-k logprobs.
        """

         sp = SamplingParams(
            max_tokens=1,
            logprobs=int(20),
            top_p=1.0,
            seed=9,
            detokenize=False,
        )
       sp = SamplingParams(
            max_tokens=1,
            logprobs=int(20),
            top_p=1.0,
            seed=SEED,
            detokenize=False,
        )

        sp = SamplingParams(
            max_tokens=1,
            logprobs=int(k),
            temperature=float(BASE_TEMP),
            top_p=1.0,
            seed=SEED,
            detokenize=False,
        )
        # Run synchronously inside a thread to avoid blocking the event loop
        loop = asyncio.get_running_loop()
        out = await loop.run_in_executor(
            None,
            lambda: self.llm.generate([TokensPrompt(prompt_token_ids=self._context)], sp),
        )
        seq_out = out[0].outputs[0]
        next_token = int(seq_out.token_ids[0])
        # Dict[int, Logprob] for the single generated position
        logprobs_dict: Dict[int, object] = seq_out.logprobs[0]
        return _BaseStep(next_token=next_token, topk=logprobs_dict)

    async def verify(self, req: VerifyRequest) -> VerifyResponse:
        """
        Implements the same verify contract as your MLX server:
          - Accept as many drafted tokens as possible using the speculative criteria.
          - If we didn't hit EOS inside the accepted draft, append base fallback token.
          - Update committed context accordingly.
        """
        draft_toks = [int(t) for t in req.draft_toks]
        d_topk_idx = [list(map(int, row)) for row in req.draft_topk_idx]
        d_topk_vals = [[float(v) for v in row] for row in req.draft_topk_vals]

        # Infer base top-k from the draft's rows; fall back if K=0
        base_k = int(d_topk_idx[0] and len(d_topk_idx[0]) or DEFAULT_TOPK) if d_topk_idx else DEFAULT_TOPK

        accepted = 0
        hit_eos = False
        base_token: Optional[int] = None

        # Work on a local buffer; only commit back to self._context at the end
        ctx = list(self._context)

        # Step through each drafted token
        for i, tok in enumerate(draft_toks):
            # 1) Ask base for distribution at the current context
            step = await self._base_step(k=base_k)

            # 2) Get base log-prob of the drafted token (if in base top-k)
            base_lp = float(step.topk[tok].logprob) if tok in step.topk else float("-inf")

            # 3) Get draft "log-prob" for that token using the draft's top-k row (approx via log-softmax on subset)
            d_ids = d_topk_idx[i] if i < len(d_topk_idx) else []
            d_vals = d_topk_vals[i] if i < len(d_topk_vals) else []
            try:
                j = d_ids.index(tok)
            except ValueError:
                # The draft's sampled token should appear exactly once in its own top-k row;
                # if not, treat as immediate reject.
                base_token = step.next_token
                break

            draft_lps = _log_softmax_topk_row(d_vals)
            draft_lp = float(draft_lps[j])

            # 4) Speculative acceptance test (probability form)
            # Accept if base_lp >= draft_lp; otherwise accept with probability exp(base_lp - draft_lp).
            accept = False
            if base_lp == float("-inf"):
                accept = False
            elif base_lp >= draft_lp:
                accept = True
            else:
                u = float(np.random.uniform(0.0, 1.0))
                accept = (u <= np.exp(base_lp - draft_lp))

            if not accept:
                # Reject here. Append base fallback (the base's one-step choice for this position).
                base_token = step.next_token
                break

            # Accepted the draft token; advance context
            accepted += 1
            ctx.append(tok)

            # EOS handling
            if self.eos is not None and tok == self.eos:
                hit_eos = True
                base_token = None   # by contract: no base fallback appended when EOS is hit in accepted draft
                break

        # If we accepted ALL drafted tokens and did not hit EOS,
        # append the base fallback one more step ahead.
        if (accepted == len(draft_toks)) and (not hit_eos):
            step = await self._base_step(k=base_k)
            base_token = step.next_token
            ctx.append(base_token)

        # Commit context back
        self._context = ctx

        return VerifyResponse(
            accepted_len=int(accepted),
            base_token=base_token,
            hit_eos=bool(hit_eos),
        )


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
    print(f"Loading base model with vLLM: {BASE_MODEL}")
    llm = LLM(model=BASE_MODEL, trust_remote_code=True)  # adjust dtype/gpu_memory_utilization as needed
    tok: PreTrainedTokenizer = llm.get_tokenizer()

    server = await asyncio.start_server(lambda r, w: handle_client(r, w, llm, tok), HOST, PORT)
    addrs = ", ".join(str(sock.getsockname()) for sock in server.sockets)
    print(f"Verifier (vLLM) listening on {addrs}")
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())

