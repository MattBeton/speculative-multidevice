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
BASE_MODEL = "/root/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95/"
HOST = "0.0.0.0"
PORT = 7070

SEED = 90
BASE_TEMP = 1.0
BASE_TOPK = 20   # top-k for both prompt_logprobs and fallback logprobs

# If your draft client still sends top-k *logits* (as in your MLX code), keep this True.
# If you switch the client to send top-k *logprobs*, set this to False.
DRAFT_VALS_ARE_LOGITS = True


# ----------------- Utilities -------------------
def _logsumexp(x: np.ndarray) -> float:
    m = np.max(x)
    return float(m + np.log(np.exp(x - m).sum()))

def _to_logprobs_from_topk_vals(vals: List[float]) -> np.ndarray:
    """Convert a top-k row of either logits or logprobs to (approx) logprobs over the k tokens."""
    arr = np.array(vals, dtype=np.float64)
    if not DRAFT_VALS_ARE_LOGITS:
        # Assume already logprobs of the top-k subset; just ensure it's finite.
        return arr
    # Treat as logits → apply log-softmax over the subset
    lse = _logsumexp(arr)
    return arr - lse

def _sample_from_logprob_dict(lp_dict: Dict[int, object], rng: np.random.Generator) -> int:
    """Sample a token ID from a vLLM logprob dict (token_id -> Logprob(dataclass with .logprob))."""
    toks, lps = [], []
    for tid, obj in lp_dict.items():
        toks.append(int(tid))
        lps.append(float(obj.logprob))
    toks = np.array(toks, dtype=np.int64)
    lps = np.array(lps, dtype=np.float64)
    # Normalize over subset
    m = np.max(lps)
    p = np.exp(lps - m)
    p /= p.sum()
    idx = int(rng.choice(len(toks), p=p))
    return int(toks[idx])

def _get_prompt_logprobs(request_output) -> List[Optional[Dict[int, object]]]:
    """vLLM attaches prompt_logprobs on the RequestOutput (version-dependent)."""
    # Prefer request-level attribute (current vLLM behavior)
    # TODO: Do we need to add the final logprobs as well?
    return getattr(request_output.outputs[0], "prompt_logprobs", None)


# ----------------- Session ---------------------
@dataclass
class _VerifyOutcome:
    accepted_len: int
    base_token: Optional[int]
    hit_eos: bool
    new_context: List[int]

class VerifierSession:
    """
    vLLM-based verifier that matches your MLX server interface, but verifies K draft tokens
    in *one* call using `prompt_logprobs` and uses `logprobs` for the fallback.
    """
    def __init__(self, llm: LLM, tok: PreTrainedTokenizer):
        self.llm = llm
        self.tok = tok
        self.eos: Optional[int] = getattr(tok, "eos_token_id", None)
        self._ctx: List[int] = []  # committed context
        self.rng = np.random.default_rng(SEED)

    async def reset(self) -> None:
        self._ctx = []

    async def prefill(self, prompt: List[int]) -> None:
        if not prompt:
            raise ValueError("empty prompt")
        # Store full prompt as committed context; vLLM will prefix-cache internally.
        self._ctx = [int(t) for t in prompt]

    async def verify(self, req: VerifyRequest) -> VerifyResponse:
        draft_toks = [int(t) for t in req.draft_toks]
        if len(draft_toks) == 0:
            # K = 0: ask base for one token (pure decoding)
            outcome = await self._verify_k0()
            self._ctx = outcome.new_context
            return VerifyResponse(
                accepted_len=outcome.accepted_len,
                base_token=outcome.base_token,
                hit_eos=outcome.hit_eos,
            )

        # 1) Single vLLM call with teacher-forced prompt = context + draft_toks
        prompt_ids = self._ctx + draft_toks
        sp = SamplingParams(
            max_tokens=1,
            prompt_logprobs=int(BASE_TOPK),
            logprobs=int(BASE_TOPK),
            temperature=float(BASE_TEMP),
            top_p=1.0,
            seed=SEED,
            detokenize=False,
        )
        loop = asyncio.get_running_loop()
        req_out = (await loop.run_in_executor(
            None,
            lambda: self.llm.generate([TokensPrompt(prompt_token_ids=prompt_ids)], sp),
        ))[0]

        # 2) Pull prompt_logprobs; slice to the last K (the drafted tokens we appended)
        full_prompt_lp = _get_prompt_logprobs(req_out)
        if full_prompt_lp is None:
            raise RuntimeError("vLLM did not return prompt_logprobs (ensure prompt_logprobs>0).")

        K = len(draft_toks)
        if len(full_prompt_lp) < len(prompt_ids):
            # Some versions may not provide the first token's logprobs; align by tail slice.
            tail = full_prompt_lp[-K:]
        else:
            tail = full_prompt_lp[-K:]  # last K correspond to drafted positions

        # 3) Iterate left→right applying the speculative accept rule
        accepted = 0
        hit_eos = False
        new_ctx = list(self._ctx)

        for i, tok in enumerate(draft_toks):
            base_row = tail[i] or {}
            # Base logprob for the drafted token at this position (or -inf if not in top-k)
            base_lp = float(base_row[tok].logprob) if tok in base_row else float("-inf")

            # Draft logprob for that token: derive from MLX top-k vals (logits → logprobs if needed)
            d_vals = req.draft_topk_vals[i]
            d_ids = req.draft_topk_idx[i]
            # Find the drafted token within the draft's own row
            try:
                j = d_ids.index(tok)
            except ValueError:
                # Draft must include its sampled token exactly once; treat as reject if not found
                # Fallback: sample from base distribution at this position (from base_row)
                fallback = _sample_from_logprob_dict(base_row, self.rng) if base_row else None
                if fallback is None:
                    # Extremely unlikely; give up without appending
                    return VerifyResponse(accepted_len=accepted, base_token=None, hit_eos=False)
                new_ctx.append(fallback)
                return VerifyResponse(accepted_len=accepted, base_token=int(fallback), hit_eos=False)

            draft_logps = _to_logprobs_from_topk_vals(d_vals)
            draft_lp = float(draft_logps[j])

            # Accept if base_lp >= draft_lp; else accept w.p. exp(base_lp - draft_lp)
            if base_lp == float("-inf"):
                # Reject
                fallback = _sample_from_logprob_dict(base_row, self.rng) if base_row else None
                if fallback is None:
                    return VerifyResponse(accepted_len=accepted, base_token=None, hit_eos=False)
                new_ctx.append(fallback)
                return VerifyResponse(accepted_len=accepted, base_token=int(fallback), hit_eos=False)

            if base_lp >= draft_lp:
                accept = True
            else:
                u = float(self.rng.uniform(0.0, 1.0))
                accept = (u <= np.exp(base_lp - draft_lp))

            if not accept:
                # Reject here at position i ⇒ sample fallback from *this position's* base distribution.
                fallback = _sample_from_logprob_dict(base_row, self.rng)
                new_ctx.append(fallback)
                return VerifyResponse(accepted_len=accepted, base_token=int(fallback), hit_eos=False)

            # Accepted this drafted token
            accepted += 1
            new_ctx.append(tok)

            # EOS handling
            if self.eos is not None and tok == self.eos:
                hit_eos = True
                # By contract: no base fallback when EOS is among accepted draft tokens
                self._ctx = new_ctx
                return VerifyResponse(accepted_len=accepted, base_token=None, hit_eos=True)

        # 4) If we made it here, we accepted all K and didn’t hit EOS → append base fallback
        # Use the generated step’s logprobs for sampling the fallback (position K)
        seq = req_out.outputs[0]
        gen_row = seq.logprobs[0] if seq.logprobs else None
        if not gen_row:
            # vLLM returned no generated logprobs (shouldn't happen with logprobs>0)
            # Fall back to using the token vLLM sampled
            base_token = int(seq.token_ids[0])
        else:
            base_token = _sample_from_logprob_dict(gen_row, self.rng)

        new_ctx.append(base_token)
        self._ctx = new_ctx
        return VerifyResponse(accepted_len=accepted, base_token=int(base_token), hit_eos=False)

    async def _verify_k0(self) -> _VerifyOutcome:
        """K=0: ask base for one token (pure decoding step)."""
        sp = SamplingParams(
            max_tokens=1,
            logprobs=int(BASE_TOPK),
            temperature=float(BASE_TEMP),
            top_p=1.0,
            seed=SEED,
            detokenize=False,
        )
        loop = asyncio.get_running_loop()
        req_out = (await loop.run_in_executor(
            None,
            lambda: self.llm.generate([TokensPrompt(prompt_token_ids=self._ctx)], sp),
        ))[0]
        seq = req_out.outputs[0]
        # Sample from logprobs to keep the RNG path consistent
        gen_row = seq.logprobs[0] if seq.logprobs else None
        if gen_row:
            base_token = _sample_from_logprob_dict(gen_row, self.rng)
        else:
            base_token = int(seq.token_ids[0])
        new_ctx = self._ctx + [base_token]
        return _VerifyOutcome(accepted_len=0, base_token=int(base_token), hit_eos=False, new_context=new_ctx)


# ----------------- Server plumbing -----------------
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
    llm = LLM(model=BASE_MODEL, trust_remote_code=True)
    tok: PreTrainedTokenizer = llm.get_tokenizer()

    server = await asyncio.start_server(lambda r, w: handle_client(r, w, llm, tok), HOST, PORT)
    addrs = ", ".join(str(sock.getsockname()) for sock in server.sockets)
    print(f"Verifier (vLLM, teacher-forced K) listening on {addrs}")
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
