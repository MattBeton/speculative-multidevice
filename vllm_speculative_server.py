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
    PrefillBatchRequest,
    PrefillBatchResponse,
    VerifyRequest,
    VerifyResponse,
    VerifyBatchRequest,
    VerifyBatchResponse,
    VerifyResponseItem,
    ResetRequest,
)

# ---------------- Configuration ----------------
BASE_MODEL = "/root/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95/"
HOST = "0.0.0.0"
PORT = 7070

SEED = 90
BASE_TEMP = 1.0
DEFAULT_TOPK = 20  # used only if we can't infer top-k from the draft row


# ----------------- Utilities -------------------
def _normalize_from_logprobs(logps: np.ndarray) -> np.ndarray:
    m = np.max(logps)
    probs = np.exp(logps - m)
    s = probs.sum()
    return probs / s if s > 0 else np.full_like(probs, 1.0 / probs.size)


def _draft_lp_from_row(
    d_ids: List[int],
    d_vals: List[float],
    tok_id: int,
) -> Tuple[float, int]:
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
    items = [(int(t), float(o.logprob)) for t, o in lp_dict.items()]
    items.sort(key=lambda kv: kv[1], reverse=True)
    ids = np.array([t for t, _ in items], dtype=np.int64)
    lps = np.array([lp for _, lp in items], dtype=np.float64)
    probs = _normalize_from_logprobs(lps)
    idx = int(rng.choice(len(ids), p=probs))
    return int(ids[idx])


def _extract_prompt_rows(req_out, k: int) -> List[Optional[Dict[int, object]]]:
    prompt_rows = getattr(req_out, "prompt_logprobs", None)
    if prompt_rows is None:
        if req_out.outputs:
            prompt_rows = getattr(req_out.outputs[0], "prompt_logprobs", None)
    if prompt_rows is None:
        raise RuntimeError("vLLM did not return prompt_logprobs; ensure prompt_logprobs > 0")
    return prompt_rows[-k:] if k > 0 else []


# ----------------- Multi-Stream Verifier -------------------
class _StreamState:
    __slots__ = ("prefix", "last", "eos")
    def __init__(self, eos: Optional[int]):
        self.prefix: List[int] = []
        self.last: Optional[int] = None
        self.eos: Optional[int] = eos


class Verifier:
    """
    vLLM-based verifier with MLX-identical commit/rollback semantics, now for many
    concurrent streams. Each stream keeps (prefix, last) independently.
    """
    def __init__(self, llm: LLM, tok: PreTrainedTokenizer):
        self.llm = llm
        self.tok = tok
        self._states: Dict[str, _StreamState] = {}
        self._rng = np.random.default_rng(SEED)
        self._default_id = "_default"

    def _get_or_create(self, sid: str) -> _StreamState:
        s = self._states.get(sid)
        if s is None:
            s = _StreamState(getattr(self.tok, "eos_token_id", None))
            self._states[sid] = s
        return s

    async def reset(self) -> None:
        self._states.clear()

    async def prefill_single(self, prompt: List[int], sid: Optional[str] = None) -> None:
        if not prompt:
            raise ValueError("empty prompt")
        sid = sid or self._default_id
        st = self._get_or_create(sid)
        st.prefix = [int(x) for x in prompt[:-1]]
        st.last = int(prompt[-1])

    async def prefill_batch(self, items: List[Tuple[str, List[int]]]) -> None:
        for sid, prompt in items:
            await self.prefill_single(prompt, sid=sid)

    async def verify_single(self, draft_toks: List[int], draft_rows: Tuple[List[List[int]], List[List[float]]], sid: Optional[str] = None):
        """Legacy path wrapper; returns (accepted_len, base_token, hit_eos)."""
        sid = sid or self._default_id
        resp = await self.verify_batch([(sid, draft_toks, draft_rows)])
        item = resp[0]
        return item.accepted_len, item.base_token, item.hit_eos

    async def verify_batch(self, batch: List[Tuple[str, List[int], Tuple[List[List[int]], List[List[float]]]]]) -> List[VerifyResponseItem]:
        """
        batch: list of (stream_id, draft_toks, (draft_topk_idx, draft_topk_vals))
        Returns one VerifyResponseItem per input item, same order.
        """
        # Build prompts
        prompts: List[TokensPrompt] = []
        Ks: List[int] = []
        sids: List[str] = []
        base_topk_candidates: List[int] = []

        # Collect for a single batched vLLM call
        for sid, draft_toks, (d_idx, _d_vals) in batch:
            st = self._get_or_create(sid)
            if st.last is None:
                raise RuntimeError(f"verify called before prefill for stream {sid!r}")
            K = len(draft_toks)
            Ks.append(K)
            sids.append(sid)
            base_topk_candidates.append(len(d_idx[0]) if (K > 0 and len(d_idx) > 0 and len(d_idx[0]) > 0) else 0)
            prompt_ids = st.prefix + [st.last] + draft_toks
            prompts.append(TokensPrompt(prompt_token_ids=prompt_ids))

        max_k = max(Ks) if Ks else 0
        base_topk = max(max(base_topk_candidates), DEFAULT_TOPK)
        sp = SamplingParams(
            max_tokens=1,
            prompt_logprobs=int(base_topk if max_k > 0 else 1),  # keep >=1 for simplicity
            logprobs=int(base_topk),
            temperature=BASE_TEMP,
            top_p=1.0,
            detokenize=False,
            seed=SEED,
        )
        loop = asyncio.get_running_loop()
        outs = await loop.run_in_executor(None, lambda: self.llm.generate(prompts, sp))

        results: List[VerifyResponseItem] = []

        for (sid, draft_toks, (d_idx_rows, d_val_rows)), out, K in zip(batch, outs, Ks):
            st = self._get_or_create(sid)

            if K == 0:
                seq = out.outputs[0]
                gen_row = seq.logprobs[0] if seq.logprobs else None
                base_token = _sample_from_lp_dict(gen_row, self._rng) if gen_row else int(seq.token_ids[0])

                # Commit MLX-style
                st.prefix.append(st.last)
                st.last = int(base_token)
                results.append(VerifyResponseItem(stream_id=sid, accepted_len=0, base_token=int(base_token), hit_eos=False))
                continue

            prompt_rows = _extract_prompt_rows(out, K)
            seq = out.outputs[0]
            gen_row = seq.logprobs[0] if seq.logprobs else None

            accepted = 0
            hit_eos = False
            base_token: Optional[int] = None

            for i, tok in enumerate(draft_toks):
                # Teacher-forced base distribution at this position
                row = prompt_rows[i] or {}
                base_lp = float(row[tok].logprob) if tok in row else float("-inf")

                d_ids = [int(x) for x in d_idx_rows[i]] if i < len(d_idx_rows) else []
                d_vals = [float(x) for x in d_val_rows[i]] if i < len(d_val_rows) else []
                draft_lp, _ = _draft_lp_from_row(d_ids, d_vals, tok)

                if base_lp == float("-inf"):
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

                accepted += 1

                if (st.eos is not None) and (tok == st.eos):
                    hit_eos = True
                    base_token = None
                    break

            if (accepted == K) and (not hit_eos):
                base_token = _sample_from_lp_dict(gen_row, self._rng) if gen_row else int(seq.token_ids[0])

            # Commit MLX-style to stream state
            if hit_eos:
                st.prefix.extend([st.last] + draft_toks[:accepted - 1])
                st.last = draft_toks[accepted - 1]
                results.append(VerifyResponseItem(stream_id=sid, accepted_len=accepted, base_token=None, hit_eos=True))
            else:
                if base_token is not None:
                    st.prefix.extend([st.last] + draft_toks[:accepted])
                    st.last = int(base_token)
                else:
                    if accepted > 0:
                        st.prefix.extend([st.last] + draft_toks[:accepted - 1])
                        st.last = draft_toks[accepted - 1]
                results.append(VerifyResponseItem(stream_id=sid, accepted_len=accepted, base_token=base_token, hit_eos=False))

        return results


# -------------------- Server plumbing --------------------
async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, verifier: Verifier) -> None:
    peer = writer.get_extra_info("peername")
    print(f"client connected: {peer}")
    channel = MessageChannel(reader, writer)
    try:
        while True:
            msg = await channel.recv()
            if msg is None:
                print(f"client disconnected: {peer}")
                break

            if isinstance(msg, ResetRequest):
                await verifier.reset()

            elif isinstance(msg, PrefillRequest):
                await verifier.prefill_single(msg.prompt)
                await channel.send(PrefillResponse(ok=True))

            elif isinstance(msg, VerifyRequest):
                accepted, base_tok, hit = await verifier.verify_single(
                    draft_toks=msg.draft_toks,
                    draft_rows=(msg.draft_topk_idx, msg.draft_topk_vals),
                )
                await channel.send(VerifyResponse(accepted_len=accepted, base_token=base_tok, hit_eos=hit))

            elif isinstance(msg, PrefillBatchRequest):
                await verifier.prefill_batch([(it.stream_id, it.prompt) for it in msg.items])
                await channel.send(PrefillBatchResponse(ok=True, count=len(msg.items)))

            elif isinstance(msg, VerifyBatchRequest):
                res_items = await verifier.verify_batch([
                    (it.stream_id, it.draft_toks, (it.draft_topk_idx, it.draft_topk_vals)) for it in msg.items
                ])
                await channel.send(VerifyBatchResponse(items=res_items))

            else:
                raise RuntimeError(f"unhandled message type: {type(msg)!r}")
    finally:
        await channel.close()


async def main() -> None:
    print(f"Loading base model with vLLM: {BASE_MODEL}")
    llm = LLM(model=BASE_MODEL, trust_remote_code=True, enforce_eager=True, max_seq_len_to_capture=0)
    tok: PreTrainedTokenizer = llm.get_tokenizer()

    verifier = Verifier(llm, tok)
    server = await asyncio.start_server(lambda r, w: handle_client(r, w, verifier), HOST, PORT)
    addrs = ", ".join(str(sock.getsockname()) for sock in server.sockets)
    print(f"Verifier (vLLM) listening on {addrs}")
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
