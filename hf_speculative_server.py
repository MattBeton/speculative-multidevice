# hf_speculative_server.py
# Hugging Face Transformers-based speculative decoding verifier.
# - Does a single forward over [last] + K drafted tokens per verification round (K+1 positions).
# - Interoperates with your existing MessageChannel / VerifyRequest / VerifyResponse.
#
# Requirements:
#   pip install "transformers>=4.40" torch --upgrade
#   (plus a CUDA build of torch if you want GPU)
#
# Notes:
# - Mirrors the acceptance math in your MLX server (uses raw logits, not logprobs),
#   so behavior matches your current draft client.
# - Maintains KV cache across requests; on each verify pass, we forward with
#   past_kv + [last] + K drafted tokens, then slice the returned KV to keep
#   only the committed portion (accepted + optional base fallback), *excluding*
#   the previous 'last' (as in your MLX server).
# - The base fallback token is sampled from the base model's top-k distribution
#   at the position m (the first rejected step), same as your MLX server.

import asyncio
import os
import time
from typing import List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from shared import (
    MessageChannel,
    PrefillRequest,
    PrefillResponse,
    ResetRequest,
    VerifyRequest,
    VerifyResponse,
)

# ------------------------- Configuration -------------------------
# Use the same base model your MLX server/client expects.
# You can point this to a local snapshot or an HF hub id.
BASE_MODEL = os.environ.get(
    "HF_BASE_MODEL",
    # Example local snapshot path (change to fit your setup)
    "/root/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95/",
    # Or e.g.: "meta-llama/Meta-Llama-3.2-3B-Instruct"
)

HOST = os.environ.get("HF_SPEC_HOST", "0.0.0.0")
PORT = int(os.environ.get("HF_SPEC_PORT", "7070"))

SEED = 90
TOP_K = 20  # must match the drafter's TOP_K so acceptance math lines up

# Device / dtype selection
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    # Prefer bfloat16 if supported, else float16
    if torch.cuda.is_bf16_supported():
        DTYPE = torch.bfloat16
    else:
        DTYPE = torch.float16
    torch.backends.cuda.matmul.allow_tf32 = True
else:
    DEVICE = torch.device("cpu")
    DTYPE = torch.float32

GENERATORS = {
    "cpu": torch.Generator(device="cpu").manual_seed(SEED),
    "cuda": torch.Generator(device="cuda").manual_seed(SEED) if DEVICE.type == "cuda" else None,
}


# ------------------------- Helpers -------------------------
def _attention_mask(past_kv: Optional[Tuple], add_len: int) -> torch.Tensor:
    """
    Build a full 1s attention mask of shape (1, past_len + add_len).
    """
    if past_kv is None:
        past_len = 0
    else:
        # past_kv is tuple(layers); each layer has (k, v) with shape (B, H, S, D)
        k0 = past_kv[0][0]
        past_len = k0.shape[-2]
    total = past_len + add_len
    return torch.ones((1, total), dtype=torch.long, device=DEVICE)


@torch.no_grad()
def _topk_sample_rows(
    logits: torch.Tensor, k: int, gen: torch.Generator
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Given logits of shape (T, V), returns:
      sampled:  (T,)    token ids sampled from per-row top-k
      topk_idx: (T, k)  ids of top-k per row
      topk_val: (T, k)  logits of top-k per row
    Sampling is over the top-k slice per row using softmax.
    """
    # Compute top-k along vocab dimension
    topk_val, topk_idx = torch.topk(logits, k, dim=-1)  # (T, k)
    probs = torch.softmax(topk_val, dim=-1)              # (T, k)
    # torch.multinomial expects probs >= 0 and sums to 1 per row
    # One sample per row
    choice = torch.multinomial(probs, num_samples=1, generator=gen).squeeze(-1)  # (T,)
    sampled = topk_idx.gather(-1, choice.unsqueeze(-1)).squeeze(-1)              # (T,)
    return sampled, topk_idx, topk_val


# ------------------------- Verifier Session -------------------------
class HFVerifierSession:
    """
    Minimal per-client session for speculative verification using Transformers.
    Matches your MLX server API/behavior:
      - prefill(prompt[:-1]) fills the KV cache; we keep prompt[-1] as _last.
      - verify(draft_toks, draft_topk_idx, draft_topk_vals) does a single forward
        over [last] + draft_toks and accepts as many as possible using your
        logits-based acceptance test; if EOS not hit, append base fallback at m.
    """

    def __init__(self, model: AutoModelForCausalLM, tok: AutoTokenizer):
        self.model = model
        self.tok = tok
        self.eos: Optional[int] = getattr(tok, "eos_token_id", None)

        self._past = None          # past_key_values for committed context
        self._last: Optional[int] = None  # last token not included in _past (prompt[-1])

        # Per-session RNG to make fallback sampling deterministic
        self._gen = GENERATORS["cuda" if DEVICE.type == "cuda" else "cpu"]

        # Timing statistics
        self.verify_iterations = 0
        self.total_verify_time = 0.0

    async def reset(self) -> None:
        self._past = None
        self._last = None
        # Reset timing stats
        self.verify_iterations = 0
        self.total_verify_time = 0.0

    @torch.no_grad()
    async def prefill(self, prompt: List[int]) -> None:
        if not prompt or len(prompt) < 1:
            raise ValueError("empty prompt")
        prefix = torch.tensor(prompt[:-1], dtype=torch.long, device=DEVICE).unsqueeze(0)  # (1, L-1)
        outputs = self.model(prefix, use_cache=True)  # fill KV with prefix only
        self._past = outputs.past_key_values
        self._last = int(prompt[-1])

    @torch.no_grad()
    async def verify(self, req: VerifyRequest) -> VerifyResponse:
        verify_start = time.perf_counter()

        if self._last is None:
            raise RuntimeError("verify called before prefill")

        draft_toks = [int(t) for t in req.draft_toks]
        spec_k = len(draft_toks)

        # 1) Base forward over [last] + draft_toks in one call
        toks_verify = torch.tensor([self._last] + draft_toks, dtype=torch.long, device=DEVICE).unsqueeze(0)  # (1, K+1)
        attn = _attention_mask(self._past, add_len=(spec_k + 1))
        outputs = self.model(
            toks_verify,
            use_cache=True,
            past_key_values=self._past,
            attention_mask=attn,
        )

        # logits: (1, K+1, V) -> (K+1, V)
        logits = outputs.logits.squeeze(0).to(torch.float32)
        # Sample base tokens, and obtain per-row top-k for acceptance math
        base_toks, base_topk_idx, base_topk_val = _topk_sample_rows(
            logits, k=TOP_K, gen=self._gen
        )  # shapes: (K+1,), (K+1, k), (K+1, k)

        # 2) Acceptance loop (mirror your MLX server: raw logits-based test)
        accepted = 0
        hit_eos = False

        # Convert draft top-k rows to tensors for indexing convenience
        d_topk_idx = torch.tensor(req.draft_topk_idx, dtype=torch.long) if req.draft_topk_idx else torch.empty(0, 0, dtype=torch.long)
        d_topk_val = torch.tensor(req.draft_topk_vals, dtype=torch.float32) if req.draft_topk_vals else torch.empty(0, 0, dtype=torch.float32)

        for i, tok in enumerate(draft_toks):
            # Draft row i: find the sampled token in draft top-k to get draft_logit
            if d_topk_idx.numel() == 0:
                # If client didn't send top-k rows, we must reject immediately
                break
            row_ids = d_topk_idx[i]         # (k,)
            row_vals = d_topk_val[i]        # (k,)
            # Should contain sampled token exactly once
            mask = (row_ids == tok)
            if int(mask.sum().item()) != 1:
                # Invalid draft rows (should not happen with your client)
                break
            draft_logit = float(row_vals[mask][0])

            # Base row i: does base have this token in its top-k?
            b_ids = base_topk_idx[i]        # (k,)
            b_vals = base_topk_val[i]       # (k,)
            b_mask = (b_ids == tok)
            in_base_topk = bool(b_mask.any().item())
            base_logit = float(b_vals[b_mask][0]) if in_base_topk else float("-inf")

            # Your MLX acceptance math (logits, not logprobs)
            if base_logit == float("-inf"):
                break
            elif draft_logit <= base_logit:
                accepted += 1
            else:
                u = float(torch.rand((), device=DEVICE, generator=self._gen).item())
                # NOTE: This is identical to your MLX server; it uses raw logit ratio.
                # It is not the theoretical p_base/p_draft test, but we are matching your code exactly.
                if u <= (base_logit / draft_logit):
                    accepted += 1
                else:
                    break

            if (self.eos is not None) and (tok == self.eos):
                hit_eos = True
                break

        # 3) Determine base fallback (position m) if EOS not hit
        base_token: Optional[int] = None
        base_appended = 0
        if not hit_eos:
            base_token = int(base_toks[accepted].item())
            base_appended = 1

        # 4) Commit the cache correctly using a short follow-up forward.
        #    We consumed (spec_k+1) tokens in the verify pass; we only COMMIT
        #    the first (accepted + base_appended) of those (excluding the old 'last').
        keep_extra = accepted + base_appended
        if keep_extra > 0:
            commit_ids = toks_verify[:, :keep_extra]   # shape (1, keep_extra)
            commit_attn = _attention_mask(self._past, add_len=keep_extra)
            commit_out = self.model(
                commit_ids,
                use_cache=True,
                past_key_values=self._past,
                attention_mask=commit_attn,
            )
            # Keep Hugging Face's Cache object intact
            self._past = commit_out.past_key_values

        # 5) Advance "last" pointer (used as the first token in the next verify pass)
        if base_appended == 1:
            self._last = base_token
        elif accepted > 0:
            self._last = int(draft_toks[accepted - 1])
        # else: no tokens committed → _last remains unchanged (rare edge case)
        
        verify_end = time.perf_counter()
        verify_time = verify_end - verify_start
        self.total_verify_time += verify_time
        self.verify_iterations += 1

        print(f"[SERVER TIMING] Verify iteration {self.verify_iterations}: forward pass took {verify_time:.4f}s")

        return VerifyResponse(accepted_len=int(accepted), base_token=base_token, hit_eos=bool(hit_eos))


# ------------------------- Server plumbing -------------------------
async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, model, tok) -> None:
    peer = writer.get_extra_info("peername")
    print(f"client connected: {peer}")
    channel = MessageChannel(reader, writer)
    session = HFVerifierSession(model, tok)
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
        # Print timing summary
        if session.verify_iterations > 0:
            print(f"\n[SERVER TIMING SUMMARY]")
            print(f"  Total verify iterations: {session.verify_iterations}")
            print(f"  Total verify forward pass time: {session.total_verify_time:.4f}s")
            print(f"  Average time per verify: {session.total_verify_time/session.verify_iterations:.4f}s")
        await channel.close()


async def main() -> None:
    print(f"Loading base model with Transformers: {BASE_MODEL}")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=DTYPE,
        device_map=None,  # keep single process; move to one device below
        low_cpu_mem_usage=True,
    ).to(DEVICE)
    model.eval()
    torch.manual_seed(SEED)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    server = await asyncio.start_server(
        lambda r, w: handle_client(r, w, model, tok), HOST, PORT
    )
    addrs = ", ".join(str(sock.getsockname()) for sock in server.sockets)
    print(f"HF Verifier listening on {addrs} (dtype={DTYPE}, device={DEVICE})")
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())

