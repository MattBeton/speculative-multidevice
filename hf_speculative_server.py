# hf_speculative_server.py
# Hugging Face Transformers-based speculative decoding verifier (batched).
# - Single forward per K-group across streams using left-padded KV caches.
# - Commits by slicing returned past_key_values (no extra forward).
# - Matches MLX logits-based acceptance math.
#
# Requirements:
#   pip install "transformers>=4.40" torch --upgrade
#   (plus a CUDA build of torch if you want GPU; optional FlashAttention 2)
#
# Optional env:
#   HF_BASE_MODEL=meta-llama/Llama-3.2-3B-Instruct
#   HF_SPEC_HOST=0.0.0.0
#   HF_SPEC_PORT=7070
#   HF_TOP_K=20
#   HF_ATTN_IMPL=flash_attention_2   # if installed
#   HF_TOKEN=hf_xxx                  # if the model is gated
#
# This file preserves the same public API defined in `shared.py`.

import asyncio
import os
import time
import random
from typing import Dict, List, Optional, Tuple, Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from shared import (
    MessageChannel,
    PrefillRequest,
    PrefillResponse,
    PrefillBatchRequest,
    PrefillBatchResponse,
    ResetRequest,
    VerifyRequest,
    VerifyResponse,
    VerifyBatchRequest,
    VerifyBatchResponse,
    VerifyResponseItem,
)

# ------------------------- Configuration -------------------------

BASE_MODEL = os.environ.get("HF_BASE_MODEL", "meta-llama/Llama-3.2-3B-Instruct")
HOST = os.environ.get("HF_SPEC_HOST", "0.0.0.0")
PORT = int(os.environ.get("HF_SPEC_PORT", "7070"))

SEED = 90
TOP_K = int(os.environ.get("HF_TOP_K", "20"))

ATTN_IMPL_ENV = os.environ.get("HF_ATTN_IMPL", "").strip()  # e.g., "flash_attention_2" if available

# Device / dtype selection
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    if torch.cuda.is_bf16_supported():
        DTYPE = torch.bfloat16
    else:
        DTYPE = torch.float16
    torch.backends.cuda.matmul.allow_tf32 = True
else:
    DEVICE = torch.device("cpu")
    DTYPE = torch.float32

# Deterministic generators
GENERATOR = {
    "cpu": torch.Generator(device="cpu").manual_seed(SEED),
    "cuda": torch.Generator(device="cuda").manual_seed(SEED) if DEVICE.type == "cuda" else None,
}
PY_RNG = random.Random(SEED)


# ------------------------- Helpers -------------------------

def _past_seq_len(past: Optional[Tuple]) -> int:
    """Return current sequence length from a HF past_key_values tuple."""
    if past is None:
        return 0
    # past[0] -> (k, v), k shape: (B, H, S, D)
    return int(past[0][0].shape[-2])


def _attention_mask_single(past: Optional[Tuple], add_len: int) -> torch.Tensor:
    """Build single-stream full-ones attention mask of shape (1, past_len + add_len)."""
    total = _past_seq_len(past) + add_len
    return torch.ones((1, total), dtype=torch.long, device=DEVICE)


def _stack_left_padded_past(pasts: List[Optional[Tuple]]) -> Tuple[Tuple[Tuple[torch.Tensor, torch.Tensor], ...], List[int], int]:
    """
    Left-pad and stack a list of per-stream past_key_values into a batched past.
    Returns:
      batch_past: tuple over layers, each (K_batch, V_batch) with shape (B, H, max_p, D)
      lens:       list of original past lengths per stream
      max_p:      maximum past length across streams
    """
    assert any(p is not None for p in pasts), "all streams have empty past; prefill required"
    # Use first non-None past to infer num_layers/H/D
    ref = next(p for p in pasts if p is not None)
    num_layers = len(ref)
    B = len(pasts)

    lens = [_past_seq_len(p) for p in pasts]
    max_p = max(lens)

    batch_layers = []
    for layer in range(num_layers):
        k_rows, v_rows = [], []
        for p in pasts:
            if p is None:
                # Allocate zeros with (1, H, max_p, D) using ref dims
                k_ref, v_ref = ref[layer]
                H, D = k_ref.shape[1], k_ref.shape[-1]
                k_pad = torch.zeros((1, H, max_p, D), dtype=k_ref.dtype, device=k_ref.device)
                v_pad = torch.zeros_like(k_pad)
            else:
                k_i, v_i = p[layer]  # (1, H, L, D)
                L = k_i.shape[-2]
                if L == max_p:
                    k_pad, v_pad = k_i, v_i
                else:
                    H, D = k_i.shape[1], k_i.shape[-1]
                    k_pad = torch.zeros((1, H, max_p, D), dtype=k_i.dtype, device=k_i.device)
                    v_pad = torch.zeros_like(k_pad)
                    # Left-pad: place existing sequence at the right end
                    k_pad[:, :, max_p - L:, :] = k_i
                    v_pad[:, :, max_p - L:, :] = v_i
            k_rows.append(k_pad)
            v_rows.append(v_pad)
        K_batch = torch.cat(k_rows, dim=0)  # (B, H, max_p, D)
        V_batch = torch.cat(v_rows, dim=0)  # (B, H, max_p, D)
        batch_layers.append((K_batch, V_batch))

    return tuple(batch_layers), lens, max_p


def _build_batched_attention(lens: List[int], max_p: int, new_len: int) -> torch.Tensor:
    """
    Build (B, max_p + new_len) attention with zeros on left pad,
    ones on real past and new tokens.
    """
    B = len(lens)
    attn = torch.zeros((B, max_p + new_len), dtype=torch.long, device=DEVICE)
    for i, L in enumerate(lens):
        attn[i, (max_p - L):(max_p + new_len)] = 1
    return attn


def _sample_from_topk(topk_val: torch.Tensor, topk_idx: torch.Tensor, gen: Optional[torch.Generator]) -> torch.Tensor:
    """
    Given topk_val and topk_idx of shape (B, T, k),
    sample one token per (B, T) row from softmax(topk_val).
    Returns token ids tensor of shape (B, T) on the same device as inputs.
    """
    B, T, k = topk_val.shape
    probs = torch.softmax(topk_val, dim=-1)            # (B, T, k)
    flat_probs = probs.reshape(B * T, k)               # (BT, k)
    choices = torch.multinomial(flat_probs, 1, generator=gen)  # (BT, 1)
    choices = choices.squeeze(-1)                      # (BT,)
    flat_idx = topk_idx.reshape(B * T, k)
    picked = flat_idx.gather(1, choices.unsqueeze(-1)).squeeze(-1)  # (BT,)
    return picked.reshape(B, T)


def _group_by_k(batch: List[Tuple[str, List[int], Tuple[List[List[int]], List[List[float]]]]]):
    """Group batch items by spec_k (len(draft_toks)) to avoid K-padding waste."""
    groups: Dict[int, List[Tuple[str, List[int], Tuple[List[List[int]], List[List[float]]]]]] = {}
    for item in batch:
        k = len(item[1])
        groups.setdefault(k, []).append(item)
    return groups


# ------------------------- Stream State -------------------------

class StreamState:
    """Per-stream state for batch processing."""
    __slots__ = ("past", "last", "eos")

    def __init__(self, eos: Optional[int]):
        self.past = None              # past_key_values for this stream
        self.last: Optional[int] = None
        self.eos: Optional[int] = eos


# ------------------------- Batch Verifier -------------------------

class BatchVerifier:
    """
    Batch-aware verifier using HF Transformers.
    - Batches streams by K and runs one forward per group.
    - Commits by slicing the returned KV (no extra forward).
    """
    def __init__(self, model: AutoModelForCausalLM, tok: AutoTokenizer):
        self.model = model
        self.tok = tok
        self._states: Dict[str, StreamState] = {}
        self._gen = GENERATOR["cuda" if DEVICE.type == "cuda" else "cpu"]
        self._py_rng = PY_RNG
        self._default_id = "_default"

        # Timing statistics
        self.verify_iterations = 0
        self.total_forward_time = 0.0
        self.total_e2e_time = 0.0
        self.total_tokens_processed = 0

    def _get_or_create(self, sid: str) -> StreamState:
        s = self._states.get(sid)
        if s is None:
            s = StreamState(getattr(self.tok, "eos_token_id", None))
            self._states[sid] = s
        return s

    async def reset(self) -> None:
        self._states.clear()
        self.verify_iterations = 0
        self.total_forward_time = 0.0
        self.total_e2e_time = 0.0
        self.total_tokens_processed = 0

    @torch.inference_mode()
    async def prefill_single(self, prompt: List[int], sid: Optional[str] = None) -> None:
        if not prompt:
            raise ValueError("empty prompt")
        sid = sid or self._default_id
        st = self._get_or_create(sid)

        prefix = torch.tensor(prompt[:-1], dtype=torch.long, device=DEVICE).unsqueeze(0)  # (1, L-1)
        outputs = self.model(prefix, use_cache=True)  # fill KV with prefix only
        st.past = outputs.past_key_values
        st.last = int(prompt[-1])

    async def prefill_batch(self, items: List[Tuple[str, List[int]]]) -> None:
        # Sequential is fine here; prefill is done once per stream.
        for sid, prompt in items:
            await self.prefill_single(prompt, sid=sid)

    async def verify_single(
        self,
        draft_toks: List[int],
        draft_rows: Tuple[List[List[int]], List[List[float]]],
        sid: Optional[str] = None,
    ):
        sid = sid or self._default_id
        resp = await self.verify_batch([(sid, draft_toks, draft_rows)])
        item = resp[0]
        return item.accepted_len, item.base_token, item.hit_eos

    @torch.inference_mode()
    async def verify_batch(
        self,
        batch: List[Tuple[str, List[int], Tuple[List[List[int]], List[List[float]]]]]
    ) -> List[VerifyResponseItem]:

        t0_total = time.perf_counter()
        total_forward = 0.0
        total_positions = 0

        results: List[VerifyResponseItem] = []

        # Group by K so we can batch each group tightly
        groups = _group_by_k(batch)

        for spec_k, items in groups.items():
            # Sanity
            if spec_k <= 0:
                # No drafted tokens → nothing to verify; skip
                for sid, _, _ in items:
                    results.append(VerifyResponseItem(stream_id=sid, accepted_len=0, base_token=None, hit_eos=False))
                continue

            states = [self._get_or_create(sid) for sid, _, _ in items]
            for idx, st in enumerate(states):
                if st.last is None:
                    raise RuntimeError(f"verify called before prefill for stream {items[idx][0]!r}")

            B = len(items)
            new_len = spec_k + 1  # [last] + K drafted tokens

            # Build inputs: (B, K+1)
            last_col = torch.tensor([st.last for st in states], device=DEVICE, dtype=torch.long).unsqueeze(1)  # (B,1)
            drafts = torch.stack([torch.tensor(d, device=DEVICE, dtype=torch.long) for _, d, _ in items], dim=0)  # (B,K)
            toks_verify = torch.cat([last_col, drafts], dim=1)  # (B, K+1)

            # Build batched past (left-padded to max length) and attention mask
            pasts = [st.past for st in states]
            batch_past, lens, max_p = _stack_left_padded_past(pasts)
            attn = _build_batched_attention(lens, max_p, new_len)

            # Forward (batched)
            tfw0 = time.perf_counter()
            outputs = self.model(
                toks_verify,
                use_cache=True,
                past_key_values=batch_past,
                attention_mask=attn,
            )
            tfw1 = time.perf_counter()
            forward_time = tfw1 - tfw0
            total_forward += forward_time
            total_positions += B * new_len

            logits = outputs.logits  # (B, K+1, V)

            # Base top-k on GPU
            base_val, base_idx = torch.topk(logits, k=TOP_K, dim=-1)    # (B, K+1, k)
            # Sample base tokens from top-k per row (GPU)
            base_toks = _sample_from_topk(base_val, base_idx, self._gen)  # (B, K+1)

            # Move small things to CPU for acceptance math (to avoid GPU syncs in the loop)
            base_idx_cpu = base_idx.to("cpu")
            base_val_cpu = base_val.to("cpu")
            base_toks_cpu = base_toks.to("cpu")

            # Acceptance per stream
            accepted_list: List[int] = []
            base_token_list: List[Optional[int]] = []
            base_appended_list: List[int] = []
            hit_eos_list: List[bool] = []

            for i, (sid, draft_toks, (d_idx_rows, d_val_rows)) in enumerate(items):
                st = states[i]
                accepted = 0
                hit_eos = False

                if d_idx_rows and d_val_rows:
                    # Convert once to CPU tensors
                    d_idx = torch.tensor(d_idx_rows, dtype=torch.long)        # (K, top_k) on CPU
                    d_val = torch.tensor(d_val_rows, dtype=torch.float32)     # (K, top_k) on CPU
                    for t, tok in enumerate(draft_toks):
                        # Draft row: locate tok in draft top-k
                        row_ids = d_idx[t]
                        row_vals = d_val[t]
                        where = (row_ids == int(tok)).nonzero(as_tuple=False)
                        if where.numel() != 1:
                            break
                        draft_logit = float(row_vals[where[0, 0]])

                        # Base row: does base include tok in its top-k?
                        b_ids = base_idx_cpu[i, t]  # (top_k,)
                        b_vals = base_val_cpu[i, t]
                        pos = (b_ids == int(tok)).nonzero(as_tuple=False)
                        if pos.numel() == 0:
                            break
                        base_logit = float(b_vals[pos[0, 0]])

                        # MLX-style logits-based acceptance
                        if draft_logit <= base_logit:
                            accepted += 1
                        else:
                            # deterministic Python RNG
                            u = self._py_rng.random()
                            if u <= (base_logit / draft_logit):
                                accepted += 1
                            else:
                                break

                        if (st.eos is not None) and (int(tok) == st.eos):
                            hit_eos = True
                            break
                # else: If no draft top-k rows, accept none

                # Base fallback (position m) if EOS not hit
                base_token: Optional[int] = None
                base_appended = 0
                if not hit_eos:
                    base_token = int(base_toks_cpu[i, accepted].item())
                    base_appended = 1

                accepted_list.append(accepted)
                base_token_list.append(base_token)
                base_appended_list.append(base_appended)
                hit_eos_list.append(hit_eos)

            # Commit by slicing returned KV for each stream (drop left pad; keep only committed portion)
            new_kv = outputs.past_key_values  # tuple over layers; each (K,V) is (B, H, max_p + new_len, D)
            for i, st in enumerate(states):
                keep_extra = accepted_list[i] + base_appended_list[i]  # number of tokens to add beyond existing past
                if keep_extra > 0:
                    pad_left = max_p - lens[i]
                    end = max_p + keep_extra  # slice [pad_left : end)
                    committed_layers = []
                    for (k_b, v_b) in new_kv:
                        # k_b/v_b: (B, H, max_p + new_len, D)
                        k_i = k_b[i:i+1, :, pad_left:end, :].contiguous()  # (1, H, L_i + keep_extra, D)
                        v_i = v_b[i:i+1, :, pad_left:end, :].contiguous()
                        committed_layers.append((k_i, v_i))
                    st.past = tuple(committed_layers)

                # Advance "last"
                if base_appended_list[i] == 1:
                    st.last = base_token_list[i]
                elif accepted_list[i] > 0:
                    st.last = int(items[i][1][accepted_list[i] - 1])
                # else: _last unchanged (rare)

            # Collect responses
            for i, (sid, _, _) in enumerate(items):
                results.append(VerifyResponseItem(
                    stream_id=sid,
                    accepted_len=int(accepted_list[i]),
                    base_token=base_token_list[i],
                    hit_eos=bool(hit_eos_list[i]),
                ))

        # Timing aggregation over all groups
        t1_total = time.perf_counter()
        e2e_time = t1_total - t0_total

        # Update running totals
        self.verify_iterations += 1
        self.total_forward_time += total_forward
        self.total_e2e_time += e2e_time
        self.total_tokens_processed += total_positions

        fwd_tps = (total_positions / total_forward) if total_forward > 0 else 0.0
        e2e_tps = (total_positions / e2e_time) if e2e_time > 0 else 0.0

        print(
            f"[BATCH SERVER TIMING] Verify iteration {self.verify_iterations}: "
            f"groups={len(groups)}, "
            f"forward={total_forward:.4f}s, total={e2e_time:.4f}s, "
            f"processed={total_positions} tokens, "
            f"fwd_tps={fwd_tps:.2f}, e2e_tps={e2e_tps:.2f}"
        )

        return results


# ------------------------- Single-Session Verifier -------------------------

class HFVerifierSession:
    """
    Minimal per-client (single-stream) session.
    Uses a single forward over [last] + K and commits by slicing the returned KV.
    """
    def __init__(self, model: AutoModelForCausalLM, tok: AutoTokenizer):
        self.model = model
        self.tok = tok
        self.eos: Optional[int] = getattr(tok, "eos_token_id", None)

        self._past = None
        self._last: Optional[int] = None

        self._gen = GENERATOR["cuda" if DEVICE.type == "cuda" else "cpu"]
        self._py_rng = PY_RNG

        # Basic timing
        self.verify_iterations = 0
        self.total_forward_time = 0.0
        self.total_e2e_time = 0.0
        self.total_tokens_processed = 0

    async def reset(self) -> None:
        self._past = None
        self._last = None
        self.verify_iterations = 0
        self.total_forward_time = 0.0
        self.total_e2e_time = 0.0
        self.total_tokens_processed = 0

    @torch.inference_mode()
    async def prefill(self, prompt: List[int]) -> None:
        if not prompt or len(prompt) < 1:
            raise ValueError("empty prompt")
        prefix = torch.tensor(prompt[:-1], dtype=torch.long, device=DEVICE).unsqueeze(0)  # (1, L-1)
        outputs = self.model(prefix, use_cache=True)  # fill KV with prefix only
        self._past = outputs.past_key_values
        self._last = int(prompt[-1])

    @torch.inference_mode()
    async def verify(self, req: VerifyRequest) -> VerifyResponse:
        t0 = time.perf_counter()

        if self._last is None:
            raise RuntimeError("verify called before prefill")

        draft_toks = [int(t) for t in req.draft_toks]
        K = len(draft_toks)
        new_len = K + 1  # [last] + K

        toks_verify = torch.tensor([self._last] + draft_toks, dtype=torch.long, device=DEVICE).unsqueeze(0)  # (1, K+1)
        attn = _attention_mask_single(self._past, add_len=new_len)

        # Forward once
        tfw0 = time.perf_counter()
        outputs = self.model(
            toks_verify,
            use_cache=True,
            past_key_values=self._past,
            attention_mask=attn,
        )
        tfw1 = time.perf_counter()
        forward_time = tfw1 - tfw0

        logits = outputs.logits.squeeze(0)  # (K+1, V)

        # Base top-k + sampling on GPU
        base_val, base_idx = torch.topk(logits, k=TOP_K, dim=-1)     # (K+1, k)
        probs = torch.softmax(base_val, dim=-1)                       # (K+1, k)
        choice = torch.multinomial(probs, 1, generator=self._gen).squeeze(-1)  # (K+1,)
        base_toks = base_idx.gather(1, choice.unsqueeze(-1)).squeeze(-1)       # (K+1,)

        # Move to CPU for acceptance
        base_idx_cpu = base_idx.to("cpu")
        base_val_cpu = base_val.to("cpu")
        base_toks_cpu = base_toks.to("cpu")

        accepted = 0
        hit_eos = False

        if req.draft_topk_idx and req.draft_topk_vals:
            d_idx = torch.tensor(req.draft_topk_idx, dtype=torch.long)       # (K, k)
            d_val = torch.tensor(req.draft_topk_vals, dtype=torch.float32)   # (K, k)
            for t, tok in enumerate(draft_toks):
                row_ids = d_idx[t]
                row_vals = d_val[t]
                where = (row_ids == int(tok)).nonzero(as_tuple=False)
                if where.numel() != 1:
                    break
                draft_logit = float(row_vals[where[0, 0]])

                b_ids = base_idx_cpu[t]  # (k,)
                b_vals = base_val_cpu[t]
                pos = (b_ids == int(tok)).nonzero(as_tuple=False)
                if pos.numel() == 0:
                    break
                base_logit = float(b_vals[pos[0, 0]])

                if draft_logit <= base_logit:
                    accepted += 1
                else:
                    if self._py_rng.random() <= (base_logit / draft_logit):
                        accepted += 1
                    else:
                        break

                if (self.eos is not None) and (int(tok) == self.eos):
                    hit_eos = True
                    break

        # Base fallback if EOS not hit
        base_token: Optional[int] = None
        base_appended = 0
        if not hit_eos:
            base_token = int(base_toks_cpu[accepted].item())
            base_appended = 1

        # Commit by slicing returned KV (exclude uncommitted verify steps)
        keep_extra = accepted + base_appended
        if keep_extra > 0:
            past_len = _past_seq_len(self._past)
            end = past_len + keep_extra
            committed_layers = []
            for (k_b, v_b) in outputs.past_key_values:
                # k_b/v_b: (1, H, past_len + K+1, D)
                k_i = k_b[:, :, :end, :].contiguous()
                v_i = v_b[:, :, :end, :].contiguous()
                committed_layers.append((k_i, v_i))
            self._past = tuple(committed_layers)

        # Advance last
        if base_appended == 1:
            self._last = base_token
        elif accepted > 0:
            self._last = int(draft_toks[accepted - 1])

        t1 = time.perf_counter()
        e2e = t1 - t0
        self.verify_iterations += 1
        self.total_forward_time += forward_time
        self.total_e2e_time += e2e
        self.total_tokens_processed += (K + 1)

        fwd_tps = ((K + 1) / forward_time) if forward_time > 0 else 0.0
        e2e_tps = ((K + 1) / e2e) if e2e > 0 else 0.0

        print(
            f"[SERVER TIMING] Verify iteration {self.verify_iterations}: "
            f"forward={forward_time:.4f}s, total={e2e:.4f}s, "
            f"processed={K + 1} tokens, fwd_tps={fwd_tps:.2f}, e2e_tps={e2e_tps:.2f}"
        )

        return VerifyResponse(accepted_len=int(accepted), base_token=base_token, hit_eos=bool(hit_eos))


# ------------------------- Server plumbing -------------------------

async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, verifier: BatchVerifier) -> None:
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
        # Print timing summary
        if verifier.verify_iterations > 0:
            avg_fwd = verifier.total_forward_time / verifier.verify_iterations
            avg_e2e = verifier.total_e2e_time / verifier.verify_iterations
            avg_tps_fwd = verifier.total_tokens_processed / verifier.total_forward_time if verifier.total_forward_time > 0 else 0.0
            avg_tps_e2e = verifier.total_tokens_processed / verifier.total_e2e_time if verifier.total_e2e_time > 0 else 0.0
            print(f"\n[BATCH SERVER TIMING SUMMARY]")
            print(f"  Total verify iterations: {verifier.verify_iterations}")
            print(f"  Total tokens processed:  {verifier.total_tokens_processed}")
            print(f"  Total forward time:      {verifier.total_forward_time:.4f}s")
            print(f"  Total end-to-end time:   {verifier.total_e2e_time:.4f}s")
            print(f"  Avg time/verify (fwd):   {avg_fwd:.4f}s")
            print(f"  Avg time/verify (e2e):   {avg_e2e:.4f}s")
            print(f"  Avg tokens/sec (fwd):    {avg_tps_fwd:.2f}")
            print(f"  Avg tokens/sec (e2e):    {avg_tps_e2e:.2f}")
        await channel.close()


async def main() -> None:
    print(f"Loading base model with Transformers: {BASE_MODEL}")
    hf_token = os.environ.get("HF_TOKEN", None)

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, token=hf_token)

    from_kwargs = {
        "torch_dtype": DTYPE,
        "device_map": None,           # keep single process; move to one device below
        "low_cpu_mem_usage": True,
        "token": hf_token,
    }
    # Optional attention implementation (e.g., FlashAttention 2) if requested
    if ATTN_IMPL_ENV:
        from_kwargs["attn_implementation"] = ATTN_IMPL_ENV

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        **from_kwargs,
    ).to(DEVICE)
    model.eval()

    torch.manual_seed(SEED)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    verifier = BatchVerifier(model, tok)

    server = await asyncio.start_server(
        lambda r, w: handle_client(r, w, verifier), HOST, PORT
    )
    addrs = ", ".join(str(sock.getsockname()) for sock in server.sockets)
    print(f"HF Batch Verifier listening on {addrs} (dtype={DTYPE}, device={DEVICE})")
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
