from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache


from model import GenerationModel
from const import DEVICE, DTYPE, HF_TOKEN, TOP_K, ATTN_IMPL_ENV

def _topk_sample_torch(logits: "torch.Tensor", k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Torch top-k categorical sampler.
    Accepts logits shape (..., V). Returns NumPy arrays:
      sampled (…,),
      topk_idx (…,k),
      topk_vals (…,k)
    """
    V = logits.shape[-1]
    k = min(int(k), int(V))

    # Flatten everything except V
    rows = logits.reshape(-1, V)

    vals, idx = torch.topk(rows, k=k, dim=-1)  # (R,k)
    # categorical over unnormalized logits
    probs = torch.softmax(vals, dim=-1)
    picks = torch.multinomial(probs, num_samples=1)  # (R,1)
    sampled = idx.gather(-1, picks).squeeze(-1)      # (R,)

    # Back to original batch shape
    shp = logits.shape[:-1]
    topk_idx = idx.view(*shp, k).to("cpu", copy=True).numpy().astype(np.int32)
    topk_vals = vals.view(*shp, k).to("cpu", copy=True).numpy().astype(np.float32)
    sampled = sampled.view(*shp).to("cpu", copy=True).numpy().astype(np.int32)
    return sampled, topk_idx, topk_vals

class HFGenerationModel(GenerationModel):
    """
    HuggingFace Transformers-backed model that mirrors the MLX interface.

    - Accepts/returns NumPy arrays only.
    - Maintains a DynamicCache internally.
    - Works best with single-stream use (B=1). Batched use is supported when all
      rows have the same B dimension and you only call `trim_cache(n)`; for
      per-row rollback use `rollback_tokens(r)`.
    """

    def __init__(self, model_id: Union[str, Path]):
        self.tok = AutoTokenizer.from_pretrained(str(model_id), use_fast=True, token=HF_TOKEN)
        from_kwargs = {
            "torch_dtype": DTYPE,
            "device_map": None,
            "low_cpu_mem_usage": True,
            "token": HF_TOKEN,
        }
        if ATTN_IMPL_ENV:
            from_kwargs["attn_implementation"] = ATTN_IMPL_ENV

        self.model = AutoModelForCausalLM.from_pretrained(str(model_id), **from_kwargs).to(DEVICE)
        self.model.eval()

        # Create an empty cache
        try:
            self.cache = DynamicCache()
        except TypeError:
            # Older HF APIs require the config
            self.cache = DynamicCache(config=self.model.config)

    def reset(self) -> None:
        try:
            self.cache = DynamicCache()
        except TypeError:
            self.cache = DynamicCache(config=self.model.config)

    def trim_cache(self, n: int) -> None:
        if n <= 0:
            return
        if self.cache is None or len(self.cache) == 0:
            return

        # Uniform trim across rows: slice off the tail S-n
        dst = DynamicCache()
        for layer in range(len(self.cache)):
            K = self.cache.layers[layer].keys
            V = self.cache.layers[layer].values
            assert K is not None and V is not None
            S = K.shape[2]
            keep = max(S - n, 0)
            K_new = K[:, :, :keep, :].contiguous()
            V_new = V[:, :, :keep, :].contiguous()
            dst.update(K_new, V_new, layer)
        self.cache = dst

    def rollback_tokens(self, r: list[int]) -> None:
        """
        Per-row rollback (right-aligned) for batched caches.
        Matches the semantics used in utils_hf.rollback_dynamic_per_row_simple.
        """
        if not r or self.cache is None or len(self.cache) == 0:
            return

        B = self.cache.layers[0].keys.shape[0]
        if len(r) != B:
            raise ValueError(f"rollback length mismatch: got {len(r)} but cache has B={B}")
        if any(x < 0 for x in r):
            raise ValueError("rollback values must be >= 0")

        dst = DynamicCache()
        for layer in range(len(self.cache)):
            K = self.cache.layers[layer].keys
            V = self.cache.layers[layer].values
            assert K is not None and V is not None

            _, H, S, D = K.shape
            K_new = K.new_zeros((B, H, S, D))
            V_new = V.new_zeros((B, H, S, D))

            for i in range(B):
                keep = max(S - int(r[i]), 0)
                if keep <= 0:
                    continue
                # Surviving tokens are the earliest .. latest-rollback
                K_src = K[i, :, :keep, :]
                V_src = V[i, :, :keep, :]
                # Right-align in the destination
                start = S - keep
                K_new[i, :, start:, :] = K_src
                V_new[i, :, start:, :] = V_src

            dst.update(K_new, V_new, layer)

        self.cache = dst

    def forward(self, tokens: np.ndarray, only_final: bool = True
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if tokens.ndim == 1:
            tokens = tokens[None, :]
        x = torch.as_tensor(tokens, dtype=torch.long, device=DEVICE)

        with torch.inference_mode():
            out = self.model(input_ids=x, past_key_values=self.cache, use_cache=True)

        logits = out.logits  # (B,T,V)
        if only_final:
            logits = logits[:, -1, :]  # (B,V)

        # torch top-k → NumPy
        sampled, topk_idx, topk_vals = _topk_sample_torch(logits, TOP_K)

        # Optional: sync CUDA to make timings comparable to MLX measurements
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

        return sampled, topk_idx, topk_vals

    def tokenize(self, prompt: str) -> np.ndarray:
        # Prefer chat template for chat-tuned models; fall back to plain encode
        if hasattr(self.tok, "apply_chat_template"):
            ids = self.tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=True,
            )
        else:
            ids = self.tok.encode(prompt)
        return np.asarray(ids, dtype=np.int32)

    def decode(self, generated: list[int]) -> str:
        return self.tok.decode(generated)

    def eos_token_id(self) -> Optional[int]:
        return getattr(self.tok, "eos_token_id", None)

    def pad_id(self) -> Optional[int]:
        pid = getattr(self.tok, "pad_token_id", None)
        if pid is None:
            pid = getattr(self.tok, "eos_token_id", None)
        return pid

