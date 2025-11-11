from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np

from typing import Optional, Tuple

import mlx.core as mx
from mlx_lm.tokenizer_utils import load_tokenizer
from mlx_lm.utils import load_model
from mlx_lm.models.cache import BatchKVCache

SEED = 90
TOP_K = 20

def topk_sample_numpy(logits: np.array, k: int) -> tuple[int, np.array, np.array]:
    # Keep original shape for final output
    shape = logits.shape
    vocab_size = shape[-1]
    
    # Reshape to 2D: (batch_dims, vocab_size)
    logits_2d = logits.reshape(-1, vocab_size)
    batch_size = logits_2d.shape[0]
    
    # Compute top-k values and indices per row
    topk_vals = np.partition(logits_2d, -k)[:, -k:]  # (batch_size, k)
    topk_idx = np.argpartition(logits_2d, -k)[:, -k:]  # (batch_size, k)
    
    # Mask logits: set values below top-k to -inf
    min_topk = topk_vals.min(axis=1, keepdims=True)  # (batch_size, 1)
    masked = np.where(logits_2d < min_topk, -np.inf, logits_2d)
    
    # Softmax with numerical stability
    logit_max = masked.max(axis=1, keepdims=True)
    probs = np.exp(masked - logit_max)
    probs /= probs.sum(axis=1, keepdims=True)
    
    # Sample one token per row
    samples = np.array([np.random.choice(vocab_size, p=p) for p in probs])
    
    # Reshape topk_idx and topk_vals back to original batch shape
    topk_idx = topk_idx.reshape(*shape[:-1], k)
    topk_vals = topk_vals.reshape(*shape[:-1], k)
    
    # Return first sample (as scalar) and top-k info with original batch shape
    return list(samples), topk_idx, topk_vals


def topk_sample_mlx(logits, k: int):
    """
    Top-k sampling in MLX without boolean indexing.

    Args:
        logits: MLX array of shape (V), (B, V), (B, S, V); unnormalized scores.
        k:      number of top candidates.

    Returns:
        sampled_token: MLX int array of shape (B,), token ids sampled from top-k.
        topk_idx:      MLX int32 array of shape (B, k), indices of top-k per row.
        topk_vals:     MLX float array of shape (B, k), logits of top-k per row.
    """
    V = logits.shape[-1]

    # 1) Indices of the k largest elements (order within k is unspecified)
    #    Use argpartition at position V-k so the last k positions hold the largest k.
    idx_part = mx.argpartition(logits, kth=V - k, axis=-1)              # (..., V)
    topk_idx = idx_part[..., -k:].astype(mx.int32)                        # (..., k)

    # 2) Gather their logits (aligned with topk_idx)
    topk_vals = mx.take_along_axis(logits, topk_idx, axis=-1)           # (..., k)

    # 3) Sample inside the k-slice using unnormalized logits
    #    categorical() removes the axis '-1' and returns shape (...,)
    choice_in_k = mx.random.categorical(topk_vals, axis=-1).astype(mx.int32)  # (...,)

    # 4) Map sampled positions back to vocab ids
    #    Add a dimension at the end to match the shape needed for take_along_axis
    choice_in_k_expanded = mx.expand_dims(choice_in_k, axis=-1)  # (..., 1)
    sampled_token = mx.take_along_axis(topk_idx, choice_in_k_expanded, axis=-1).squeeze(-1)  # (...,)

    return sampled_token, topk_idx, topk_vals



class GenerationModel(ABC):
    """Minimal, uniform interface for tokenization + cached forward decode."""

    # ---- model lifecycle ----
    @abstractmethod
    def reset(self) -> None:
        """Reset/clear the model's KV cache."""

    @abstractmethod
    def trim_cache(self, n: int) -> None:
        """Trim the last n generated steps from the cache (uniform across the batch)."""

    def rollback_tokens(self, r: list[int]) -> None:
        """
        Optional: per-row rollback for batched caches; default is unsupported.
        Implementations may override if they support per-row repacking.
        """
        raise NotImplementedError("Per-row rollback isn't supported for this backend.")

    # ---- tokenization / text I/O ----
    @abstractmethod
    def tokenize(self, prompt: str) -> np.ndarray:
        """Return a 1D np.int32 array of token ids (prompt prepared for generation)."""

    @abstractmethod
    def decode(self, generated: list[int]) -> str:
        """Detokenize a generated token id list."""

    @abstractmethod
    def eos_token_id(self) -> Optional[int]:
        """EOS token id or None."""

    @abstractmethod
    def pad_id(self) -> Optional[int]:
        """Pad token id or None."""

    # ---- main step ----
    @abstractmethod
    def forward(self, tokens: np.ndarray, only_final: bool = True
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Append `tokens` to the current cache and return:
          sampled_ids, topk_idx, topk_vals

        Semantics:
          - If `only_final=False`, we return one sample/top-k *per incoming position*.
          - If `only_final=True`, we return a single sample/top-k for the last position
            (per row if input is batched).
        All returns are NumPy arrays.
        """



class MLXGenerationModel(GenerationModel):
    def __init__(self, model_path: Path):
        mx.random.seed(SEED)
        self.model, self.config = load_model(model_path)
        self.tok = load_tokenizer(model_path)

        # self.cache = make_prompt_cache(self.model)  # rotating kv cache is fine
        self.cache: None | list[BatchedKVCache] = None
        self.tokens: np.array | None = None          # shape = (B, S)
        self.lengths: list[int] | None = None        # valid (non-pad) length per row
        self._pad_id = self.pad_id()                 # resolved once

    def reset(self) -> None:
        """Reset KV cache and local tracking."""
        self.cache = make_prompt_cache(self.model)
        self.tokens = None
        self.lengths = None

    def trim_cache(self, n: int) -> None:
        """Uniform rollback across the batch (kept for compatibility)."""
        for c in self.cache:
            c.trim(n)
        if self.lengths is not None:
            self.lengths = [max(0, L - n) for L in self.lengths]
        if self.tokens is not None and self.tokens.ndim == 2:
            # drop n tokens from the right on every row (keep shape)
            S_prev = self.tokens.shape[1]
            keep = max(0, S_prev - n)
            self.tokens = self.tokens[:, :keep]

    def add_tokens(self, tokens: np.array):
        """
        Track the 'committed' token matrix for the batch, keeping per-row lengths.
        - First call: tokens may be left-padded (B, S0). We detect lengths by pad_id.
        - Subsequent calls: tokens are usually (B, 1) decode steps (no pad).
        """
        t = np.asarray(tokens, dtype=np.int32)
        if t.ndim == 1:
            t = t[None, :]  # (T,) -> (1, T) batch

        assert self.tokens.shape[0] == t.shape[0], "Batch size changed between calls."
        assert self.lengths is not None

        self.tokens = np.concatenate([self.tokens, t], axis=1)
        self.lengths = [L + t.shape[1] for L in self.lengths]

    def forward(self, tokens: np.array, add_tokens=True, only_final=True) -> np.array:
        if add_tokens:
            self.add_tokens(tokens)

        tokens_mlx = mx.array(tokens, dtype=mx.int32)
        
        kwargs = {"cache":self.cache}
        # mask, pos = self._prefill_mask_and_positions(tokens)
        # kwargs['mask'] = mask
        # kwargs['cache_position'] = pos

        logits = self.model(tokens_mlx, **kwargs)  # (B, T, V)

        if only_final:
            logits = logits[:, -1, :]  # -> (B, V)

        tok, topk_idx, topk_vals = topk_sample_mlx(logits, TOP_K)
        mx.eval(tok, topk_idx, topk_vals)

        return (
            np.asarray(tok.astype(mx.int32)),
            np.asarray(topk_idx.astype(mx.int32)),
            np.asarray(topk_vals.astype(mx.float32)),
        )

    def prefill(self, tokens: list[list[int]]) -> np.array:
        self.lengths = [len(prompt) for prompt in tokens]

        n_layers = len(self.model.layers)
        self.cache = [
            BatchKVCache(left_padding=[max(self.lengths) - l for l in self.lengths]) for _ in range(n_layers)
        ]

        # Pad before prefill
        S = max(self.lengths)
        self.tokens = np.full((len(tokens), S), 0, dtype=np.int32)
        for i, prompt in enumerate(tokens):
            self.tokens[i, S - len(prompt):] = np.asarray(prompt, dtype=np.int32)

        return self.forward(self.tokens, add_tokens=False)

    # def prefill(self, tokens: list[list[int]]) -> np.array:
    #     self.lengths = [len(prompt) for prompt in tokens]
    #     self.tokens = tokens
    #
    #     # Pad before prefill
    #     S = max(self.lengths)
    #     tokens_np = np.full((len(tokens), S), 0, dtype=np.int32)
    #     for i, prompt in enumerate(tokens):
    #         tokens_np[i, S - len(prompt):] = np.asarray(prompt, dtype=np.int32)
    #
    #     return self.forward(tokens_np, add_tokens=False)

    def rollback_tokens(self, r: list[int]) -> None:
        """
        Per-row rollback for MLX caches that pre-allocate capacity.
        We:
        * compute per-row 'keep' from tracked lengths,
        * rebuild K/V inside the SAME capacity (right-aligned),
        * update per-row lengths if exposed by the cache object.
        """
        # 1) Sanity + current per-row lengths (what we've actually advanced to)
        if not self.cache or getattr(self.cache[0], "keys", None) is None:
            raise RuntimeError("Cache is empty; call forward() before rollback.")
        if self.lengths is None:
            raise RuntimeError("Internal lengths not initialized; ensure add_tokens() ran.")

        # 2) Compute per-row keep using REAL lengths, not S_cap
        L_prev = self.lengths
        L_new   = [L_prev[i] - r[i] for i in range(len(r))]

        S_prev = max(L_prev)
        S_target = max(L_new)

        B = len(r)

        # 3) Repack each layer's K/V into the same capacity, right-aligned
        for c in self.cache:
            K = c.keys   # (B, H, S_cap, D)
            V = c.values
            assert K is not None and V is not None

            _, H, S_cap, D = K.shape

            # fresh arrays with identical capacity
            K_new = mx.zeros((B, H, S_target, D), dtype=K.dtype)
            V_new = mx.zeros((B, H, S_target, D), dtype=V.dtype)

            for i in range(len(r)):
                keep = L_new[i]
                if keep <= 0:
                    continue

                lhs_start = S_prev - L_prev[i]
                lhs_end = lhs_start + keep

                K_src = K[i, :, lhs_start:lhs_end, :]
                V_src = V[i, :, lhs_start:lhs_end, :]

                # print(K_src[0, :, 0])

                rhs_start = S_target - keep
                rhs_end = S_target

                # print(S_prev, L_prev[i], r[i])

                # if i == 0:
                #     print(lhs_start, lhs_end)
                #     print(rhs_start, rhs_end)

                K_new[i, :, rhs_start:rhs_end, :] = K_src
                V_new[i, :, rhs_start:rhs_end, :] = V_src

                # print(np.array(K_new[0, 0, :, 0].tolist()))
                # raise Exception('stop')

            new_left_pad = mx.array([S_target - k for k in L_new], dtype=mx.int32)  # per-row
            new_offset   = mx.array(L_new, dtype=mx.int32)                          # per-row

            c.state = (K_new, V_new, new_offset, new_left_pad)

        # 4) Keep our own bookkeeping in sync
        self.lengths = L_new

        toks_new = np.full((B, S_target), self._pad_id, dtype=np.int32)
        for i in range(len(r)):
            lhs_start = S_prev - L_prev[i]
            lhs_end = lhs_start + L_new[i]

            rhs_start = S_target - L_new[i]
            rhs_end = S_target

            # if i == 0:
            #     print(lhs_start, lhs_end)
            #     print(rhs_start, rhs_end)

            toks_new[i, rhs_start:rhs_end] = self.tokens[i, lhs_start:lhs_end]
        self.tokens = toks_new

    def tokenize(self, prompt: str) -> np.array:
        return np.array(self.tok.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True
        ))

    def decode(self, generated: list[int]) -> str:
        return self.tok.decode(generated)

    def eos_token_id(self) -> int:
        return getattr(self.tok, "eos_token_id", None)

    def pad_id(self) -> int | None:
        pid = getattr(self.tok, "pad_token_id", None)
        if pid is None:
            pid = 0
        return pid



