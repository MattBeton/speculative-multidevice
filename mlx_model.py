from pathlib import Path
import numpy as np

from typing import final

import mlx.core as mx
from mlx_lm.tokenizer_utils import load_tokenizer
from mlx_lm.utils import load_model
from mlx_lm.models.cache import BatchKVCache, make_prompt_cache

from const import SEED, TOP_K
from model import GenerationModel


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



@final
class MLXGenerationModel(GenerationModel):
    def __init__(self, model_path: Path):
        mx.random.seed(SEED)
        self.model, self.config = load_model(model_path)
        self.tok = load_tokenizer(model_path)

        # self.cache = make_prompt_cache(self.model)  # rotating kv cache is fine
        self.cache: None | list[BatchKVCache] = None
        self.tokens: np.ndarray | None = None          # shape = (B, S)
        self.lengths: list[int] | None = None        # valid (non-pad) length per row

    def reset(self) -> None:
        """Reset KV cache and local tracking."""
        self.cache = None
        self.tokens = None
        self.lengths = None

    def _add_tokens(self, tokens: np.ndarray):
        """
        Track the 'committed' token matrix for the batch, keeping per-row lengths.
        """
        t = np.asarray(tokens, dtype=np.int32)
        if t.ndim == 1:
            t = t[None, :]  # (T,) -> (1, T) batch

        assert self.tokens.shape[0] == t.shape[0], "Batch size changed between calls."
        assert self.lengths is not None

        self.tokens = np.concatenate([self.tokens, t], axis=1)
        self.lengths = [L + t.shape[1] for L in self.lengths]

    def forward(
            self, 
            tokens: np.ndarray, 
            add_tokens=True, 
            only_final=True
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if add_tokens:
            self._add_tokens(tokens)

        tokens_mlx = mx.array(tokens, dtype=mx.int32)

        logits = self.model(tokens_mlx, cache=self.cache)  # (B, T, V)

        if only_final:
            logits = logits[:, -1, :]  # -> (B, V)

        tok, topk_idx, topk_vals = topk_sample_mlx(logits, TOP_K)
        mx.eval(tok, topk_idx, topk_vals)

        return (
            np.asarray(tok.astype(mx.int32)),
            np.asarray(topk_idx.astype(mx.int32)),
            np.asarray(topk_vals.astype(mx.float32)),
        )

    def prefill(self, tokens: list[list[int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.reset()

        self.lengths = [len(prompt) for prompt in tokens]

        n_layers = len(self.model.layers)
        self.cache = [
            BatchKVCache(left_padding=[max(self.lengths) - l for l in self.lengths]) for _ in range(n_layers)
        ]

        # Pad before prefill
        S = max(self.lengths)
        self.tokens = np.full((len(tokens), S), self.pad_id, dtype=np.int32)
        for i, prompt in enumerate(tokens):
            self.tokens[i, S - len(prompt):] = np.asarray(prompt, dtype=np.int32)

        return self.forward(self.tokens, add_tokens=False)

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
            raise RuntimeError("Internal lengths not initialized; ensure prefill() ran.")

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

                rhs_start = S_target - keep
                rhs_end = S_target

                K_new[i, :, rhs_start:rhs_end, :] = K_src
                V_new[i, :, rhs_start:rhs_end, :] = V_src

            new_left_pad = mx.array([S_target - k for k in L_new], dtype=mx.int32)  # per-row
            new_offset   = mx.array(L_new, dtype=mx.int32)                          # per-row

            c.state = (K_new, V_new, new_offset, new_left_pad)

        # 4) Update per-row lengths and token streams
        self.lengths = L_new

        toks_new = np.full((B, S_target), self.pad_id, dtype=np.int32)
        for i in range(len(r)):
            lhs_start = S_prev - L_prev[i]
            lhs_end = lhs_start + L_new[i]

            rhs_start = S_target - L_new[i]
            rhs_end = S_target

            toks_new[i, rhs_start:rhs_end] = self.tokens[i, lhs_start:lhs_end]
        self.tokens = toks_new

    def tokenize(self, prompt: str) -> np.ndarray:
        return np.asarray(self.tok.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True
        ))

    def decode(self, generated: list[int]) -> str:
        return self.tok.decode(generated)

    @property
    def eos_token_id(self) -> int:
        return getattr(self.tok, "eos_token_id", 0)

    @property
    def pad_id(self) -> int:
        pad_id = getattr(self.tok, "pad_token_id", None)
        if pad_id is None:
            pad_id = 0
        return pad_id
