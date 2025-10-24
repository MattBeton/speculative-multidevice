from abc import ABC
from pathlib import Path
from mlx.nn.layers.base import tree_flatten
from mlx.utils import tree_map
import numpy as np

import mlx.core as mx
from mlx_lm.tokenizer_utils import load_tokenizer
from mlx_lm.utils import load_model
from mlx_lm.models.cache import make_prompt_cache

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
        logits: MLX array of shape (B, V) or (1, V); unnormalized scores.
        k:      number of top candidates.

    Returns:
        sampled_token: MLX int array of shape (B,), token ids sampled from top-k.
        topk_idx:      MLX int32 array of shape (B, k), indices of top-k per row.
        topk_vals:     MLX float array of shape (B, k), logits of top-k per row.
    """
    # Ensure 2D (B, V)
    if logits.ndim == 1:
        logits = logits[None, :]
    if logits.ndim == 3:
        logits = logits.squeeze(0)

    B, V = logits.shape
    k = int(k if k <= V else V)  # clamp

    # 1) Indices of the k largest elements (order within k is unspecified)
    #    Use argpartition at position V-k so the last k positions hold the largest k.
    idx_part = mx.argpartition(logits, kth=V - k, axis=-1)              # (B, V)
    topk_idx = idx_part[:, -k:].astype(mx.int32)                        # (B, k)

    # 2) Gather their logits (aligned with topk_idx)
    topk_vals = mx.take_along_axis(logits, topk_idx, axis=-1)           # (B, k)

    # 3) Sample inside the k-slice using unnormalized logits
    #    categorical() removes the axis '-1' and returns shape (B,)
    choice_in_k = mx.random.categorical(topk_vals, axis=-1).astype(mx.int32)  # (B,)

    # 4) Map sampled positions back to vocab ids
    sampled_token = mx.take_along_axis(topk_idx, choice_in_k[:, None], axis=-1).squeeze(-1)  # (B,)

    return sampled_token, topk_idx, topk_vals



class GenerationModel(ABC):
    def forward(self, tokens: np.array) -> np.array:
        ...

    def reset(self) -> None:
        ...

    def tokenize(self, prompt: str) -> np.array:
        ...

    def eos_token_id(self) -> int:
        ...


class MLXGenerationModel(GenerationModel):
    def __init__(self, model_path: Path):
        mx.random.seed(SEED)
        self.model, self.config = load_model(model_path)
        self.tok = load_tokenizer(model_path)
        self.cache = make_prompt_cache(self.model) # for rotating kv cache: make_prompt_cache(model, max_kv_size=4096)

    def reset(self) -> None:
        """Reset KV cache to empty."""
        self.cache = make_prompt_cache(self.model)

    def trim_cache(self, n: int) -> None:
        """
        Reset the KV cache and prefill it with `tokens` (1D array).
        Useful for speculative decoding rollback to the last accepted prefix.
        """
        # tree_map(lambda x: print(x.keys.shape) if x.keys is not None else None, self.cache)
        for c in self.cache:
            c.trim(n)

    def forward(self, tokens: np.array, only_final=True) -> np.array:
        tokens_mlx = mx.array(tokens.reshape((1, -1)), dtype=mx.int32)

        logits = self.model(tokens_mlx, cache=self.cache)                  # (1, T0, V), cache is filled in-place

        if only_final:
            logits = logits[:, -1, :]

        ## TODO: When we figure out MLX topk
        tok, topk_idx, topk_vals = topk_sample_mlx(logits, TOP_K)
        mx.eval(tok, topk_idx, topk_vals)

        return np.asarray(tok.astype(mx.int32)), np.asarray(topk_idx.astype(mx.int32)), np.asarray(topk_vals.astype(mx.float32))

    def tokenize(self, prompt: str) -> np.array:
        return np.array(self.tok.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True
        ))
    
    def decode(self, generated: list[int]) -> str:
        return self.tok.decode(generated)

    def eos_token_id(self) -> int:
        return getattr(self.tok, "eos_token_id", None)
