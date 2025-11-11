from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np


import mlx.core as mx
from mlx_lm.tokenizer_utils import load_tokenizer
from mlx_lm.utils import load_model
from mlx_lm.models.cache import BatchKVCache

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

class GenerationModel(ABC):
    """Minimal, uniform interface for tokenization + cached forward decode."""

    # ---- model lifecycle ----
    @abstractmethod
    def reset(self) -> None:
        """Reset/clear the model's KV cache."""

    # ---- main step ----
    @abstractmethod
    def forward(
        self, 
        tokens: np.ndarray, 
        only_final: bool = True
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Append `tokens` to the current cache and return:
          sampled_ids, topk_idx, topk_vals

        Semantics:
          - If `only_final=False`, we return one sample/top-k *per incoming position*.
          - If `only_final=True`, we return a single sample/top-k for the last position
            (per row if input is batched).
        All returns are NumPy arrays.
        """

    @abstractmethod
    def prefill(self, tokens: list[list[int]]) -> None:
        """Prefill the model's KV cache with the given tokens."""

    @abstractmethod
    def rollback_tokens(self, r: list[int]) -> None:
        """Per-row rollback for the model's KV cache."""

    # ---- tokenization / text I/O ----
    @abstractmethod
    def tokenize(self, prompt: str) -> np.ndarray:
        """Return a 1D np.int32 array of token ids (prompt prepared for generation)."""

    @abstractmethod
    def decode(self, generated: list[int]) -> str:
        """Detokenize a generated token id list."""

    @property
    @abstractmethod
    def eos_token_id(self) -> int:
        """EOS token id or None."""

    @property
    @abstractmethod
    def pad_id(self) -> int:
        """Pad token id or None."""

