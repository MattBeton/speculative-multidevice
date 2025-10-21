from abc import ABC
from pathlib import Path
import numpy as np

import mlx.core as mx
from mlx_lm.tokenizer_utils import load_tokenizer
from mlx_lm.utils import load_model
from mlx_lm.models.cache import make_prompt_cache

SEED = 0
TOP_K = 40

def topk_sample_numpy(logits: np.array, k: int) -> tuple[int, np.array, np.array]:
    topk_vals = np.partition(logits, -k)[-k:]  # k largest values
    topk_idx = np.argpartition(logits, -k)[-k:]  # their indices
    masked = np.where(logits < topk_vals.min(), -np.inf, logits)
    probs = np.exp(masked - masked.max())  # numerical stability
    probs /= probs.sum()
    sample = np.random.choice(len(logits), p=probs)
    return int(sample), topk_idx, topk_vals

# def topk_sample_mlx(logits: mx.array, k: int) -> tuple[int, mx.array, mx.array]:
#     logits = logits.reshape(-1)  # Ensure 1D
#     topk_vals = mx.topk(logits, k)
#     topk_idx = mx.argpartition(logits, -k)[-k:]
#     masked = mx.where(logits < topk_vals.min(), float("-inf"), logits)
#     probs = mx.exp(masked - masked.max())
#     probs /= mx.sum(probs)
#     sample = mx.random.categorical(probs)
#     return sample, topk_idx, topk_vals

def topk_sample_mlx(logits, k: int) -> int:
    """
    logits: (1, V) MLX array (unnormalized)
    Currently spoofing the topk idx&val return. Get an LM to fix this when we are landed.
    """
    kth = mx.min(mx.topk(logits, k), axis=-1, keepdims=True)  # (1,1)
    masked = mx.where(logits < kth, float("-inf"), logits)    # (1,V)
    return mx.random.categorical(masked, axis=-1)[0], mx.arange(TOP_K), logits[:, :TOP_K].flatten()


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

    def forward(self, tokens: np.array, only_final=True) -> np.array:
        tokens_mlx = mx.array(tokens.reshape((1, -1)), dtype=mx.int32)

        logits = self.model(tokens_mlx, cache=self.cache)                  # (1, T0, V), cache is filled in-place

        if only_final:
            logits = logits[:, -1, :]
        else:
            raise NotImpelementedError()

        tok, topk_idx, topk_vals = topk_sample_mlx(logits, TOP_K)

        mx.eval(tok, topk_idx, topk_vals)

        return int(tok.item()), np.asarray(topk_idx.astype(mx.int32)), np.asarray(topk_vals.astype(mx.float32))

    def tokenize(self, prompt: str) -> np.array:
        return np.array(self.tok.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True
        ))
    
    def decode(self, generated: list[int]) -> str:
        return self.tok.decode(generated)

    def eos_token_id(self) -> int:
        return getattr(tok, "eos_token_id", None)
