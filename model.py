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


def topk_sample_mlx(logits, k: int) -> int:
    """
    logits: (1, V) MLX array (unnormalized)
    Currently spoofing the topk idx&val return. Get an LM to fix this when we are landed.
    """
    V = logits.shape[-1]  # vocab size
    kth = mx.min(mx.topk(logits, k), axis=-1, keepdims=True)  # (1,1)
    masked = mx.where(logits < kth, float("-inf"), logits)    # (1,V)
    sampled_token = mx.random.categorical(masked, axis=-1)[0]

    # topk_idx = mx.arange(V)[logits < kth]
    # topk_val = logits[logits < kth]

    # return sampled_token, topk_idx, topk_val
    #
    return sampled_token, mx.arange(TOP_K), logits[:, :TOP_K].flatten()



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

        tok, topk_idx, topk_vals = topk_sample_numpy(np.asarray(logits.astype(mx.float32)), TOP_K)
        return tok, topk_idx, topk_vals

        ### TODO: When we figure out MLX topk
        # tok, topk_idx, topk_vals = topk_sample_mlx(logits, TOP_K)
        # mx.eval(tok, topk_idx, topk_vals)
        #
        # return int(tok.item()), np.asarray(topk_idx.astype(mx.int32)), np.asarray(topk_vals.astype(mx.float32))

    def tokenize(self, prompt: str) -> np.array:
        return np.array(self.tok.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True
        ))
    
    def decode(self, generated: list[int]) -> str:
        return self.tok.decode(generated)

    def eos_token_id(self) -> int:
        return getattr(tok, "eos_token_id", None)
