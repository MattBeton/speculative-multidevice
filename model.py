from abc import ABC
from pathlib import Path
import numpy as np

import mlx.core as mx
from mlx_lm.tokenizer_utils import load_tokenizer
from mlx_lm.utils import load_model
from mlx_lm.models.cache import make_prompt_cache

SEED = 0

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

    def forward(self, tokens: np.array) -> np.array:
        tokens_mlx = mx.array(tokens.reshape((1, -1)), dtype=mx.int32)

        logits = self.model(tokens_mlx, cache=self.cache)                  # (1, T0, V), cache is filled in-place
        mx.eval(logits)

        return np.asarray(logits.astype(mx.float32))

    def tokenize(self, prompt: str) -> np.array:
        return np.array(self.tok.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True
        ))
    
    def decode(self, generated: list[int]) -> str:
        return self.tok.decode(generated)

    def eos_token_id(self) -> int:
        return getattr(tok, "eos_token_id", None)
