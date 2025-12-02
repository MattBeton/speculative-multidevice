import pytest

from pathlib import Path
import numpy as np

from model import GenerationModel
from mlx_model import MLXGenerationModel

# Fixed prompts list
PROMPTS: list[str] = [
    "Why is the sky blue?",
    # "Explain speculative decoding in simple terms.",
    # "Write a sonnet about the iPhone.",
    # "What are the benefits of renewable energy?",
    # "Describe the process of photosynthesis.",
    # "How does machine learning work?",
    # "What is the difference between AI and AGI?",
    # "Explain the theory of relativity.",
]

@pytest.fixture
def model() -> GenerationModel:
    DRAFT_MODEL_PATH = next(Path(
        "~/.cache/huggingface/hub/models--mlx-community--Llama-3.2-1B-Instruct-bf16/snapshots/"
    ).expanduser().glob("*"))

    return MLXGenerationModel(DRAFT_MODEL_PATH)

def test_model_generation(model: GenerationModel):
    model.reset()
    eos = model.eos_token_id

    ids = model.tokenize(PROMPTS[0])

    model.prefill([ids[:-1]])

    generated = []
    last = ids[-1]
    for _ in range(20):
        y = np.array([[last]], dtype=np.int32)

        toks, _, _ = model.forward(y)

        last = int(toks[-1])
        generated.append(last)

        if last == eos:
            break

    print(model.decode(model.tokens.reshape(-1).tolist()))
