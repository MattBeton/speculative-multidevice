import argparse
import asyncio
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from model import MLXGenerationModel
from timing import TokenTimer
from shared import (
    MessageChannel,
    PrefillRequest,
    PrefillResponse,
    ResetRequest,
    ResetResponse,
    VerifyRequest,
    VerifyResponse,
    run_mlx,
)

from const import (
    PAD_ID
)

# ---- Configure the draft (client) model ----
DRAFT_MODEL_PATH = next(
    Path("~/.cache/huggingface/hub/models--mlx-community--Llama-3.2-1B-Instruct-bf16/snapshots/").expanduser().glob("*")
)

# Fixed prompts list
PROMPTS: List[str] = [
    "Why is the sky blue?",
    "Explain speculative decoding in simple terms.",
    "Write a sonnet about the iPhone.",
    "What are the benefits of renewable energy?",
    "Describe the process of photosynthesis.",
    "How does machine learning work?",
    "What is the difference between AI and AGI?",
    "Explain the theory of relativity.",
]

SPEC_K = 8
MAX_NEW_TOKENS = 64

model = MLXGenerationModel(DRAFT_MODEL_PATH)

prompts_tokens = [
    list(model.tokenize(prompt)) for prompt in PROMPTS
]

def pad(prompts_tokens: list[list[int]]) -> list[list[int]]:
    maxlen = max([len(x) for x in prompts_tokens])

    prompts_tokens = [
        [PAD_ID] * (maxlen - len(prompt)) + prompt for prompt in prompts_tokens
    ]

    return np.array(prompts_tokens)

tokens = pad(prompts_tokens)

print(f'executing forwards pass')

print(model.forward(tokens))

