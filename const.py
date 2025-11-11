import os
import random
import torch

# Model configuration
BASE_MODEL = os.environ.get("HF_BASE_MODEL", "meta-llama/Llama-3.2-3B-Instruct")
TOP_K = int(os.environ.get("HF_TOP_K", "20"))
ATTN_IMPL_ENV = os.environ.get("HF_ATTN_IMPL", "").strip()  # e.g., "flash_attention_2" if available

# speculative decoding config
SPEC_K = 8
MAX_NEW_TOKENS = 64

# Server configuration
HOST = os.environ.get("HF_SPEC_HOST", "0.0.0.0")
PORT = int(os.environ.get("HF_SPEC_PORT", "7070"))

# Random seed
SEED = 90

# Device and dtype configuration
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    if torch.cuda.is_bf16_supported():
        DTYPE = torch.bfloat16
    else:
        DTYPE = torch.float16
    torch.backends.cuda.matmul.allow_tf32 = True
else:
    DEVICE = torch.device("cpu")
    DTYPE = torch.float32

# Deterministic generators
GENERATOR = {
    "cpu": torch.Generator(device="cpu").manual_seed(SEED),
    "cuda": torch.Generator(device="cuda").manual_seed(SEED) if DEVICE.type == "cuda" else None,
}
PY_RNG = random.Random(SEED)

# HF token for authentication
HF_TOKEN = os.environ.get("HF_TOKEN", None)

# Pad token ID - can be set after loading tokenizer
# Defaults to 0 if not set
PAD_ID = 0

