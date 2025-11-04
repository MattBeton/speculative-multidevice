import os
import torch

# Model configuration
BASE_MODEL = os.environ.get("HF_BASE_MODEL", "meta-llama/Llama-3.1-8B")
TOP_K = int(os.environ.get("HF_TOP_K", "20"))
ATTN_IMPL_ENV = os.environ.get("HF_ATTN_IMPL", "").strip()  # e.g., "flash_attention_2" if available

# Device and dtype configuration
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')
# DTYPE = torch.float16
DTYPE = torch.float32

# HF token for authentication
HF_TOKEN = os.environ.get("HF_TOKEN", None)

# Pad token ID - can be set after loading tokenizer
# Defaults to 0 if not set
PAD_ID = 0

