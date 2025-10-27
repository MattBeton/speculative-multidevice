import torch, transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AttentionInterface
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

device = "cuda:0"
model_id = "meta-llama/Llama-3.2-3B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,                  # or torch.float16
    attn_implementation="flash_attention_2",
).to(device).eval()

tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)

# --- Wrap FA2 ---
orig_fa2 = ALL_ATTENTION_FUNCTIONS["flash_attention_2"]
calls = {"n": 0}

def traced_fa2(*args, **kwargs):
    calls["n"] += 1
    return orig_fa2(*args, **kwargs)

AttentionInterface.register("fa2_traced", traced_fa2)
model.set_attn_implementation("fa2_traced")

# Use a longer prompt so we take the varlen prefill path
prompt = "hello " * 1024
inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to(device)

with torch.no_grad():
    _ = model(**inputs)

print("FlashAttention-2 calls =", calls["n"])  # > 0 means FA2 path is used

