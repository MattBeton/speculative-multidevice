# pip install mlx-lm
import time
from pathlib import Path
import numpy as np

from model import MLXGenerationModel

# MODEL = "/Users/frank/.cache/huggingface/hub/models--mlx-community--Qwen3-4B-Instruct-2507-8bit/snapshots/0e42af58449718de7931ee04f28191fbe6c43a56"
MODEL_PATH = Path("/Users/frank/.cache/huggingface/hub/models--mlx-community--Qwen3-0.6B-4bit/snapshots/73e3e38d981303bc594367cd910ea6eb48349da8")
PROMPT = "Write a terse haiku about Apple MLX."
MAX_NEW_TOKENS = 64
TOP_K = 40

def topk_sample(logits: np.array, k: int) -> int:
    kth = np.partition(logits, -k)[-k]
    masked = np.where(logits < kth, -np.inf, logits)
    probs = np.exp(masked - np.max(masked))
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

def main():
    model = MLXGenerationModel(MODEL_PATH)

    ids = model.tokenize(PROMPT)

    # --- PREFILL ---
    t0 = time.perf_counter()
    logits = model.forward(ids)
    t1 = time.perf_counter()
    prefill_tokens = len(ids)

    # First generated token comes from the last prefill logits (no extra model call)
    last_logits = logits[0, -1, :]                 # (1, V)
    next_id = topk_sample(last_logits, TOP_K)
    generated = [next_id]
    eos = model.eos_token_id

    # --- DECODE one token at a time with KV cache (only pass the LAST token) ---
    t2 = time.perf_counter()
    last = next_id
    for _ in range(MAX_NEW_TOKENS - 1):
        y = np.array([[last]], dtype=np.int32)    # (1, 1)
        logits = model.forward(y)                 # uses the cache, returns (1, 1, V)
        last = topk_sample(logits[0, -1, :], TOP_K)
        generated.append(last)
        if eos is not None and last == eos:
            break
    t3 = time.perf_counter()


    # --- Report & show text ---
    decode_tokens = len(generated)

    prefill_tps = prefill_tokens / max(t1 - t0, 1e-9)
    decode_tps = decode_tokens / max(t3 - t2, 1e-9)

    print(f"[prefill] {prefill_tokens} toks in {t1 - t0:.3f}s  → {prefill_tps:.1f} tok/s")
    print(f"[decode ] {decode_tokens} toks in {t3 - t2:.3f}s  → {decode_tps:.1f} tok/s")
    print('\n' + model.decode(generated))

if __name__ == "__main__":
    main()

