# pip install mlx-lm
import time
import mlx.core as mx
from mlx_lm.tokenizer_utils import load_tokenizer
from mlx_lm.utils import load_model
from mlx_lm.models.cache import make_prompt_cache
from pathlib import Path
import numpy as np

# MODEL = "/Users/frank/.cache/huggingface/hub/models--mlx-community--Qwen3-4B-Instruct-2507-8bit/snapshots/0e42af58449718de7931ee04f28191fbe6c43a56"
MODEL_PATH = Path("/Users/frank/.cache/huggingface/hub/models--mlx-community--Qwen3-0.6B-4bit/snapshots/73e3e38d981303bc594367cd910ea6eb48349da8")
PROMPT = "Write a terse haiku about Apple MLX."
MAX_NEW_TOKENS = 64
TOP_K = 40
SEED = 0

def topk_sample(logits, k: int) -> int:
    """
    logits: (1, V) MLX array (unnormalized)
    Returns an int token id sampled from the top-k.
    """
    # Compute per-row kth value, mask everything else to -inf, then categorical sample
    kth = mx.min(mx.topk(logits, k), axis=-1, keepdims=True)  # (1,1)
    masked = mx.where(logits < kth, float("-inf"), logits)    # (1,V)
    return int(mx.random.categorical(masked, axis=-1)[0].item())

def main():
    mx.random.seed(SEED)
    model, config = load_model(MODEL_PATH)
    tok = load_tokenizer(MODEL_PATH)

    ids = tok.apply_chat_template(
        [{"role": "user", "content": PROMPT}], add_generation_prompt=True, tokenize=True
    )
    x = mx.array([ids], dtype=mx.int32)            # (1, T0)

    cache = make_prompt_cache(model)  # for rotating kv cache: make_prompt_cache(model, max_kv_size=4096)

    # --- PREFILL ---
    t0 = time.perf_counter()
    logits = model(x, cache=cache)                 # (1, T0, V), cache is filled in-place
    mx.eval(logits)                                 # force lazy compute for accurate timing
    t1 = time.perf_counter()
    prefill_tokens = x.shape[1]

    # First generated token comes from the last prefill logits (no extra model call)
    last_logits = logits[:, -1, :]                 # (1, V)
    next_id = topk_sample(last_logits, TOP_K)
    generated = [next_id]
    eos = getattr(tok, "eos_token_id", None)

    # --- DECODE one token at a time with KV cache (only pass the LAST token) ---
    t2 = time.perf_counter()
    last = next_id
    for _ in range(MAX_NEW_TOKENS - 1):
        y = mx.array([[last]], dtype=mx.int32)    # (1, 1)
        logits = model(y, cache=cache)            # uses the cache, returns (1, 1, V)
        mx.eval(logits)
        last = topk_sample(logits[:, -1, :], TOP_K)
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
    print('\n' + tok.decode(generated))

if __name__ == "__main__":
    main()

