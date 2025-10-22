# pip install mlx-lm
import time
from pathlib import Path
import numpy as np

from model import MLXGenerationModel, TOP_K


DRAFT_MODEL_PATH = Path("/Users/frank/.cache/huggingface/hub/models--mlx-community--Llama-3.2-1B-Instruct-bf16/snapshots/").glob("*")
DRAFT_MODEL_PATH = next(DRAFT_MODEL_PATH)
# BASE_MODEL_PATH = Path("/Users/frank/.cache/huggingface/hub/models--mlx-community--Llama-3.2-1B-Instruct-bf16/snapshots/").glob("*")
BASE_MODEL_PATH = Path("/Users/frank/.cache/huggingface/hub/models--mlx-community--Llama-3.2-3B-Instruct/snapshots/").glob("*")
BASE_MODEL_PATH = next(BASE_MODEL_PATH)

# PROMPT = "Write a terse haiku about Apple MLX."
PROMPT = "Why is the sky blue?"

MAX_NEW_TOKENS = 64
SPEC_K = 8

    
def main():
    draft_model = MLXGenerationModel(DRAFT_MODEL_PATH)
    base_model = MLXGenerationModel(BASE_MODEL_PATH)

    ids_draft = draft_model.tokenize(PROMPT)
    ids = base_model.tokenize(PROMPT)
    assert (ids == ids_draft).all()
    eos = base_model.eos_token_id()

    # --- PREFILL (cache = prefix without last token) ---
    prefill_start = time.perf_counter()
    draft_model.forward(ids[:-1])
    base_model.forward(ids[:-1])
    prefill_time = time.perf_counter() - prefill_start
    prefill_tokens = len(ids) - 1
    prefill_tps = (prefill_tokens / prefill_time) if prefill_time else float("inf")
    print(f"Prefill: {prefill_tokens} tokens in {prefill_time:.3f}s ({prefill_tps:.2f} tok/s)")

    valid_idx = len(ids) - 1
    prompt_length = valid_idx


    # --- DECODE one token at a time with KV cache (only pass the LAST token) ---
    decode_start = time.perf_counter()
    while valid_idx < prompt_length + MAX_NEW_TOKENS:
        last = int(ids[-1])  # ensure scalar and refresh each iteration from current sequence
        draft_toks = []
        draft_topk_idx = []
        draft_topk_vals = []

        for _ in range(SPEC_K):
            y = np.array([[last]], dtype=np.int32)    # (1, 1)
            tok, topk_idx, topk_vals = draft_model.forward(y)
            t = int(tok[0])
            last = t
            draft_toks.append(t)
            draft_topk_idx.append(topk_idx[0])
            draft_topk_vals.append(topk_vals[0])

        draft_toks = np.array(draft_toks, dtype=np.int32)                    # (K,)
        draft_topk_idx = np.stack(draft_topk_idx, axis=0)                    # (K, TOP_K)
        draft_topk_vals = np.stack(draft_topk_vals, axis=0)                  # (K, TOP_K)

        # Verify against base: start from the CURRENT last accepted token
        toks_verify = np.concatenate((np.array([ids[-1]], dtype=np.int32), draft_toks))

        if SPEC_K > 1:
            # Base cache currently at ids[:-1]; advance across last+K for top-k rows per step
            _, base_topk_idx, base_topk_vals = base_model.forward(
                toks_verify,
                only_final=False
            )
        else:
            base_topk_idx = base_topk_vals = None

        print('-'*100)

        accepted_tokens = []
        hit_eos = False
        for i, tok in enumerate(draft_toks):
            # draft top-k info for this step (tok is guaranteed to be within it)
            d_idx_row = draft_topk_idx[i]
            d_val_row = draft_topk_vals[i]
            d_mask = (d_idx_row == tok)
            if d_mask.sum() != 1:
                raise RuntimeError("Draft top-k should contain the sampled token exactly once.")
            draft_logit = d_val_row[d_mask][0]

            # base top-k row aligned to *this* token
            b_idx_row, b_val_row = base_topk_idx[i], base_topk_vals[i]
            b_mask = (b_idx_row == tok)
            in_base_topk = b_mask.any()
            base_logit = b_val_row[b_mask][0] if in_base_topk else float('-inf')

            if base_logit == float('-inf'):
                accepted = False
            else:
                if draft_logit <= base_logit:
                    accepted = True
                else:
                    u = np.random.uniform(0, 1)
                    if u <= base_logit / draft_logit:
                        accepted = True
                    else:
                        accepted = False

            # print(f"i={i} tok={tok} in_base_topk={in_base_topk} draft_logit={float(draft_logit):.3f} base_logit={float(base_logit):.3f}")
            # print(f'{accepted=}')

            if not accepted:
                break

            if eos is not None and tok == eos:
                accepted_tokens.append(tok)
                hit_eos = True
                break

            accepted_tokens.append(tok)

        # Incorporate accepted tokens into the running sequence
        if accepted_tokens:
            ids = np.concatenate((ids, np.array(accepted_tokens, dtype=np.int32)))
            valid_idx += len(accepted_tokens)

        # If EOS hit or token budget reached, stop
        if hit_eos or (valid_idx >= prompt_length + MAX_NEW_TOKENS):
            break

        # Finish up & get ready for next iteration.
        # Roll back KV caches of both models to the prefix *excluding* the new last token.
        # That way the next step can pass only the (new) last token.
        draft_model.trim_cache(SPEC_K - i)
        base_model.trim_cache(SPEC_K - i)

    decode_time = time.perf_counter() - decode_start
    generated_tokens = max(len(ids) - prompt_length, 0)
    decode_tps = (generated_tokens / decode_time) if decode_time else float("inf")
    print(f"Decode: {generated_tokens} tokens in {decode_time:.3f}s ({decode_tps:.2f} tok/s)")
    print(base_model.decode(ids[prompt_length:]))

if __name__ == "__main__":
    main()
