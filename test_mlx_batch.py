# test_mlx_batch.py (demo: batched prefill → gen 10 → per-row rollback → continue)
import numpy as np
from pathlib import Path
import mlx.core as mx

from model import MLXGenerationModel

# ---- Draft/base config for local MLX model ----
DRAFT_MODEL_PATH = next(
    Path("~/.cache/huggingface/hub/models--mlx-community--Llama-3.2-1B-Instruct-bf16/snapshots/")
    .expanduser().glob("*")
)

PROMPTS = [
    "Why is the sky blue?",
    "Explain speculative decoding in simple terms.",
    "Write a sonnet about the iPhone.",
    "What are the benefits of renewable energy?",
    "Describe the process of photosynthesis.",
    "How does machine learning work?",
    "What is the difference between AI and AGI?",
    "Explain the theory of relativity.",
]

N_GEN_BEFORE = 10
N_GEN_AFTER  = 10

def left_pad_to_matrix(rows: list[list[int]], pad_id: int) -> np.ndarray:
    S = max(len(x) for x in rows)
    out = np.full((len(rows), S), pad_id, dtype=np.int32)
    for i, r in enumerate(rows):
        out[i, S - len(r):] = np.asarray(r, dtype=np.int32)
    return out

def last_nonpad_token_ids(mat: np.ndarray, pad_id: int) -> np.ndarray:
    # returns (B,) last token per row (assumes at least one non-pad)
    out = []
    for row in mat:
        idx = np.max(np.nonzero(row != pad_id)[0])  # last non-pad index
        out.append(int(row[idx]))
    return np.asarray(out, dtype=np.int32)

def decode_rows(model: MLXGenerationModel, rows_ids: list[list[int]]) -> list[str]:
    return [model.decode(ids) for ids in rows_ids]

def main():
    model = MLXGenerationModel(DRAFT_MODEL_PATH)
    pad_id = model.pad_id()

    # --- Tokenize and split into prefix (all but last) + last prompt token
    ids_list = [list(model.tokenize(p)) for p in PROMPTS]
    prefixes = [ids[:-1] for ids in ids_list]
    last_tok  = [ids[-1] for ids in ids_list]  # (B,)

    # Prefill: only prefixes (left-padded batch)
    X = left_pad_to_matrix(prefixes, pad_id)
    _ = model.forward(X, only_final=False)  # fill KV; ignore the returned sample

    # ---- Generate N_GEN_BEFORE tokens in batch ----
    B = len(PROMPTS)
    generated_before: list[list[int]] = [[] for _ in range(B)]

    cur = np.asarray(last_tok, dtype=np.int32).reshape(B, 1)  # feed last prompt token
    for _ in range(N_GEN_BEFORE):
        toks, _, _ = model.forward(cur)                       # (B,)
        step = [int(t) for t in toks]
        for i in range(B):
            generated_before[i].append(step[i])
        cur = np.asarray(step, dtype=np.int32).reshape(B, 1)  # next step seeds

    print("\n=== Before rollback (first 2 streams for brevity) ===")
    for i in range(min(2, B)):
        txt = model.decode(generated_before[i])
        print(f"[{i}] {txt}")

    # Print KV cache slice before rollback
    print("\n=== KV cache slice before rollback (batch idx=0, layer=0, head=0, dim=0) ===")
    if model.cache and len(model.cache) > 0:
        cache_keys = model.cache[0].keys
        if cache_keys is not None:
            slice_val = cache_keys[0, 0, :, 0]
            mx.eval(slice_val)
            print(np.asarray(slice_val))

    # ---- Per-row rollback by variable amounts (0..7 here; feel free to set up to 10) ----
    # Example rollback vector: [0,1,2,3,4,5,6,7]
    r = [(i + 3) % 5 for i in range(B)]  # <= 10 as requested
    print(f"\nRollback vector per row: {r}")
    model.rollback_tokens(r)

    # Print KV cache slice after rollback
    print("\n=== KV cache slice after rollback (batch idx=0, layer=0, head=0, dim=0) ===")
    if model.cache and len(model.cache) > 0:
        cache_keys = model.cache[0].keys
        if cache_keys is not None:
            slice_val = cache_keys[0, 0, :, 0]
            mx.eval(slice_val)
            print(np.asarray(slice_val))

    # After rollback, compute per-row 'last' token to feed the next step
    # last_after = last_nonpad_token_ids(model.tokens, pad_id).reshape(B, 1)
    last_after = model.tokens[:, -1].reshape(B, 1)

    # ---- Continue generating N_GEN_AFTER tokens ----
    generated_after: list[list[int]] = [[] for _ in range(B)]
    cur = last_after
    for _ in range(N_GEN_AFTER):
        toks, _, _ = model.forward(cur)
        step = [int(t) for t in toks]
        for i in range(B):
            generated_after[i].append(step[i])
        cur = np.asarray(step, dtype=np.int32).reshape(B, 1)

    print("\n=== After rollback & resume (first 2 streams) ===")
    for i in range(min(2, B)):
        txt = model.decode(generated_after[i])
        print(f"[{i}] {txt}")

    # You can also inspect full resumed sequences by decoding:
    #  - the *entire* kept prefix + resumed tail:
    kept_plus_after = []
    for i in range(B):
        # Reconstruct row i's kept prefix from model.tokens (drop pad)
        row = [int(t) for t in model.tokens[i].tolist() if t != pad_id]
        kept_plus_after.append(row + generated_after[i])

    print("\n=== Kept+Resumed (full rows; first 2 streams) ===")
    for i in range(min(2, B)):
        print(f"[{i}] {model.decode(kept_plus_after[i])}\n")

if __name__ == "__main__":
    main()

