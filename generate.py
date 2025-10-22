# pip install mlx-lm
import time
from pathlib import Path
import numpy as np

from model import MLXGenerationModel, TOP_K

DRAFT_MODEL_PATH = Path("/Users/frank/.cache/huggingface/hub/models--mlx-community--Llama-3.2-1B-Instruct-4bit/snapshots/").glob("*")
DRAFT_MODEL_PATH = next(DRAFT_MODEL_PATH)
BASE_MODEL_PATH = Path("/Users/frank/.cache/huggingface/hub/models--mlx-community--Llama-3.2-1B-Instruct-bf16/snapshots/").glob("*")
BASE_MODEL_PATH = next(BASE_MODEL_PATH)
PROMPT = "Write a terse haiku about Apple MLX."

MAX_NEW_TOKENS = 64
SPEC_K = 8

    
def main():
    draft_model = MLXGenerationModel(DRAFT_MODEL_PATH)
    base_model = MLXGenerationModel(BASE_MODEL_PATH)

    ids_draft = draft_model.tokenize(PROMPT)
    ids = base_model.tokenize(PROMPT)
    assert (ids == ids_draft).all()
    eos = base_model.eos_token_id()

    # --- PREFILL ---
    draft_model.forward(ids[:-1], only_final=False)
    base_model.forward(ids[:-1], only_final=False)  # warm cache

    valid_idx = len(ids) - 1
    draft_idx = len(ids) - 1


    # --- DECODE one token at a time with KV cache (only pass the LAST token) ---
    last = int(ids[-1])  # ensure scalar
    while valid_idx < len(ids) + MAX_NEW_TOKENS:
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



        toks_verify = np.concatenate((np.array([ids[-1]]), draft_toks)) # TODO: This needs changing to work on iterations of the while loop past the first one.

        if SPEC_K > 1:
            _, base_topk_idx, base_topk_vals = base_model.forward(
                toks_verify,
                only_final=False
            )
        else:
            base_topk_idx = base_topk_vals = None

        print('-'*100)

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

            print(f"i={i} tok={tok} in_base_topk={in_base_topk} draft_logit={float(draft_logit):.3f} base_logit={float(base_logit):.3f}")
            print(f'{accepted=}')

            if not accepted:
                break

            if eos is not None and tok == eos:
                break

        if eos is not None and tok == eos:
            break

        # TODO: Finish up & get ready for next iteration.
        # If we didn't accept all tokens then we need to roll back kv cache of both models.

if __name__ == "__main__":
    main()

