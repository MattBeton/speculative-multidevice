# pip install mlx-lm
import time
from pathlib import Path
import numpy as np

from model import MLXGenerationModel, TOP_K

DRAFT_MODEL_PATH = Path("/Users/frank/.cache/huggingface/hub/models--mlx-community--Qwen3-0.6B-4bit/snapshots/73e3e38d981303bc594367cd910ea6eb48349da8")
BASE_MODEL_PATH = Path("/Users/frank/.cache/huggingface/hub/models--mlx-community--Qwen3-4B-Instruct-2507-8bit/snapshots/0e42af58449718de7931ee04f28191fbe6c43a56")
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
    _, base_pref_idx, base_pref_vals = base_model.forward(ids[:-1], only_final=False)
    draft_model.forward(ids[:-1], only_final=False)

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
            print(f'generated token {draft_model.decode(tok)}')
            print(topk_idx.shape)
            print(f'topk tokens {[draft_model.decode(x) for x in topk_idx.flatten()]}')
            t = int(tok[0])
            last = t
            draft_toks.append(t)
            draft_topk_idx.append(topk_idx[0])
            draft_topk_vals.append(topk_vals[0])

        draft_toks = np.array(draft_toks, dtype=np.int32)                    # (K,)
        draft_topk_idx = np.stack(draft_topk_idx, axis=0)                    # (K, TOP_K)
        draft_topk_vals = np.stack(draft_topk_vals, axis=0)                  # (K, TOP_K)

        # Base distributions aligned to *verify* each token:
        #  - t1 uses base_pref_* last row (p(next|prompt))
        #  - t2..tK use base run on t1..t_{K-1} (rows 0..K-2)
        base_idx_0 = base_pref_idx[-1]                                      # (TOP_K,)
        base_vals_0 = base_pref_vals[-1]                                    # (TOP_K,)

        toks_verify = np.concatenate((np.array([ids[-1]]), draft_toks))

        if SPEC_K > 1:
            _, base_seq_idx, base_seq_vals = base_model.forward(
                toks_verify,
                only_final=False
            )
        else:
            base_seq_idx = base_seq_vals = None

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
            if i == 0:
                b_idx_row, b_val_row = base_idx_0, base_vals_0
            else:
                b_idx_row, b_val_row = base_seq_idx[i], base_seq_vals[i]
            b_mask = (b_idx_row == tok)
            in_base_topk = b_mask.any()
            base_logit = b_val_row[b_mask][0] if in_base_topk else float('-inf')

            print(f'tok {draft_model.decode(tok)}')
            print(f'topk tokens - base {[draft_model.decode(x) for x in b_idx_row]}')

            print(f"i={i} tok={tok} in_base_topk={in_base_topk} draft_logit={float(draft_logit):.3f} base_logit={float(base_logit):.3f}")

        # stop here for debugging
        raise SystemExit(0)

        # if eos is not None and last == eos:
        #     break


    # --- Report & show text ---
    # TODO: implement accept/commit and final decode
    pass

if __name__ == "__main__":
    main()

