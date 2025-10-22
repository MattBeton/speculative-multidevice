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
    eos = base_model.eos_token_id

    # --- PREFILL ---
    toks, topk_idx, topk_vals = base_model.forward(ids, only_final=False)
    draft_model.forward(ids[:-1], only_final=False)

    valid_idx = len(ids) - 1
    draft_idx = len(ids) - 1


    # --- DECODE one token at a time with KV cache (only pass the LAST token) ---
    last = ids[-1]
    while valid_idx < len(ids) + MAX_NEW_TOKENS:
        draft_toks = np.empty((0), dtype=np.int32)
        draft_idx = np.empty((0, TOP_K), dtype=np.int32)
        draft_vals = np.empty((0, TOP_K), dtype=np.float32)

        for _ in range(SPEC_K):
            y = np.array([[last]], dtype=np.int32)    # (1, 1)
            toks, topk_idx, topk_vals = draft_model.forward(y)
            last = toks[-1:]
            draft_toks = np.concatenate([draft_toks, last])
            draft_idx = np.concatenate([draft_idx, topk_idx], axis=0)
            draft_vals = np.concatenate([draft_vals, topk_vals], axis=0)

        # print(list(draft_toks))
        # print(draft_idx)

        _, base_idx, base_vals = base_model.forward(draft_toks, only_final=False)

        for i in range(SPEC_K):
            tok = draft_toks[i]
            print(f'{tok=}')

            print(base_idx)

            draft_idx_mask = (draft_idx[i,:] == tok)
            if draft_idx_mask.sum() == 0:
                print('reject!')
            elif draft_idx_mask.sum() > 1:
                raise Exception('WHAT')
            p_draft = np.where(draft_idx_mask, draft_vals[i,:], 0).sum()
            print(f'{p_draft=}')

            base_idx_mask = (base_idx[i,:] == tok)
            if base_idx_mask.sum() == 0:
                print('reject!')
            elif base_idx_mask.sum() > 1:
                raise Exception('WHAT')
            p_base = np.where(base_idx_mask, base_vals[i,:], 0).sum()
            print(f'{p_base=}')

            raise Exception()

        # if eos is not None and last == eos:
        #     break


    # --- Report & show text ---
    decode_tokens = len(draft)

    print(draft_model.decode(draft))

if __name__ == "__main__":
    main()

