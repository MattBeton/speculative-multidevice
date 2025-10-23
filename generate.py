# pip install mlx-lm
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

import numpy as np

from model import MLXGenerationModel
from timing import TokenTimer

DRAFT_MODEL_PATH = Path("/Users/frank/.cache/huggingface/hub/models--mlx-community--Llama-3.2-1B-Instruct-bf16/snapshots/").glob("*")
DRAFT_MODEL_PATH = next(DRAFT_MODEL_PATH)
# BASE_MODEL_PATH = Path("/Users/frank/.cache/huggingface/hub/models--mlx-community--Llama-3.2-1B-Instruct-bf16/snapshots/").glob("*")
BASE_MODEL_PATH = Path("/Users/frank/.cache/huggingface/hub/models--mlx-community--Llama-3.2-3B-Instruct/snapshots/").glob("*")
BASE_MODEL_PATH = next(BASE_MODEL_PATH)

# PROMPT = "Write a terse haiku about Apple MLX."
PROMPT = "Why is the sky blue?"

MAX_NEW_TOKENS = 64
SPEC_K_VALUES = [4, 8]
RUNS = 3


def _summarize_phase(results: List[Dict], key: str) -> Dict[str, float]:
    runs = len(results)
    tokens_total = sum(r[key].tokens for r in results if r[key] is not None)
    seconds_total = sum(r[key].seconds for r in results if r[key] is not None)
    tokens_avg = tokens_total / runs if runs else 0
    seconds_avg = seconds_total / runs if runs else 0
    tok_per_sec = (tokens_total / seconds_total) if seconds_total else float("inf")
    return {
        "tokens_avg": tokens_avg,
        "seconds_avg": seconds_avg,
        "tok_per_sec": tok_per_sec,
    }


def _speculative_single_run(
    draft_model: MLXGenerationModel,
    base_model: MLXGenerationModel,
    spec_k: int,
):
    if spec_k < 2:
        raise ValueError("spec_k must be at least 2 for speculative decoding.")

    draft_model.reset()
    base_model.reset()

    timer = TokenTimer()

    ids_draft = draft_model.tokenize(PROMPT)
    ids = base_model.tokenize(PROMPT)
    assert (ids == ids_draft).all()
    eos = base_model.eos_token_id()

    prefill_tokens = len(ids) - 1
    with timer.measure("prefill", lambda: prefill_tokens):
        draft_model.forward(ids[:-1])
        base_model.forward(ids[:-1])

    valid_idx = len(ids) - 1
    prompt_length = valid_idx

    total_drafted = 0
    total_accepted = 0

    with timer.measure("decode", lambda: max(len(ids) - prompt_length, 0)):
        while valid_idx < prompt_length + MAX_NEW_TOKENS:
            last = int(ids[-1])
            draft_toks = []
            draft_topk_idx = []
            draft_topk_vals = []

            for _ in range(spec_k):
                y = np.array([[last]], dtype=np.int32)
                tok, topk_idx, topk_vals = draft_model.forward(y)
                t = int(tok[0])
                last = t
                draft_toks.append(t)
                draft_topk_idx.append(topk_idx[0])
                draft_topk_vals.append(topk_vals[0])

            draft_toks = np.array(draft_toks, dtype=np.int32)
            draft_topk_idx = np.stack(draft_topk_idx, axis=0)
            draft_topk_vals = np.stack(draft_topk_vals, axis=0)
            total_drafted += len(draft_toks)

            toks_verify = np.concatenate((np.array([ids[-1]], dtype=np.int32), draft_toks))

            _, base_topk_idx, base_topk_vals = base_model.forward(
                toks_verify,
                only_final=False,
            )

            accepted_tokens = []
            hit_eos = False
            for i, tok in enumerate(draft_toks):
                d_idx_row = draft_topk_idx[i]
                d_val_row = draft_topk_vals[i]
                d_mask = (d_idx_row == tok)
                if d_mask.sum() != 1:
                    raise RuntimeError("Draft top-k should contain the sampled token exactly once.")
                draft_logit = d_val_row[d_mask][0]

                b_idx_row, b_val_row = base_topk_idx[i], base_topk_vals[i]
                b_mask = (b_idx_row == tok)
                in_base_topk = b_mask.any()
                base_logit = b_val_row[b_mask][0] if in_base_topk else float("-inf")

                if base_logit == float("-inf"):
                    accepted = False
                elif draft_logit <= base_logit:
                    accepted = True
                else:
                    u = np.random.uniform(0, 1)
                    accepted = u <= base_logit / draft_logit

                if not accepted:
                    break

                accepted_tokens.append(tok)

                if eos is not None and tok == eos:
                    hit_eos = True
                    break

            if accepted_tokens:
                total_accepted += len(accepted_tokens)
                ids = np.concatenate((ids, np.array(accepted_tokens, dtype=np.int32)))
                valid_idx += len(accepted_tokens)

            if hit_eos or (valid_idx >= prompt_length + MAX_NEW_TOKENS):
                break

            draft_model.trim_cache(spec_k - i)
            base_model.trim_cache(spec_k - i)

    generated_ids = ids[prompt_length:]
    return {
        "prefill": timer.get("prefill"),
        "decode": timer.get("decode"),
        "drafted": total_drafted,
        "accepted": total_accepted,
        "generated_ids": generated_ids.tolist(),
        "text": base_model.decode(generated_ids),
    }


def run_speculative(spec_k: int, runs: int = RUNS):
    draft_model = MLXGenerationModel(DRAFT_MODEL_PATH)
    base_model = MLXGenerationModel(BASE_MODEL_PATH)

    results: List[Dict] = []
    for _ in tqdm(range(runs)):
        results.append(_speculative_single_run(draft_model, base_model, spec_k))

    prefill_summary = _summarize_phase(results, "prefill")
    decode_summary = _summarize_phase(results, "decode")
    drafted_total = sum(r["drafted"] for r in results)
    accepted_total = sum(r["accepted"] for r in results)
    accept_rate = (accepted_total / drafted_total) if drafted_total else 0.0

    return {
        "spec_k": spec_k,
        "runs": runs,
        "prefill": prefill_summary,
        "decode": decode_summary,
        "accept": {
            "drafted_total": drafted_total,
            "accepted_total": accepted_total,
            "drafted_avg": drafted_total / runs if runs else 0,
            "accepted_avg": accepted_total / runs if runs else 0,
            "accept_rate": accept_rate,
        },
        "text": results[-1]["text"] if results else "",
    }


def _print_phase_summary(name: str, summary: Dict[str, float]) -> None:
    print(
        f"[{name:7}] avg {summary['tokens_avg']:.1f} toks in {summary['seconds_avg']:.3f}s  → {summary['tok_per_sec']:.1f} tok/s"
    )


def main():
    for idx, spec_k in enumerate(SPEC_K_VALUES):
        summary = run_speculative(spec_k)
        if idx:
            print()
        print(f"SPEC_K={spec_k} (runs={summary['runs']})")
        _print_phase_summary("prefill", summary["prefill"])
        _print_phase_summary("decode", summary["decode"])
        accept = summary["accept"]
        if accept["drafted_total"]:
            print(
                f"[spec    ] drafted {accept['drafted_avg']:.1f} toks/run, accepted {accept['accepted_avg']:.1f}  → {accept['accept_rate']:.1%} accept rate"
            )
        # print("\n" + summary["text"])


if __name__ == "__main__":
    main()
