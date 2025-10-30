# pip install mlx-lm
from pathlib import Path
import numpy as np

from model import MLXGenerationModel
from timing import TokenTimer

DRAFT_MODEL_PATH = next(Path("/Users/frank/.cache/huggingface/hub/models--mlx-community--Llama-3.2-1B-Instruct-bf16/snapshots/").glob("*"))
BASE_MODEL_PATH = next(Path("/Users/frank/.cache/huggingface/hub/models--mlx-community--Llama-3.2-3B-Instruct/snapshots/").glob("*"))

PROMPT = "Why is the sky blue?"
MAX_NEW_TOKENS = 64
SPEC_K_VALUES = [4, 8]
RUNS = 10


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

            base_toks, base_topk_idx, base_topk_vals = base_model.forward(
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

            # Number of drafted tokens accepted
            m = len(accepted_tokens)  # m in [0, spec_k]

            # Append the base fallback only if we didn't end on EOS in the draft.
            base_appended = 0
            if not hit_eos:
                # Base returned K+1 samples (positions 0..K); choose index m.
                base_tok = int(base_toks[m])
                accepted_tokens.append(base_tok)
                base_appended = 1

            # Count only the accepted draft tokens (exclude the base fallback).
            total_accepted += m

            ids = np.concatenate((ids, np.array(accepted_tokens, dtype=np.int32)))
            valid_idx += (m + base_appended)

            # ---- Align caches (trim BY delta, not TO absolute length) ----
            # Drafter consumed K steps; Base consumed K+1 steps during verification.
            # We committed (m + base_appended) steps into `ids`.
            draft_trim = spec_k - (m + base_appended)
            if draft_trim > 0:
                draft_model.trim_cache(draft_trim)
            elif draft_trim < 0:
                # Only possible when m==spec_k and base_appended==1.
                # Catch up the draft cache with the last accepted draft token (d_{K-1}).
                draft_model.forward(np.array([[int(draft_toks[m - 1])]], dtype=np.int32))

            base_trim = (spec_k + 1) - (m + base_appended)
            if base_trim > 0:
                base_model.trim_cache(base_trim)

            if hit_eos or (valid_idx >= prompt_length + MAX_NEW_TOKENS):
                break

    generated_ids = ids[prompt_length:]
    generated_list = generated_ids.tolist()
    return {
        "prefill": timer.get("prefill"),
        "decode": timer.get("decode"),
        "drafted": total_drafted,
        "accepted": total_accepted,
        "generated": generated_list,
        "text": base_model.decode(generated_list),
    }


def _summarize_phase(results, key: str):
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


def run_speculative(spec_k: int, runs: int = RUNS):
    draft_model = MLXGenerationModel(DRAFT_MODEL_PATH)
    base_model = MLXGenerationModel(BASE_MODEL_PATH)

    results = []
    for _ in range(runs):
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


def _print_phase_summary(name: str, summary: dict):
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
        print("\n" + summary["text"])


if __name__ == "__main__":
    main()
