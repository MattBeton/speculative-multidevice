from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

import numpy as np

from model import MLXGenerationModel
from timing import TokenTimer

DRAFT_MODEL_PATH = Path("/Users/frank/.cache/huggingface/hub/models--mlx-community--Llama-3.2-1B-Instruct-bf16/snapshots/").glob("*")
DRAFT_MODEL_PATH = next(DRAFT_MODEL_PATH)

PROMPT = "Why is the sky blue?"

MAX_NEW_TOKENS = 64
SPEC_K_VALUES = [4, 8]
RUNS = 10




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
        print("\n" + summary["text"])


if __name__ == "__main__":
    main()
