# pip install mlx-lm
from pathlib import Path
import numpy as np

from model import MLXGenerationModel
from timing import TokenTimer

MODEL_PATH = next(Path("/Users/frank/.cache/huggingface/hub/models--mlx-community--Llama-3.2-3B-Instruct/snapshots/").glob("*"))

PROMPT = "Write a terse haiku about Apple MLX."
MAX_NEW_TOKENS = 64
RUNS = 3


def _pure_single_run(model: MLXGenerationModel):
    model.reset()
    ids = model.tokenize(PROMPT)

    timer = TokenTimer()

    # --- PREFILL ---
    with timer.measure("prefill", lambda: len(ids)):
        toks, _, _ = model.forward(ids, only_final=False)

    # First generated token comes from the last prefill logits (no extra model call)
    generated = [int(toks[-1])]
    eos = model.eos_token_id()

    # --- DECODE one token at a time with KV cache (only pass the LAST token) ---
    last = generated[-1]
    with timer.measure("decode", lambda: len(generated)):
        for _ in range(MAX_NEW_TOKENS - 1):
            y = np.array([[last]], dtype=np.int32)    # (1, 1)
            toks, _, _ = model.forward(y)
            last = int(toks[-1])
            generated.append(last)
            if eos is not None and last == eos:
                break

    return {
        "prefill": timer.get("prefill"),
        "decode": timer.get("decode"),
        "generated": generated,
        "text": model.decode(generated),
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


def run_pure(runs: int = RUNS):
    model = MLXGenerationModel(MODEL_PATH)
    results = []
    for _ in range(runs):
        results.append(_pure_single_run(model))

    prefill_summary = _summarize_phase(results, "prefill")
    decode_summary = _summarize_phase(results, "decode")

    return {
        "runs": runs,
        "prefill": prefill_summary,
        "decode": decode_summary,
        "text": results[-1]["text"] if results else "",
    }


def _print_phase_summary(name: str, summary: dict):
    print(
        f"[{name:7}] avg {summary['tokens_avg']:.1f} toks in {summary['seconds_avg']:.3f}s  → {summary['tok_per_sec']:.1f} tok/s"
    )


def main():
    summary = run_pure()
    _print_phase_summary("prefill", summary["prefill"])
    _print_phase_summary("decode", summary["decode"])
    print("\n" + summary["text"])


if __name__ == "__main__":
    main()
