#!/usr/bin/env python3
import os, json, time, argparse, requests, itertools

BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
API_KEY  = os.environ.get("VLLM_API_KEY", "dummy")
MODEL    = os.environ.get("MODEL", "meta-llama/Llama-3.2-1B")
HDRS     = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

def make_batch_prompts(base_prompt: str, batch_size: int, prompts_file: str | None):
    if prompts_file:
        with open(prompts_file, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if not lines:
            lines = [base_prompt]
        # Cycle if fewer lines than batch_size
        return list(itertools.islice(itertools.cycle(lines), batch_size))
    return [base_prompt] * batch_size

def stream_batched(prompts, max_tokens: int, temperature: float = 0.0):
    """
    Stream a single batched request. Returns:
      ttft_by_idx: dict[idx] = seconds to first token (per sequence)
      decode_tokens_by_idx: dict[idx] = generated_tokens - 1 (exclude first token)
      decode_window_s: (t_last - t_first) across all sequences
    """
    url = f"{BASE_URL}/completions"
    payload = {
        "model": MODEL,
        "prompt": prompts,          # <-- batched prompts
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "logprobs": 0,              # keep payload tiny for timing
        "echo": False,
    }
    t_request = time.monotonic()
    first_tok_time = {}   # idx -> timestamp of first token
    last_tok_time  = {}   # idx -> timestamp of last token
    tok_counts     = {}   # idx -> total tokens streamed

    with requests.post(url, headers=HDRS, json=payload, stream=True, timeout=1200) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue

            choices = obj.get("choices") or []
            if not choices:
                continue
            ch = choices[0]
            idx = ch.get("index", 0)

            # Heuristic: count 1 token per chunk (cheap & robust). If your build
            # includes logprobs.token_ids, you could do len(token_ids) instead.
            delta_text = ch.get("text", "")
            step_tokens = 1 if delta_text else 0

            if step_tokens > 0:
                now = time.monotonic()
                if idx not in first_tok_time:
                    first_tok_time[idx] = now
                last_tok_time[idx] = now
                tok_counts[idx] = tok_counts.get(idx, 0) + step_tokens

    if not first_tok_time:
        # No tokens across the batch
        return {}, {}, float("nan")

    # Decode window across the batch
    t_first_any = min(first_tok_time.values())
    t_last_any  = max(last_tok_time.values())
    decode_window_s = max(t_last_any - t_first_any, 1e-9)

    # Per-seq TTFT and decode token counts (exclude first token)
    ttft_by_idx = {i: (first_tok_time[i] - t_request) for i in first_tok_time}
    decode_tokens_by_idx = {i: max(tok_counts.get(i, 0) - 1, 0) for i in tok_counts}

    return ttft_by_idx, decode_tokens_by_idx, decode_window_s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="Why is the sky blue?")
    ap.add_argument("--prompts-file", default=None, help="optional file with one prompt per line")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    args = ap.parse_args()

    print(f"BASE_URL   : {BASE_URL}")
    print(f"MODEL      : {MODEL}")
    print(f"BATCH_SIZE : {args.batch_size}")
    print(f"max_tokens : {args.max_tokens}, runs={args.runs}, warmup={args.warmup}\n")

    # Prepare batch prompts
    batch_prompts = make_batch_prompts(args.prompt, args.batch_size, args.prompts_file)

    # Warmup (short)
    for i in range(args.warmup):
        try:
            stream_batched(batch_prompts, min(8, args.max_tokens), temperature=0.0)
        except Exception as e:
            print(f"(warmup {i+1}/{args.warmup}) error: {e}")

    total_decode_tokens = 0
    total_decode_time   = 0.0
    ok_runs = 0

    for i in range(args.runs):
        try:
            ttft_by_idx, decode_tokens_by_idx, window_s = stream_batched(
                batch_prompts, args.max_tokens, temperature=0.0
            )
            batch_dec_toks = sum(decode_tokens_by_idx.values())
            batch_dec_tps  = batch_dec_toks / window_s if window_s > 0 else float("nan")

            # (Optional) show a compact per-run summary
            avg_ttft = sum(ttft_by_idx.values()) / len(ttft_by_idx) if ttft_by_idx else float("nan")
            print(f"run {i+1:02d}: avg_TTFT={avg_ttft:.3f}s, "
                  f"decode_tokens={batch_dec_toks}, window={window_s:.3f}s "
                  f"→ batch_decode_TPS={batch_dec_tps:.1f}")

            if window_s > 0:
                total_decode_tokens += batch_dec_toks
                total_decode_time   += window_s
                ok_runs += 1
        except Exception as e:
            print(f"run {i+1:02d} error: {e}")

    if ok_runs:
        avg_tps = total_decode_tokens / total_decode_time if total_decode_time > 0 else float("nan")
        print(f"\nAverages over {ok_runs} runs:")
        print(f"  avg batch decode TPS: {avg_tps:.1f} tok/s (sum of decoded tokens across batch / wall time)")
        print(f"  total decoded tokens: {total_decode_tokens}")
        print(f"  total decode walltime: {total_decode_time:.3f}s")
    else:
        print("\nNo successful runs recorded.")

if __name__ == "__main__":
    main()

