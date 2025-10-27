#!/usr/bin/env python3
import os, json, time, argparse, requests

BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
API_KEY  = os.environ.get("VLLM_API_KEY", "dummy")
MODEL    = os.environ.get("MODEL", "meta-llama/Llama-3.2-3B")

HDRS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

def stream_completion(prompt: str, max_tokens: int, temperature: float = 0.0):
    """Stream tokens and return (ttft_s, decode_tokens, decode_duration_s)."""
    url = f"{BASE_URL}/completions"
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,          # stream tokens
        "logprobs": 0,           # avoid extra payload to keep timing clean
        "echo": False,
    }
    t_req = time.monotonic()
    with requests.post(url, headers=HDRS, json=payload, stream=True, timeout=600) as r:
        r.raise_for_status()
        first_token_time = None
        last_token_time  = None
        gen_tokens = 0

        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue

            # vLLM sends one token per chunk typically
            choice = obj.get("choices", [{}])[0]
            finish_reason = choice.get("finish_reason")
            if finish_reason is not None:
                # finish_reason set at the end as well; ignore here
                pass

            delta_text = choice.get("text", "")
            # Count token(s). Streaming usually delivers one token; be robust if token_ids exist
            logprobs = choice.get("logprobs", {})
            token_ids = logprobs.get("token_ids")
            step_tokens = len(token_ids) if token_ids else (1 if delta_text else 0)
            if step_tokens > 0:
                now = time.monotonic()
                if first_token_time is None:
                    first_token_time = now
                last_token_time = now
                gen_tokens += step_tokens

        if first_token_time is None:
            # No tokens returned
            return float("nan"), 0, float("nan")

        ttft = first_token_time - t_req
        # decode excludes the *first* token time-window
        decode_tokens = max(gen_tokens - 1, 0)
        decode_duration = max((last_token_time - first_token_time), 1e-9)
        return ttft, decode_tokens, decode_duration

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="Why is the sky blue?")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    args = ap.parse_args()

    print(f"BASE_URL: {BASE_URL}")
    print(f"MODEL   : {MODEL}")
    print(f"PROMPT  : {args.prompt!r}")
    print(f"max_tokens={args.max_tokens}, runs={args.runs}, warmup={args.warmup}\n")

    # Warmup
    for i in range(args.warmup):
        try:
            stream_completion(args.prompt, min(8, args.max_tokens), temperature=0.0)
        except Exception as e:
            print(f"(warmup {i+1}/{args.warmup}) error: {e}")

    ttft_sum = 0.0
    dtoks_sum = 0
    dtime_sum = 0.0
    ok_runs = 0

    for i in range(args.runs):
        try:
            ttft, dtoks, dtime = stream_completion(args.prompt, args.max_tokens, temperature=0.0)
            d_tps = (dtoks / dtime) if dtime and dtoks >= 0 else float("nan")
            print(f"run {i+1:02d}: TTFT={ttft:.3f}s, decode_tokens={dtoks}, decode_time={dtime:.3f}s → decode_TPS={d_tps:.1f}")
            if not (ttft != ttft or d_tps != d_tps):  # NaN check
                ttft_sum += ttft
                dtoks_sum += dtoks
                dtime_sum += dtime
                ok_runs += 1
        except Exception as e:
            print(f"run {i+1:02d} error: {e}")

    if ok_runs:
        avg_ttft = ttft_sum / ok_runs
        avg_decode_tps = (dtoks_sum / dtime_sum) if dtime_sum > 0 else float("nan")
        print(f"\nAverages over {ok_runs} runs:")
        print(f"  avg TTFT          : {avg_ttft:.3f}s")
        print(f"  avg decode TPS    : {avg_decode_tps:.1f} tok/s (excludes prefill)")
        print(f"  total decode toks : {dtoks_sum}")
        print(f"  total decode time : {dtime_sum:.3f}s")
    else:
        print("\nNo successful runs recorded.")

if __name__ == "__main__":
    main()

