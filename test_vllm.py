#!/usr/bin/env python3
import os, json, argparse, requests, math

BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
API_KEY  = os.environ.get("VLLM_API_KEY", "dummy")
MODEL    = os.environ.get("MODEL", "meta-llama/Llama-3.2-1B")

HDRS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

def completions(prompt: str, max_tokens: int, logprobs_k: int):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "logprobs": int(logprobs_k),
    }
    r = requests.post(f"{BASE_URL}/completions", headers=HDRS, data=json.dumps(payload), timeout=120)
    if r.status_code != 200:
        print("Request failed:", r.status_code, r.text)
        r.raise_for_status()
    return r.json()

def normalize_topk_entry_list(entry):
    """
    Normalize one step of top-k alternatives into a list of dicts:
    [{token: str, logprob: float, token_id: Optional[int]} ...]
    Accepts either:
      - list[dict(token, logprob, token_id?)]
      - dict[str->float]
    """
    if entry is None:
        return []
    if isinstance(entry, list):
        # Already a list of dicts; ensure keys exist
        out = []
        for it in entry:
            if isinstance(it, dict):
                out.append({
                    "token": it.get("token", ""),
                    "logprob": float(it.get("logprob", -math.inf)),
                    "token_id": it.get("token_id"),
                })
        return out
    if isinstance(entry, dict):
        # Dict of token->logprob
        return [{"token": k, "logprob": float(v), "token_id": None} for k, v in entry.items()]
    # Unknown shape
    return []

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="Why is the sky blue?")
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=10)
    args = ap.parse_args()

    print(f"Base URL: {BASE_URL}")
    print(f"Model:    {MODEL}")
    print(f"Prompt:   {args.prompt!r}\n")

    resp = completions(args.prompt, args.max_tokens, args.top_k)
    ch = resp["choices"][0]
    text_out = ch.get("text", "").strip()
    lp = ch.get("logprobs", {}) or {}

    token_ids   = lp.get("token_ids", [])           # may or may not be present
    tokens      = lp.get("tokens", [])              # strings
    token_lps   = lp.get("token_logprobs", [])      # sampled token logprob per step
    top_logprob = lp.get("top_logprobs", [])        # list[...] where each step may be list[dict] or dict

    print("Generated text:\n", text_out, "\n", sep="")

    # Step-wise table (be defensive about lengths)
    n = min(len(tokens), len(token_lps), len(token_ids) if token_ids else len(tokens))
    print("Step (i) | token_id | token | sampled_logprob")
    for i in range(n):
        tid = token_ids[i] if token_ids else None
        tok = (tokens[i] or "").encode("utf-8", "replace").decode("utf-8")
        lpi = token_lps[i]
        tid_str = f"{tid:>8}" if tid is not None else "   (none)"
        print(f"{i:02d}       {tid_str}   {tok!r:>12}   {lpi: .4f}")

    # Top-k alternatives for the first couple of steps
    steps_to_show = min(2, len(top_logprob))
    for s in range(steps_to_show):
        alts_norm = normalize_topk_entry_list(top_logprob[s])
        if not alts_norm:
            continue
        alts_sorted = sorted(alts_norm, key=lambda x: x.get("logprob", -math.inf), reverse=True)
        print(f"\nTop-{args.top_k} alternatives for step {s}:")
        for j, it in enumerate(alts_sorted[: min(10, len(alts_sorted))]):
            tid = it.get("token_id")
            tok = (it.get("token") or "").encode("utf-8", "replace").decode("utf-8")
            lpv = it.get("logprob")
            tid_str = f"{tid:>7}" if tid is not None else "  (none)"
            print(f"  {j:02d}  id={tid_str}  tok={tok!r:>12}  logprob={lpv: .4f}")

if __name__ == "__main__":
    main()
