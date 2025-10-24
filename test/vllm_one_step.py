# vllm_one_step.py
# pip install vllm transformers
from __future__ import annotations

import math
import time
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from transformers import PreTrainedTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


@dataclass
class StepTopK:
    token_ids: List[int]         # length <= top_k+1 (vLLM also returns the chosen token)
    logprobs: List[float]        # log P(token | context) for corresponding ids
    decoded: List[Optional[str]] # optional decoded token strings (for debugging)

@dataclass
class GenTrace:
    chosen_token_ids: List[int]  # tokens we (client) sampled and appended
    per_step_topk: List[StepTopK]
    text: str
    prefill_time: float          # time for first token generation (seconds)
    decode_time: float           # total time for subsequent tokens (seconds)
    prefill_tps: float           # prefill tokens per second
    decode_tps: float            # decode tokens per second


def _encode_prompt_ids(
    tok: PreTrainedTokenizer,
    messages: List[Dict[str, str]],
    *,
    add_generation_prompt: bool = True,
    fallback_role_key: str = "content",
) -> List[int]:
    """
    Try chat template; if unavailable, fall back to plain encoding of the last user message.
    """
    # Prefer chat templates (Instruct models)
    try:
        tmpl = getattr(tok, "chat_template", None)
        if tmpl:  # non-empty template string
            return tok.apply_chat_template(
                messages, add_generation_prompt=add_generation_prompt, tokenize=True
            )
    except Exception:
        # Fall through to raw encode
        pass

    # Fallback: encode the last message's content verbatim
    if not messages:
        raise ValueError("messages must be non-empty")
    last = messages[-1]
    text = last.get(fallback_role_key) or last.get("content") or ""
    if not isinstance(text, str) or not text:
        raise ValueError("fallback could not find a text string to encode")
    # Include special tokens so BOS/EOS (if defined) are handled
    return tok.encode(text, add_special_tokens=True)


def _normalize_from_logprobs(logps: np.ndarray) -> np.ndarray:
    """Convert (subset) log-probs to a proper probability vector."""
    # numerically stable softmax over the subset we have
    m = np.max(logps)
    probs = np.exp(logps - m)
    s = probs.sum()
    return probs / s if s > 0 else np.full_like(probs, 1.0 / probs.size)


def _sample_from_topk_dict(
    topk: Dict[int, "LogprobLike"], rng: np.random.Generator
) -> Tuple[int, StepTopK]:
    """
    topk: dict[token_id -> object with .logprob and .decoded_token (optional)]
    Returns (chosen_token_id, StepTopK record).
    """
    # vLLM may return up to K+1 entries (to include the actually chosen token). Sort by logprob desc.
    items = sorted(
        [(tid, obj) for tid, obj in topk.items()],
        key=lambda t: float(t[1].logprob),
        reverse=True,
    )

    token_ids = np.array([int(tid) for tid, _ in items], dtype=np.int64)
    logps = np.array([float(obj.logprob) for _, obj in items], dtype=np.float64)
    decoded = [getattr(obj, "decoded_token", None) for _, obj in items]

    probs = _normalize_from_logprobs(logps)
    idx = int(rng.choice(len(token_ids), p=probs))
    chosen = int(token_ids[idx])

    return chosen, StepTopK(
        token_ids=token_ids.tolist(),
        logprobs=logps.tolist(),
        decoded=decoded,
    )


def autoreg_one_step_vllm(
    model: str,
    messages: List[Dict[str, str]],
    *,
    top_k: int = 20,
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    seed: int = 90,
    trust_remote_code: bool = True,
) -> GenTrace:
    """
    Autoregressively generate by making exactly one vLLM call per token.
    Sampling is performed on the client from vLLM's returned top-k logprobs.

    Args:
      model: HF model id or local path (e.g., "meta-llama/Meta-Llama-3.2-3B-Instruct").
      messages: chat messages to encode with the model's chat template.
      top_k: number of top candidates to fetch per step (client samples from these).
      max_new_tokens: generation length cap.
      temperature: used only to compute vLLM's internal sample (we ignore it); logprobs are still provided.
      seed: RNG seed for client-side sampling.
      trust_remote_code/dtype/gpu_memory_utilization: forwarded to vLLM LLM().

    Returns:
      GenTrace with our chosen token ids, per-step top-k, and final decoded text.
    """
    # 1) Spin up vLLM
    llm = LLM(
        model=model,
        trust_remote_code=trust_remote_code,
    )
    tok: PreTrainedTokenizer = llm.get_tokenizer()

    # 2) Encode prompt as token IDs, with a safe fallback when no chat_template exists.
    prompt_ids: List[int] = _encode_prompt_ids(tok, messages, add_generation_prompt=True)

    rng = np.random.default_rng(seed)
    chosen: List[int] = []
    per_step: List[StepTopK] = []

    eos_id = getattr(tok, "eos_token_id", None)

    # Timing tracking
    prefill_time = 0.0
    decode_time = 0.0
    prompt_len = len(prompt_ids)

    # 3) One-call-per-token loop
    for step_idx in range(max_new_tokens):
        sp = SamplingParams(
            max_tokens=1,
            logprobs=int(top_k),
            temperature=float(temperature),
            top_p=1.0,             # do not truncate on server; we'll sample client-side
            detokenize=False,       # we track token ids and decode at the end
        )

        # Time the generation step
        start_time = time.perf_counter()
        out = llm.generate([TokensPrompt(prompt_token_ids=prompt_ids)], sp)
        elapsed = time.perf_counter() - start_time

        # First step is prefill (processing prompt + first token), rest is decode
        if step_idx == 0:
            prefill_time = elapsed
        else:
            decode_time += elapsed

        # RequestOutput -> outputs[0] is the only sequence; logprobs for the single generated step at index 0
        seq_out = out[0].outputs[0]
        step_logprobs: Dict[int, object] = seq_out.logprobs[0]  # dict[token_id -> Logprob dataclass]

        # Client-side sample from top-k distribution
        next_id, topk_record = _sample_from_topk_dict(step_logprobs, rng=rng)

        # Commit our choice and advance the context
        chosen.append(next_id)
        per_step.append(topk_record)
        prompt_ids.append(next_id)

        if eos_id is not None and next_id == int(eos_id):
            break

    text = tok.decode(chosen)

    # Calculate TPS
    # Prefill TPS: prompt tokens per second
    prefill_tps = prompt_len / prefill_time if prefill_time > 0 else 0.0
    # Decode TPS: generated tokens per second (excluding first token)
    num_decode_tokens = len(chosen) - 1 if len(chosen) > 1 else 0
    decode_tps = num_decode_tokens / decode_time if decode_time > 0 else 0.0

    return GenTrace(
        chosen_token_ids=chosen,
        per_step_topk=per_step,
        text=text,
        prefill_time=prefill_time,
        decode_time=decode_time,
        prefill_tps=prefill_tps,
        decode_tps=decode_tps,
    )


if __name__ == "__main__":
    import logging
    import os

    # Reduce vLLM logging verbosity to minimize latency from logging overhead
    os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
    logging.getLogger("vllm").setLevel(logging.ERROR)

    # Example usage
    MODEL = "meta-llama/Meta-Llama-3.2-3B"
    MODEL = "/root/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B/snapshots/13afe5124825b4f3751f836b40dafda64c1ed062/"
    MESSAGES = [{"role": "user", "content": "Why is the sky blue?"}]

    trace = autoreg_one_step_vllm(
        model=MODEL,
        messages=MESSAGES,
        top_k=20,
        max_new_tokens=64,
        temperature=1.0,
        seed=90,
    )

    # Display results
    print(f"Generated text: {trace.text}\n")
    print(f"Prefill TPS: {trace.prefill_tps:.2f} tokens/sec ({trace.prefill_time:.4f}s)")
    print(f"Decode TPS: {trace.decode_tps:.2f} tokens/sec ({trace.decode_time:.4f}s)")
    print(f"Total tokens generated: {len(trace.chosen_token_ids)}")

    # If you want to inspect the per-step top-k:
    # for i, step in enumerate(trace.per_step_topk[:3]):
    #     print(f"step {i} top-k ids:", step.token_ids[:10])
    #     print(f"step {i} top-k logps:", step.logprobs[:10])

