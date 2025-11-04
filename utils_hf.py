from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import torch
import torch.nn as nn
from transformers import DynamicCache
from const import DEVICE, DTYPE, ATTN_IMPL_ENV, HF_TOKEN, PAD_ID

def get_pad_id(tokenizer):
    """Get the pad token ID from tokenizer, falling back to eos_token_id or 0."""
    return tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (
        tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    )

def print_cache(cache: DynamicCache, idx: int, layer: Optional[int]=0):
    print(cache.layers[layer].keys[idx, 0, :, 0])

def load_model(model_id: str):
    from_kwargs = {
        "dtype": DTYPE,
        "device_map": None,           # keep single process; move to one device below
        "low_cpu_mem_usage": True,
        "token": HF_TOKEN,
        "local_files_only": True,
    }
    if ATTN_IMPL_ENV:
        from_kwargs["attn_implementation"] = ATTN_IMPL_ENV

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        **from_kwargs,
    ).to(DEVICE) # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, 
        use_fast=True, 
        token=HF_TOKEN, 
        local_files_only=True,
    )

    return model, tokenizer

def mask_and_pos_ids(L: list[int]): # TODO: Should this be a list or a tensor?
    max_len = max(L)

    attention_mask = torch.zeros((len(L), max_len), dtype=torch.long, device=DEVICE)
    for i, l in enumerate(L):
        attention_mask[i, max_len - l:] = 1

    # position_ids / cache_position: 0..L_i-1 for non-pad tokens, 0 for pads
    # This works for both absolute and RoPE-style position handling.
    position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp_min(0)
    position_ids = position_ids.masked_fill(attention_mask == 0, 0)

    return attention_mask, position_ids 

@torch.inference_mode()
def prefill(model: nn.Module, tokens: list[list[int]], tokenizer=None):
    """Prefill the model with tokens. If tokenizer is provided, uses its pad_id, otherwise uses const.PAD_ID."""
    if tokenizer is not None:
        pad_id = get_pad_id(tokenizer)
    else:
        pad_id = PAD_ID
    max_len = max(len(x) for x in tokens)
    padded = [[pad_id] * (max_len - len(prompt)) + prompt for prompt in tokens]
    x = torch.tensor(padded, dtype=torch.long, device=DEVICE)

    cache = DynamicCache(
        config=model.config, 
    )

    attention_mask, position_ids = mask_and_pos_ids([len(x) for x in tokens])
    # print(attention_mask, position_ids)

    outputs = model(
        input_ids=x, 
        attention_mask=attention_mask,
        past_key_values=cache, 
        use_cache=True,
        position_ids=position_ids,
    )

    return outputs.past_key_values

def rollback_dynamic_per_row_simple(cache: DynamicCache, tokens: list[list[int]], r: list[int]):
    """
    Roll back r[i] tokens for each batch row i in a DynamicCache.
    The output cache maintains the same sequence length as the input, padding with zeros where needed.
    """
    assert cache.layers[0].keys is not None and cache.layers[0].values is not None
    assert all([x >= 0 for x in r])
    B = cache.layers[0].keys.shape[0]
    device = cache.layers[0].keys.device
    dtype = cache.layers[0].keys.dtype

    # Prepare destination cache
    dst = DynamicCache()

    for layer in range(len(cache)):
        K = cache.layers[layer].keys
        V = cache.layers[layer].values
        assert K is not None and V is not None

        _, H, S, D = K.shape

        K_new = K.new_zeros((B, H, S, D))
        V_new = V.new_zeros((B, H, S, D))

        # Copy per row
        for i in range(B):
            keep = S - r[i]
            if keep <= 0:
                continue
            # surviving tokens are the first 'keep' positions (earliest..latest-rollback)
            K_src = K[i, :, :keep, :]
            V_src = V[i, :, :keep, :]

            # right-aligned → write to the right, pad on the left implicitly
            start = S - keep
            K_new[i, :, start:, :] = K_src
            V_new[i, :, start:, :] = V_src
            # print(K_new[i, 0, :, 0])

        # print(dst.layers[layer].keys[i, 0, :, 0])
        dst.update(K_new, V_new, layer)

    tokens = [x[:len(x) - trim] for x, trim in zip(tokens, r)]

    return dst, tokens

@torch.inference_mode()
def generate_step(
    model: nn.Module,
    cache: DynamicCache,
    tokens: list[list[int]],
    lengths: torch.LongTensor,
):
    x = torch.tensor(tokens, dtype=torch.long, device=DEVICE).view(-1, 1)

    B = lengths.size(0)
    S_prev = cache.layers[0].keys.shape[2]

    # Use int64 (not bool) on MPS; build past + current token mask
    attn_mask = torch.zeros((B, S_prev + 1), dtype=torch.long, device=DEVICE)

    starts = (S_prev - lengths).clamp_min(0)                # (B,)
    idx = torch.arange(S_prev, device=DEVICE)[None, :]      # (1, S_prev)
    attn_mask[:, :-1] = (idx >= starts[:, None]).to(torch.long)
    attn_mask[:, -1] = 1

    pos_ids = lengths.view(B, 1)  # new token's RoPE position (0..L_i)

    out = model(
        input_ids=x,
        past_key_values=cache,
        use_cache=True,
        attention_mask=attn_mask,  # int mask
        position_ids=pos_ids,
    )

    next_tok = out.logits[:, -1].argmax(dim=-1).tolist()
    lengths = lengths + 1
    return [[t] for t in next_tok], lengths
