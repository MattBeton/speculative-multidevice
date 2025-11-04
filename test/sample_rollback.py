import torch
from transformers import DynamicCache

def rollback_dynamic_per_row(cache: DynamicCache, r: torch.LongTensor):
    """
    Roll back r[i] tokens for each batch row i in a DynamicCache.
    Returns a *new* DynamicCache with time dim equal to max(L_i - r_i).
    """
    assert cache.layers[0].keys is not None
    B = cache.layers[0].keys.shape[0]
    device = cache.layers[0].keys.device
    uniq = torch.unique(r)

    # Build empty destination (we'll fill layer by layer)
    dst = DynamicCache()
    for layer in range(len(cache)):
        K = cache.layers[layer].keys
        V = cache.layers[layer].keys
        assert K is not None
        assert V is not None

        B, H, S_old, D = K.shape
        # Compute new per-row lengths after rollback
        L_after = torch.full((B,), S_old, dtype=torch.long, device=device) - r
        S_new = int(L_after.max().item())

        K_new = K.new_zeros(B, H, S_new, D)
        V_new = V.new_zeros(B, H, S_new, D)

        # For each rollback bucket, crop and scatter back
        for rv in uniq.tolist():
            idx = (r == rv).nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() == 0: 
                continue
            # Select sub-batch, crop rv tokens from the right
            K_sub = K.index_select(0, idx)
            V_sub = V.index_select(0, idx)

            # physical crop for this bucket
            S_keep = S_old - rv
            K_sub = K_sub[..., :S_keep, :]
            V_sub = V_sub[..., :S_keep, :]

            # place back into dst; zero-filling beyond S_keep keeps them "rolled back"
            K_new.index_copy_(0, idx, torch.nn.functional.pad(K_sub, (0,0,0,0,0, S_new - S_keep)))
            V_new.index_copy_(0, idx, torch.nn.functional.pad(V_sub, (0,0,0,0,0, S_new - S_keep)))

        dst.update(K_new, V_new, layer)

    return dst
