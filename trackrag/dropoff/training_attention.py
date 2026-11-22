"""Bound temporary self-attention memory for the local Qwen training experiment.

All prompt positions and all causally available keys participate. Query tiles are
recomputed during backward, so a complete sequence-square score matrix is never
stored for the 32 Qwen heads together. Inference continues using native SDPA.
"""
from copy import copy


def attention_rows(query, key, value, mask, *, start, end, causal, scale):
    import torch
    import torch.nn.functional as F

    # Native SDPA assumes query position zero when is_causal=True. Number the
    # rows from their actual starting position for all noninitial tiles.
    if mask is None:
        allowed = None
        if causal:
            qi = torch.arange(start, end, device=query.device)[:, None]
            ki = torch.arange(key.shape[-2], device=query.device)[None, :]
            allowed = qi >= ki
    else:
        allowed = mask[..., start:end, :] if mask.shape[-2] != 1 else mask
    return F.scaled_dot_product_attention(query, key, value, attn_mask=allowed, dropout_p=0.,
        is_causal=False, scale=scale, enable_gqa=(query.shape[1] != key.shape[1]))


def kernel():
    import torch

    class TiledAttention(torch.autograd.Function):
        @staticmethod
        def forward(ctx, query, key, value, mask, causal, scale, tile):
            if (query.ndim != 4 or key.shape != value.shape or query.shape[0] != key.shape[0]
                    or query.shape[-2:] != key.shape[-2:] or query.shape[1] % key.shape[1]
                    or (mask is not None and (mask.ndim < 2 or mask.requires_grad
                                              or mask.shape[-2] not in (1, query.shape[-2])
                                              or mask.shape[-1] != key.shape[-2]))
                    or not isinstance(tile, int) or tile <= 0):
                raise ValueError('Unsupported training self-attention dimensions or mask')
            ctx.save_for_backward(query, key, value, mask)
            ctx.causal, ctx.scale, ctx.tile = causal, scale, tile
            output = torch.empty_like(query)
            length = query.shape[-2]
            for start in range(0, length, tile):
                end = min(start + tile, length)
                keys = end if causal and mask is None else length
                output[..., start:end, :] = attention_rows(query[..., start:end, :], key[..., :keys, :],
                    value[..., :keys, :], mask, start=start, end=end, causal=causal, scale=scale)
            return output

        @staticmethod
        def backward(ctx, gradient):
            query, key, value, mask = ctx.saved_tensors
            q_grad = torch.empty_like(query)
            # Key/value gradients overlap between query tiles. Accumulate those
            # contributions in FP32 for BF16/FP16 inputs before a single cast.
            accumulator = torch.float32 if key.dtype in (torch.bfloat16, torch.float16) else key.dtype
            k_grad = torch.zeros_like(key, dtype=accumulator)
            v_grad = torch.zeros_like(value, dtype=accumulator)
            length = query.shape[-2]
            for start in range(0, length, ctx.tile):
                end = min(start + ctx.tile, length)
                keys = end if ctx.causal and mask is None else length
                with torch.enable_grad():
                    q = query[..., start:end, :].detach().requires_grad_(True)
                    k = key[..., :keys, :].detach().requires_grad_(True)
                    v = value[..., :keys, :].detach().requires_grad_(True)
                    out = attention_rows(q, k, v, mask, start=start, end=end,
                                         causal=ctx.causal, scale=ctx.scale)
                    dq, dk, dv = torch.autograd.grad(out, (q, k, v), gradient[..., start:end, :])
                q_grad[..., start:end, :] = dq
                k_grad[..., :keys, :].add_(dk)
                v_grad[..., :keys, :].add_(dv)
                del out, q, k, v, dq, dk, dv
            return q_grad, k_grad.to(key.dtype), v_grad.to(value.dtype), None, None, None, None

    return TiledAttention.apply


def install_qwen_training_attention(layers, *, tile=64):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    bounded = kernel()
    def attention(module, query, key, value, attention_mask, dropout=0., scaling=None, **kwargs):
        if dropout or kwargs.get('is_causal', True) is not True or not module.is_causal:
            raise ValueError('The tiled training profile requires zero-dropout causal Qwen self attention')
        out = bounded(query, key, value, attention_mask, attention_mask is None, scaling, tile)
        return out.transpose(1, 2).contiguous(), None

    name = 'trackrag_tiled_training_sdpa_v1'
    ALL_ATTENTION_FUNCTIONS.register(name, attention)
    for layer in layers:
        attn = layer.self_attn
        if not attn.is_causal or attn.attention_dropout != 0 or attn.config._attn_implementation != 'sdpa':
            raise ValueError('Unexpected native attention module in the Qwen text stack')
        # Keep the parent language model's native SDPA mask builder unchanged.
        # Only this layer's dispatch configuration selects the training kernel.
        attn.config = copy(attn.config)
        attn.config._attn_implementation = name
