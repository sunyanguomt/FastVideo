from einops import rearrange
from flash_attn import flash_attn_varlen_qkvpacked_func, flash_attn_qkvpacked_func
from flash_attn.flash_attn_interface import (
    _flash_attn_forward,
    _flash_attn_backward
)
from flash_attn.bert_padding import pad_input, unpad_input

import torch
import torch.nn.functional as F


def flash_attn_no_pad(qkv, key_padding_mask, causal=False, dropout_p=0.0, softmax_scale=None):
    """
    Original function that accepts stacked qkv [B, S, 3, H, D].
    """
    output_unpad = flash_attn_qkvpacked_func(
            qkv,
            dropout_p,
            softmax_scale,
            causal
    )
    return output_unpad


class FlashAttnSeparateQKVFunc(torch.autograd.Function):
    """
    Custom autograd function for separate q, k, v input.
    Mimics FlashAttnQKVPackedFunc but for separate tensors.
    """
    @staticmethod
    def forward(ctx, q, k, v, dropout_p, softmax_scale, causal):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        
        # Direct call to low-level forward (same as packed version)
        out, q_saved, k_saved, v_saved, out_padded, softmax_lse, S_dmask, rng_state = _flash_attn_forward(
            q, k, v,
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size=(-1, -1),
            softcap=0.0,
            alibi_slopes=None,
            return_softmax=False,
        )
        
        # Save for backward (same as packed version)
        ctx.save_for_backward(q_saved, k_saved, v_saved, out_padded, softmax_lse, rng_state)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        
        return out
    
    @staticmethod
    def backward(ctx, *grad_outputs):
        dout = grad_outputs[0]
        q, k, v, out, softmax_lse, rng_state = ctx.saved_tensors
        
        # Allocate gradient buffers (same pattern as packed version)
        # dq = torch.empty_like(q)
        # dk = torch.empty_like(k)
        # dv = torch.empty_like(v)
        dqkv = torch.empty((*q.shape[:2], 3, *q.shape[2:]), dtype=q.dtype, device=q.device)
        
        # Direct call to low-level backward (same as packed version)
        _flash_attn_backward(
            dout,
            q, k, v,
            out,
            softmax_lse,
            dqkv[:, :, 0],
            dqkv[:, :, 1],
            dqkv[:, :, 2],
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            (-1, -1),  # window_size
            0.0,       # softcap
            None,      # alibi_slopes
            False,     # deterministic
            rng_state=rng_state,
        )
        
        def maybe_contiguous(x):
            return x.contiguous() if x is not None and x.stride(-1) != 1 else x
        
        dqkv = dqkv[..., : dout.shape[-1]]  # We could have padded the head dimension
        dqkv = maybe_contiguous(dqkv.transpose(1, 3))
        dq = dqkv[:,:,0,...]
        dk = dqkv[:,:,1,...]
        dv = dqkv[:,:,2,...]
        
        return dq, dk, dv, None, None, None


def flash_attn_no_pad_separate_qkv(q, k, v, key_padding_mask=None, causal=False, dropout_p=0.0, softmax_scale=None):
    """
    Optimized version that accepts separate q, k, v tensors.
    No torch.stack overhead in forward, no recompute in backward.
    Directly uses low-level flash-attention functions.
    
    Args:
        q: (batch_size, seqlen, nheads, headdim)
        k: (batch_size, seqlen, nheads, headdim)  
        v: (batch_size, seqlen, nheads, headdim)
        key_padding_mask: (batch_size, seqlen) - not used currently
        causal: whether to use causal attention
        dropout_p: dropout probability
        softmax_scale: scale for softmax, defaults to 1/sqrt(headdim)
    
    Returns:
        output: (batch_size, seqlen, nheads, headdim)
    """
    return FlashAttnSeparateQKVFunc.apply(
        q, k, v,
        dropout_p,
        softmax_scale,
        causal
    )
