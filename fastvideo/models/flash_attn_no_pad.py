from einops import rearrange
from flash_attn import flash_attn_varlen_qkvpacked_func, flash_attn_qkvpacked_func, flash_attn_qkv_separate
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
    return flash_attn_qkv_separate(
        q, k, v,
        dropout_p,
        softmax_scale,
        causal
    )
