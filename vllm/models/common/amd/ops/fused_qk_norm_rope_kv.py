# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AMD AITER helpers for fused QK-norm, RoPE, and KV-cache updates."""

import functools

import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import is_quantized_kv_cache


@functools.cache
def has_aiter_qk_norm_rope_kv_cache_update() -> bool:
    try:
        from aiter.ops.fused_qk_norm_rope_cache_quant import (  # noqa: F401
            fused_qk_norm_rope_cache_pts_quant_shuffle,
        )
    except ImportError:
        return False
    return True


def split_shuffled_kv_cache(
    kv_cache: torch.Tensor,
    num_kv_heads: int,
    head_dim: int,
    kv_cache_dtype: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return K/V views for AITER's shuffled KV-cache writer."""
    key_cache, value_cache = kv_cache.unflatten(-1, (num_kv_heads, head_dim)).unbind(1)
    if is_quantized_kv_cache(kv_cache_dtype):
        fp8_dtype = current_platform.fp8_dtype()
        key_cache = key_cache.view(fp8_dtype)
        value_cache = value_cache.view(fp8_dtype)
    return key_cache, value_cache


def fused_qk_norm_rope_kv_cache_update(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    num_heads_q: int,
    num_heads_k: int,
    head_dim: int,
    is_neox: bool,
    rms_norm_eps: float,
    q_out: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    kv_cache_dtype: str,
    rotary_dim: int,
    use_shuffle_layout: bool,
    return_kv: bool = False,
    k_out: torch.Tensor | None = None,
    v_out: torch.Tensor | None = None,
    q_out_fp8: torch.Tensor | None = None,
    per_tensor_q_scale: torch.Tensor | None = None,
) -> None:
    """Fuse Q/K norm, RoPE, and an AITER KV update."""
    key_cache, value_cache = split_shuffled_kv_cache(
        kv_cache, num_heads_k, head_dim, kv_cache_dtype
    )
    rocm_aiter_ops.fused_qk_norm_rope_and_cache(
        qkv=qkv,
        q_weight=q_weight,
        k_weight=k_weight,
        cos_sin_cache=cos_sin_cache,
        positions=positions,
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_k,
        num_heads_v=num_heads_k,
        head_dim=head_dim,
        is_neox=is_neox,
        rms_norm_eps=rms_norm_eps,
        q_out=q_out,
        k_cache=key_cache,
        v_cache=value_cache,
        slot_mapping=slot_mapping,
        k_scale=k_scale,
        v_scale=v_scale,
        k_out=k_out,
        v_out=v_out,
        return_kv=return_kv,
        use_shuffle_layout=use_shuffle_layout,
        block_size=key_cache.shape[1],
        x=16 // key_cache.element_size(),
        rotary_dim=rotary_dim,
        q_out_fp8=q_out_fp8,
        per_tensor_q_scale=per_tensor_q_scale,
    )
