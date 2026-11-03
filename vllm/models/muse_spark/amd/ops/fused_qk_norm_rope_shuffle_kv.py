# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse-Spark policy for the shared AITER shuffled KV-cache insert."""

import functools

import torch
import torch.nn.functional as F

from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.attention.attention import get_attention_context
from vllm.models.common.ops.fused_qk_norm_rope_kv_amd import (
    fused_qk_norm_rope_kv_cache_update,
    has_aiter_qk_norm_rope_kv_cache_update,
)
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

# AITER's fp8 cache is e4m3 only.
SUPPORTED_KV_CACHE_DTYPES = frozenset(
    {"auto", "bfloat16", "float16", "fp8", "fp8_e4m3"}
)


def supports_shuffled_fused_insert(attn_layer: torch.nn.Module) -> bool:
    """Whether AITER can write this layer's shuffled KV cache from the model."""
    if not rocm_aiter_ops.is_shuffle_kv_cache_enabled():
        return False
    if not has_aiter_qk_norm_rope_kv_cache_update():
        return False
    if get_current_vllm_config().parallel_config.decode_context_parallel_size > 1:
        return False
    from vllm.v1.attention.backends.rocm_aiter_fa import AiterFlashAttentionBackend

    if attn_layer.attn_backend is not AiterFlashAttentionBackend:
        return False
    if attn_layer.kv_sharing_target_layer_name is not None:
        return False
    return attn_layer.kv_cache_dtype in SUPPORTED_KV_CACHE_DTYPES


@functools.cache
def unit_norm_weight(
    head_dim: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    return torch.ones(head_dim, dtype=dtype, device=device)


@functools.cache
def _identity_rope(
    head_dim: int, dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    half = head_dim // 2
    table = torch.cat([torch.ones(half), torch.zeros(half)]).view(1, head_dim)
    return table.to(device, dtype), torch.zeros(1, dtype=torch.int64, device=device)


def kernel_rope_table(rotary_emb: torch.nn.Module, dtype: torch.dtype) -> torch.Tensor:
    """``cos||sin`` in ``dtype``, cached on the module; its own table is kept."""
    table = rotary_emb.cos_sin_cache
    if table.dtype == dtype:
        return table
    cached = getattr(rotary_emb, "_aiter_cos_sin_cache", None)
    if cached is None or cached.dtype != dtype or cached.device != table.device:
        cached = table.to(dtype)
        rotary_emb._aiter_cos_sin_cache = cached
    return cached


def kernel_rope_args(
    rotary_emb: torch.nn.Module | None,
    positions: torch.Tensor,
    head_dim: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """``(cos_sin, positions, rotary_dim)``; rotary_dim 0 rotates the whole head.

    NoPE rotates by cos 1 and sin 0, read at position 0 for every token.
    """
    if rotary_emb is None:
        table, zero = _identity_rope(head_dim, dtype, positions.device)
        return table, zero.expand(positions.shape[0]), 0
    rotary_dim = rotary_emb.rotary_dim
    return (
        kernel_rope_table(rotary_emb, dtype),
        positions,
        0 if rotary_dim == head_dim else rotary_dim,
    )


def qkv_rows(qkv: torch.Tensor, qkv_size: int) -> torch.Tensor:
    """The ``[q | k | v]`` prefix with rows packed, as the kernel indexes it."""
    return qkv[:, :qkv_size].contiguous()


def pad_slot_mapping(slot_mapping: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """One slot per qkv row; the kernel reads a slot for every row it rotates."""
    num_slots = slot_mapping.shape[0]
    if num_slots >= num_tokens:
        return slot_mapping[:num_tokens]
    return F.pad(slot_mapping, (0, num_tokens - num_slots), value=PAD_SLOT_ID)


def aiter_qk_norm_rope_shuffled_insert(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rotary_dim: int,
    norm_weight: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    eps: float,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    kv_cache_dtype: str,
) -> torch.Tensor:
    """Norm and rotate packed ``[q | k | v]`` rows, write K/V, return Q.

    ``k_scale``/``v_scale`` are the per-tensor CPU scalars.
    """
    q_out = qkv.new_empty(qkv.shape[0], num_heads * head_dim)
    fused_qk_norm_rope_kv_cache_update(
        qkv=qkv,
        q_weight=norm_weight,
        k_weight=norm_weight,
        cos_sin_cache=cos_sin_cache,
        positions=positions,
        num_heads_q=num_heads,
        num_heads_k=num_kv_heads,
        head_dim=head_dim,
        is_neox=False,
        rms_norm_eps=eps,
        q_out=q_out,
        kv_cache=kv_cache,
        slot_mapping=slot_mapping,
        k_scale=k_scale,
        v_scale=v_scale,
        kv_cache_dtype=kv_cache_dtype,
        rotary_dim=rotary_dim,
        use_shuffle_layout=True,
    )
    return q_out


def shuffled_fused_qk_norm_rope_kv_insert(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    rotary_emb: torch.nn.Module | None,
    layer_name: str,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    eps: float,
) -> torch.Tensor | None:
    """Q after norm and RoPE with K/V inserted, or None with no cache bound."""
    _, attn_layer, kv_cache, slot_mapping = get_attention_context(layer_name)
    if slot_mapping is None or kv_cache.numel() == 0:
        return None
    rows = qkv_rows(qkv, (num_heads + 2 * num_kv_heads) * head_dim)
    cos_sin_cache, kernel_positions, rotary_dim = kernel_rope_args(
        rotary_emb, positions, head_dim, qkv.dtype
    )
    return aiter_qk_norm_rope_shuffled_insert(
        rows,
        kernel_positions,
        cos_sin_cache,
        rotary_dim,
        unit_norm_weight(head_dim, qkv.dtype, qkv.device),
        num_heads,
        num_kv_heads,
        head_dim,
        eps,
        kv_cache,
        pad_slot_mapping(slot_mapping, rows.shape[0]),
        attn_layer._k_scale_cpu,
        attn_layer._v_scale_cpu,
        attn_layer.kv_cache_dtype,
    )
