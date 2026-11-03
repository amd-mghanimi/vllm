# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn.functional as F

from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.attention.attention import get_attention_context
from vllm.models.common.amd.ops.fused_qk_norm_rope_kv import (
    fused_qk_norm_rope_kv_cache_update,
    has_aiter_qk_norm_rope_kv_cache_update,
)
from vllm.models.muse_spark.common.ops.fused_qk_norm_rope_kv import (
    fused_qk_norm_rope_kv_insert,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.kv_cache_interface import is_quantized_kv_cache

# AITER's fp8 cache is e4m3 only.
_SHUFFLED_KV_CACHE_DTYPES = frozenset(
    {"auto", "bfloat16", "float16", "fp8", "fp8_e4m3"}
)


def rocm_fused_qk_norm_rope_kv_insert(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor | None,
    layer_name: str,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    eps: float,
) -> None:
    if torch.compiler.is_compiling():
        torch.ops.vllm.muse_spark_fused_qk_norm_rope_kv_insert(
            qkv,
            positions,
            cos_sin_cache,
            _encode_layer_name(layer_name),
            num_heads,
            num_kv_heads,
            head_dim,
            rotary_dim,
            eps,
        )
    else:
        fused_qk_norm_rope_kv_insert(
            qkv,
            positions,
            cos_sin_cache,
            layer_name,
            num_heads,
            num_kv_heads,
            head_dim,
            rotary_dim,
            eps,
        )


def _muse_spark_fused_qk_norm_rope_kv_insert(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor | None,
    layer_name: LayerNameType,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    eps: float,
) -> None:
    fused_qk_norm_rope_kv_insert(
        qkv,
        positions,
        cos_sin_cache,
        _resolve_layer_name(layer_name),
        num_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        eps,
    )


def _muse_spark_fused_qk_norm_rope_kv_insert_fake(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor | None,
    layer_name: LayerNameType,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    eps: float,
) -> None:
    return


if current_platform.is_rocm():
    direct_register_custom_op(
        op_name="muse_spark_fused_qk_norm_rope_kv_insert",
        op_func=_muse_spark_fused_qk_norm_rope_kv_insert,
        fake_impl=_muse_spark_fused_qk_norm_rope_kv_insert_fake,
        mutates_args=["qkv"],
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
    return attn_layer.kv_cache_dtype in _SHUFFLED_KV_CACHE_DTYPES


def graph_capture_active() -> bool:
    """Whether this forward is recording a CUDA/HIP graph; replays run no Python."""
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


_SHUFFLED_INSERT_CONSTANTS: dict[tuple, torch.Tensor | tuple[torch.Tensor, ...]] = {}


def _shuffled_insert_constant(key: tuple, make):
    """Read-only constants, built eagerly so no graph pool owns them."""
    value = _SHUFFLED_INSERT_CONSTANTS.get(key)
    if value is None:
        if graph_capture_active():
            raise RuntimeError(
                "Muse shuffled KV insert constants must be built before graph capture"
            )
        value = make()
        _SHUFFLED_INSERT_CONSTANTS[key] = value
    return value


def unit_norm_weight(
    head_dim: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    return _shuffled_insert_constant(
        ("norm", head_dim, dtype, device),
        lambda: torch.ones(head_dim, dtype=dtype, device=device),
    )


def _identity_rope(
    head_dim: int, dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    def make() -> tuple[torch.Tensor, torch.Tensor]:
        half = head_dim // 2
        table = torch.cat([torch.ones(half), torch.zeros(half)]).view(1, head_dim)
        zero = torch.zeros(1, dtype=torch.int64, device=device)
        return table.to(device, dtype), zero

    return _shuffled_insert_constant(("nope", head_dim, dtype, device), make)


def kernel_rope_table(rotary_emb: torch.nn.Module, dtype: torch.dtype) -> torch.Tensor:
    """``cos||sin`` in ``dtype``, cached on the module; its own table is kept."""
    table = rotary_emb.cos_sin_cache
    if table.dtype == dtype:
        return table
    cached = getattr(rotary_emb, "_aiter_cos_sin_cache", None)
    if cached is None or cached.dtype != dtype or cached.device != table.device:
        if graph_capture_active():
            raise RuntimeError("Muse AITER RoPE table must be built before capture")
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


def shuffled_insert_buffers(
    qkv: torch.Tensor,
    slot_mapping: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-forward ``(rows, q_out, slots)``.

    Eager calls allocate normally. Under capture these come from vLLM's graph
    pool and keep their addresses for replay; they are never cached here.
    """
    rows = qkv_rows(qkv, (num_heads + 2 * num_kv_heads) * head_dim)
    q_out = rows.new_empty(rows.shape[0], num_heads * head_dim)
    return rows, q_out, pad_slot_mapping(slot_mapping, rows.shape[0])


def _emit_fp8_query(metadata: object, attn_layer: torch.nn.Module) -> bool:
    """fp8 Q for a decode-only step on a full-attention fp8 shuffled layer.

    Sliding-window decode stays on ``pa_decode_gluon`` with a bf16 query.
    A step that also prefills or extends keeps a bf16 query, because those
    paths run flash-attn on bf16 K/V. The attention backend quantizes the
    decode slice itself in that case.
    """
    if metadata is None or not is_quantized_kv_cache(attn_layer.kv_cache_dtype):
        return False
    impl = getattr(attn_layer, "impl", None)
    if impl is None or impl.sliding_window[0] != -1:
        return False
    return (
        getattr(metadata, "num_prefills", 1) == 0
        and getattr(metadata, "num_extends", 1) == 0
        and getattr(metadata, "num_decodes", 0) > 0
    )


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
    """Q after norm and RoPE with shuffled K/V inserted, or None with no cache."""
    norm_weight = unit_norm_weight(head_dim, qkv.dtype, qkv.device)
    cos_sin_cache, kernel_positions, rotary_dim = kernel_rope_args(
        rotary_emb, positions, head_dim, qkv.dtype
    )
    metadata, attn_layer, kv_cache, slot_mapping = get_attention_context(layer_name)
    if slot_mapping is None or kv_cache.numel() == 0:
        return None
    rows, q_out, slots = shuffled_insert_buffers(
        qkv, slot_mapping, num_heads, num_kv_heads, head_dim
    )
    # Same per-tensor scale the attention kernel dequantizes with. Allocated
    # per forward, like q_out, so a decode CUDA graph owns a stable address.
    q_out_fp8 = None
    if _emit_fp8_query(metadata, attn_layer):
        q_out_fp8 = torch.empty(
            q_out.shape, dtype=current_platform.fp8_dtype(), device=q_out.device
        )
    fused_qk_norm_rope_kv_cache_update(
        qkv=rows,
        q_weight=norm_weight,
        k_weight=norm_weight,
        cos_sin_cache=cos_sin_cache,
        positions=kernel_positions,
        num_heads_q=num_heads,
        num_heads_k=num_kv_heads,
        head_dim=head_dim,
        is_neox=False,
        rms_norm_eps=eps,
        q_out=q_out,
        kv_cache=kv_cache,
        slot_mapping=slots,
        k_scale=attn_layer._k_scale_cpu,
        v_scale=attn_layer._v_scale_cpu,
        kv_cache_dtype=attn_layer.kv_cache_dtype,
        rotary_dim=rotary_dim,
        use_shuffle_layout=True,
        q_out_fp8=q_out_fp8,
        per_tensor_q_scale=attn_layer._q_scale if q_out_fp8 is not None else None,
    )
    return q_out if q_out_fp8 is None else q_out_fp8
