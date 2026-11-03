# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse-Spark QK-norm + RoPE + KV insert into the ROCm shuffle layout via AITER.

The fused result must match the unfused shuffle path: weightless fp32 QK
RMSNorm, GPT-J RoPE, then ``reshape_and_cache_shuffle_triton``.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm() or not torch.cuda.is_available():
    pytest.skip("AITER shuffle KV insert requires ROCm", allow_module_level=True)

import vllm.models.common.amd.ops.fused_qk_norm_rope_kv as shared_ops
import vllm.models.muse_spark.amd.ops.fused_qk_norm_rope_kv as shuffle_ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.models.muse_spark.common.layers import MuseSpark1xRMSNorm
from vllm.v1.attention.backends.rocm_aiter_fa import (
    AiterFlashAttentionBackend,
    reshape_and_cache_shuffle_triton,
)

pytestmark = pytest.mark.skipif(
    not shared_ops.has_aiter_qk_norm_rope_kv_cache_update(),
    reason="AITER fused_qk_norm_rope_cache_pts_quant_shuffle unavailable",
)

HEAD_DIM = 64
EPS = 1e-5
MAX_POS = 8192
DEVICE = "cuda"


@pytest.fixture(autouse=True)
def vllm_config():
    with set_current_vllm_config(VllmConfig()):
        yield


@pytest.fixture
def shuffle_on(monkeypatch):
    monkeypatch.setattr(rocm_aiter_ops, "_SHUFFLE_KV_CACHE_ENABLED", True)


def make_rope(rotary_dim):
    return RotaryEmbedding(
        head_size=HEAD_DIM,
        rotary_dim=rotary_dim,
        max_position_embeddings=MAX_POS,
        base=500000.0,
        is_neox_style=False,
        dtype=torch.float32,
    ).to(DEVICE)


def make_kv_cache(num_blocks, block_size, num_kv_heads, kv_cache_dtype, planes):
    """Random ``(B, 2, N, H*hs)`` cache; ``planes`` puts K/V outermost (LHBNC)."""
    dtype = torch.uint8 if kv_cache_dtype.startswith("fp8") else torch.bfloat16
    shape = [num_blocks, 2, block_size, num_kv_heads * HEAD_DIM]
    if planes:
        shape[0], shape[1] = shape[1], shape[0]
    if dtype == torch.uint8:
        # Skip NaN codes so equality checks stay meaningful.
        raw = torch.randint(0, 0x70, shape, dtype=dtype, device=DEVICE)
    else:
        raw = torch.randn(shape, dtype=dtype, device=DEVICE)
    return raw.transpose(0, 1) if planes else raw


def make_slot_mapping(num_tokens, num_blocks, block_size, num_pad, seed):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    slots = torch.randperm(num_blocks * block_size, generator=gen)[:num_tokens]
    if num_pad:
        slots[-num_pad:] = -1
    return slots.to(DEVICE)


def unfused_shuffle_reference(
    qkv,
    positions,
    rotary_emb,
    num_heads,
    num_kv_heads,
    key_cache,
    value_cache,
    slot_mapping,
    kv_cache_dtype,
    k_scale,
):
    """``MuseSpark1xAttention`` unfused path, then the backend's shuffle writer."""
    q_size, kv_size = num_heads * HEAD_DIM, num_kv_heads * HEAD_DIM
    q, k, v = qkv[:, : q_size + 2 * kv_size].split([q_size, kv_size, kv_size], -1)
    norm = MuseSpark1xRMSNorm(HEAD_DIM, eps=EPS, has_weight=False)
    q = norm(q.reshape(-1, HEAD_DIM), out_dtype=torch.float32).reshape(-1, q_size)
    k = norm(k.reshape(-1, HEAD_DIM), out_dtype=torch.float32).reshape(-1, kv_size)
    q, k = q.to(v.dtype), k.to(v.dtype)
    if rotary_emb is not None:
        q, k = rotary_emb.forward_cuda(positions, q, k)
    scale = torch.tensor([k_scale], dtype=torch.float32, device=DEVICE)
    reshape_and_cache_shuffle_triton(
        k.view(-1, num_kv_heads, HEAD_DIM),
        v.reshape(-1, num_kv_heads, HEAD_DIM),
        key_cache,
        value_cache,
        slot_mapping,
        kv_cache_dtype,
        scale,
        scale,
    )
    return q


def run_fused(
    qkv,
    positions,
    rotary_emb,
    num_heads,
    num_kv_heads,
    kv_cache,
    slot_mapping,
    k_scale,
    kv_cache_dtype,
):
    rows = shuffle_ops.qkv_rows(qkv, (num_heads + 2 * num_kv_heads) * HEAD_DIM)
    cos_sin, kernel_positions, rotary_dim = shuffle_ops.kernel_rope_args(
        rotary_emb, positions, HEAD_DIM, qkv.dtype
    )
    weight = shuffle_ops.unit_norm_weight(HEAD_DIM, qkv.dtype, qkv.device)
    scale = torch.tensor(k_scale, dtype=torch.float32)
    q_out = rows.new_empty(rows.shape[0], num_heads * HEAD_DIM)
    shared_ops.fused_qk_norm_rope_kv_cache_update(
        qkv=rows,
        q_weight=weight,
        k_weight=weight,
        cos_sin_cache=cos_sin,
        positions=kernel_positions,
        num_heads_q=num_heads,
        num_heads_k=num_kv_heads,
        head_dim=HEAD_DIM,
        is_neox=False,
        rms_norm_eps=EPS,
        q_out=q_out,
        kv_cache=kv_cache,
        slot_mapping=shuffle_ops.pad_slot_mapping(slot_mapping, rows.shape[0]),
        k_scale=scale,
        v_scale=scale,
        kv_cache_dtype=kv_cache_dtype,
        rotary_dim=rotary_dim,
        use_shuffle_layout=True,
    )
    return q_out


def bf16_ulp_diff(got, expected):
    """Distance in bf16 representable steps (sign-magnitude order)."""

    def ordered(x):
        bits = x.contiguous().view(torch.int16).int()
        return torch.where(bits < 0, -(bits & 0x7FFF), bits)

    return (ordered(got) - ordered(expected)).abs()


def assert_fp8_codes_close(got, expected):
    diff = (got.view(torch.uint8).int() - expected.view(torch.uint8).int()).abs()
    assert int(diff.max()) <= 1, f"fp8 cache differs by {int(diff.max())} codes"
    return int((diff != 0).sum())


def check_parity(
    num_tokens,
    kv_cache_dtype,
    block_size,
    rotary_dim,
    with_gate,
    num_pad=0,
    planes=False,
    k_scale=1.0,
    num_heads=16,
    num_kv_heads=2,
    seed=0,
):
    torch.manual_seed(seed)
    q_size, kv_size = num_heads * HEAD_DIM, num_kv_heads * HEAD_DIM
    width = 2 * q_size + 2 * kv_size if with_gate else q_size + 2 * kv_size
    qkv = torch.randn(num_tokens, width, dtype=torch.bfloat16, device=DEVICE) * 4
    positions = torch.randint(0, MAX_POS, (num_tokens,), device=DEVICE)
    num_blocks = 2 * num_tokens // block_size + 3
    slot_mapping = make_slot_mapping(num_tokens, num_blocks, block_size, num_pad, seed)
    ref_cache = make_kv_cache(
        num_blocks, block_size, num_kv_heads, kv_cache_dtype, planes
    )
    fused_cache = ref_cache.clone()
    if planes:
        fused_cache = ref_cache.transpose(0, 1).clone().transpose(0, 1)
    rope = make_rope(rotary_dim) if rotary_dim else None
    ref_q = unfused_shuffle_reference(
        qkv.clone(),
        positions,
        make_rope(rotary_dim) if rotary_dim else None,
        num_heads,
        num_kv_heads,
        *shared_ops.split_shuffled_kv_cache(
            ref_cache, num_kv_heads, HEAD_DIM, kv_cache_dtype
        ),
        slot_mapping,
        kv_cache_dtype,
        k_scale,
    )
    qkv_before = qkv.clone()
    fused_q = run_fused(
        qkv,
        positions,
        rope,
        num_heads,
        num_kv_heads,
        fused_cache,
        slot_mapping,
        k_scale,
        kv_cache_dtype,
    )
    torch.accelerator.synchronize()

    torch.testing.assert_close(qkv, qkv_before, rtol=0, atol=0)
    if rope is not None:
        assert rope.cos_sin_cache.dtype == torch.float32
    q_ulps = bf16_ulp_diff(fused_q, ref_q)
    stats = {"q_max_abs": float((fused_q.float() - ref_q.float()).abs().max())}
    stats["q_max_ulp"] = int(q_ulps.max())
    ref_k, ref_v = shared_ops.split_shuffled_kv_cache(
        ref_cache, num_kv_heads, HEAD_DIM, kv_cache_dtype
    )
    got_k, got_v = shared_ops.split_shuffled_kv_cache(
        fused_cache, num_kv_heads, HEAD_DIM, kv_cache_dtype
    )
    if kv_cache_dtype.startswith("fp8"):
        stats["k_code_mismatch"] = assert_fp8_codes_close(got_k, ref_k)
        stats["v_code_mismatch"] = assert_fp8_codes_close(got_v, ref_v)
    else:
        k_ulps = bf16_ulp_diff(got_k, ref_k)
        stats["k_max_ulp"] = int(k_ulps.max())
        stats["k_byte_mismatch"] = int((k_ulps != 0).sum())
        stats["v_byte_mismatch"] = int(
            (got_v.view(torch.int16) != ref_v.view(torch.int16)).sum()
        )
        assert stats["k_max_ulp"] <= 2, stats
        assert stats["v_byte_mismatch"] == 0, stats
    assert stats["q_max_ulp"] <= 2, stats
    print(
        f"T={num_tokens} {kv_cache_dtype} bs={block_size} rot={rotary_dim} "
        f"gate={with_gate} pad={num_pad} planes={planes}: {stats}"
    )
    return stats


@pytest.mark.parametrize("num_tokens", [1, 19, 300])
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
@pytest.mark.parametrize("block_size", [16, 32])
@pytest.mark.parametrize("rotary_dim", [HEAD_DIM, 0], ids=["rope", "nope"])
@pytest.mark.parametrize("with_gate", [True, False], ids=["gate", "nogate"])
def test_matches_unfused_shuffle_path(
    shuffle_on, num_tokens, kv_cache_dtype, block_size, rotary_dim, with_gate
):
    check_parity(num_tokens, kv_cache_dtype, block_size, rotary_dim, with_gate)


@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
@pytest.mark.parametrize("num_tokens", [1, 19])
def test_partial_rotary(shuffle_on, kv_cache_dtype, num_tokens):
    check_parity(num_tokens, kv_cache_dtype, 16, HEAD_DIM // 2, with_gate=True)


@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
def test_padding_rows_do_not_write(shuffle_on, kv_cache_dtype):
    check_parity(19, kv_cache_dtype, 16, HEAD_DIM, True, num_pad=7)


@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
def test_kv_planes_outermost(shuffle_on, kv_cache_dtype):
    check_parity(19, kv_cache_dtype, 32, HEAD_DIM, True, planes=True)


def test_fp8_non_unit_scale(shuffle_on):
    check_parity(300, "fp8", 16, HEAD_DIM, True, k_scale=0.75)


def test_other_head_counts(shuffle_on):
    check_parity(19, "auto", 16, HEAD_DIM, True, num_heads=4, num_kv_heads=1)


def test_qkv_rows_drops_the_gate_and_packs_rows():
    qkv = torch.randn(5, 7 * HEAD_DIM, device=DEVICE)
    rows = shuffle_ops.qkv_rows(qkv, 4 * HEAD_DIM)
    assert rows.shape == (5, 4 * HEAD_DIM) and rows.stride(0) == 4 * HEAD_DIM
    torch.testing.assert_close(rows, qkv[:, : 4 * HEAD_DIM], rtol=0, atol=0)
    packed = torch.randn(5, 4 * HEAD_DIM, device=DEVICE)
    assert shuffle_ops.qkv_rows(packed, 4 * HEAD_DIM).data_ptr() == packed.data_ptr()
    # A single row is contiguous whatever its stride, so the gate costs no copy.
    one = torch.randn(1, 7 * HEAD_DIM, device=DEVICE)
    one_rows = shuffle_ops.qkv_rows(one, 4 * HEAD_DIM)
    assert one_rows.is_contiguous() and one_rows.data_ptr() == one.data_ptr()


def test_pad_slot_mapping():
    slots = torch.arange(3, device=DEVICE)
    assert shuffle_ops.pad_slot_mapping(slots, 3).data_ptr() == slots.data_ptr()
    assert shuffle_ops.pad_slot_mapping(slots, 2).tolist() == [0, 1]
    assert shuffle_ops.pad_slot_mapping(slots, 5).tolist() == [0, 1, 2, -1, -1]


def test_kernel_rope_args():
    positions = torch.arange(6, device=DEVICE)
    table, pos, rotary_dim = shuffle_ops.kernel_rope_args(
        None, positions, HEAD_DIM, torch.bfloat16
    )
    assert rotary_dim == 0 and pos.shape == (6,) and pos.stride(0) == 0
    assert table.shape == (1, HEAD_DIM) and table.dtype == torch.bfloat16
    assert (
        table[0, : HEAD_DIM // 2].eq(1).all() and table[0, HEAD_DIM // 2 :].eq(0).all()
    )

    rope = make_rope(HEAD_DIM)
    table, pos, rotary_dim = shuffle_ops.kernel_rope_args(
        rope, positions, HEAD_DIM, torch.bfloat16
    )
    assert rotary_dim == 0 and pos is positions
    assert table.dtype == torch.bfloat16 and rope.cos_sin_cache.dtype == torch.float32
    torch.testing.assert_close(table, rope.cos_sin_cache.to(torch.bfloat16))
    again, _, _ = shuffle_ops.kernel_rope_args(
        rope, positions, HEAD_DIM, torch.bfloat16
    )
    assert again is table

    partial = make_rope(HEAD_DIM // 2)
    _, _, rotary_dim = shuffle_ops.kernel_rope_args(
        partial, positions, HEAD_DIM, torch.bfloat16
    )
    assert rotary_dim == HEAD_DIM // 2


@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
def test_split_shuffled_kv_cache(shuffle_on, kv_cache_dtype):
    cache = make_kv_cache(3, 16, 2, kv_cache_dtype, planes=False)
    key_cache, value_cache = shared_ops.split_shuffled_kv_cache(
        cache, 2, HEAD_DIM, kv_cache_dtype
    )
    assert key_cache.shape == value_cache.shape == (3, 16, 2, HEAD_DIM)
    if kv_cache_dtype == "fp8":
        assert key_cache.dtype == current_platform.fp8_dtype()
    assert key_cache.data_ptr() == cache.data_ptr()


def make_gate_layer(**overrides):
    attrs = dict(
        attn_backend=AiterFlashAttentionBackend,
        kv_sharing_target_layer_name=None,
        kv_cache_dtype="auto",
    )
    attrs.update(overrides)
    return SimpleNamespace(**attrs)


def test_supports_shuffled_fused_insert(monkeypatch):
    monkeypatch.setattr(rocm_aiter_ops, "_SHUFFLE_KV_CACHE_ENABLED", False)
    assert not shuffle_ops.supports_shuffled_fused_insert(make_gate_layer())
    monkeypatch.setattr(rocm_aiter_ops, "_SHUFFLE_KV_CACHE_ENABLED", True)
    assert shuffle_ops.supports_shuffled_fused_insert(make_gate_layer())
    assert shuffle_ops.supports_shuffled_fused_insert(
        make_gate_layer(kv_cache_dtype="fp8")
    )
    for overrides in (
        dict(kv_cache_dtype="fp8_e5m2"),
        dict(kv_sharing_target_layer_name="model.layers.0.self_attn.attn"),
        dict(attn_backend=SimpleNamespace()),
    ):
        assert not shuffle_ops.supports_shuffled_fused_insert(
            make_gate_layer(**overrides)
        )
    monkeypatch.setattr(
        shuffle_ops, "has_aiter_qk_norm_rope_kv_cache_update", lambda: False
    )
    assert not shuffle_ops.supports_shuffled_fused_insert(make_gate_layer())


@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
def test_layer_entry_point_reads_the_forward_context(
    monkeypatch, shuffle_on, kv_cache_dtype
):
    num_heads, num_kv_heads, num_tokens, block_size = 16, 2, 19, 16
    q_size, kv_size = num_heads * HEAD_DIM, num_kv_heads * HEAD_DIM
    qkv = torch.randn(
        num_tokens, 2 * q_size + 2 * kv_size, dtype=torch.bfloat16, device=DEVICE
    )
    positions = torch.randint(0, MAX_POS, (num_tokens,), device=DEVICE)
    cache = make_kv_cache(4, block_size, num_kv_heads, kv_cache_dtype, planes=True)
    expected_cache = cache.transpose(0, 1).clone().transpose(0, 1)
    # Slots for the real rows only, as an unpadded eager batch provides them.
    slot_mapping = make_slot_mapping(num_tokens - 3, 4, block_size, 0, seed=1)
    layer = SimpleNamespace(
        kv_cache_dtype=kv_cache_dtype,
        _k_scale_cpu=torch.tensor(1.0),
        _v_scale_cpu=torch.tensor(1.0),
    )
    context = [(None, layer, cache, None)]
    monkeypatch.setattr(shuffle_ops, "get_attention_context", lambda name: context[0])
    rope = make_rope(HEAD_DIM)
    args = (qkv, positions, rope, "layer", num_heads, num_kv_heads, HEAD_DIM, EPS)
    assert shuffle_ops.shuffled_fused_qk_norm_rope_kv_insert(*args) is None

    context[0] = (None, layer, cache, slot_mapping)
    q = shuffle_ops.shuffled_fused_qk_norm_rope_kv_insert(*args)
    ref_q = unfused_shuffle_reference(
        qkv.clone(),
        positions,
        make_rope(HEAD_DIM),
        num_heads,
        num_kv_heads,
        *shared_ops.split_shuffled_kv_cache(
            expected_cache, num_kv_heads, HEAD_DIM, kv_cache_dtype
        ),
        shuffle_ops.pad_slot_mapping(slot_mapping, num_tokens),
        kv_cache_dtype,
        1.0,
    )
    assert int(bf16_ulp_diff(q, ref_q).max()) <= 2
    got = shared_ops.split_shuffled_kv_cache(
        cache, num_kv_heads, HEAD_DIM, kv_cache_dtype
    )
    want = shared_ops.split_shuffled_kv_cache(
        expected_cache, num_kv_heads, HEAD_DIM, kv_cache_dtype
    )
    if kv_cache_dtype == "fp8":
        assert_fp8_codes_close(got[0], want[0])
        assert_fp8_codes_close(got[1], want[1])
    else:
        assert int(bf16_ulp_diff(got[0], want[0]).max()) <= 2
        torch.testing.assert_close(got[1], want[1], rtol=0, atol=0)
