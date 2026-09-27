# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Accuracy test for the ROCm AITER FP8 MLA prefill path.

The FP8 MLA prefill path
(``AiterMLAImpl._mla_fp8_prefill_attn`` -> ``mla_prefill_ps_asm_fwd`` +
``mla_reduce_v1``) is auto-enabled on gfx950 (MI355) when AITER ships those
kernels, and it has no other test coverage. This test drives the *real*
metadata builder and impl methods (via ``object.__new__``, mirroring
``test_rocm_aiter_mla_head_padding.py::test_h12_aiter_mla_decode_matches_reference``)
so it exercises the actual persistent-scheduling metadata contract
(``get_ps_metadata_v1``) plus both kernels -- rather than re-implementing them --
and compares the output against a causal SDPA reference.

gfx950-only: ``mla_prefill_ps_asm_fwd`` has no gfx942 build, so this is skipped
on every other platform.
"""

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform


def _fp8_prefill_available() -> bool:
    if not (current_platform.is_rocm() and torch.cuda.is_available()):
        return False
    from vllm.v1.attention.backends.mla.rocm_aiter_mla import (
        _fp8_mla_prefill_supported,
    )

    return _fp8_mla_prefill_supported()


pytestmark = pytest.mark.skipif(
    not _fp8_prefill_available(),
    reason="FP8 MLA prefill needs gfx950 (MI355) + an AITER build exporting "
    "mla_prefill_ps_asm_fwd / mla_reduce_v1",
)

# DeepSeek-style MLA head dims after kv_b_proj decompression (q/k carry the
# nope+rope dims, v carries v_head_dim).
#
# The kernel requires 16-aligned heads, so _mla_fp8_prefill_attn replicate-pads
# q/k/v up to get_fp8_prefill_num_heads(num_heads) and slices the output back.
# Cover all three regimes:
#   12 -> padded to 16 (non-divisor, tile+slice). Live case: a K3 rank at TP8.
#   16 -> no padding, output aliases the caller's buffer.
#   24 -> padded to 32. No current model and TP reaches this band (96 heads
#         would need TP4, which does not fit in 288 GiB; 128-head models give
#         128/64/32/16/8), so this case exists to keep the general path honest
#         for future architectures: it is the only one of the three where
#         padding above 16 actually happens, so it is what proves the
#         replicate-pad/slice-back argument holds for a target other than 16.
HEAD_COUNTS = [12, 16, 24]
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM  # 192
V_HEAD_DIM = 128
SCALE = 1.0 / math.sqrt(QK_HEAD_DIM)

# Dominated by fp8 (e4m3) rounding of q/k/v. Measured on MI355: worst-element
# abs error ~0.054, so atol=0.1 is ~2x that noise floor; rtol=0.05 bounds larger
# elements.
ATOL, RTOL = 1e-1, 5e-2

# The assembly kernel writes bf16 through a raw output pointer. Keep its tensors
# and workspace reservation on one dtype.
ATTN_OUT_DTYPE = torch.bfloat16


@pytest.fixture(autouse=True)
def _workspace_manager():
    """The FP8 prefill path draws per-call scratch from the global workspace
    manager, which ``GPUModelRunner.__init__`` normally initializes. Set it up
    for the test and tear it down so no global state leaks to other tests.
    """
    from vllm.v1.worker.workspace import (
        init_workspace_manager,
        reset_workspace_manager,
    )

    init_workspace_manager(torch.device("cuda"))
    yield
    reset_workspace_manager()


def _make_impl(num_heads: int):
    """Minimal AiterMLAImpl exposing only what _mla_fp8_prefill_attn reads."""
    from aiter import mla_prefill_ps_asm_fwd, mla_reduce_v1

    from vllm.v1.attention.backends.mla.rocm_aiter_mla import AiterMLAImpl

    impl = object.__new__(AiterMLAImpl)
    impl.num_heads = num_heads
    impl.v_head_dim = V_HEAD_DIM
    impl.scale = SCALE
    impl._mla_prefill_ps_asm_fwd = mla_prefill_ps_asm_fwd
    impl._mla_reduce_v1 = mla_reduce_v1
    return impl


def _make_builder(
    num_heads: int,
    max_num_reqs: int,
    max_prefill_qlen: int,
    max_num_batched_tokens: int,
    device: torch.device,
    attn_out_dtype: torch.dtype = ATTN_OUT_DTYPE,
):
    """The production builder with its FP8 prefill PS buffers allocated."""
    from vllm.v1.attention.backends.mla.rocm_aiter_mla import AiterMLAMetadataBuilder

    builder = object.__new__(AiterMLAMetadataBuilder)
    builder.num_heads = num_heads
    builder.mla_dims = SimpleNamespace(v_head_dim=V_HEAD_DIM)
    builder._init_fp8_prefill_ps_buffers(
        max_num_reqs=max_num_reqs,
        max_prefill_qlen=max_prefill_qlen,
        max_num_batched_tokens=max_num_batched_tokens,
        attn_out_dtype=attn_out_dtype,
        device=device,
    )
    return builder


def _step_inputs(seq_lens: list[int], device: torch.device):
    """Metadata inputs for one batch of prefills (does an H2D copy, so may sync)."""
    qo_indptr_cpu = torch.zeros(len(seq_lens) + 1, dtype=torch.int32)
    qo_indptr_cpu[1:] = torch.tensor(seq_lens, dtype=torch.int32).cumsum(0)
    qo_indptr = qo_indptr_cpu.to(device)

    # SimpleNamespace stands in for the metadata dataclasses: the builder only
    # reads prefill.{query_start_loc,max_query_len}, num_decodes, and
    # common.query_start_loc_cpu, then populates metadata.fp8_prefill_*.
    prefill = SimpleNamespace(query_start_loc=qo_indptr, max_query_len=max(seq_lens))
    metadata = SimpleNamespace(prefill=prefill, num_decodes=0)
    common = SimpleNamespace(query_start_loc_cpu=qo_indptr_cpu)
    return metadata, common


def _build_step(builder, seq_lens: list[int], device: torch.device):
    """Run the builder's FP8 prefill metadata build for one batch of prefills."""
    metadata, common = _step_inputs(seq_lens, device)
    builder._build_fp8_prefill_ps_metadata(metadata, common)
    return metadata


def _build_prefill_metadata(
    seq_lens: list[int],
    device: torch.device,
    num_heads: int,
    attn_out_dtype: torch.dtype = ATTN_OUT_DTYPE,
):
    """Build metadata with the production builder and output dtype."""
    total_q = sum(seq_lens)
    builder = _make_builder(
        num_heads, len(seq_lens), max(seq_lens), total_q, device, attn_out_dtype
    )
    return _build_step(builder, seq_lens, device), total_q, builder


def _reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Causal SDPA reference on fp8-quantized inputs (matches kernel dtype)."""
    fp8 = current_platform.fp8_dtype()
    qf = q.to(fp8).float().transpose(0, 1)  # [H, S, Dqk]
    kf = k.to(fp8).float().transpose(0, 1)
    vf = v.to(fp8).float().transpose(0, 1)  # [H, S, Dv]
    out = F.scaled_dot_product_attention(qf, kf, vf, is_causal=True, scale=SCALE)
    return out.transpose(0, 1)  # [S, H, Dv]


@pytest.mark.parametrize("num_heads", HEAD_COUNTS)
@pytest.mark.parametrize("seq_len", [128, 512])
@torch.inference_mode()
def test_fp8_prefill_matches_reference(seq_len: int, num_heads: int) -> None:
    device = torch.device("cuda")
    torch.manual_seed(0)

    metadata, total_q, _ = _build_prefill_metadata([seq_len], device, num_heads)

    # Lock the workspace, as GPUModelRunner does after warmup, so the forward
    # cannot quietly grow it. The builder reserves scratch at num_head_k and
    # the forward requests it at its own padded width; if a future change makes
    # the forward's width exceed the builder's, that is an under-reservation
    # which is invisible on an unlocked workspace (it just grows) and fatal in
    # a real serve. Locking turns it into a failure here.
    #
    # Note this does not catch the reverse -- a builder wider than the forward
    # merely over-reserves -- so it would not have flagged the pre-PR
    # max(16, num_heads) sizing, which over-reserves 2.8x at 24 heads.
    from vllm.v1.worker.workspace import lock_workspace

    lock_workspace()

    qkv_kwargs = dict(dtype=ATTN_OUT_DTYPE, device=device)
    q = torch.randn(total_q, num_heads, QK_HEAD_DIM, **qkv_kwargs)
    k = torch.randn(total_q, num_heads, QK_HEAD_DIM, **qkv_kwargs)
    v = torch.randn(total_q, num_heads, V_HEAD_DIM, **qkv_kwargs)
    out = torch.zeros(
        total_q, num_heads * V_HEAD_DIM, dtype=ATTN_OUT_DTYPE, device=device
    )

    impl = _make_impl(num_heads)
    impl._mla_fp8_prefill_attn(q, k, v, metadata, out)

    out_ref = _reference(q, k, v)

    assert torch.isfinite(out).all()
    torch.testing.assert_close(
        out.view(total_q, num_heads, V_HEAD_DIM).float(),
        out_ref.float(),
        atol=ATOL,
        rtol=RTOL,
    )


@pytest.mark.parametrize("num_heads", HEAD_COUNTS)
def test_fp8_prefill_metadata_width_matches_forward(num_heads: int) -> None:
    """The PS metadata must be built for the head count the kernel is handed.

    ``_init_fp8_prefill_ps_buffers``/``build()`` size the work and reduce maps
    from ``num_head_k``, while ``_mla_fp8_prefill_attn`` pads q/k/v to its own
    target. Divergence is not a numerical bug in either direction -- maps built
    narrower than the kernel width still cover the real heads, since padding
    only ever appends duplicate heads above them. It is a sizing bug: narrower
    maps mean a lower head alignment, hence more partial tiles and a larger
    reservation, while wider ones under-reserve the forward's scratch. Pin the
    two to one helper rather than to independent expressions.
    """
    from vllm.v1.attention.backends.mla.rocm_aiter_mla import AiterMLAHelper

    width = AiterMLAHelper.get_fp8_prefill_num_heads(num_heads)
    assert width % AiterMLAHelper._AITER_MIN_MLA_HEADS == 0
    assert width >= num_heads
    assert width - num_heads < AiterMLAHelper._AITER_MIN_MLA_HEADS
    # Every count the enable gate accepts must have a usable padded width.
    assert AiterMLAHelper.is_valid_num_heads(num_heads)

    # The builder must actually size its maps at that width. Assert against the
    # buffer it allocated rather than recomputing the width here, or the test
    # passes no matter what _init_fp8_prefill_ps_buffers does.
    #
    # fp8_ps_reduce_partial_map is sized by the partial-tile count, which is
    # gcd-driven and so is a sharp probe of the width used: a lower head
    # alignment yields *more* tiles. At 24 heads, sizing at the unpadded count
    # gives 4x the tiles (64 vs 16) and ~2.8x the scratch reservation (193.6
    # vs 68.6 MiB at batch=1/qlen=512) -- not wrong numerically (the 24-wide
    # maps still cover the real heads) but the reason to pin the widths
    # together.
    from aiter import get_ps_metadata_info_v1

    from vllm.v1.attention.backends.mla.rocm_aiter_mla import _FP8_PREFILL_TILE_Q

    def _tiles(nhk: int) -> int:
        return get_ps_metadata_info_v1(
            batch_size=1,
            num_head_k=nhk,
            max_qlen=512,
            qlen_granularity=_FP8_PREFILL_TILE_Q,
        )[5][0]

    _, _, builder = _build_prefill_metadata([512], torch.device("cuda"), num_heads)
    assert builder.fp8_ps_reduce_partial_map.numel() == _tiles(width)


# Ordering of the persistent PS plan across steps (vllm-project/vllm#58886).
#
# The builder keeps one set of device buffers for the FP8 prefill plan, and every
# step's metadata aliases them. Under async scheduling the host builds step N+1
# while step N's prefill kernels are still queued, so the rewrite must be ordered
# after them on the stream. These tests make that overlap deterministic with a GPU
# delay instead of waiting for serving to hit it:
#   - Kernels of step N are replaced by clones of the plan buffers queued behind the
#     delay. They read the plan like mla_prefill_ps_asm_fwd / mla_reduce_v1 do but
#     cannot fault, so a stale plan shows up as a comparison failure, not a GPU
#     memory fault that would take down the test process.
#   - Everything runs on a non-blocking stream, as vLLM does
#     (vllm.utils.torch_utils.current_stream). A stream-less hipMemcpy waits for the
#     null stream, so on the default stream an unordered rewrite would look correct.
#   - Only one extra stream is created. With more streams than HIP hardware queues
#     (GPU_MAX_HW_QUEUES, default 4) the runtime shares queues, and sharing one with
#     the null stream can order an unordered copy by accident.
#   - Step inputs are prepared before the delay: moving qo_indptr to the device is a
#     blocking copy that drains the stream, which would remove the overlap. Only the
#     builder call itself runs while the delay is pending.
#   - The check that the delay is still pending right before the rewrite guards
#     against a vacuous pass on a fast host or a slow launch: without overlap there
#     is nothing to order.
ORDERING_BATCHES = ([8175], [6992, 6896, 2495], [4096, 4096, 4096, 4096])
ORDERING_DELAY_MS = 200.0


def _plan_buffers(builder) -> tuple[torch.Tensor, ...]:
    """Device buffers get_ps_metadata_v1 writes and the prefill kernels read."""
    return (
        builder.fp8_ps_work_indptr,
        builder.fp8_ps_work_info,
        builder.fp8_ps_reduce_indptr,
        builder.fp8_ps_reduce_final_map,
        builder.fp8_ps_reduce_partial_map,
    )


def _ordering_builder(device: torch.device):
    return _make_builder(
        num_heads=16,
        max_num_reqs=max(len(b) for b in ORDERING_BATCHES),
        max_prefill_qlen=max(max(b) for b in ORDERING_BATCHES),
        max_num_batched_tokens=max(sum(b) for b in ORDERING_BATCHES),
        device=device,
    )


def _expected_plans(device: torch.device) -> list[list[torch.Tensor]]:
    """Plan for each batch, built with nothing in flight."""
    builder = _ordering_builder(device)
    plans = []
    for batch in ORDERING_BATCHES:
        _build_step(builder, batch, device)
        torch.cuda.synchronize()
        plans.append([t.clone() for t in _plan_buffers(builder)])
    assert not all(torch.equal(a, b) for a, b in zip(plans[0], plans[1])), (
        "batches must produce different plans, or a stale read is undetectable"
    )
    return plans


def _delay_cycles(stream: torch.cuda.Stream, ms: float) -> int:
    """torch.cuda._sleep cycle count for about ``ms`` on this GPU."""
    with torch.cuda.stream(stream):
        torch.cuda._sleep(1_000_000)
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        torch.cuda._sleep(50_000_000)
        end.record()
    end.synchronize()
    return int(50_000_000 * ms / start.elapsed_time(end))


def _assert_plan(seen: list[torch.Tensor], expected: list[torch.Tensor], what: str):
    for got, exp in zip(seen, expected):
        assert torch.equal(got, exp), what


@torch.inference_mode()
def test_fp8_prefill_ps_metadata_rewrite_is_stream_ordered() -> None:
    """Step N+1's plan rewrite must not reach step N's queued prefill kernels.

    Before the fix, get_ps_metadata_v1 wrote the device buffers with a stream-less
    blocking hipMemcpy: the rewrite landed at once and step N's queued reads saw
    step N+1's plan (in serving: a GPU memory fault, or silently wrong attention).
    With the fix the plan is staged on the host and copied on the current stream,
    so it lands only after step N's kernels.
    """
    device = torch.device("cuda")
    expected = _expected_plans(device)
    builder = _ordering_builder(device)
    stream = torch.cuda.Stream()
    cycles = _delay_cycles(stream, ORDERING_DELAY_MS)

    with torch.cuda.stream(stream):
        _build_step(builder, ORDERING_BATCHES[0], device)  # step N
        next_metadata, next_common = _step_inputs(ORDERING_BATCHES[1], device)
        torch.cuda._sleep(cycles)  # step N's earlier layers
        seen_by_step_n = [t.clone() for t in _plan_buffers(builder)]
        step_n_done = torch.cuda.Event()
        step_n_done.record()

        overlapped = not step_n_done.query()
        builder._build_fp8_prefill_ps_metadata(next_metadata, next_common)  # step N+1
    torch.cuda.synchronize()

    if not overlapped:
        pytest.skip("GPU drained before step N+1 was built; nothing to order")
    _assert_plan(seen_by_step_n, expected[0], "step N read step N+1's plan")
    _assert_plan(
        [t.clone() for t in _plan_buffers(builder)], expected[1], "step N+1's plan lost"
    )


@torch.inference_mode()
def test_fp8_prefill_ps_metadata_staging_reuse_waits_for_copy() -> None:
    """A host staging slot must not be replanned while its copy is still queued.

    The fix alternates two pinned host staging slots, so the third build in a row
    reuses the first build's slot. If that slot were overwritten before its queued
    copy ran, the copy would carry the later plan and the first consumer would
    read it. Queue three builds behind one GPU delay, each followed by a reader,
    and check every reader sees its own step's plan.
    """
    device = torch.device("cuda")
    expected = _expected_plans(device)
    builder = _ordering_builder(device)
    stream = torch.cuda.Stream()
    cycles = _delay_cycles(stream, ORDERING_DELAY_MS)

    seen = []
    with torch.cuda.stream(stream):
        inputs = [_step_inputs(batch, device) for batch in ORDERING_BATCHES]
        torch.cuda._sleep(cycles)
        delay_done = torch.cuda.Event()
        delay_done.record()
        overlapped = not delay_done.query()
        for metadata, common in inputs:
            builder._build_fp8_prefill_ps_metadata(metadata, common)
            seen.append([t.clone() for t in _plan_buffers(builder)])
    torch.cuda.synchronize()

    if not overlapped:
        pytest.skip("GPU drained before the first build; nothing to order")
    for i, (got, exp) in enumerate(zip(seen, expected)):
        _assert_plan(got, exp, f"step {i} read another step's plan")
