# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host side of the mono KDA launch (``layer.py``): the launcher cache, the
per-device workspace and ``mono_kda``, which takes the tensors vLLM's Kimi-K3
KDA layer holds and returns o_proj's TP-local (unreduced) output.

``mono_kda`` replaces, for a non-speculative decode step of M <= ``M_MAX`` rows,

    in_proj_qkvgfab(x)                        rocm_unquantized_gemm
    f_b_proj(f_a)                             rocm_unquantized_gemm
    ops.fused_kda_decode(...)                 conv + delta rule + gated RMSNorm
    o_proj(core_attn_out) before its all-reduce

Callers check ``supported`` first.
"""

import functools

import torch

from vllm.models.kimi_k3.amd.mono_kda.plan import (
    CONV_WIDTH,
    HEAD_DIM,
    M_MAX,
    Dims,
    ctrl_words,
    ws_layout,
)


@functools.cache
def _launcher(heads, hidden, lower_bound, eps):
    from vllm.models.kimi_k3.amd.mono_kda.layer import compile_mono_kda

    return compile_mono_kda(
        heads=heads, hidden=hidden, lower_bound=lower_bound, eps=eps
    )


class Workspace:
    """Per-device buffers sized for M_MAX rows.

    The control words must start at zero; the kernel's last workgroup returns
    them to zero, so one zero fill at allocation covers every later launch.
    Sharing is safe while launches are ordered, as on one stream.
    """

    def __init__(self, device, dims: Dims):
        total = ws_layout(dims)[1]
        self.buf = torch.zeros(total, dtype=torch.uint8, device=device)
        self.ctrl = self.buf[: ctrl_words(dims.heads) * 4].view(torch.int32)


_WORKSPACES: dict[tuple, Workspace] = {}


def _workspace(device, dims: Dims) -> Workspace:
    key = (device, dims)
    ws = _WORKSPACES.get(key)
    if ws is None:
        ws = _WORKSPACES[key] = Workspace(device, dims)
    return ws


@functools.cache
def _num_cus(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def supported(
    x: torch.Tensor,
    w_in: torch.Tensor,
    w_fb: torch.Tensor,
    w_o: torch.Tensor,
    conv_w: torch.Tensor,
    conv_state: torch.Tensor,
    state: torch.Tensor,
    state_indices: torch.Tensor,
) -> bool:
    """Shapes and layouts the launch takes. conv_state is the [slot, 3P, 3]
    view fused_kda_decode gets (width-major storage); conv_w the width-major
    [3, 4, P] fp32 decode weight."""
    m, hidden = x.shape
    proj = w_fb.shape[0]
    heads = proj // HEAD_DIM
    dims = Dims(heads=heads, hidden=hidden)
    bf = torch.bfloat16
    return (
        1 <= m <= M_MAX
        and proj % HEAD_DIM == 0
        and hidden % 128 == 0
        and x.dtype == w_in.dtype == w_fb.dtype == w_o.dtype == bf
        and x.is_contiguous()
        and w_in.is_contiguous()
        and w_fb.is_contiguous()
        and w_o.is_contiguous()
        and tuple(w_in.shape) == (dims.n_in, hidden)
        and tuple(w_fb.shape) == (proj, HEAD_DIM)
        and tuple(w_o.shape) == (hidden, proj)
        and conv_w.dtype == torch.float32
        and conv_w.is_contiguous()
        and tuple(conv_w.shape) == (3, CONV_WIDTH, proj)
        and conv_state.dtype == bf
        and conv_state.dim() == 3
        and tuple(conv_state.shape[1:]) == (3 * proj, CONV_WIDTH - 1)
        and conv_state.stride(1) == 1
        and conv_state.stride(2) == 3 * proj
        and state.dtype == torch.float32
        and state.dim() == 4
        and tuple(state.shape[1:]) == (heads, HEAD_DIM, HEAD_DIM)
        and state.stride()[1:] == (HEAD_DIM * HEAD_DIM, HEAD_DIM, 1)
        and state_indices.dtype == torch.int32
        and state_indices.is_contiguous()
        and state_indices.numel() >= m
        and conv_state.stride(0) < 2**31
        and state.stride(0) < 2**31
    )


def mono_kda(
    x: torch.Tensor,
    w_in: torch.Tensor,
    w_fb: torch.Tensor,
    w_o: torch.Tensor,
    conv_w: torch.Tensor,
    conv_state: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    norm_w: torch.Tensor,
    state_indices: torch.Tensor,
    state: torch.Tensor,
    *,
    lower_bound: float | None,
    eps: float,
    out: torch.Tensor | None = None,
    workspace: Workspace | None = None,
    grid: int | None = None,
) -> torch.Tensor:
    """o_proj's TP-local output [M, hidden] bf16 of the decode step; updates
    conv_state and state in place for the rows whose slot is > 0.

    x [M, hidden] bf16 (in_proj's input); w_in [n_in, hidden], w_fb [P, 128],
    w_o [hidden, P] bf16; conv_w [3, 4, P] fp32 width-major; A_log [heads],
    dt_bias [P], norm_w [128] fp32; state_indices [>= M] int32.
    """
    m, hidden = x.shape
    heads = w_fb.shape[0] // HEAD_DIM
    dims = Dims(heads=heads, hidden=hidden)
    assert supported(x, w_in, w_fb, w_o, conv_w, conv_state, state, state_indices)
    if out is None:
        out = torch.empty((m, hidden), dtype=torch.bfloat16, device=x.device)
    launch = _launcher(
        heads, hidden, None if lower_bound is None else float(lower_bound), float(eps)
    )
    ws = workspace or _workspace(x.device, dims)
    if grid is None:
        grid = min(dims.tickets(m), _num_cus(x.device.index))
    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

    _run_compiled(
        launch,
        x.data_ptr(),
        w_in.data_ptr(),
        w_fb.data_ptr(),
        w_o.data_ptr(),
        conv_w.data_ptr(),
        conv_state.data_ptr(),
        A_log.data_ptr(),
        dt_bias.data_ptr(),
        norm_w.data_ptr(),
        state_indices.data_ptr(),
        state.data_ptr(),
        out.data_ptr(),
        ws.buf.data_ptr(),
        m,
        grid,
        conv_state.stride(0),
        state.stride(0),
        torch.cuda.current_stream(x.device),
    )
    return out
