# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerics of the mono KDA launch against the path it replaces: in_proj, f_b_proj
(bf16 GEMMs), ops.fused_kda_decode, o_proj, at Kimi-K3's TP8 shapes."""

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCm only", allow_module_level=True)

from vllm.platforms.rocm import on_gfx950  # noqa: E402

if not on_gfx950():
    pytest.skip("gfx950 only", allow_module_level=True)

import vllm._custom_ops as ops  # noqa: E402
from vllm.models.kimi_k3.amd.mono_kda import runner  # noqa: E402
from vllm.models.kimi_k3.amd.mono_kda.plan import Dims  # noqa: E402

HIDDEN, HEADS, D, EPS = 7168, 12, 128, 1e-5
SLOTS = 40


class Layer:
    def __init__(self, seed):
        g = torch.Generator(device="cuda").manual_seed(seed)
        dev, bf, f32 = "cuda", torch.bfloat16, torch.float32
        dims = Dims(heads=HEADS, hidden=HIDDEN)
        self.P, self.n_in = dims.proj, dims.n_in

        def rn(*shape, s=1.0, dtype=f32):
            return (torch.randn(*shape, generator=g, device=dev) * s).to(dtype)

        self.w_in = rn(self.n_in, HIDDEN, s=0.02, dtype=bf)
        self.w_in[4 * self.P + D + HEADS :] = 0  # vLLM's padding rows
        self.w_fb = rn(self.P, D, s=0.05, dtype=bf)
        self.w_o = rn(HIDDEN, self.P, s=0.02, dtype=bf)
        self.conv_w = rn(3, 4, self.P, s=0.3)
        self.A_log = rn(HEADS, s=0.5)
        self.dt_bias = rn(self.P, s=0.5)
        self.norm_w = 1 + rn(D, s=0.1)
        # SD layout [slot, width-1, 3P] viewed as [slot, 3P, width-1]
        self.conv_state = rn(SLOTS, 3, 3 * self.P, s=0.5, dtype=bf).transpose(-1, -2)
        self.state = rn(SLOTS, HEADS, D, D, s=0.05)

    def clone_states(self):
        return self.conv_state.clone(), self.state.clone()


def stock(layer, x, idx, conv_state, state, lb):
    P = layer.P
    y = F.linear(x, layer.w_in)[:, : 4 * P + D + HEADS]
    mixed_qkv, g2, f_a, beta = y.split([3 * P, P, D, HEADS], dim=-1)
    m = x.shape[0]
    g1 = F.linear(f_a, layer.w_fb).view(1, m, HEADS, D)
    out = torch.empty(1, m, HEADS, D, dtype=x.dtype, device=x.device)
    ops.fused_kda_decode(
        x=mixed_qkv, weight=layer.conv_w, bias=None, conv_state=conv_state,
        raw_g=g1, raw_beta=beta.unsqueeze(0), A_log=layer.A_log,
        dt_bias=layer.dt_bias, state_indices=idx, state=state, out=out,
        lower_bound=lb, output_gate=g2.view(m, HEADS, D),
        norm_weight=layer.norm_w, norm_eps=EPS,
    )  # fmt: skip
    return F.linear(out.view(m, P), layer.w_o)


def mono(layer, x, idx, conv_state, state, lb, grid=None):
    return runner.mono_kda(
        x, layer.w_in, layer.w_fb, layer.w_o, layer.conv_w, conv_state,
        layer.A_log, layer.dt_bias, layer.norm_w, idx, state,
        lower_bound=lb, eps=EPS, grid=grid,
    )  # fmt: skip


def _close(a, b, what, atol, rtol):
    torch.testing.assert_close(a.float(), b.float(), atol=atol, rtol=rtol, msg=what)


@pytest.mark.parametrize("lb", [None, -5.0])
@pytest.mark.parametrize("m", [1, 2, 3, 5, 8, 13, 16])
def test_mono_kda_matches_stock(m, lb):
    torch.manual_seed(m)
    layer = Layer(seed=m)
    idx = (torch.randperm(SLOTS - 1)[:m] + 1).to(torch.int32).cuda()
    if m >= 3:
        idx[-1] = 0  # a padded row: zero output, states untouched
    cs_ref, st_ref = layer.clone_states()
    cs, st = layer.clone_states()
    # three steps: the states carry over and the control words must reset
    for step in range(3):
        x = (torch.randn(m, HIDDEN, device="cuda") * 0.5).to(torch.bfloat16)
        ref = stock(layer, x, idx, cs_ref, st_ref, lb)
        got = mono(layer, x, idx, cs, st, lb)
        torch.accelerator.synchronize()
        _close(got, ref, f"out, step {step}", atol=3e-2, rtol=3e-2)
        _close(st, st_ref, f"state, step {step}", atol=2e-3, rtol=2e-2)
        _close(cs, cs_ref, f"conv state, step {step}", atol=1e-2, rtol=1e-2)
    if m >= 3:
        assert got[-1].abs().max().item() == 0.0
        assert torch.equal(st[0], layer.state[0])


@pytest.mark.parametrize("grid", [1, 7, 64])
def test_mono_kda_any_grid(grid):
    """Every ticket waits only on smaller ones: one workgroup suffices."""
    m, lb = 6, -5.0
    layer = Layer(seed=100 + grid)
    idx = (torch.arange(m) + 1).to(torch.int32).cuda()
    cs_ref, st_ref = layer.clone_states()
    cs, st = layer.clone_states()
    for _ in range(2):
        x = (torch.randn(m, HIDDEN, device="cuda") * 0.5).to(torch.bfloat16)
        ref = stock(layer, x, idx, cs_ref, st_ref, lb)
        got = mono(layer, x, idx, cs, st, lb, grid=grid)
        torch.accelerator.synchronize()
        _close(got, ref, "out", atol=3e-2, rtol=3e-2)
        _close(st, st_ref, "state", atol=2e-3, rtol=2e-2)
