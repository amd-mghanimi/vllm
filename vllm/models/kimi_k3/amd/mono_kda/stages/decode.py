# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The decode task of head h, row m: f_b_proj, the causal conv update, the
gated delta rule and the gated RMSNorm, as vLLM computes them for a
non-speculative decode row (``f_b_proj`` then ``ops.fused_kda_decode``,
``csrc/libtorch_stable/kimi_k3/fused_kda_decode_kernel_rocm.cu``).

With h' = S * decay (S the head's [v, k] fp32 state, decay per k):

    u = h'.k    vn = (v - u) * beta    S <- h' + k vn^T    o = h'.q + vn (k.q)

Everything that does not depend on the step's in_proj columns -- the state
slice, the W_fb rows, the conv weights and the conv state -- is loaded before
the task waits for them. A row whose state slot is <= 0 (vLLM's padding)
writes zeros and leaves both states alone.
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels import communication_ops_utils as comm_ops
from aiter.ops.flydsl.kernels.act import sigmoid_f32
from aiter.ops.flydsl.kernels.mxfp4_gemm_common import _raw
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from vllm.models.deepseek_v41.amd.mono.common.ops import butterfly, hw_exp2, hw_rsq
from vllm.models.kimi_k3.amd.mono_kda.plan import CONV_WIDTH, HEAD_DIM, N_WAVES
from vllm.models.kimi_k3.amd.mono_kda.stages.gemv import CM_NT
from vllm.models.kimi_k3.amd.mono_kda.sync import lds_f32, st_wt

D = HEAD_DIM
LOG2E = 1.4426950408889634
LANES = 16  # lanes a state row
ROWS = 256 // LANES  # rows a pass
PASSES = D // ROWS
VEC = D // LANES // 4  # f32x4 a lane a row

# LDS words of a decode task
L_G1 = 0  # 2 x D f_b partials
L_Q = L_G1 + 2 * D
L_K = L_Q + D
L_DECAY = L_K + D
L_V = L_DECAY + D
L_O = L_V + D
L_RED = L_O + D  # 3 sums a wave (4 words each), then the norm's sum a wave
LDS_WORDS = L_RED + 5 * N_WAVES


def _rsrc(addr, nbytes=None):
    """nbytes bounds the buffer: a masked buffer_store writes to byte offset
    0x7FFFFFFF and only the bound drops it."""
    return buffer_ops.create_buffer_resource_from_addr(
        fx.Int64(addr), num_records_bytes=nbytes
    )


def _state_rsrc(s_base):
    return _rsrc(s_base, D * D * 4)


def _conv_rsrc(arg_conv_state, slot, conv_stride, P):
    return _rsrc(
        fx.Int64(arg_conv_state) + fx.Int64(slot) * fx.Int64(conv_stride) * fx.Int64(2),
        (CONV_WIDTH - 1) * 3 * P * 2,
    )


def _ld(rsrc, off, dtype, cm=0):
    return buffer_ops.buffer_load(
        rsrc, _raw(fx.Int32(off)), vec_width=1, dtype=dtype, cache_modifier=cm
    )


def _ld_bf(rsrc, off):
    return fx.Float32(fx.BFloat16(_ld(rsrc, off, T.bf16)))


def _ld_f32(rsrc, off):
    return fx.Float32(_ld(rsrc, off, T.f32))


def _bf16x8(rsrc, off_dw):
    v = buffer_ops.buffer_load(rsrc, _raw(off_dw), vec_width=4, dtype=T.i32)
    return fx.Vector(v).bitcast(fx.BFloat16)


def _silu(x):
    return x * sigmoid_f32(x)


def _exp(x):
    return hw_exp2(x * fx.Float32(LOG2E))


def _softplus(x):
    return (x > fx.Float32(20.0)).select(x, fx.Float32(fmath.log1p(_exp(x))))


def preload(
    tid, h, slot, arg_state, state_stride, arg_wfb, arg_conv_w, arg_conv_state,
    conv_stride, arg_dtb, *, P,
):  # fmt: skip
    """Loads independent of the step's in_proj columns. slot >= 0."""
    rlane = tid % fx.Int32(LANES)
    rgroup = tid // fx.Int32(LANES)
    s_base = fx.Int64(arg_state) + (
        fx.Int64(slot) * fx.Int64(state_stride) + fx.Int64(h) * fx.Int64(D * D)
    ) * fx.Int64(4)
    s_rsrc = _state_rsrc(s_base)
    state = []
    for p in range_constexpr(PASSES):
        for j in range_constexpr(VEC):
            off = (fx.Int32(p * ROWS) + rgroup) * fx.Int32(D) + rlane * fx.Int32(4)
            off = off + fx.Int32(j * 4 * LANES)
            state.append(
                buffer_ops.buffer_load(
                    s_rsrc, _raw(off), vec_width=4, dtype=T.f32, cache_modifier=CM_NT
                )
            )

    # f_b: thread tid takes K half tid / D of output k = tid % D
    k = tid % fx.Int32(D)
    half = tid // fx.Int32(D)
    w_rsrc = _rsrc(arg_wfb)
    w_dw = (h * fx.Int32(D) + k) * fx.Int32(D // 2) + half * fx.Int32(D // 4)
    wfb = [_bf16x8(w_rsrc, w_dw + fx.Int32(4 * i)) for i in range_constexpr(D // 16)]

    # conv: threads < D own channel k of q and of k, the others channel k of v
    cw_rsrc = _rsrc(arg_conv_w)
    cs_rsrc = _conv_rsrc(arg_conv_state, slot, conv_stride, P)
    ch = h * fx.Int32(D) + k
    taps = []
    for seg in (0, 1):
        # seg 0 / 1 for the q / k thread; the v thread reads seg 2 for its first
        sg = (half == fx.Int32(0)).select(fx.Int32(seg), fx.Int32(2))
        c = sg * fx.Int32(P) + ch
        w = [
            _ld_f32(
                cw_rsrc, (sg * fx.Int32(CONV_WIDTH) + fx.Int32(i)) * fx.Int32(P) + ch
            )
            for i in range_constexpr(CONV_WIDTH)
        ]
        st = [
            _ld_bf(cs_rsrc, fx.Int32(i * 3 * P) + c)
            for i in range_constexpr(CONV_WIDTH - 1)
        ]
        taps.append((w, st))
    dtb = _ld_f32(_rsrc(arg_dtb), ch)
    return state, wfb, taps, dtb, s_base


@comm_ops.traced
def decode_task(
    tid, lane, wave, lds_base, h, m, slot, pre, arg_proj, arg_conv_state,
    conv_stride, arg_a_log, arg_normw, arg_o, *, P, ROW, FA_COL, BETA_COL,
    lower_bound, eps, scale,
):  # fmt: skip
    """After the waits: the rest of the task. pre = preload(...)."""
    state, wfb, taps, dtb, s_base = pre
    live = slot > fx.Int32(0)
    k = tid % fx.Int32(D)
    half = tid // fx.Int32(D)
    live_qk = (half == fx.Int32(0)).select(slot, fx.Int32(0)) > fx.Int32(0)
    rlane = tid % fx.Int32(LANES)
    rgroup = tid // fx.Int32(LANES)
    ch = h * fx.Int32(D) + k
    p_rsrc = _rsrc(arg_proj)
    row = m * fx.Int32(ROW)

    # f_b_proj: g1[k] = bf16(f_a . W_fb[h * D + k]), f_a in-proj columns
    fa_dw = (row + fx.Int32(FA_COL) + half * fx.Int32(D // 2)) // fx.Int32(2)
    fa = [_bf16x8(p_rsrc, fa_dw + fx.Int32(4 * i)) for i in range_constexpr(D // 16)]
    part = fx.Float32(0.0)
    for i in range_constexpr(D // 16):
        a = fx.Vector(fa[i])
        b = fx.Vector(wfb[i])
        for e in range_constexpr(8):
            part = part + fx.Float32(a[e]) * fx.Float32(b[e])
    g1s = lds_f32(lds_base, L_G1)
    g1s[tid] = part

    # conv update: q, k (threads < D) or v (the others), new tap from in_proj
    acc = []
    for si in range_constexpr(2):
        w, st = taps[si]
        sg = (half == fx.Int32(0)).select(fx.Int32(si), fx.Int32(2))
        x_new = _ld_bf(p_rsrc, row + sg * fx.Int32(P) + ch)
        a = fx.Float32(0.0)
        for i in range_constexpr(CONV_WIDTH - 1):
            a = a + st[i] * w[i]
        a = a + x_new * w[CONV_WIDTH - 1]
        acc.append(_silu(a))
        cs_rsrc = _conv_rsrc(arg_conv_state, slot, conv_stride, P)
        c = sg * fx.Int32(P) + ch
        # the v thread runs seg 2 twice: it stores once
        store = live if si == 0 else live_qk
        new = (st[1], st[2], x_new)
        for i in range_constexpr(CONV_WIDTH - 1):
            buffer_ops.buffer_store(
                _raw(new[i].to(fx.BFloat16)),
                cs_rsrc,
                _raw(fx.Int32(i * 3 * P) + c),
                mask=_raw(store),
            )
    gpu.barrier()

    beta = sigmoid_f32(_ld_bf(p_rsrc, row + fx.Int32(BETA_COL) + h))
    g1 = (fx.Float32(g1s[k]) + fx.Float32(g1s[k + fx.Int32(D)])).to(fx.BFloat16)
    g_raw = fx.Float32(g1) + dtb
    exp_a = _exp(_ld_f32(_rsrc(arg_a_log), h))
    if const_expr(lower_bound is None):
        decay = _exp(-exp_a * _softplus(g_raw))
    else:
        decay = _exp(fx.Float32(lower_bound) * sigmoid_f32(exp_a * g_raw))

    is_qk = half == fx.Int32(0)
    qv = is_qk.select(acc[0], fx.Float32(0.0))
    kv = is_qk.select(acc[1], fx.Float32(0.0))
    sums = [butterfly(v, (32, 16, 8, 4, 2, 1)) for v in (qv * qv, kv * kv, qv * kv)]
    red = lds_f32(lds_base, L_RED)
    if lane == fx.Int32(0):
        for i in range_constexpr(3):
            red[wave * fx.Int32(4) + fx.Int32(i)] = sums[i]
    sv = lds_f32(lds_base, L_V)
    if half == fx.Int32(1):
        sv[k] = acc[0]
    gpu.barrier()
    tot = [fx.Float32(red[i]) + fx.Float32(red[4 + i]) for i in range_constexpr(3)]
    rq = hw_rsq(tot[0] + fx.Float32(1e-6)) * fx.Float32(scale)
    rk = hw_rsq(tot[1] + fx.Float32(1e-6))
    kq = tot[2] * rq * rk
    sq = lds_f32(lds_base, L_Q)
    sk = lds_f32(lds_base, L_K)
    sd = lds_f32(lds_base, L_DECAY)
    if half == fx.Int32(0):
        sq[k] = qv * rq
        sk[k] = kv * rk
        sd[k] = decay
    gpu.barrier()

    rqv, rkv, rdv = [], [], []
    for j in range_constexpr(VEC):
        for e in range_constexpr(4):
            kk = rlane * fx.Int32(4) + fx.Int32(j * 4 * LANES + e)
            rqv.append(fx.Float32(sq[kk]))
            rkv.append(fx.Float32(sk[kk]))
            rdv.append(fx.Float32(sd[kk]))

    s_rsrc = _state_rsrc(s_base)
    so = lds_f32(lds_base, L_O)
    for p in range_constexpr(PASSES):
        hv = []
        for j in range_constexpr(VEC):
            vec = fx.Vector(state[p * VEC + j])
            for e in range_constexpr(4):
                hv.append(fx.Float32(vec[e]) * rdv[4 * j + e])
        pk = [fx.Float32(0.0)] * 4
        pq = [fx.Float32(0.0)] * 4
        for e in range_constexpr(len(hv)):
            pk[e % 4] = pk[e % 4] + hv[e] * rkv[e]
            pq[e % 4] = pq[e % 4] + hv[e] * rqv[e]
        dot_hk = butterfly((pk[0] + pk[1]) + (pk[2] + pk[3]), (1, 2, 4, 8))
        dot_hq = butterfly((pq[0] + pq[1]) + (pq[2] + pq[3]), (1, 2, 4, 8))
        v = fx.Int32(p * ROWS) + rgroup
        vn = (fx.Float32(sv[v]) - dot_hk) * beta
        for j in range_constexpr(VEC):
            upd = [
                _raw(hv[4 * j + e] + rkv[4 * j + e] * vn) for e in range_constexpr(4)
            ]
            off = v * fx.Int32(D) + rlane * fx.Int32(4) + fx.Int32(j * 4 * LANES)
            buffer_ops.buffer_store(
                fx.Vector.from_elements(upd, fx.Float32),
                s_rsrc,
                _raw(off),
                mask=_raw(live),
                cache_modifier=CM_NT,
            )
        if rlane == fx.Int32(0):
            so[v] = dot_hq + vn * kq
    gpu.barrier()

    o = fx.Float32(so[k])
    o2 = is_qk.select(o * o, fx.Float32(0.0))
    ss = butterfly(o2, (32, 16, 8, 4, 2, 1))
    if lane == fx.Int32(0):
        red[fx.Int32(4 * N_WAVES) + wave] = ss
    gpu.barrier()
    rstd = hw_rsq(
        (fx.Float32(red[4 * N_WAVES]) + fx.Float32(red[4 * N_WAVES + 1]))
        * fx.Float32(1.0 / D)
        + fx.Float32(eps)
    )
    if half == fx.Int32(0):
        gate = sigmoid_f32(_ld_bf(p_rsrc, row + fx.Int32(3 * P) + ch))
        nw = _ld_f32(_rsrc(arg_normw), k)
        y = live.select(o * rstd * nw * gate, fx.Float32(0.0))
        st_wt(arg_o, m * fx.Int32(P) + ch, y.to(fx.BFloat16), 2)
