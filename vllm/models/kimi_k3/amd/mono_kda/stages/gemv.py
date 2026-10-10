# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""bf16 GEMV tiles: y[t, c0 + j] = x[t, :] . W[c0 + j, :] for the step's rows
t < M <= 16 and 16 weight rows j, as vLLM's unquantized linear computes them
(fp32 accumulation, bf16 output).

A tile is one workgroup: wave w takes the K quarter w and runs
mfma_f32_16x16x32_bf16 straight from global memory. Lane l feeds token row
l % 16 of x and weight row l % 16, K elements (l / 16) * 8 .. + 8, and gets back
tokens (l / 16) * 4 + r of column l % 16. The four quarters meet in LDS and are
summed in wave order. x's buffer resource covers M rows, so padding rows load
zeros.
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.mxfp4_gemm_common import _raw
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import T

from vllm.models.kimi_k3.amd.mono_kda.plan import N_WAVES, TILE
from vllm.models.kimi_k3.amd.mono_kda.sync import lds_f32

CM_NT = 2  # nontemporal: a weight is read once a step
CHUNK = 14  # K steps whose loads go out together


def _bf16x8(rsrc, off_dw, cm=0):
    v = buffer_ops.buffer_load(
        rsrc, _raw(off_dw), vec_width=4, dtype=T.i32, cache_modifier=cm
    )
    return fx.Vector(v).bitcast(fx.BFloat16)


def _mfma(a8, b8, acc):
    return rocdl.mfma_f32_16x16x32_bf16(
        T.f32x4, [_raw(a8), _raw(b8), _raw(acc), 0, 0, 0]
    )


def _zero():
    return fx.Vector.from_elements([_raw(fx.Float32(0.0))] * 4, fx.Float32)


def _k_dw(lane, wave, K):
    """Dword offset, within a K-long bf16 row, of this lane's first K step."""
    return wave * fx.Int32(K // N_WAVES // 2) + lane // fx.Int32(16) * fx.Int32(4)


def load_w(arg_w, c0, lane, wave, *, K):
    """This lane's weight operands of the tile at column c0, every K step."""
    steps = K // N_WAVES // 32
    w_rsrc = buffer_ops.create_buffer_resource_from_addr(fx.Int64(arg_w))
    w_dw = (c0 + lane % fx.Int32(16)) * fx.Int32(K // 2) + _k_dw(lane, wave, K)
    return [_bf16x8(w_rsrc, w_dw + fx.Int32(s * 16), CM_NT) for s in range(steps)]


def _x_rsrc(arg_x, i32_M, K):
    return buffer_ops.create_buffer_resource_from_addr(
        fx.Int64(arg_x), num_records_bytes=fx.Int64(i32_M) * fx.Int64(K * 2)
    )


def _x_dw(lane, wave, K):
    return lane % fx.Int32(16) * fx.Int32(K // 2) + _k_dw(lane, wave, K)


def mma_loaded(arg_x, wv, lane, wave, i32_M, *, K):
    """The wave's partial tile from weight operands already loaded."""
    x_rsrc = _x_rsrc(arg_x, i32_M, K)
    x_dw = _x_dw(lane, wave, K)
    acc = _zero()
    for s in range_constexpr(len(wv)):
        acc = _mfma(_bf16x8(x_rsrc, x_dw + fx.Int32(s * 16)), wv[s], acc)
    return acc


def mma_streamed(arg_x, arg_w, c0, lane, wave, i32_M, *, K):
    """The wave's partial tile, weights streamed CHUNK K steps at a time."""
    steps = K // N_WAVES // 32
    x_rsrc = _x_rsrc(arg_x, i32_M, K)
    w_rsrc = buffer_ops.create_buffer_resource_from_addr(fx.Int64(arg_w))
    x_dw = _x_dw(lane, wave, K)
    w_dw = (c0 + lane % fx.Int32(16)) * fx.Int32(K // 2) + _k_dw(lane, wave, K)
    acc = _zero()
    for s0 in range_constexpr(0, steps, CHUNK):
        idx = range(s0, min(s0 + CHUNK, steps))
        wv = [_bf16x8(w_rsrc, w_dw + fx.Int32(s * 16), CM_NT) for s in idx]
        xv = [_bf16x8(x_rsrc, x_dw + fx.Int32(s * 16)) for s in idx]
        for j in range_constexpr(len(wv)):
            acc = _mfma(xv[j], wv[j], acc)
    return acc


def reduce_tile(acc, tid, lane, wave, lds_base, red_word):
    """Sum the waves' partial tiles in LDS; thread tid gets token tid / 16,
    column tid % 16 of the tile. The caller's next barrier frees the LDS."""
    red = lds_f32(lds_base, red_word)
    base = (wave * fx.Int32(64) + lane) * fx.Int32(4)
    for r in range_constexpr(4):
        red[base + fx.Int32(r)] = fx.Float32(fx.Vector(acc)[r])
    rocdl.s_waitcnt(lgkmcnt=0)
    fx.gpu.barrier()
    t = tid // fx.Int32(TILE)
    c = tid % fx.Int32(TILE)
    src = (t // fx.Int32(4) * fx.Int32(16) + c) * fx.Int32(4) + t % fx.Int32(4)
    tot = fx.Float32(red[src])
    for w in range_constexpr(1, N_WAVES):
        tot = tot + fx.Float32(red[src + fx.Int32(w * 256)])
    return t, c, tot


RED_WORDS = N_WAVES * 64 * 4
