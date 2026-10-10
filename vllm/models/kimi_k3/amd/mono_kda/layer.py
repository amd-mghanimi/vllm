# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The mono KDA launch: a Kimi-K3 KDA layer's decode step -- in_proj_qkvgfab,
f_b_proj, the conv update, the gated delta rule, the gated RMSNorm and o_proj --
in one persistent FlyDSL launch (README.md).

Workgroups loop on a global ticket counter and the ticket picks the work item:

    0 .. S-1                 in_proj tiles of f_a and beta (every head needs them)
    S .. IN-1                in_proj tiles, head by head: q_h, k_h, v_h, g_h
    IN .. IN+H*M-1           decode task (head h, row m), head-major
    .. +OUT                  o_proj tiles

A decode task loads its state slice, W_fb rows and conv taps, then waits for
the f_a / beta tiles and its head's tiles; an o_proj tile loads its weights,
then waits for every decode task. Each wait names items of smaller tickets, and
a workgroup holding a ticket works on it until done, so the grid cannot
deadlock however many workgroups are resident.
"""

import hashlib
import pathlib

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.mxfp4_gemm_common import _raw, global_typed_ptr
from flydsl.expr import gpu, ptrtoint, rocdl
from flydsl.expr.typing import T

from vllm.models.kimi_k3.amd.mono_kda.plan import (
    HEAD_DIM,
    LINE_WORDS,
    SPIN_SLEEP,
    THREADS,
    TILE,
    W_DECODE,
    W_HEAD,
    W_SHARED,
    W_TICKET,
    Dims,
    ws_layout,
)
from vllm.models.kimi_k3.amd.mono_kda.stages import decode, gemv
from vllm.models.kimi_k3.amd.mono_kda.sync import bump, grab, st_wt, wait_ge

# FlyDSL keys its compile cache on the kernel's source and scalar closure
# values, not on the stages it calls: the package's sources go into the key.
_SOURCES = hashlib.sha256(
    b"".join(
        p.read_bytes() for p in sorted(pathlib.Path(__file__).parent.rglob("*.py"))
    )
).hexdigest()[:16]


def compile_mono_kda(*, heads, hidden, lower_bound, eps):
    """Compile the launch for one layer shape; returns the launcher.

    lower_bound: the KDA gate's lower bound (safe gate), or None.
    """
    d = Dims(heads=heads, hidden=hidden)
    P, ROW, S, HT = d.proj, d.proj_row, d.shared_tiles, d.head_tiles
    assert P % 64 == 0 and d.fa_col % 64 == 0 and d.beta_col % 64 == 0
    IN, OUT = d.in_tiles, d.out_tiles
    assert hidden % (4 * 32) == 0 and P % (4 * 32) == 0
    offs = ws_layout(d)[0]
    RED = max(gemv.RED_WORDS, decode.LDS_WORDS)
    S_FIRST, S_IN, S_DEC, S_OUT = RED, RED + 1, RED + 2, RED + 3
    lds_words = RED + 4
    dec_kw = dict(
        P=P, ROW=ROW, FA_COL=d.fa_col, BETA_COL=d.beta_col,
        lower_bound=lower_bound, eps=eps, scale=HEAD_DIM**-0.5,
    )  # fmt: skip

    lb = "none" if lower_bound is None else f"{lower_bound:g}"
    # an assembler symbol: no '-' or '.'
    name = f"k3_mono_kda_h{heads}_d{hidden}_lb{lb}_e{eps:g}"
    name = name.replace("-", "m").replace(".", "p")
    cache_tag = repr((name, _SOURCES))

    @fx.struct
    class Storage:
        raw: fx.Array[fx.Float32, lds_words, 16]

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def mono_kda_kernel(
        arg_x: fx.Int64,
        arg_w_in: fx.Int64,
        arg_w_fb: fx.Int64,
        arg_w_o: fx.Int64,
        arg_conv_w: fx.Int64,
        arg_conv_state: fx.Int64,
        arg_a_log: fx.Int64,
        arg_dtb: fx.Int64,
        arg_normw: fx.Int64,
        arg_idx: fx.Int64,
        arg_state: fx.Int64,
        arg_out: fx.Int64,
        arg_ws: fx.Int64,
        i32_M: fx.Int32,
        i32_grid: fx.Int32,
        i32_conv_stride: fx.Int32,
        i32_state_stride: fx.Int32,
    ):
        _ = cache_tag
        tid = fx.Int32(gpu.thread_id("x"))
        lane = tid % fx.Int32(64)
        wave = rocdl.readfirstlane(T.i32, tid // fx.Int32(64))
        smem = fx.SharedAllocator()
        lds = smem.allocate(Storage).peek().raw.ptr
        lds_base = fx.Int32(ptrtoint(lds))

        ctrl = fx.Int64(arg_ws)
        arg_proj = ctrl + fx.Int64(offs["proj"])
        arg_o = ctrl + fx.Int64(offs["o"])
        a_ticket = ctrl + fx.Int64(W_TICKET * 4)
        a_shared = ctrl + fx.Int64(W_SHARED * 4)
        a_decode = ctrl + fx.Int64(W_DECODE * 4)
        a_head = ctrl + fx.Int64(W_HEAD * 4)
        n_dec = fx.Int32(heads) * i32_M
        t_out = fx.Int32(IN) + n_dec
        n_work = t_out + fx.Int32(OUT)

        t = grab(wave, lane, lds_base, a_ticket, S_FIRST)

        while t < fx.Int32(IN):
            is_sh = t < fx.Int32(S)
            j = t - fx.Int32(S)
            hh = j // fx.Int32(HT)
            r = j - hh * fx.Int32(HT)
            part = r // fx.Int32(HEAD_DIM // TILE)
            sub = r - part * fx.Int32(HEAD_DIM // TILE)
            col0 = is_sh.select(
                fx.Int32(d.fa_col) + t * fx.Int32(TILE),
                part * fx.Int32(P) + hh * fx.Int32(HEAD_DIM) + sub * fx.Int32(TILE),
            )
            acc = gemv.mma_streamed(arg_x, arg_w_in, col0, lane, wave, i32_M, K=hidden)
            tt, c, v = gemv.reduce_tile(acc, tid, lane, wave, lds_base, 0)
            if tt < i32_M:
                st_wt(arg_proj, tt * fx.Int32(ROW) + col0 + c, v.to(fx.BFloat16), 2)
            word = is_sh.select(
                fx.Int32(W_SHARED), fx.Int32(W_HEAD) + hh * fx.Int32(LINE_WORDS)
            )
            bump(wave, lane, ctrl + fx.Int64(word) * fx.Int64(4))
            t = grab(wave, lane, lds_base, a_ticket, S_IN)

        while t < t_out:
            u = t - fx.Int32(IN)
            h = u // i32_M
            m = u - h * i32_M
            idx = buffer_ops.create_buffer_resource_from_addr(fx.Int64(arg_idx))
            slot = rocdl.readfirstlane(
                T.i32,
                fx.Int32(
                    buffer_ops.buffer_load(idx, _raw(m), vec_width=1, dtype=T.i32)
                ),
            )
            pre = decode.preload(
                tid, h, fx.max(slot, fx.Int32(0)), arg_state, i32_state_stride,
                arg_w_fb, arg_conv_w, arg_conv_state, i32_conv_stride, arg_dtb, P=P,
            )  # fmt: skip
            wait_ge(wave, a_shared, S, SPIN_SLEEP)
            wait_ge(
                wave, a_head + fx.Int64(h) * fx.Int64(LINE_WORDS * 4), HT, SPIN_SLEEP
            )
            decode.decode_task(
                tid, lane, wave, lds_base, h, m, slot, pre, arg_proj, arg_conv_state,
                i32_conv_stride, arg_a_log, arg_normw, arg_o, **dec_kw,
            )  # fmt: skip
            bump(wave, lane, a_decode)
            t = grab(wave, lane, lds_base, a_ticket, S_DEC)

        while t < n_work:
            n0 = (t - t_out) * fx.Int32(TILE)
            wv = gemv.load_w(arg_w_o, n0, lane, wave, K=P)
            wait_ge(wave, a_decode, n_dec, SPIN_SLEEP)
            acc = gemv.mma_loaded(arg_o, wv, lane, wave, i32_M, K=P)
            tt, c, v = gemv.reduce_tile(acc, tid, lane, wave, lds_base, 0)
            o_rsrc = buffer_ops.create_buffer_resource_from_addr(
                fx.Int64(arg_out),
                num_records_bytes=fx.Int64(i32_M) * fx.Int64(hidden * 2),
            )
            buffer_ops.buffer_store(
                _raw(v.to(fx.BFloat16)),
                o_rsrc,
                _raw(tt * fx.Int32(hidden) + n0 + c),
                mask=_raw(tt < i32_M),
            )
            t = grab(wave, lane, lds_base, a_ticket, S_OUT)

        # Every workgroup ends holding exactly one ticket >= n_work, taken after
        # its last item: the holder of the largest is the last to use the words.
        if t == n_work + i32_grid - fx.Int32(1):
            ctrl32 = global_typed_ptr(arg_ws, T.i32)
            if tid < fx.Int32(heads):
                ctrl32[fx.Int32(W_HEAD) + tid * fx.Int32(LINE_WORDS)] = fx.Int32(0)
            if tid == fx.Int32(0):
                ctrl32[W_TICKET] = fx.Int32(0)
                ctrl32[W_SHARED] = fx.Int32(0)
                ctrl32[W_DECODE] = fx.Int32(0)

    @flyc.jit
    def launch_mono_kda(
        arg_x: fx.Int64,
        arg_w_in: fx.Int64,
        arg_w_fb: fx.Int64,
        arg_w_o: fx.Int64,
        arg_conv_w: fx.Int64,
        arg_conv_state: fx.Int64,
        arg_a_log: fx.Int64,
        arg_dtb: fx.Int64,
        arg_normw: fx.Int64,
        arg_idx: fx.Int64,
        arg_state: fx.Int64,
        arg_out: fx.Int64,
        arg_ws: fx.Int64,
        i32_M: fx.Int32,
        i32_grid: fx.Int32,
        i32_conv_stride: fx.Int32,
        i32_state_stride: fx.Int32,
        stream: fx.Stream,
    ):
        mono_kda_kernel(
            arg_x,
            arg_w_in,
            arg_w_fb,
            arg_w_o,
            arg_conv_w,
            arg_conv_state,
            arg_a_log,
            arg_dtb,
            arg_normw,
            arg_idx,
            arg_state,
            arg_out,
            arg_ws,
            i32_M,
            i32_grid,
            i32_conv_stride,
            i32_state_stride,
        ).launch(  # fmt: skip
            grid=(fx.Int64(i32_grid), 1, 1), block=(THREADS, 1, 1), stream=stream
        )

    launch_mono_kda.work_meta = dict(IN=IN, OUT=OUT)
    return launch_mono_kda
