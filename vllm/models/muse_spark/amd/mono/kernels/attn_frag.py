# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: ignore-errors
# ruff: noqa: B002, E501, E741, SIM102, SIM108

# Vendored from AMD's Muse Spark FlyDSL monokernel kernels (mk/attn_frag.py):
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""The verify attention of one Muse Spark layer as a fragment of the persistent kernels: K3
and HA run it at their head (attn = "swa" / "full"), so the layer's attention and its decoder
work are one launch.

Shapes (a TP8 rank): 16 q heads, 2 kv heads (QPK = 8), head dim 64, 8 query tokens a request,
NR requests (K3 S8: 1, S16: 2; HA S32: 4). K / V bf16 in the layer's vLLM LBHNC cache
`[block][2][16][K 64 | V 64]` (block 16); the page table is the layer group's block_table
(int32 `[NR][stride]`); seq_used = vLLM's seq_lens (int32 [NR], keys incl. the 8 new ones).
q [8 NR, 16, 64] bf16 (row stride 1024, without the QK scale); out attn_o [8 NR, 1024] bf16.
SWA: vLLM's window (2047, 0), keys qpos - key < 2048, and only keys in the window are read
(vLLM frees the blocks before it: their block-table entries are stale).

Split (every CTA): group = (request, kv head), CPG CTAs a group; CTA j of a group takes tiles
[lo_t + j tps, +tps) of 64 keys (tps = ceil(n_t / CPG) at run time) for all 64 rows (8 tokens x 8
heads) of the kv head. 8 waves = 4 row quarters (16 rows) x 2 tile lanes; a step = 2 tiles.
K / V tiles: global -> VGPR (PD steps ahead; one 64-bit block base a wave, a wave loads the 16
keys of one block) -> LDS. S^T = K Q^T (16x16x32 bf16 MFMA, a lane holds one q row), online
softmax (exp2 domain, the -inf -> 0 rule), O^T = V^T P^T (P^T lane-local by a key permutation
inside the 32-key k block; V^T from LDS with ds_read_b64_tr_b16). The 2 tile lanes merge in LDS;
the CTA writes (m, l, unnormalized o) for 64 rows (fp32, device scope) + one tagged flag.

Merge: item = (group, row) -> one output row of 64 d. NR = 1: CTA 2i merges item i with 512
threads (32 segment lanes x 16 d-quads); NR >= 2: NR / 2 items a CTA. Poll the group's flags,
load the partials, merge (max trick) per thread, over lanes and over waves, out = o / l -> bf16
-> attn_o and the same rows as tagged pairs (P_AO), which the consumer (K3 stage O, HA ng)
polls directly.

Descriptor (int64 words, device memory; vllm_desc): 0 q, 1 the KV cache base (shuffled: the K
plane), 2 its block stride in elements (a value), 3 block table, 4 seq_lens, 5 attn_o (out), 6 part
(scratch of part_bytes()), 7 block-table row stride (elements), 8 real requests (requests >= it are
graph padding: no split, zero output rows, their flags still set), 9 the softmax scale x log2(e) as
fp32 bits, 10 the V plane (shuffled), 11 / 12 the fp32 [1] k_scale / v_scale (fp8), 13 unused. A
fragment's descriptor is row `layer` of a table of TABLE_ROWS rows.

Shuffled cache (make(shuffle=True); AITER's VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT, x = 8 bf16): a block
of each plane is [kv head][64 d / 8][16 keys][8] for K and [kv head][16 keys / 8][64 d][8] for V,
both planes with the same block stride. K tiles land in LDS as in the dense build; V lands as V^T
([d][key]), whose two 4-key runs of an MFMA A operand are plain ds_read_b64.

fp8 cache (make(fp8=True); vLLM's kv_cache_dtype fp8 / fp8_e4m3, OCP e4m3 bytes): the same element
placement with 1-byte elements, shuffled with x = 16 (K [kv head][64 d / 16][16 keys][16], V
[kv head][64 d][16 keys]). The loads move the fp8 bytes; the step dequantizes them (fp8 x scale,
RNE bf16) into the bf16 words the bf16 build loads, so the LDS tiles and the MFMAs are the bf16 ones.
"""

from types import SimpleNamespace

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.kernels_common import LOG2E
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T, as_ir_value

from .dev import (
    BLOCKS,
    CM_DEV,
    CM_NT,
    OOB_BOUND,
    THREADS,
    bf16_pair,
    bf_hi,
    bf_lo,
    hw_exp2,
    ld_f32,
    ld_i32,
    ld_i32_if,
    ld_vec,
    ld_vec_if,
    mb_poll,
    mb_put,
    mb_put2,
    mfma_bf16,
    permlane_swap,
    rsrc,
    traced,
    uniform,
    xred,
)

HKV, QPK, HD, BS, TILE, NQ = 2, 8, 64, 16, 64, 8
NH = HKV * QPK
WIN_SWA = 2048  # vLLM's window (2047, 0) -> keys qpos - key < 2048
KP = 72  # LDS row stride of a K / V tile (bf16)
KPW = KP // 2
NEG = float("-inf")
PGMAX = 512  # page ids a CTA staged in one round (128 tiles)
# keys a request the page staging covers (max_position_embeddings)
VLLM_MAX_KEYS = 1 << 20
REGION_IDS = {"aseg": 40}
DESC_WORDS = 14
TABLE_ROWS = 64
SX = 8  # bf16 a 16 B shuffle group
SX8 = 16  # fp8 a 16 B shuffle group
E4M3_MAX = 448.0

# parts of the scratch buffer (bytes); the segment count is always BLOCKS (G x CPG <= 256)
P_O = 0  # partial o, fp32 [seg][row][64]
P_ML = BLOCKS * 64 * HD * 4  # (m, l) [seg][row]
P_SF = P_ML + BLOCKS * 64 * 2 * 4  # one tagged flag a segment
P_IF = BLOCKS * 64 * (HD + 4) * 8  # one tagged flag an output row
P_AO = P_IF + 16 * 128 * 8  # attn_o as tagged pairs [row][512 words]
P_END = P_AO + 128 * 512 * 8
assert P_SF + BLOCKS * 8 <= P_IF


def part_bytes():
    return P_END


def default_cpg(NR, swa):
    g = NR * HKV
    if swa:
        return min(BLOCKS // g, 17)  # 2 tiles a CTA (34 tiles at most in the window)
    return BLOCKS // g


@traced
def store_kv_shuffled(w2, isv, kv, tok, p, k_blk, v_blk):
    """Pair (2p, 2p + 1) of head kv (w2: bf16 low, high) of token tok into the shuffled block
    whose K / V plane bytes start at k_blk / v_blk (64-bit, wave-uniform): K as one dword, V as
    two bf16 a 16 B group apart."""
    if isv:
        e = kv * (BS * HD) + (tok // SX) * (HD * SX) + 2 * p * SX + tok % SX
        pair = fx.Vector.from_elements([w2], fx.Int32).bitcast(fx.BFloat16)
        bo.buffer_store(pair[0], rsrc(v_blk), e * 2, offset_is_bytes=True)
        bo.buffer_store(pair[1], rsrc(v_blk), e * 2 + 2 * SX, offset_is_bytes=True)
    else:
        e = kv * (BS * HD) + (p // 4) * (BS * SX) + tok * SX + 2 * (p % 4)
        bo.buffer_store(w2, rsrc(k_blk), e // 2)


def _sat(x):
    # v_cvt_pk_fp8_f32 turns a finite overflow into NaN; HIP's SATFINITE clamps first
    x = (x > fx.Float32(E4M3_MAX)).select(fx.Float32(E4M3_MAX), x)
    return (x < fx.Float32(-E4M3_MAX)).select(fx.Float32(-E4M3_MAX), x)


def fp8_pair(w2, inv):
    """The bf16 pair w2 (low, high) times inv -> two e4m3 bytes in the low 16 bits, as the ROCm
    fused insert's scaled_convert (fp32 product, RNE, saturated)."""
    return fx.Int32(
        rocdl.cvt_pk_fp8_f32(
            T.i32,
            as_ir_value(_sat(bf_lo(w2) * inv)),
            as_ir_value(_sat(bf_hi(w2) * inv)),
            as_ir_value(fx.Int32(0)),
            False,
        )
    )


def dequant16(w4, scale):
    """4 dwords of 16 e4m3 bytes -> 8 bf16 pair words, byte order kept: fp8 x scale, RNE bf16."""
    out = []
    for i in range(4):
        for sel in (False, True):
            f = fx.Vector(
                rocdl.cvt_pk_f32_fp8(T.vec(2, T.f32), as_ir_value(fx.Int32(w4[i])), sel)
            )
            out.append(bf16_pair(f[0] * scale, f[1] * scale))
    return out


@traced
def store_kv_fp8(w2, isv, kv, tok, p, k_blk, v_blk, k_scale, v_scale, shuffle):
    """Pair (2p, 2p + 1) of head kv (w2: bf16 low, high) of token tok, quantized with 1 / k_scale or
    1 / v_scale (fp32 [1] pointers), into the fp8 block whose K / V plane bytes start at k_blk / v_blk
    (64-bit, wave-uniform; dense: the page [kv head][16][K 64 | V 64] at k_blk). K and dense V as one
    16-bit store, shuffled V as two bytes a 16 B group apart (x = 16)."""
    inv = fx.Float32(1.0) / ld_f32(isv.select(v_scale, k_scale), 0)
    b2 = fp8_pair(w2, inv)
    bytes4 = fx.Vector.from_elements([b2], fx.Int32).bitcast(fx.Int8)
    half = fx.Vector.from_elements([b2], fx.Int32).bitcast(fx.BFloat16)[0]
    if const_expr(not shuffle):
        e = kv * (BS * 2 * HD) + tok * (2 * HD) + isv.select(fx.Int32(HD), fx.Int32(0))
        bo.buffer_store(half, rsrc(k_blk), e + 2 * p, offset_is_bytes=True)
    else:
        if isv:
            ev = kv * (BS * HD) + 2 * p * SX8 + tok
            bo.buffer_store(bytes4[0], rsrc(v_blk), ev, offset_is_bytes=True)
            bo.buffer_store(bytes4[1], rsrc(v_blk), ev + SX8, offset_is_bytes=True)
        else:
            ek = kv * (BS * HD) + (p // 8) * (BS * SX8) + tok * SX8 + 2 * (p % 8)
            bo.buffer_store(half, rsrc(k_blk), ek, offset_is_bytes=True)


def make(NR, swa, maxk=VLLM_MAX_KEYS, shuffle=False, fp8=False):
    """The fragment for NR requests; swa = SWA (window 2048) or full attention. It stages enough
    page ids a CTA for maxk keys a request (PGN); shuffle = AITER's shuffled KV cache; fp8 = an e4m3
    cache with per-tensor k / v scales (descriptor words 11 / 12)."""
    # K / V load cache policy: cached for SWA, nontemporal for full
    KCM = 0 if swa else CM_NT

    G = NR * HKV
    CPG = default_cpg(NR, swa)
    # steps a CTA at the window / at 32k context
    NST = -(-(-(-(34 if swa else 512) // CPG)) // 2)
    PD = max(1, min(4, NST))  # K / V steps in flight
    assert G * CPG <= BLOCKS
    NI = NR * NQ * NH  # output rows
    if NI >= BLOCKS:
        IPC = NI // BLOCKS
        STR = 1
    else:
        IPC = 1
        STR = BLOCKS // NI
    TPI = THREADS // IPC
    WPI = TPI // 64
    SL = TPI // 16
    SPT = -(-CPG // SL)
    assert IPC * TPI == THREADS and WPI >= 1 and NI <= 16 * 128 and PGMAX == THREADS
    WIN = WIN_SWA if swa else 0
    PGN = PGMAX  # page ids staged a CTA (LDS); full attention: maxk keys a request (PGN // THREADS rounds)
    if not swa:
        tiles = -(-(-(-maxk // TILE)) // CPG)  # tiles a CTA at maxk keys
        PGN = max(PGMAX, -(-tiles * (TILE // BS) // THREADS) * THREADS)

    @fx.struct
    class Lds:
        kv: fx.Array[fx.Int32, 2 * 2 * 64 * KPW, 16]  # [tile lane][K | V][key][KP bf16]
        # tile lane 1 -> 0: [quarter][lane][m, l, acc 16]
        mrg: fx.Array[fx.Float32, 4 * 64 * 18, 16]
        mg2: fx.Array[fx.Float32, 8 * 16 * 6, 16]  # merge: [wave][d quad][M, L, A 4]
        # the CTA's page ids, staged once (off the K / V vmcnt chain)
        pg: fx.Array[fx.Int32, PGN, 16]

    def fmax(a, b):
        return (a > b).select(a, b)

    def fadd(a, b):
        return a + b

    def p64(dv, i):
        lo = uniform(dv[2 * i])
        hi = uniform(dv[2 * i + 1])
        return (fx.Int64(hi) << fx.Int64(32)) | (fx.Int64(lo) & fx.Int64(0xFFFFFFFF))

    def desc(dptr):
        dv = []
        for i in range(DESC_WORDS // 2):
            v4 = ld_vec(dptr, 4 * i, 4)
            dv += [v4[e] for e in range(4)]
        ks = vs = None
        if const_expr(fp8):
            ks = uniform(ld_i32(p64(dv, 11), 0)).bitcast(fx.Float32)
            vs = uniform(ld_i32(p64(dv, 12), 0)).bitcast(fx.Float32)
        return SimpleNamespace(
            ks=ks,
            vs=vs,
            q=p64(dv, 0),
            k=p64(dv, 1),
            v=p64(dv, 2),
            pt=p64(dv, 3),
            used=p64(dv, 4),
            out=p64(dv, 5),
            part=p64(dv, 6),
            pts=uniform(dv[14]),
            nreq=uniform(dv[16]),
            qks=uniform(dv[18]).bitcast(fx.Float32),
            vp=p64(dv, 10),
        )

    def geom(d, req):
        """Run-time key-tile range of request req (uniform)."""
        seq_len = uniform(ld_i32(d.used, req))
        ctx = seq_len - NQ
        if const_expr(swa):
            lo = fx.max(ctx - WIN + 1, 0)
        else:
            lo = fx.Int32(0)
        lo_t = lo // TILE
        hi_t = (seq_len + TILE - 1) // TILE
        n_t = hi_t - lo_t
        tps = fx.max((n_t + CPG - 1) // CPG, 1)
        nseg = (n_t + tps - 1) // tps
        return seq_len, ctx, lo_t, hi_t, tps, nseg

    def tr16(kvb, byte):
        ptr = fx.to_llvm_ptr(fx.inttoptr(kvb.type, fx.Int64(byte)))
        return fx.Vector(
            rocdl.ds_read_tr16_b64(T.vec(4, T.bf16), ptr).result, (4,), fx.BFloat16
        )

    def key_mask(key, seq_len, wlo):
        ok = key < seq_len
        if const_expr(swa):
            ok = ok & (key >= wlo)
        return ok

    def v_masks(k0, seq_len, wlo):
        """V^T words j = keys (k0 + 2j, + 1) of a block -> the bf16 halves of the keys in
        [wlo, seq_len)."""
        out = []
        for j in range_constexpr(8):
            key = k0 + 2 * j
            lo = key_mask(key, seq_len, wlo).select(fx.Int32(0xFFFF), fx.Int32(0))
            hi = key_mask(key + 1, seq_len, wlo).select(fx.Int32(-65536), fx.Int32(0))
            out.append(lo | hi)
        return out

    def kv_load_shuffled(d, tile, t1, seq_len, kvh, pg, lkey, lch, ok, wlo):
        """K: key lkey's d [16 lch, 16 lch + 16) (two 16 B groups; fp8: one). V^T: d = 4 (lkey % 16)
        + lch of the block's 16 keys (two 16 B groups; fp8: one); the loads take the whole block, so
        keys outside [wlo, seq_len) are zeroed as the dense loads leave them (fp8: in step)."""
        k0 = tile * TILE + (lkey // BS) * BS
        bok = (tile < t1) & (k0 < seq_len)
        if const_expr(swa):
            bok = bok & (k0 + BS > wlo)
        dcol = (lkey % BS) * 4 + lch
        if const_expr(fp8):
            hoff = fx.Int64(uniform(pg)) * d.v + fx.Int64(kvh * (BS * HD))
            koff = lch * (BS * SX8 // 4) + (lkey % BS) * (SX8 // 4)
            kq = ld_vec_if(ok, d.k + hoff, koff, 4, OOB_BOUND, KCM)
            vq = ld_vec_if(bok, d.vp + hoff, dcol * (BS // 4), 4, OOB_BOUND, KCM)
            return fx.Vector.from_elements(
                [kq[i] for i in range(4)] + [vq[i] for i in range(4)], fx.Int32
            )
        hoff = (fx.Int64(uniform(pg)) * d.v + fx.Int64(kvh * (BS * HD))) * fx.Int64(2)
        kb = d.k + hoff
        vb = d.vp + hoff
        koff = lch * (BS * SX) + (lkey % BS) * (SX // 2)
        k = [ld_vec_if(ok, kb, koff + o, 4, OOB_BOUND, KCM) for o in (0, BS * SX // 2)]
        masks = v_masks(k0, seq_len, wlo)
        v = []
        for tg in range_constexpr(2):
            w4 = ld_vec_if(
                bok, vb, dcol * (SX // 2) + tg * (HD * SX // 2), 4, OOB_BOUND, KCM
            )
            for i in range_constexpr(4):
                v.append(w4[i] & masks[4 * tg + i])
        return fx.Vector.from_elements(
            [k[j][i] for j in range(2) for i in range(4)] + v, fx.Int32
        )

    def kv_words(d, s, kvc, t0, seq_len, ltl, lkey, wlo):
        """Step s's loaded words -> the 16 bf16 pair words (K 8 | V 8) of the bf16 build; fp8:
        dequantized with the k / v scales (shuffled V masked to [wlo, seq_len) after)."""
        if const_expr(not fp8):
            return [kvc[i] for i in range(16)]
        kw = dequant16([kvc[i] for i in range(4)], d.ks)
        vw = dequant16([kvc[4 + i] for i in range(4)], d.vs)
        if const_expr(shuffle):
            k0 = (t0 + 2 * s + ltl) * TILE + (lkey // BS) * BS
            vw = [w & m for w, m in zip(vw, v_masks(k0, seq_len, wlo))]
        return kw + vw

    def kv_load(d, t0, t1, seq_len, kvh, step, pgl, ltl, lkey, lch, wlo=None):
        tile = t0 + 2 * step + ltl
        kg = tile * TILE + lkey
        ok = (tile < t1) & (kg < seq_len)
        pg = fx.ptr_load(
            pgl + fx.min((2 * step + ltl) * (TILE // BS) + lkey // BS, PGN - 1)
        )
        if const_expr(swa):
            # keys before the window: no access (vLLM frees their blocks)
            ok = ok & (kg >= wlo)
        if const_expr(shuffle):
            return kv_load_shuffled(d, tile, t1, seq_len, kvh, pg, lkey, lch, ok, wlo)
        if const_expr(fp8):
            # dwords: row (kg % 16) of 128 fp8, K / V channels [16 lch, 16 lch + 16)
            base = d.k + fx.Int64(uniform(pg)) * d.v + fx.Int64(kvh * (BS * 2 * HD))
            off = (kg % BS) * (HD // 2) + 4 * lch
            kv = [ld_vec_if(ok, base, off + o, 4, OOB_BOUND, KCM) for o in (0, HD // 4)]
            return fx.Vector.from_elements(
                [kv[j][i] for j in range(2) for i in range(4)], fx.Int32
            )
        base = d.k + (
            fx.Int64(uniform(pg)) * d.v + fx.Int64(kvh * (BS * 2 * HD))
        ) * fx.Int64(2)
        # dwords: row (kg % 16) of 128 bf16, channels [16 lch, 16 lch + 16)
        off = (kg % BS) * HD + 8 * lch
        kv = [
            ld_vec_if(ok, base, off + o, 4, OOB_BOUND, KCM)
            for o in (0, 4, HD // 2, HD // 2 + 4)
        ]
        return fx.Vector.from_elements(
            [kv[j][i] for j in range(4) for i in range(4)], fx.Int32
        )

    def step(
        d,
        s,
        kvc,
        m,
        l,
        acc,
        t0,
        t1,
        seq_len,
        kvh,
        pgl,
        ltl,
        lkey,
        lch,
        wtl,
        r16,
        g,
        qpos,
        qb,
        kvl,
        kvbase,
        wlo=None,
    ):
        """One step (2 tiles): K / V of step s (kvc) -> LDS, the loads of step s + PD issued, then this wave's tile."""
        gpu.barrier()
        kvw = kv_words(d, s, kvc, t0, seq_len, ltl, lkey, wlo)
        for kind in range_constexpr(2):
            words = kvw[8 * kind : 8 * kind + 8]
            b0 = ((ltl * 2 + kind) * 64 + lkey) * KPW + 8 * lch
            if const_expr(shuffle and kind == 1):
                dcol = (lkey % BS) * 4 + lch
                b0 = ((ltl * 2 + 1) * 64 + dcol) * KPW + 8 * (lkey // BS)
            fx.ptr_store(fx.Vector.from_elements(words[:4], fx.Int32), kvl + b0)
            fx.ptr_store(fx.Vector.from_elements(words[4:], fx.Int32), kvl + b0 + 4)
        gpu.barrier()
        rocdl.sched_barrier(0)
        kvn = kv_load(d, t0, t1, seq_len, kvh, s + PD, pgl, ltl, lkey, lch, wlo)
        rocdl.sched_barrier(0)
        tile = t0 + 2 * s + wtl
        kb = tile * TILE
        tin = tile < t1
        # S^T = K Q^T: lane holds q row r16, keys 16 n + 4 g + i
        xs = []
        mx = fx.Float32(NEG)
        for n in range_constexpr(4):
            sv = fx.Vector.filled(4, 0.0, fx.Float32)
            for c in range_constexpr(2):
                a = fx.Vector(
                    fx.ptr_load(
                        kvl
                        + ((wtl * 2) * 64 + 16 * n + r16) * KPW
                        + (32 * c + 8 * g) // 2,
                        result_type=fx.Vector.make_type(4, fx.Int32),
                    )
                ).bitcast(fx.BFloat16)
                sv = mfma_bf16(a, qb[c], sv)
            for i in range_constexpr(4):
                key = kb + 16 * n + 4 * g + i
                ok = tin & (key <= qpos)
                if const_expr(swa):
                    ok = ok & ((qpos - key) < WIN)
                x = ok.select(sv[i] * d.qks, fx.Float32(NEG))
                xs.append(x)
                mx = fmax(mx, x)
        mx = xred(mx, 16, fmax)
        mx = xred(mx, 32, fmax)
        mj = fmax(m, mx)
        mj = (mj > fx.Float32(NEG)).select(mj, fx.Float32(0.0))
        ps = [hw_exp2(x - mj) for x in xs]
        alpha = hw_exp2(m - mj)
        psum = fx.Float32(0.0)
        for x in ps:
            psum = psum + x
        l = l * alpha + psum
        acc = [
            fx.Vector.from_elements([acc[j][i] * alpha for i in range(4)], fx.Float32)
            for j in range(4)
        ]
        m = mj
        # O^T += V^T P^T
        for c in range_constexpr(2):
            pb = fx.Vector.from_elements(
                [
                    bf16_pair(ps[(2 * c) * 4 + 0], ps[(2 * c) * 4 + 1]),
                    bf16_pair(ps[(2 * c) * 4 + 2], ps[(2 * c) * 4 + 3]),
                    bf16_pair(ps[(2 * c + 1) * 4 + 0], ps[(2 * c + 1) * 4 + 1]),
                    bf16_pair(ps[(2 * c + 1) * 4 + 2], ps[(2 * c + 1) * 4 + 3]),
                ],
                fx.Int32,
            ).bitcast(fx.BFloat16)
            for j in range_constexpr(4):
                if const_expr(shuffle):
                    # V^T row d = 16 j + r16: keys 32 c + 4 g + [0, 4) and 32 c + 16 + 4 g + [0, 4)
                    vr = (
                        kvl + ((wtl * 2 + 1) * 64 + 16 * j + r16) * KPW + 16 * c + 2 * g
                    )
                    v2 = fx.Vector.make_type(2, fx.Int32)
                    lo_ = fx.Vector(fx.ptr_load(vr, result_type=v2))
                    hi_ = fx.Vector(fx.ptr_load(vr + 8, result_type=v2))
                    a = lo_.shuffle(hi_, [0, 1, 2, 3]).bitcast(fx.BFloat16)
                else:
                    row1 = 32 * c + 4 * g + r16 // 4
                    col = 16 * j + 4 * (r16 % 4)
                    vb = kvbase + (((wtl * 2 + 1) * 64 + row1) * KP + col) * 2
                    lo_ = tr16(kvl, vb)
                    hi_ = tr16(kvl, vb + 16 * KP * 2)
                    a = lo_.shuffle(hi_, [0, 1, 2, 3, 4, 5, 6, 7])
                acc[j] = mfma_bf16(a, pb, acc[j])
        return m, l, acc, kvn

    @traced
    def split(d, tag, tid, bid, lane, wave, L):
        kvl = L.kv.ptr
        kvbase = fx.Int32(fx.ptrtoint(kvl))
        grp = bid // CPG
        jseg = bid % CPG
        req = grp // HKV
        kvh = grp % HKV
        seq_len, ctx, lo_t, hi_t, tps, nseg = geom(d, req)
        t0 = lo_t + jseg * tps
        t1 = fx.min(t0 + tps, hi_t)
        wlo = None
        if const_expr(swa):
            wlo = fx.max(ctx - WIN + 1, 0)  # the first key in the window
        if (bid < G * CPG) & (req < d.nreq):
            if t0 < t1:
                nst = (t1 - t0 + 1) // 2
                # load lanes: tile lane, key, 16 B chunk
                ltl = tid // 256
                lkey = (tid % 256) // 4
                lch = tid % 4

                # this wave: row quarter qr (16 q rows), tile lane wtl
                qr = wave % 4
                wtl = wave // 4
                r16 = lane % 16
                g = lane // 16
                m_row = 16 * qr + r16
                tok = m_row // QPK
                head = kvh * QPK + m_row % QPK
                qpos = ctx + tok
                qb = [
                    ld_vec(
                        d.q,
                        ((NQ * req + tok) * (NH * HD) + head * HD + 32 * c + 8 * g)
                        // 2,
                        4,
                    ).bitcast(fx.BFloat16)
                    for c in range(2)
                ]
                pgl = L.pg.ptr
                # full attention past 128 tiles a CTA
                for r in range_constexpr(1, PGN // THREADS):
                    tr = tid + r * THREADS
                    pk = t0 * (TILE // BS) + tr
                    pok = (tr < (t1 - t0) * (TILE // BS)) & (pk * BS < seq_len)
                    fx.ptr_store(
                        ld_i32_if(pok, d.pt, req * d.pts + pk, OOB_BOUND), pgl + tr
                    )
                pk = t0 * (TILE // BS) + tid  # page tid of the CTA's range
                pok = (tid < (t1 - t0) * (TILE // BS)) & (pk * BS < seq_len)
                if const_expr(swa):
                    # blocks before the window: stale ids, not read
                    pok = pok & ((pk + 1) * BS > wlo)
                pgv = ld_i32_if(pok, d.pt, req * d.pts + pk, OOB_BOUND)
                fx.ptr_store(pgv, pgl + fx.min(tid, PGMAX - 1))
                gpu.barrier()
                kvq = []
                # in step order: the scheduler must not move slot 0 behind the others
                for i in range_constexpr(PD):
                    kvq.append(
                        kv_load(d, t0, t1, seq_len, kvh, i, pgl, ltl, lkey, lch, wlo)
                    )
                    rocdl.sched_barrier(0)
                m = fx.Float32(NEG)
                l = fx.Float32(0.0)
                acc = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(4)]
                s = fx.Int32(0)
                while s < nst:
                    # PD steps a trip with fixed register slots (a rotated list makes LLVM copy the slots at the back
                    # edge behind a vmcnt wait); steps past nst are fully masked (no loads, no change of m / l / acc)
                    for j in range_constexpr(PD):
                        m, l, acc, kvn = step(
                            d,
                            s + j,
                            kvq[j],
                            m,
                            l,
                            acc,
                            t0,
                            t1,
                            seq_len,
                            kvh,
                            pgl,
                            ltl,
                            lkey,
                            lch,
                            wtl,
                            r16,
                            g,
                            qpos,
                            qb,
                            kvl,
                            kvbase,
                            wlo,
                        )
                        kvq = kvq[:j] + [kvn] + kvq[j + 1 :]
                    s = s + PD
                lt = xred(l, 16, fadd)
                lt = xred(lt, 32, fadd)
                mrg = L.mrg.ptr
                mb = (qr * 64 + lane) * 18
                if wtl == 1:
                    fx.ptr_store(m, mrg + mb)
                    fx.ptr_store(lt, mrg + mb + 1)
                    for j in range_constexpr(4):
                        for i in range_constexpr(4):
                            fx.ptr_store(acc[j][i], mrg + mb + 2 + 4 * j + i)
                gpu.barrier()
                if wtl == 0:
                    m1 = fx.ptr_load(mrg + mb)
                    l1 = fx.ptr_load(mrg + mb + 1)
                    mm = fmax(m, m1)
                    a0 = hw_exp2(m - mm)
                    a1 = hw_exp2(m1 - mm)
                    lf = lt * a0 + l1 * a1
                    seg = grp * CPG + jseg
                    po = d.part + fx.Int64(P_O)
                    for j in range_constexpr(4):
                        ov = [
                            acc[j][i] * a0 + fx.ptr_load(mrg + mb + 2 + 4 * j + i) * a1
                            for i in range(4)
                        ]
                        bo.buffer_store(
                            fx.Vector.from_elements(
                                [v_.bitcast(fx.Int32) for v_ in ov], fx.Int32
                            ),
                            rsrc(po),
                            ((seg * 64 + m_row) * HD + 16 * j + 4 * g),
                            cache_modifier=CM_DEV,
                        )
                    if g == 0:
                        bo.buffer_store(
                            fx.Vector.from_elements(
                                [mm.bitcast(fx.Int32), lf.bitcast(fx.Int32)], fx.Int32
                            ),
                            rsrc(d.part + fx.Int64(P_ML)),
                            (seg * 64 + m_row) * 2,
                            cache_modifier=CM_DEV,
                        )
                rocdl.s_waitcnt(vmcnt=0, expcnt=0, lgkmcnt=0)
                gpu.barrier()
                if tid == 0:
                    mb_put(tag, d.part + fx.Int64(P_SF), grp * CPG + jseg, fx.Int32(1))

    def comb(Ma, La, Aa, Mb, Lb, Ab):
        M = fmax(Ma, Mb)
        wa = (Ma > fx.Float32(NEG)).select(hw_exp2(Ma - M), fx.Float32(0.0))
        wb = (Mb > fx.Float32(NEG)).select(hw_exp2(Mb - M), fx.Float32(0.0))
        return M, La * wa + Lb * wb, [Aa[i] * wa + Ab[i] * wb for i in range(4)]

    def swap(off, v):
        a, b = permlane_swap(off, v.bitcast(fx.Int32), v.bitcast(fx.Int32))
        return a.bitcast(fx.Float32), b.bitcast(fx.Float32)

    @traced
    def merge(d, tag, health, tid, bid, lane, wave, L):
        if const_expr(STR > 1):
            has = (bid % STR) == 0
            item = bid // STR
        else:
            has = bid >= 0
            item = bid * IPC + tid // TPI
        if has:
            itl = tid // TPI
            tt = tid % TPI
            sl = tt // 16
            dq = tt % 16
            grp = item // 64
            row = item % 64
            req = grp // HKV
            kvh = grp % HKV
            real = req < d.nreq
            seq_len, ctx, lo_t, hi_t, tps, nseg = geom(d, fx.min(req, d.nreq - 1))
            mls, os_ = [], []
            sf = d.part + fx.Int64(P_SF)
            specs, oks = [], []
            for k in range_constexpr(SPT):
                sg = sl + SL * k
                specs.append((sf, grp * CPG + sg, 1))
                oks.append((sg < nseg) & real)
            mb_poll(tag, health, specs, REGION_IDS["aseg"], oks=oks)
            for k in range_constexpr(SPT):
                seg = grp * CPG + sl + SL * k
                mls.append(
                    ld_vec_if(
                        oks[k],
                        d.part + fx.Int64(P_ML),
                        (seg * 64 + row) * 2,
                        2,
                        P_SF - P_ML,
                        CM_DEV,
                    )
                )
                os_.append(
                    ld_vec_if(
                        oks[k],
                        d.part + fx.Int64(P_O),
                        (seg * 64 + row) * HD + 4 * dq,
                        4,
                        P_ML,
                        CM_DEV,
                    )
                )
            M = fx.Float32(NEG)
            for k in range_constexpr(SPT):
                M = oks[k].select(fmax(M, mls[k][0].bitcast(fx.Float32)), M)
            Ls = fx.Float32(0.0)
            A = [fx.Float32(0.0) for _ in range(4)]
            for k in range_constexpr(SPT):
                w = oks[k].select(
                    hw_exp2(mls[k][0].bitcast(fx.Float32) - M), fx.Float32(0.0)
                )
                Ls = Ls + w * mls[k][1].bitcast(fx.Float32)
                A = [A[i] + w * os_[k][i].bitcast(fx.Float32) for i in range(4)]
            for off in (16, 32):
                ma, mb_ = swap(off, M)
                la, lb = swap(off, Ls)
                aa, ab = [], []
                for i in range(4):
                    x_, y_ = swap(off, A[i])
                    aa.append(x_)
                    ab.append(y_)
                M, Ls, A = comb(ma, la, aa, mb_, lb, ab)
            g2 = L.mg2.ptr
            wi = tt // 64
            if lane < 16:
                b = ((itl * WPI + wi) * 16 + dq) * 6
                fx.ptr_store(M, g2 + b)
                fx.ptr_store(Ls, g2 + b + 1)
                for i in range_constexpr(4):
                    fx.ptr_store(A[i], g2 + b + 2 + i)
            gpu.barrier()
            if tt < 16:
                b0 = ((itl * WPI) * 16 + dq) * 6
                M = fx.ptr_load(g2 + b0)
                Ls = fx.ptr_load(g2 + b0 + 1)
                A = [fx.ptr_load(g2 + b0 + 2 + i) for i in range(4)]
                for w_ in range_constexpr(1, WPI):
                    b = ((itl * WPI + w_) * 16 + dq) * 6
                    M, Ls, A = comb(
                        M,
                        Ls,
                        A,
                        fx.ptr_load(g2 + b),
                        fx.ptr_load(g2 + b + 1),
                        [fx.ptr_load(g2 + b + 2 + i) for i in range(4)],
                    )
                f = fx.Float32(1.0) / (Ls > fx.Float32(0.0)).select(Ls, fx.Float32(1.0))
                tok = row // QPK
                head = kvh * QPK + row % QPK
                w2 = fx.Vector.from_elements(
                    [bf16_pair(A[0] * f, A[1] * f), bf16_pair(A[2] * f, A[3] * f)],
                    fx.Int32,
                )
                bo.buffer_store(
                    w2,
                    rsrc(d.out),
                    ((NQ * req + tok) * (NH * HD) + head * HD + 4 * dq) // 2,
                    cache_modifier=CM_DEV,
                )
                mb_put2(
                    tag,
                    d.part + fx.Int64(P_AO),
                    ((NQ * req + tok) * (NH * HD) + head * HD + 4 * dq) // 2,
                    w2[0],
                    w2[1],
                )
                rocdl.s_waitcnt(vmcnt=0, expcnt=0, lgkmcnt=0)
                if tt == 0:
                    mb_put(tag, d.part + fx.Int64(P_IF), item, fx.Int32(1))

    def run(dptr, tag, health, tid, bid, lane, wave, L):
        """The whole fragment; returns (the attn_o pointer, the tagged attn_o rows)."""
        d = desc(dptr)
        split(d, tag, tid, bid, lane, wave, L)
        merge(d, tag, health, tid, bid, lane, wave, L)
        return d.out, d.part + fx.Int64(P_AO)

    return SimpleNamespace(
        Lds=Lds,
        run=run,
        key=f"attnfrag_nr{NR}_{'swa' if swa else 'full'}{'_shuf' if shuffle else ''}{'_fp8' if fp8 else ''}_c{CPG}_p{PD}_k{KCM}_{_SRC_HASH}",
    )


with open(__file__, "rb") as _f:
    import hashlib as _hl

    _SRC_HASH = _hl.sha1(_f.read()).hexdigest()[:12]


def vllm_scale_word(scale):
    """Descriptor word 9: scale x log2(e) as fp32 bits (an int64 word)."""
    import struct

    return struct.unpack("<i", struct.pack("<f", float(scale) * LOG2E))[0]


FP8_STORAGE = ("uint8", "float8_e4m3fn")  # vLLM's fp8 / fp8_e4m3 cache storage


def kv_planes(kv_cache, shuffled=False):
    """(K base, V base, block stride in elements) of a layer's bf16 or fp8 (FP8_STORAGE) cache
    view: dense `[block][2 kv heads][16][K 64 | V 64]` (V base = K base), or shuffled
    `[block][K | V][16][2 x 64]` (vLLM's two head slots, AITER reinterprets each slot's block in
    place)."""
    import torch

    assert (
        kv_cache.dtype == torch.bfloat16
        or str(kv_cache.dtype).removeprefix("torch.") in FP8_STORAGE
    ), kv_cache.dtype
    if not shuffled:
        assert tuple(kv_cache.shape[1:]) == (HKV, BS, 2 * HD), kv_cache.shape
        assert tuple(kv_cache.stride()[1:]) == (BS * 2 * HD, 2 * HD, 1), (
            kv_cache.stride()
        )
        return kv_cache.data_ptr(), kv_cache.data_ptr(), kv_cache.stride(0)
    assert tuple(kv_cache.shape[1:]) == (2, BS, HKV * HD), kv_cache.shape
    assert tuple(kv_cache.stride()[2:]) == (HKV * HD, 1), kv_cache.stride()
    v_ptr = kv_cache.data_ptr() + kv_cache.stride(1) * kv_cache.element_size()
    return kv_cache.data_ptr(), v_ptr, kv_cache.stride(0)


def vllm_desc(
    q,
    kv_cache,
    block_table,
    seq_lens,
    out,
    part,
    nreq,
    scale,
    shuffled=False,
    k_scale=None,
    v_scale=None,
):
    """The DESC_WORDS descriptor words of one layer: q [8 nreq.., 16 x 64] bf16 rows,
    kv_cache the layer's bf16 or fp8 view (kv_planes), block_table int32
    `[>= nreq][stride]` of the layer's group, seq_lens int32 [>= nreq], out attn_o [rows, 1024] bf16, part the scratch
    (int32, part_bytes()), nreq the requests of the step (graph padding included: seq_len 0 = no keys, zero rows);
    k_scale / v_scale the layer's fp32 [1] scales, given exactly when the cache is fp8."""
    import torch

    assert (
        q.dtype == torch.bfloat16
        and q.stride(-1) == 1
        and q.shape[-1] * (q.shape[-2] if q.dim() == 3 else 1) % (NH * HD) == 0
    )
    k_ptr, v_ptr, bstride = kv_planes(kv_cache, shuffled)
    fp8 = kv_cache.dtype != torch.bfloat16
    assert (k_scale is not None) == fp8 and (v_scale is not None) == fp8
    if fp8:
        for t in (k_scale, v_scale):
            assert t.dtype == torch.float32 and t.numel() == 1 and t.is_cuda, t
    assert (
        block_table.dtype == torch.int32
        and block_table.stride(1) == 1
        and block_table.shape[0] >= nreq
    )
    assert (
        seq_lens.dtype == torch.int32
        and seq_lens.is_contiguous()
        and seq_lens.shape[0] >= nreq
    )
    assert (
        out.dtype == torch.bfloat16
        and out.is_contiguous()
        and part.numel() * part.element_size() >= part_bytes()
    )
    return [
        q.data_ptr(),
        k_ptr,
        bstride,
        block_table.data_ptr(),
        seq_lens.data_ptr(),
        out.data_ptr(),
        part.data_ptr(),
        block_table.stride(0),
        nreq,
        vllm_scale_word(scale),
        v_ptr,
        k_scale.data_ptr() if fp8 else 0,
        v_scale.data_ptr() if fp8 else 0,
        0,
    ]
