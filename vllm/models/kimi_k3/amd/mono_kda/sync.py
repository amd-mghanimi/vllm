# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tickets and counters between the launch's workgroups.

Control-word updates branch on the wave-uniform ``wave == 0``, never on
``tid == 0``: LLVM tail-merges identical divergent blocks across the barriers
that follow them, which deadlocks the workgroup. Only lane 0 issues an update,
so each call site passes its own LDS slot and no two of those blocks are
identical.

Data a counter guards is written through to memory (``st_wt``) and the
counter is bumped after the stores drain; a waiter then only drops its CU's L1,
since nothing in the launch read those lines before.
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import communication_ops_utils as comm_ops
from aiter.ops.flydsl.kernels.mxfp4_gemm_common import lds_typed_ptr
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import gpu, rocdl
from flydsl.expr.typing import T


def lds_i32(base, off_words):
    return lds_typed_ptr(base, T.i32, byte_offset=fx.Int32(off_words * 4))


def lds_f32(base, off_words):
    return lds_typed_ptr(base, T.f32, byte_offset=fx.Int32(off_words * 4))


def st_wt(base_i64, index, val, nbytes):
    """Device-scope store: writes through the XCD's L2."""
    _llvm.StoreOp(
        val.ir_value(),
        comm_ops._ptr_plus(base_i64, index, nbytes),
        alignment=nbytes,
        ordering=_llvm.AtomicOrdering.monotonic,
        syncscope=fx.rocdl.SyncScope.AgentOneAs,
    )


def l1_invalidate():
    """Drop this CU's L1 (lines another XCD wrote through to memory)."""
    _llvm.InlineAsmOp(None, [], "buffer_inv sc0", "", has_side_effects=True)


@comm_ops.traced
def _fetch_add(lane, addr, slot):
    if lane == fx.Int32(0):
        slot[0] = fx.Int32(comm_ops.atomic_add_agent(addr, fx.Int32(1)))


@comm_ops.traced
def grab(wave, lane, lds_base, a_ticket, slot_word):
    """Next ticket for the whole workgroup."""
    slot = lds_i32(lds_base, slot_word)
    gpu.barrier()
    if wave == fx.Int32(0):
        _fetch_add(lane, a_ticket, slot)
    gpu.barrier()
    return rocdl.readfirstlane(T.i32, fx.Int32(slot[0]))


@comm_ops.traced
def bump(wave, lane, addr):
    """Add one to a control word once this workgroup's stores have drained."""
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    # traced conditions: nested ifs, as `and` would short-circuit on the host
    if wave == fx.Int32(0):  # noqa: SIM102
        if lane == fx.Int32(0):
            comm_ops.atomic_add_agent(addr, fx.Int32(1))


@comm_ops.traced
def wait_ge(wave, addr, val, sleep):
    """Spin until a control word reaches val, then drop this CU's L1."""
    if wave == fx.Int32(0):
        cur = fx.Int32(comm_ops.load_i32_global_agent(addr))
        while cur < fx.Int32(val):
            rocdl.s_sleep(sleep)
            cur = fx.Int32(comm_ops.load_i32_global_agent(addr))
        rocdl.s_waitcnt(vmcnt=0)
        l1_invalidate()
    gpu.barrier()
