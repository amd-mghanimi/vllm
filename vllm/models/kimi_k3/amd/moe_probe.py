# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 MoE fault probe (debug, off unless VLLM_ROCM_K3_PROBE is set).

CUDA-graph-safe checks of every small MoE call's inputs and expert ids, written to a
host-mapped file that survives a GPU memory fault. Read with read_ring().
"""

import os
import time

import torch

from vllm.distributed import get_tensor_model_parallel_rank
from vllm.logger import init_logger

logger = init_logger(__name__)


class GraphProbe:
    """Debug: non-finite inputs and out-of-range expert ids per MoE call,
    CUDA-graph safe.

    Enabled by VLLM_ROCM_K3_PROBE=<dir>. Every call of at most MAX_M tokens runs a
    few device ops (no host sync): it counts rows of router logits / routed input
    with a NaN or inf (first and last bad row: padded graph rows sit at the tail)
    and, after the mono MoE launch, rows with an expert id outside [0, E). A call with
    any of these appends an event to a device ring and snapshots its inputs into
    a per-layer buffer. A host thread drains the ring every second, logs new
    events and saves the snapshots to <dir>/rank<r>/, so evidence survives a
    GPU fault.
    """

    RING = 4096
    MAX_M = 32
    EV = (
        "seq",
        "layer",
        "m",
        "bad_logit_rows",
        "first_bad",
        "last_bad",
        "bad_x_rows",
        "bad_id_rows",
    )

    def __init__(self, root: str):
        self.root = root
        self.ready = False

    def _mapped(self, path: str, n: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        """int64[n] in a MAP_SHARED file registered with the HIP runtime, as a host
        tensor and a device view. Device writes reach the file's pages when the
        writing kernel ends, and the kernel flushes them even if a GPU fault kills
        the process (layout: calls, events, then the ring, see read_ring() below)."""
        import mmap

        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o666)
        os.ftruncate(fd, n * 8)
        self._mm = mmap.mmap(
            fd, n * 8, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE
        )
        host = torch.frombuffer(self._mm, dtype=torch.int64)

        class _Iface:
            __cuda_array_interface__ = {
                "shape": (n,),
                "typestr": "<i8",
                "data": (host.data_ptr(), False),
                "version": 3,
            }

        # as_tensor silently copies (and the probe sees nothing) unless the
        # pointer resolves to `device`, so register and wrap with it current.
        with torch.cuda.device(device):
            rc = torch.cuda.cudart().cudaHostRegister(host.data_ptr(), n * 8, 3)
            assert int(rc) == 0, f"K3 MoE probe: hostRegister failed ({rc})"
            dev = torch.as_tensor(_Iface(), device=device)
        assert dev.data_ptr() == host.data_ptr(), "K3 MoE probe: device view is a copy"
        return host, dev

    def _alloc(self, device, ne: int, hidden: int) -> None:
        z = dict(device=device)
        self.dir = os.path.join(self.root, f"rank{get_tensor_model_parallel_rank()}")
        os.makedirs(self.dir, exist_ok=True)
        nf = len(self.EV)
        host, dev = self._mapped(
            os.path.join(self.dir, "ring.bin"), 2 + self.RING * nf, device
        )
        self.h_calls, self.calls = host[0:1], dev[0:1]
        self.h_n_ev, self.n_ev = host[1:2], dev[1:2]
        self.h_ring, self.ring = (
            host[2:].view(self.RING, nf),
            dev[2:].view(self.RING, nf),
        )
        self.snap_lg = torch.zeros(128, self.MAX_M, ne, **z)
        self.snap_x = torch.zeros(128, self.MAX_M, hidden, dtype=torch.bfloat16, **z)
        self.snap_meta = torch.full((128, 2), -1, dtype=torch.int64, **z)
        self.rows = torch.arange(self.MAX_M, device=device)
        self.ready = True
        __import__("threading").Thread(target=self._drain, daemon=True).start()
        logger.info("K3 MoE probe: events -> %s", self.dir)

    def _rows_span(self, bad: torch.Tensor, m: int):
        r = self.rows[:m]
        first = torch.where(bad, r, m).min()
        last = torch.where(bad, r, -1).max()
        return first, last

    def check(
        self,
        layer: int | None,
        logits: torch.Tensor,
        x: torch.Tensor,
        ids: torch.Tensor | None = None,
        ne: int = 0,
    ) -> None:
        """Inputs (ids None, before the MoE) or the chain's expert ids (after it)."""
        m = logits.shape[0]
        if layer is None or m > self.MAX_M:
            return
        if not self.ready:
            if torch.cuda.is_current_stream_capturing():
                return
            self._alloc(logits.device, logits.shape[1], x.shape[1])
        if ids is None:
            bad_lg = ~logits.isfinite().all(1)
            bad_x = ~x.isfinite().all(1)
            bad_id = torch.zeros_like(bad_lg)
        else:
            bad_id = ((ids < 0) | (ids >= ne)).any(1)
            bad_lg = bad_x = torch.zeros_like(bad_id)
        first, last = self._rows_span(bad_lg | bad_x | bad_id, m)
        any_bad = (bad_lg | bad_x | bad_id).any()
        seq = self.calls.clone()
        ev = torch.stack(
            [
                seq[0],
                torch.full_like(seq[0], layer),
                torch.full_like(seq[0], m),
                bad_lg.sum(),
                first,
                last,
                bad_x.sum(),
                bad_id.sum(),
            ]
        )
        slot = self.n_ev % self.RING
        self.ring.index_copy_(
            0, slot, torch.where(any_bad, ev, self.ring[slot][0]).unsqueeze(0)
        )
        self.n_ev.add_(any_bad.long())
        if ids is None:
            self.calls.add_(1)
        if ids is None and layer < self.snap_meta.shape[0]:
            self.snap_lg[layer, :m].copy_(
                torch.where(any_bad, logits, self.snap_lg[layer, :m])
            )
            self.snap_x[layer, :m].copy_(
                torch.where(any_bad, x, self.snap_x[layer, :m])
            )
            meta = torch.stack([seq[0], torch.full_like(seq[0], m)])
            self.snap_meta[layer].copy_(
                torch.where(any_bad, meta, self.snap_meta[layer])
            )

    def _drain(self) -> None:
        stream = torch.cuda.Stream(device=self.calls.device)
        done, beat = 0, 0
        while True:
            time.sleep(1.0)
            n, calls = int(self.h_n_ev[0]), int(self.h_calls[0])
            beat += 1
            if beat % 60 == 0:
                logger.info("K3 MoE probe: %d calls, %d events", calls, n)
            if n == done:
                continue
            ring = self.h_ring.clone()
            new = [
                ring[i % self.RING].tolist() for i in range(max(done, n - self.RING), n)
            ]
            for ev in new[:32]:
                logger.warning(
                    "K3 MoE probe event: %s (calls so far %d)",
                    dict(zip(self.EV, ev)),
                    calls,
                )
            out = {"events": new, "fields": self.EV, "calls": calls}
            torch.save(out, os.path.join(self.dir, f"events_{n:08d}.pt"))
            with torch.cuda.stream(stream):
                out.update(
                    snap_logits=self.snap_lg.cpu(),
                    snap_x=self.snap_x.cpu(),
                    snap_meta=self.snap_meta.cpu(),
                )
            torch.save(out, os.path.join(self.dir, f"events_{n:08d}.pt"))
            done = n


PROBE = (
    GraphProbe(os.environ["VLLM_ROCM_K3_PROBE"])
    if os.environ.get("VLLM_ROCM_K3_PROBE")
    else None
)


def read_ring(path: str) -> tuple[int, int, list[dict]]:
    """(calls, events, the last events) of a probe's <dir>/rank<r>/ring.bin."""
    import numpy as np

    a = np.fromfile(path, dtype=np.int64)
    calls, n = int(a[0]), int(a[1])
    nf = len(GraphProbe.EV)
    ring = a[2 : 2 + GraphProbe.RING * nf].reshape(GraphProbe.RING, nf)
    return (
        calls,
        n,
        [
            dict(zip(GraphProbe.EV, ring[i % GraphProbe.RING].tolist()))
            for i in range(max(0, n - GraphProbe.RING), n)
        ],
    )
