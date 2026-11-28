# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AITER MegaMoE (fused dispatch, MXFP4 GEMMs, post-route norm, combine) for
Muse Spark's token-sharded prefill under TP + EP.

Each rank feeds its own rows of the token-sharded residual stream and gets
back its own rows of `sum_k w_k * rms_norm(gain * y_k)`, so the MoE needs no
all-gather of the pooled rows or router logits and no reduce-scatter after.

Optional expert replicas (VLLM_ROCM_MUSE_MEGA_MOE_SLOTS=B, 1|2|4|8): each rank's
weight window gets B replica slots (AITER `ReplicaWeightPool`). Batches of at
least VLLM_ROCM_MUSE_MEGA_MOE_BALANCE_MIN_TOKENS tokens run a second MegaMoEV2
runtime over the windows; the planner (`ReplicaPlanner`) splits the hot experts'
routes between their owner and the slots holding them, and after each call a
side stream picks the next forward's slots from that call's histogram. Slot pulls
are issued from the MoE layers only, two replica layers ahead of the layer that
needs them. Smaller batches use the plain runtime on the home rows of the same
windows, so they are unaffected by the replicas.
"""

import os
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.distributed import get_ep_group
from vllm.logger import init_logger
from vllm.utils.math_utils import next_power_of_2

logger = init_logger(__name__)

# A replica layer's slot pull is issued when the replica layer this many
# positions earlier in the call order runs.
_LOOKAHEAD = 2
_PLANNER_GATE_PCT = 110
_PLANNER_SELECT_HYST = 0.0


def mega_moe_enabled() -> bool:
    return os.environ.get("VLLM_ROCM_MUSE_MEGA_MOE", "0") == "1"


def mega_moe_min_tokens() -> int:
    return int(os.environ.get("VLLM_ROCM_MUSE_MEGA_MOE_MIN_TOKENS", "4096"))


def mega_moe_slots() -> int:
    """Replica slots per rank: 0 (off), 1, 2, 4 or 8."""
    slots = int(os.environ.get("VLLM_ROCM_MUSE_MEGA_MOE_SLOTS", "0"))
    if slots not in (0, 1, 2, 4, 8):
        raise ValueError(
            f"VLLM_ROCM_MUSE_MEGA_MOE_SLOTS must be 0, 1, 2, 4 or 8, got {slots}"
        )
    return slots


def mega_moe_balance_min_tokens() -> int:
    return int(os.environ.get("VLLM_ROCM_MUSE_MEGA_MOE_BALANCE_MIN_TOKENS", "8192"))


def mega_moe_refresh_pct() -> int:
    """A replica slot is kept until it carries more than this percent of mean load."""
    return int(os.environ.get("VLLM_ROCM_MUSE_MEGA_MOE_REFRESH", "125"))


def shuffle_mega_weights(
    layer: torch.nn.Module,
) -> tuple[tuple[torch.Tensor, ...], Any]:
    """This rank's experts, as loaded (`[E, 2I, H/2]` packed MXFP4 + e8m0
    scales), in MegaMoEV2's layout. Returns (weights, pool): without replicas
    the weights are the home experts and the pool is None; with replicas the
    pool owns the windows (home rows first, then the slots) and the weights are
    views of its home rows, so the plain runtime needs no second copy."""
    from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

    fp4 = torch.float4_e2m1fn_x2
    w13, s13 = layer.w13_weight.data, layer.w13_weight_scale.data
    w2, s2 = layer.w2_weight.data, layer.w2_weight_scale.data
    n, two_i, half_h = w13.shape
    hidden, inter = 2 * half_h, two_i // 2
    if w2.shape != (n, hidden, inter // 2) or s13.shape != (n, two_i, hidden // 32):
        raise ValueError(
            f"unexpected MXFP4 expert shapes w13 {tuple(w13.shape)} "
            f"s13 {tuple(s13.shape)} w2 {tuple(w2.shape)} s2 {tuple(s2.shape)}"
        )
    parts = (
        shuffle_weight_a16w4(w13.view(fp4).clone(), 16, True).contiguous(),
        shuffle_scale_a16w4(
            s13.reshape(-1, hidden // 32).clone(), n, True
        ).contiguous(),
        shuffle_weight_a16w4(w2.view(fp4).clone(), 16, False).contiguous(),
        shuffle_scale_a16w4(s2.reshape(-1, inter // 32).clone(), n, False).contiguous(),
    )
    slots = mega_moe_slots()
    if not slots:
        return parts, None
    from aiter.ops.flydsl.kernels.mega_moe.replica_pool import ReplicaWeightPool

    ep = get_ep_group()
    pool = ReplicaWeightPool(
        parts,
        rank=ep.rank_in_group,
        world_size=ep.world_size,
        experts_per_rank=n,
        slots=slots,
        group=ep.cpu_group,
    )
    plain = tuple(win[: part.shape[0]] for win, part in zip(pool.windows, parts))
    logger.info_once(
        "Muse Spark MegaMoE: %d replica slots per rank (%.2f GiB per layer pool)",
        slots,
        pool.buf.numel() / 2**30,
    )
    return plain, pool


_SHMEM_READY = False


def _init_shmem() -> None:
    global _SHMEM_READY
    if _SHMEM_READY:
        return
    os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "8G")
    import mori

    torch._C._distributed_c10d._register_process_group(
        "muse_mega_moe", get_ep_group().cpu_group
    )
    mori.shmem.shmem_torch_process_group_init("muse_mega_moe")
    _SHMEM_READY = True


class MuseSparkMegaMoE:
    """One layer's MegaMoEV2 weights and norm gain over shared runtimes (one per
    replica count; their symmetric-heap buffers are large)."""

    _runtimes: dict[tuple, Any] = {}
    _planners: dict[tuple, Any] = {}
    _side: torch.cuda.Stream | None = None
    _layers: list["MuseSparkMegaMoE"] = []  # replica layers, in call order
    _fills = 0
    _pulls = 0

    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        weights: tuple[torch.Tensor, ...],
        norm_weight: torch.Tensor,
        norm_gain_center: float,
        norm_eps: float,
        pool: Any = None,
    ) -> None:
        ep = get_ep_group()
        self.ep_rank, self.ep_size = ep.rank_in_group, ep.world_size
        self.num_experts, self.top_k = num_experts, top_k
        self.hidden_size, self.intermediate_size = hidden_size, intermediate_size
        self.weights = weights
        self.gain = (norm_weight.float() + norm_gain_center).contiguous()
        self.eps = float(norm_eps)
        tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.max_tok_per_rank = next_power_of_2(-(-tokens // self.ep_size))
        self.pool = pool
        self.slots = 0 if pool is None else pool.slots
        self.balance_min_tokens = mega_moe_balance_min_tokens()
        if pool is not None:
            dev = norm_weight.device
            self.refresh_pct = mega_moe_refresh_pct()
            held = lambda: torch.full(  # noqa: E731
                (self.ep_size, self.slots), -1, dtype=torch.int32, device=dev
            )
            self.held, self.placed = held(), held()
            # Expert totals and refresh flag of the last plan (ReplicaPlanner.place).
            self.ecount = torch.zeros(num_experts + 1, dtype=torch.int32, device=dev)
            self.placed_host = torch.empty(
                (self.ep_size, self.slots), dtype=torch.int32, pin_memory=True
            )
            self.selected = torch.cuda.Event()
            self.ready: torch.cuda.Event | None = None
            self.fill_due = False
            self.index: int | None = None

    def use_replicas(self, tokens: int) -> bool:
        """Whether a batch of `tokens` tokens takes the replica runtime. The
        choice depends only on the batch size, so every EP rank makes the same one."""
        return self.pool is not None and tokens >= self.balance_min_tokens

    def _planner(self) -> Any:
        key = (self.slots, self.max_tok_per_rank)
        planner = self._planners.get(key)
        if planner is None:
            from aiter.ops.flydsl.kernels.mega_moe.replica_plan import ReplicaPlanner

            planner = ReplicaPlanner(
                rank=self.ep_rank,
                world_size=self.ep_size,
                experts=self.num_experts,
                slots=self.slots,
                max_routes=self.max_tok_per_rank * self.top_k,
                device=self.ecount.device,
                gate_pct=_PLANNER_GATE_PCT,
                select_hyst=_PLANNER_SELECT_HYST,
                refresh_pct=self.refresh_pct,
            )
            self._planners[key] = planner
        if MuseSparkMegaMoE._side is None:
            MuseSparkMegaMoE._side = torch.cuda.Stream(device=self.ecount.device)
        return planner

    def _runtime(self, slots: int, weights: tuple[torch.Tensor, ...]) -> Any:
        key = (self.max_tok_per_rank, self.hidden_size, self.intermediate_size, slots)
        rt = self._runtimes.get(key)
        if rt is None:
            _init_shmem()
            from aiter.ops.flydsl.kernels.mega_moe import MegaMoEV2

            w13, s13, w2, s2 = weights
            rt = MegaMoEV2(
                rank=self.ep_rank,
                world_size=self.ep_size,
                model_dim=self.hidden_size,
                inter_dim=self.intermediate_size,
                experts=self.num_experts + self.ep_size * slots,
                topk=self.top_k,
                quant="a4w4",
                w1=w13,
                w1_scale=s13,
                w2=w2,
                w2_scale=s2,
                max_tok_per_rank=self.max_tok_per_rank,
                stage2_p2p_quant=os.environ.get(
                    "VLLM_ROCM_MUSE_MEGA_MOE_P2P_QUANT", "int8_blockwise_1x32"
                ),
                swiglu_limit=0.0,
                post_route_norm=(self.gain, self.eps),
            )
            logger.info_once(
                "Muse Spark MegaMoE: MegaMoEV2 mtpr=%d slots=%d",
                self.max_tok_per_rank,
                slots,
            )
            self._runtimes[key] = rt
        return rt

    def __call__(
        self,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        replica: bool = False,
    ) -> torch.Tensor:
        """This rank's MoE output rows: a view into the shared runtime's
        buffer, valid until the next layer's call."""
        x = x.contiguous()
        topk_weights = topk_weights.to(torch.float32).contiguous()
        topk_ids = topk_ids.to(torch.int32).contiguous()
        if not replica:
            rt = self._runtime(0, self.weights)
            rt.bind_weights(*self.weights)
            rt.post_route_norm_gain = self.gain
            return rt.forward(x, topk_weights, topk_ids)
        return self._forward_replica(x, topk_weights, topk_ids)

    def _forward_replica(
        self, x: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor
    ) -> torch.Tensor:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Muse Spark MegaMoE replica slots are not CUDA-graph capturable"
            )
        windows = self.pool.windows
        rt = self._runtime(self.slots, windows)
        rt.bind_weights(*windows)
        rt.post_route_norm_gain = self.gain
        planner = self._planner()
        main = torch.cuda.current_stream()
        self._enter()
        if self.ready is not None:
            main.wait_event(self.ready)  # this layer's slot pull
        vids = planner.plan(topk_ids, self.held, self.ecount)
        out = rt.forward(x, topk_weights, vids)
        # After this call's last read of the slots: the next forward's slots.
        side = self._side
        side.wait_stream(main)
        with torch.cuda.stream(side):
            planner.select(self.held, self.placed, self.ecount)
            self.placed_host.copy_(self.placed, non_blocking=True)
            self.selected.record(side)
        self.fill_due = True
        ahead = self._layers[(self.index + _LOOKAHEAD) % len(self._layers)]
        if ahead is not self and ahead.fill_due:
            ahead._issue_fill()
        return out

    def _enter(self) -> None:
        if self.index is None:
            self.index = len(self._layers)
            self._layers.append(self)
        if self.fill_due:  # not issued by an earlier layer
            self._issue_fill()

    def _issue_fill(self) -> int:
        """Queue this layer's slot pull on the side stream (after its last select,
        hence after its last read of the slots); returns the experts pulled."""
        self.selected.synchronize()  # placed_host; normally long done
        cls = MuseSparkMegaMoE
        row = self.placed_host[self.ep_rank].tolist()
        pulls = sum(e >= 0 for e in row)
        self.pool.fill_dma(row, cls._side)
        cls._fills += 1
        cls._pulls += pulls
        if cls._fills % (4 * len(cls._layers)) == 0:
            logger.info(
                "Muse Spark MegaMoE balance: rank %d pulled %.3f experts per layer "
                "call over %d calls",
                self.ep_rank,
                cls._pulls / cls._fills,
                cls._fills,
            )
        if self.ready is None:
            self.ready = torch.cuda.Event()
        self.ready.record(cls._side)
        self.fill_due = False
        return pulls
