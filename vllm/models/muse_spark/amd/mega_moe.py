# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AITER MegaMoE (fused dispatch, MXFP4 GEMMs, post-route norm, combine) for
Muse Spark's token-sharded prefill under TP + EP.

Each rank feeds its own rows of the token-sharded residual stream and gets
back its own rows of `sum_k w_k * rms_norm(gain * y_k)`, so the MoE needs no
all-gather of the pooled rows or router logits and no reduce-scatter after.
"""

import os
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.distributed import get_ep_group
from vllm.logger import init_logger
from vllm.utils.math_utils import next_power_of_2

logger = init_logger(__name__)


def mega_moe_enabled() -> bool:
    return os.environ.get("VLLM_ROCM_MUSE_MEGA_MOE", "0") == "1"


def mega_moe_min_tokens() -> int:
    return int(os.environ.get("VLLM_ROCM_MUSE_MEGA_MOE_MIN_TOKENS", "4096"))


def shuffle_mega_weights(layer: torch.nn.Module) -> tuple[torch.Tensor, ...]:
    """This rank's experts, as loaded (`[E, 2I, H/2]` packed MXFP4 + e8m0
    scales), in MegaMoEV2's layout. Copies: the loaded tensors are then
    converted in place for the AITER experts that serve the other batches."""
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
    return (
        shuffle_weight_a16w4(w13.view(fp4).clone(), 16, True).contiguous(),
        shuffle_scale_a16w4(
            s13.reshape(-1, hidden // 32).clone(), n, True
        ).contiguous(),
        shuffle_weight_a16w4(w2.view(fp4).clone(), 16, False).contiguous(),
        shuffle_scale_a16w4(s2.reshape(-1, inter // 32).clone(), n, False).contiguous(),
    )


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
    """One layer's MegaMoEV2 weights and norm gain over a runtime shared by
    every layer (its symmetric-heap buffers are large)."""

    _runtimes: dict[tuple, Any] = {}

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

    def _runtime(self) -> Any:
        key = (self.max_tok_per_rank, self.hidden_size, self.intermediate_size)
        rt = self._runtimes.get(key)
        if rt is None:
            _init_shmem()
            from aiter.ops.flydsl.kernels.mega_moe import MegaMoEV2

            w13, s13, w2, s2 = self.weights
            rt = MegaMoEV2(
                rank=self.ep_rank,
                world_size=self.ep_size,
                model_dim=self.hidden_size,
                inter_dim=self.intermediate_size,
                experts=self.num_experts,
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
                "Muse Spark MegaMoE: MegaMoEV2 mtpr=%d", self.max_tok_per_rank
            )
            self._runtimes[key] = rt
        return rt

    def __call__(
        self, x: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor
    ) -> torch.Tensor:
        """This rank's MoE output rows: a view into the shared runtime's
        buffer, valid until the next layer's call."""
        rt = self._runtime()
        rt.bind_weights(*self.weights)
        rt.post_route_norm_gain = self.gain
        return rt.forward(
            x.contiguous(),
            topk_weights.to(torch.float32).contiguous(),
            topk_ids.to(torch.int32).contiguous(),
        )
