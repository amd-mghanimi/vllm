# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import cast

import torch

from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_reduce,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.models.common.amd.ops.fused_allreduce_rms_norm import (
    fused_allreduce_rms_norm_out,
)

logger = init_logger(__name__)

# Decode token counts where the latent-AG mailbox replaces the second
# all-reduce (c4, c16). Prefill and c64 stay on tier 2.
_LATENT_AG_MAX_TOKENS = 16


class ROCmLatentMoERunner(MoERunner):
    """MoE runner for latent MoE with a replicated routed up-projection.

    Mirrors CUDA's LatentMoERunner. Tier 2 is the column-parallel up-projection
    plus a full all-reduce. At ``M <= 16`` the second all-reduce is an
    all-gather of the 896-column shards via AITER's latent-AG mailbox.

    Native path: the replicated up-proj produces the full hidden dim on every
    rank, so the base runner combines routed + shared correctly at any TP size.

    The latent all-reduce is fused with the following RMSNorm via AITER's
    one-stage custom AR when that kernel's gate admits the tensor; see
    ``vllm.models.common.amd.ops.fused_allreduce_rms_norm``. Everything
    else keeps the unfused all-reduce.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        transform = self.routed_output_transform
        up_proj = getattr(transform, "up_proj", None)
        tp_size = get_tensor_model_parallel_world_size()

        self._up_proj_shard_size = 0
        self._tail_shardable = (
            up_proj is not None
            and tp_size > 1
            and up_proj.weight.shape[0] % tp_size == 0
            and self._shared_experts is not None
            and not self.moe_config.is_sequence_parallel
            and self.routed_scaling_factor == 1.0
        )
        if self._tail_shardable:
            assert up_proj is not None
            self._up_proj_shard_size = up_proj.weight.shape[0] // tp_size
        else:
            logger.warning_once(
                "K3 latent-MoE tail is not shardable under this config, "
                "falling back to the replicated up-projection.",
                scope="global",
            )
        self._logged_sharded_tail = False
        self._logged_mailbox_tail = False
        self._latent_ag_mailbox = None

    def _latent_ag(self):
        """One IPC mailbox per runner, allocated before graph capture."""
        if self._latent_ag_mailbox is not None:
            return self._latent_ag_mailbox
        from aiter.ops.latent_ag_mailbox import LatentAgMailbox

        tp = get_tp_group()
        self._latent_ag_mailbox = LatentAgMailbox(
            rank=get_tensor_model_parallel_rank(),
            world_size=get_tensor_model_parallel_world_size(),
            device=torch.device("cuda", torch.cuda.current_device()),
            max_m=_LATENT_AG_MAX_TOKENS,
            shard_n=self._up_proj_shard_size,
            group=tp.cpu_group,
        )
        logger.info_once(
            "Kimi-K3 latent-MoE tail: mailbox all-gather for M<=%d "
            "(%d bytes/rank).",
            _LATENT_AG_MAX_TOKENS,
            self._latent_ag_mailbox.mailbox_bytes,
            scope="global",
        )
        return self._latent_ag_mailbox

    def _shard_up_proj_tail(
        self,
        fused_output: torch.Tensor,
        shared_output: torch.Tensor,
        trunc_size: int | None,
    ) -> torch.Tensor:
        """Tier 2: column-parallel up-projection folded into the final reduce."""
        if not self._logged_sharded_tail:
            self._logged_sharded_tail = True
            logger.info_once(
                "Kimi-K3 latent-MoE tail: up-projecting only this rank's "
                "hidden shard into the shared output.",
                scope="global",
            )

        transform = self.routed_output_transform
        assert transform is not None

        if transform.norm is not None:
            latent = fused_allreduce_rms_norm_out(fused_output, transform.norm)
        else:
            latent = tensor_model_parallel_all_reduce(fused_output)

        shard_size = self._up_proj_shard_size
        shard_start = get_tensor_model_parallel_rank() * shard_size
        up_proj_shard = transform.up_proj.weight.narrow(0, shard_start, shard_size)
        hidden_shard = shared_output.narrow(-1, shard_start, shard_size)

        # hidden_shard += latent @ up_proj_shard.T, accumulated in the GEMM's
        # beta-add epilogue so folding in the shared partial costs no kernel.
        hidden_shard.addmm_(latent, up_proj_shard.t())

        num_tokens = fused_output.shape[0]
        if 0 < num_tokens <= _LATENT_AG_MAX_TOKENS:
            if not self._logged_mailbox_tail:
                self._logged_mailbox_tail = True
                logger.info_once(
                    "Kimi-K3 latent-MoE tail: all-gathering the up-proj shard "
                    "via the latent-AG mailbox (M<=%d).",
                    _LATENT_AG_MAX_TOKENS,
                    scope="global",
                )
            out = self._latent_ag().allgather(hidden_shard.contiguous())
            # Already the full hidden. Strip padding here; the late all-reduce
            # would slice the wrong axis of a fresh [M, hidden] tensor only if
            # trunc_size were a routed dim, which it is not for this tail.
            if trunc_size is not None:
                out = out[..., :trunc_size]
            return out

        return self._maybe_reduce_final_output(
            shared_output, trunc_size, output_is_reduced=False
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._tail_shardable and not self._fused_output_is_reduced:
            return self._fused_forward(
                hidden_states, router_logits, input_ids, shared_experts_input
            )
        return super().forward(
            hidden_states, router_logits, input_ids, shared_experts_input
        )

    def _fused_forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        # When the caller pre-applies the routed input transform outside the
        # runner (e.g. to overlap it on a separate stream), it passes the
        # already-transformed routed input as ``hidden_states`` and the original
        # hidden states as ``shared_experts_input``; skip the transform then.
        if shared_experts_input is None:
            hidden_states, shared_experts_input = self.apply_routed_input_transform(
                hidden_states
            )

        hidden_states, og_hidden_dim_pre_xform, og_hidden_dim_post_xform = (
            self._maybe_pad_hidden_states(
                shared_experts_input,
                hidden_states,
            )
        )

        result = self._forward_entry(
            hidden_states,
            router_logits,
            shared_experts_input,
            input_ids,
            self._encode_layer_name(),
            self.moe_config.hidden_dim_unpadded
            if self._quant_method.has_unpadded_output
            else 0,
        )

        shared_output, fused_output = cast(tuple[torch.Tensor, torch.Tensor], result)

        if og_hidden_dim_pre_xform is not None:
            fused_output = fused_output[..., :og_hidden_dim_pre_xform]

        result = self._shard_up_proj_tail(
            fused_output, shared_output, og_hidden_dim_post_xform
        )

        return self._maybe_add_zero_expert_output(result)
