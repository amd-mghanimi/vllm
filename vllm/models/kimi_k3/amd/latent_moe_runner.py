# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import time
from functools import cached_property
from typing import cast

import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SituAndMul
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    FusedTopKBiasRouter,
)
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
    GroupedTopKRouter,
)
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExpertsOrder,
)

logger = init_logger(__name__)

# AITER's routed_chain takes M <= 32; past 16 tokens it loses more to the
# four-launch path than it saves.
_ROUTED_CHAIN_MAX_TOKENS = 16


def _tensor_blob(t: torch.Tensor) -> dict:
    """A tensor as raw bytes plus dtype, so fp4/e8m0 weights survive torch.save."""
    t = t.detach()
    return {
        "bytes": t.contiguous().reshape(-1).view(torch.uint8).cpu(),
        "dtype": str(t.dtype).removeprefix("torch."),
        "shape": tuple(t.shape),
        "contiguous": t.is_contiguous(),
        "is_shuffled": getattr(t, "is_shuffled", None),
    }


class _MoERecorder:
    """Debug: record the stock MoE path's inputs and outputs for offline replay.

    Enabled by VLLM_ROCM_K3_RECORD=<dir> (eager mode only: copies to host).
    Forces the stock path. For TP ranks in VLLM_ROCM_K3_RECORD_RANKS (default 0)
    and layers in VLLM_ROCM_K3_RECORD_LAYERS (default 1,30,60) it writes the
    layer's processed weights once, then every VLLM_ROCM_K3_RECORD_EVERY-th
    call with at most VLLM_ROCM_K3_RECORD_MAX_TOKENS tokens, up to
    VLLM_ROCM_K3_RECORD_MAX calls per layer, in chunks of 64 calls.
    """

    CHUNK = 64

    def __init__(self, root: str):
        env = os.environ.get
        self.root = root
        self.layers = {int(v) for v in env("VLLM_ROCM_K3_RECORD_LAYERS", "1,30,60").split(",")}
        self.ranks = {int(v) for v in env("VLLM_ROCM_K3_RECORD_RANKS", "0").split(",")}
        self.every = int(env("VLLM_ROCM_K3_RECORD_EVERY", "1"))
        self.max_calls = int(env("VLLM_ROCM_K3_RECORD_MAX", "2048"))
        self.max_tokens = int(env("VLLM_ROCM_K3_RECORD_MAX_TOKENS", "32"))
        self.seen: dict[int, int] = {}
        self.saved: dict[int, int] = {}
        self.buf: dict[int, list] = {}

    def layer_dir(self, layer: int) -> str:
        rank = get_tensor_model_parallel_rank()
        path = os.path.join(self.root, f"rank{rank}", f"layer{layer:02d}")
        os.makedirs(path, exist_ok=True)
        return path

    def wants(self, layer: int | None, m: int) -> bool:
        if (
            layer is None
            or layer not in self.layers
            or get_tensor_model_parallel_rank() not in self.ranks
            or m > self.max_tokens
            or self.saved.get(layer, 0) >= self.max_calls
        ):
            return False
        n = self.seen.get(layer, 0)
        self.seen[layer] = n + 1
        return n % self.every == 0

    def save_weights(self, layer: int, weights: dict) -> None:
        path = os.path.join(self.layer_dir(layer), "weights.pt")
        if not os.path.exists(path):
            torch.save(
                {k: _tensor_blob(v) if isinstance(v, torch.Tensor) else v
                 for k, v in weights.items()},
                path,
            )
            logger.info("K3 MoE record: weights of layer %d -> %s", layer, path)

    def add(self, layer: int, rec: dict) -> None:
        rec = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
               for k, v in rec.items()}
        buf = self.buf.setdefault(layer, [])
        buf.append(rec)
        self.saved[layer] = self.saved.get(layer, 0) + 1
        if len(buf) == self.CHUNK or self.saved[layer] == self.max_calls:
            n = (self.saved[layer] - 1) // self.CHUNK
            torch.save(buf, os.path.join(self.layer_dir(layer), f"calls_{n:05d}.pt"))
            buf.clear()


_RECORDER = (
    _MoERecorder(os.environ["VLLM_ROCM_K3_RECORD"])
    if os.environ.get("VLLM_ROCM_K3_RECORD")
    else None
)


class ROCmLatentMoERunner(MoERunner):
    """MoE runner for latent MoE with a replicated routed up-projection.

    Mirrors CUDA's LatentMoERunner, but currently only the up projection
    -sharded path is implemented. (Tier 2)

    Native path: the replicated up-proj produces the full hidden dim on every
    rank, so the base runner combines routed + shared correctly at any TP size.
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

        latent = tensor_model_parallel_all_reduce(fused_output)
        if transform.norm is not None:
            latent = transform.norm(latent)

        shard_size = self._up_proj_shard_size
        shard_start = get_tensor_model_parallel_rank() * shard_size
        up_proj_shard = transform.up_proj.weight.narrow(0, shard_start, shard_size)
        hidden_shard = shared_output.narrow(-1, shard_start, shard_size)

        # hidden_shard += latent @ up_proj_shard.T, accumulated in the GEMM's
        # beta-add epilogue so folding in the shared partial costs no kernel.
        hidden_shard.addmm_(latent, up_proj_shard.t())

        return self._maybe_reduce_final_output(
            shared_output, trunc_size, output_is_reduced=False
        )

    @cached_property
    def _routed_chain_layer_ok(self) -> bool:
        """Whether this layer's routing and experts match AITER's routed_chain.

        routed_chain is biased sigmoid top-k, renormalised and unscaled, then the
        a4w4 SiTUv2 MoE on [gate; up] a16w4-shuffled weights, with no EP, bias
        or padding. Read on the first forward, after weights are processed.
        """
        if not (
            rocm_aiter_ops.is_fused_moe_enabled()
            and rocm_aiter_ops.get_fused_moe_situv2_activation() == "a4w4"
        ):
            return False
        try:
            from aiter.ops.flydsl.moe_routed_chain import routed_chain  # noqa: F401
        except ImportError:
            return False
        quant_method = self._quant_method
        router = self.router
        cfg = self.moe_config
        parallel = cfg.moe_parallel_config
        quant_config = quant_method.moe_quant_config
        ok = (
            getattr(quant_method, "is_k3_situ_aiter", False)
            and not quant_method.is_monolithic
            and not quant_method.mk_can_overlap_shared_experts
            and quant_config is not None
            and quant_config.w1_bias is None
            and quant_config.w2_bias is None
            and (
                isinstance(router, FusedTopKBiasRouter)
                or (
                    isinstance(router, GroupedTopKRouter)
                    and router.num_expert_group == 1
                    and router.topk_group == 1
                )
            )
            and router.e_score_correction_bias is not None
            and router.scoring_func == "sigmoid"
            and router.renormalize
            and router.routed_scaling_factor == 1.0
            and router.num_fused_shared_experts == 0
            and getattr(router, "bias_vl", None) is None
            and getattr(router, "_hash_indices_table", None) is None
            and router.eplb_state is None
            and self.routed_experts.expert_map is None
            and not parallel.use_ep
            and not parallel.enable_eplb
            and parallel.dp_size == 1
            and not cfg.is_sequence_parallel
            and cfg.swiglu_limit is None
            and cfg.hidden_dim == cfg.hidden_dim_unpadded
            and cfg.intermediate_size_per_partition
            == cfg.intermediate_size_per_partition_unpadded
            and not cfg.intermediate_pad
        )
        if ok:
            logger.info_once(
                "Kimi-K3 MoE: AITER routed_chain (top-k + a4w4 MoE in one launch) "
                "for small batches.",
                scope="global",
            )
        return ok

    def _use_routed_chain(
        self, hidden_states: torch.Tensor, router_logits: torch.Tensor
    ) -> bool:
        if (
            _RECORDER is not None
            or not self._routed_chain_layer_ok
            or self.router.capture_fn is not None
            or hidden_states.shape[0] > _ROUTED_CHAIN_MAX_TOKENS
        ):
            return False
        if (
            hidden_states.dtype != torch.bfloat16
            or router_logits.dtype != torch.float32
        ):
            return False
        from aiter.ops.flydsl.moe_routed_chain import fused_supported

        w13 = self.routed_experts.w13_weight
        return fused_supported(
            hidden_states.shape[0],
            w13.shape[0],
            self.router.top_k,
            hidden_states.shape[1],
            w13.shape[1] // 2,
        )

    @cached_property
    def _chain_shared_weights(self) -> tuple | None:
        """(gate_up weight, down weight, beta, linear_beta) when routed_chain can run
        this layer's shared expert in the same launch (VLLM_ROCM_K3_CHAIN_SHARED=1).

        The shared expert must be KimiMLP as vLLM builds it for K3: unquantized bf16,
        no bias, SiTU, and an unreduced down projection (the tail reduces it).
        """
        if os.environ.get("VLLM_ROCM_K3_CHAIN_SHARED", "0") != "1":
            return None
        if not self._routed_chain_layer_ok or self.gate is not None:
            return None
        try:
            from aiter.ops.flydsl.moe_routed_chain import shared_supported  # noqa: F401
        except ImportError:
            return None
        weights = self._shared_mlp_weights
        if weights is not None:
            logger.info_once(
                "Kimi-K3 MoE: shared expert runs inside AITER routed_chain.",
                scope="global",
            )
        return weights

    @cached_property
    def _shared_mlp_weights(self) -> tuple | None:
        """(gate_up weight, down weight, beta, linear_beta) of a KimiMLP shared expert."""
        from vllm.model_executor.layers.linear import UnquantizedLinearMethod

        mlp = getattr(self._shared_experts, "_layer", None)
        gate_up = getattr(mlp, "gate_up_proj", None)
        down = getattr(mlp, "down_proj", None)
        act = getattr(mlp, "act_fn", None)
        if not (
            isinstance(act, SituAndMul)
            and gate_up is not None
            and down is not None
            and isinstance(gate_up.quant_method, UnquantizedLinearMethod)
            and isinstance(down.quant_method, UnquantizedLinearMethod)
            and gate_up.bias is None
            and down.bias is None
            and not down.reduce_results
            and gate_up.weight.dtype == down.weight.dtype == torch.bfloat16
        ):
            return None
        return gate_up.weight, down.weight, act.beta, act.linear_beta

    @cached_property
    def _layer_index(self) -> int | None:
        parts = [p for p in str(getattr(self, "layer_name", "")).split(".") if p.isdigit()]
        return int(parts[0]) if parts else None

    def _record_weights(self) -> dict:
        quant_config = self._quant_method.moe_quant_config
        weights = dict(
            layer_name=str(self.layer_name),
            w13=self.routed_experts.w13_weight,
            w2=self.routed_experts.w2_weight,
            w1_scale=quant_config.w1_scale,
            w2_scale=quant_config.w2_scale,
            bias=self.router.e_score_correction_bias.data,
            topk=self.router.top_k,
            situ_beta=self.moe_config.activation_situ_beta,
            situ_linear_beta=self.moe_config.activation_situ_linear_beta,
            chain_layer_ok=self._routed_chain_layer_ok,
        )
        if self._shared_mlp_weights is not None:
            w_gu, w_dn, beta, linear_beta = self._shared_mlp_weights
            weights.update(shared_w_gu=w_gu, shared_w_dn=w_dn, shared_beta=beta,
                           shared_linear_beta=linear_beta)
        return weights

    def _recorded_forward_impl(
        self, hidden_states, router_logits, shared_experts_input, input_ids
    ):
        """The stock path, with its inputs and outputs saved for offline replay."""
        layer = self._layer_index
        assert _RECORDER is not None
        if not _RECORDER.wants(layer, hidden_states.shape[0]):
            return super()._forward_impl(
                hidden_states, router_logits, shared_experts_input, input_ids
            )
        _RECORDER.save_weights(layer, self._record_weights())
        rec = dict(
            t=time.time(),
            m=hidden_states.shape[0],
            x=hidden_states.clone(),
            logits=router_logits.clone(),
            shared_x=None if shared_experts_input is None else shared_experts_input.clone(),
        )
        result = super()._forward_impl(
            hidden_states, router_logits, shared_experts_input, input_ids
        )
        shared_out, fused_out = result if isinstance(result, tuple) else (None, result)
        if isinstance(fused_out, torch.Tensor):
            rec.update(fused_out=fused_out, shared_out=shared_out)
            _RECORDER.add(layer, rec)
        return result

    def _routed_chain(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_experts_input: torch.Tensor | None = None,
    ):
        """Routed output, or (routed, shared) outputs with shared_experts_input."""
        from aiter.ops.flydsl.moe_routed_chain import routed_chain

        quant_config = self._quant_method.moe_quant_config
        shared = {}
        if shared_experts_input is not None:
            w_gu, w_dn, beta, linear_beta = self._chain_shared_weights
            shared = dict(
                shared_x=shared_experts_input.contiguous(),
                shared_w_gu=w_gu,
                shared_w_dn=w_dn,
                shared_beta=beta,
                shared_linear_beta=linear_beta,
            )
        return routed_chain(
            router_logits.contiguous(),
            self.router.e_score_correction_bias.data,
            hidden_states.contiguous(),
            self.routed_experts.w13_weight,
            self.routed_experts.w2_weight,
            quant_config.w1_scale,
            quant_config.w2_scale,
            topk=self.router.top_k,
            situ_beta=self.moe_config.activation_situ_beta,
            situ_linear_beta=self.moe_config.activation_situ_linear_beta,
            **shared,
        )

    def _forward_impl(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
        input_ids: torch.Tensor | None = None,
    ):
        """Small batches with a chain-able shared expert: one launch runs routing,
        the routed experts and the shared expert, so the shared expert is not
        forked to the aux stream."""
        if _RECORDER is not None:
            return self._recorded_forward_impl(
                hidden_states, router_logits, shared_experts_input, input_ids
            )
        if (
            shared_experts_input is None
            or self._chain_shared_weights is None
            or shared_experts_input.dtype != torch.bfloat16
            or not self._use_routed_chain(hidden_states, router_logits)
        ):
            return super()._forward_impl(
                hidden_states, router_logits, shared_experts_input, input_ids
            )
        self.routed_experts._ensure_moe_quant_config_init()
        with self._sequence_parallel_context():
            hidden_states, router_logits = self._maybe_dispatch(
                hidden_states, router_logits
            )
            fused_out, shared_out = self._routed_chain(
                hidden_states, router_logits, shared_experts_input
            )
            return self._maybe_combine(shared_out, fused_out)

    def _apply_quant_method(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
        input_ids: torch.Tensor | None = None,
        shared_experts_overlapping: bool = False,
    ):
        """Small batches run routing and the routed experts as one AITER launch.

        Shared experts run exactly as in the base runner; only the router's
        select_experts and the quant method's experts call are replaced.
        """
        if not self._use_routed_chain(hidden_states, router_logits):
            return super()._apply_quant_method(
                hidden_states,
                router_logits,
                shared_experts_input,
                input_ids,
                shared_experts_overlapping,
            )
        self._maybe_apply_shared_experts(
            shared_experts_input, SharedExpertsOrder.NO_OVERLAP
        )
        fused_out = self._routed_chain(hidden_states, router_logits)
        if shared_experts_overlapping:
            assert self._shared_experts is not None
            self._shared_experts.wait()
        return (
            self._shared_experts.output if self._shared_experts is not None else None,
            fused_out,
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
