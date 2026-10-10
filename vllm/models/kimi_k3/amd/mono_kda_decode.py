# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 mono KDA decode layer on ROCm CDNA4 (``VLLM_ROCM_MONO_DECODE=1``).

A KDA layer's decode step -- in_proj_qkvgfab, f_b_proj, the conv update, the
gated delta rule, the gated RMSNorm and o_proj -- runs as one persistent FlyDSL
launch (``mono_kda``, this directory); o_proj's TP all-reduce follows as
before. A step takes the launch when it is decode only, non-speculative and at
most ``M_MAX`` rows, run eagerly or as a FULL CUDA graph (a PIECEWISE /
breakable capture is replayed for mixed batches too). Anything else runs the
layer as before; both paths read and write the same states, so they interleave
freely. The conditions read only metadata every TP rank holds alike.
"""

from functools import cached_property
from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
from vllm.config import CUDAGraphMode
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.models.kimi_k3.amd.kda import KimiK3DeltaAttention

logger = init_logger(__name__)


class MonoKDA:
    """The mono KDA path of one ``KimiK3DeltaAttention`` layer."""

    def __init__(self, layer: "KimiK3DeltaAttention") -> None:
        self.layer = layer

    @cached_property
    def layer_ok(self) -> bool:
        """Static conditions: the flag, the device, and a layer the launch
        computes as vLLM does (the fused decode kernel's weights staged, bf16
        unquantized projections without bias, no speculative tokens)."""
        if not envs.VLLM_ROCM_MONO_DECODE or not current_platform.is_rocm():
            return False
        from vllm.platforms.rocm import on_gfx950

        a = self.layer
        why = None
        if not on_gfx950():
            why = "not gfx950"
        elif a.decode_conv1d_weight is None or a.decode_norm_weight is None:
            why = "fused KDA decode kernel unavailable"
        elif a.conv1d.bias is not None:
            why = "conv bias"
        elif a.num_spec != 0:
            why = "speculative decoding"
        elif any(
            p.quant_method is None
            or type(p.quant_method).__name__ != "UnquantizedLinearMethod"
            or p.bias is not None
            or p.weight.dtype != torch.bfloat16
            for p in (a.in_proj_qkvgfab, a.f_b_proj, a.o_proj)
        ):
            why = "projections not bf16 unquantized without bias"
        if why is not None:
            logger.info_once("Kimi-K3 mono KDA decode off: %s", why)
            return False
        logger.info_once("Kimi-K3 mono KDA decode on (decode steps of <= 16 rows)")
        return True

    def _decode_metadata(self, rows: int):
        """The layer's GDN metadata when this step takes the launch, else None."""
        from vllm.models.kimi_k3.amd.mono_kda.plan import M_MAX
        from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

        if not 1 <= rows <= M_MAX or not is_forward_context_available():
            return None
        fc = get_forward_context()
        if fc.cudagraph_runtime_mode not in (CUDAGraphMode.NONE, CUDAGraphMode.FULL):
            return None
        md = fc.attn_metadata
        if not isinstance(md, dict):
            return None
        m = md.get(self.layer.prefix)
        if (
            not isinstance(m, GDNAttentionMetadata)
            or m.spec_sequence_masks is not None
            or m.num_prefills != 0
            or m.num_decodes == 0
            # FULL graphs pad tokens and state slots alike (NULL slot 0)
            or m.num_actual_tokens != rows
            or m.non_spec_state_indices_tensor is None
        ):
            return None
        return m

    def __call__(self, hidden_states) -> torch.Tensor | None:
        """The layer's output (o_proj, all-reduced) by the launch, or None when
        this step takes the original path."""
        if torch.compiler.is_compiling():
            return None
        if not self.layer_ok or not isinstance(hidden_states, torch.Tensor):
            return None
        if hidden_states.dim() != 2 or not hidden_states.is_contiguous():
            return None
        m = self._decode_metadata(hidden_states.shape[0])
        if m is None:
            return None
        from vllm.models.kimi_k3.amd.mono_kda import runner

        a = self.layer
        conv_state, state = a.kv_cache
        conv_state = conv_state.transpose(-1, -2)
        idx = m.non_spec_state_indices_tensor
        args = (
            hidden_states,
            a.in_proj_qkvgfab.weight,
            a.f_b_proj.weight,
            a.o_proj.weight,
            a.decode_conv1d_weight,
            conv_state,
        )
        if not runner.supported(*args, state, idx):
            logger.debug_once("Kimi-K3 mono KDA decode skipped: unsupported layout")
            return None
        out = runner.mono_kda(
            *args,
            a.A_log,
            a.dt_bias,
            a.decode_norm_weight,
            idx,
            state,
            lower_bound=a.gate_lower_bound,
            eps=a.o_norm.eps,
        )
        if a.o_proj.reduce_results and a.o_proj.tp_size > 1:
            out = tensor_model_parallel_all_reduce(out)
        return out
