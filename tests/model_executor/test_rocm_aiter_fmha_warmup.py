# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import torch

from vllm.model_executor.warmup import kernel_warmup as mod
from vllm.v1.attention.backends.registry import AttentionBackendEnum


def _config(*, enable_jit_warmup: bool = True):
    return SimpleNamespace(
        kernel_config=SimpleNamespace(enable_jit_warmup=enable_jit_warmup),
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            multimodal_config=SimpleNamespace(
                mm_encoder_attn_backend=AttentionBackendEnum.ROCM_AITER_FA
            ),
        ),
    )


def test_warmup_rocm_aiter_mm_encoder_fmha(monkeypatch):
    monkeypatch.setenv("AITER_ENABLE_FMHA_OPUS", "1")
    monkeypatch.setattr(mod.current_platform, "is_rocm", lambda: True)

    import vllm.platforms.rocm as rocm

    monkeypatch.setattr(rocm, "on_gfx950", lambda: True)

    warmup = Mock()
    aiter = ModuleType("aiter")
    aiter_ops = ModuleType("aiter.ops")
    aiter_mha = ModuleType("aiter.ops.mha")
    aiter_mha.fmha_fwd_bf16_opus_fwd = warmup
    monkeypatch.setitem(sys.modules, "aiter", aiter)
    monkeypatch.setitem(sys.modules, "aiter.ops", aiter_ops)
    monkeypatch.setitem(sys.modules, "aiter.ops.mha", aiter_mha)

    tensor = object()
    monkeypatch.setattr(mod.torch, "empty", Mock(return_value=tensor))
    synchronize = Mock()
    monkeypatch.setattr(mod.torch.accelerator, "synchronize", synchronize)

    mod.warmup_rocm_aiter_mm_encoder_fmha(_config(), torch.device("cuda"))

    warmup.assert_called_once_with(
        tensor,
        tensor,
        tensor,
        softmax_scale=128**-0.5,
        causal=False,
    )
    synchronize.assert_called_once_with()


def test_warmup_rocm_aiter_mm_encoder_fmha_honors_jit_gate(monkeypatch):
    monkeypatch.setenv("AITER_ENABLE_FMHA_OPUS", "1")
    monkeypatch.setattr(mod.current_platform, "is_rocm", lambda: True)
    empty = Mock()
    monkeypatch.setattr(mod.torch, "empty", empty)

    mod.warmup_rocm_aiter_mm_encoder_fmha(
        _config(enable_jit_warmup=False), torch.device("cuda")
    )

    empty.assert_not_called()
