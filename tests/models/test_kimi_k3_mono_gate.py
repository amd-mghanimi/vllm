# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which Kimi-K3 MoE calls take the mono MoE launch (CPU, no AITER needed).

The runner's predicates are called on a stand-in object; the kernel module is
replaced by a fake that records the shapes it is asked about.
"""

import sys
import types

import pytest
import torch

from vllm.models.kimi_k3.amd import latent_moe_runner as lmr

R = lmr.ROCmLatentMoERunner
RUNNER = "vllm.models.kimi_k3.amd.mono.runner"


class _Router:
    top_k = 16
    capture_fn = None


class _Experts:
    w13_weight = torch.empty(896, 768, 1792, dtype=torch.uint8)


def _stub(layer_ok=True):
    s = types.SimpleNamespace(router=_Router(), routed_experts=_Experts())
    s._routed_chain_layer_ok = layer_ok
    return s


@pytest.fixture
def fake_runner(monkeypatch):
    asked = []
    mod = types.ModuleType(RUNNER)

    def supported(m, ne, topk, inter):
        asked.append((m, ne, topk, inter))
        return m <= 32

    mod.supported = supported
    monkeypatch.setitem(sys.modules, RUNNER, mod)
    monkeypatch.setattr(lmr, "_RECORDER", None)
    return asked


def _call(s, m, x_dtype=torch.bfloat16, logits_dtype=torch.float32):
    x = torch.empty(m, 3584, dtype=x_dtype)
    logits = torch.empty(m, 896, dtype=logits_dtype)
    return R._use_routed_chain(s, x, logits)


def test_off_without_switch(monkeypatch):
    monkeypatch.delenv("VLLM_ROCM_MONO_DECODE", raising=False)
    assert R._routed_chain_layer_ok.func(_stub()) is False


def test_off_on_other_platforms(monkeypatch):
    monkeypatch.setenv("VLLM_ROCM_MONO_DECODE", "1")
    monkeypatch.setattr(lmr.current_platform, "is_rocm", lambda: False)
    assert R._routed_chain_layer_ok.func(_stub()) is False


def test_small_decode_batches(fake_runner):
    s = _stub()
    assert _call(s, 1)
    assert _call(s, 16)
    assert fake_runner == [(1, 896, 16, 384), (16, 896, 16, 384)]


def test_declined(fake_runner):
    assert not _call(_stub(), 17)
    assert not _call(_stub(), 4, x_dtype=torch.float16)
    assert not _call(_stub(), 4, logits_dtype=torch.bfloat16)
    assert not _call(_stub(layer_ok=False), 4)
    s = _stub()
    s.router.capture_fn = lambda *a: None
    assert not _call(s, 4)
    assert fake_runner == []
