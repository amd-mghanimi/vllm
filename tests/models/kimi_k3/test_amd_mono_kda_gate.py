# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which steps and layouts take the mono KDA launch (CPU only)."""

from types import SimpleNamespace

import pytest
import torch

from vllm.config import CUDAGraphMode
from vllm.models.kimi_k3.amd import mono_kda_decode
from vllm.models.kimi_k3.amd.mono_kda import runner
from vllm.models.kimi_k3.amd.mono_kda.plan import Dims
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

PREFIX = "model.layers.0.self_attn"


def _metadata(**kw):
    md = object.__new__(GDNAttentionMetadata)
    base = dict(
        spec_sequence_masks=None,
        num_prefills=0,
        num_decodes=4,
        num_actual_tokens=4,
        non_spec_state_indices_tensor=torch.arange(4, dtype=torch.int32),
    )
    base.update(kw)
    for k, v in base.items():
        object.__setattr__(md, k, v)
    return md


@pytest.fixture
def mono(monkeypatch):
    def use(mode=CUDAGraphMode.FULL, md=None):
        fc = SimpleNamespace(
            cudagraph_runtime_mode=mode,
            attn_metadata={PREFIX: _metadata() if md is None else md},
        )
        monkeypatch.setattr(
            mono_kda_decode, "is_forward_context_available", lambda: True
        )
        monkeypatch.setattr(mono_kda_decode, "get_forward_context", lambda: fc)

    s = mono_kda_decode.MonoKDA(SimpleNamespace(prefix=PREFIX))
    s.use = use
    return s


def test_decode_steps_take_the_launch(mono):
    for mode in (CUDAGraphMode.NONE, CUDAGraphMode.FULL):
        mono.use(mode)
        assert mono._decode_metadata(4) is not None


@pytest.mark.parametrize(
    "why, mode, md, rows",
    [
        ("piecewise", CUDAGraphMode.PIECEWISE, None, 4),
        ("spec", CUDAGraphMode.FULL, _metadata(spec_sequence_masks=torch.ones(4)), 4),
        ("prefill", CUDAGraphMode.FULL, _metadata(num_prefills=1), 4),
        ("no decodes", CUDAGraphMode.FULL, _metadata(num_decodes=0), 4),
        ("rows", CUDAGraphMode.FULL, _metadata(num_actual_tokens=3), 4),
        ("too many", CUDAGraphMode.FULL, _metadata(num_actual_tokens=17), 17),
        ("no slots", CUDAGraphMode.FULL,
         _metadata(non_spec_state_indices_tensor=None), 4),
    ],
)  # fmt: skip
def test_other_steps_keep_the_layer(mono, why, mode, md, rows):
    mono.use(mode, md)
    assert mono._decode_metadata(rows) is None, why


def test_flag_off_keeps_the_layer(monkeypatch, mono):
    monkeypatch.setattr(
        mono_kda_decode.envs, "VLLM_ROCM_MONO_DECODE", False, raising=False
    )
    assert mono.layer_ok is False
    assert mono(torch.zeros(4, 7168, dtype=torch.bfloat16)) is None


def _tensors(m=4, heads=12, hidden=7168):
    d = Dims(heads=heads, hidden=hidden)
    bf = torch.bfloat16
    P = d.proj
    return dict(
        x=torch.zeros(m, hidden, dtype=bf),
        w_in=torch.zeros(d.n_in, hidden, dtype=bf),
        w_fb=torch.zeros(P, 128, dtype=bf),
        w_o=torch.zeros(hidden, P, dtype=bf),
        conv_w=torch.zeros(3, 4, P),
        conv_state=torch.zeros(8, 3, 3 * P, dtype=bf).transpose(-1, -2),
        state=torch.zeros(8, heads, 128, 128),
        state_indices=torch.zeros(m, dtype=torch.int32),
    )


def test_supported_layouts():
    assert runner.supported(**_tensors())
    assert runner.supported(**_tensors(m=16))
    assert not runner.supported(**_tensors(m=17))
    t = _tensors()
    t["conv_state"] = torch.zeros(8, 3 * 1536, 3, dtype=torch.bfloat16)  # dim-first
    assert not runner.supported(**t)
    t = _tensors()
    t["w_in"] = t["w_in"].float()
    assert not runner.supported(**t)
    t = _tensors()
    t["state_indices"] = t["state_indices"][:3]
    assert not runner.supported(**t)
