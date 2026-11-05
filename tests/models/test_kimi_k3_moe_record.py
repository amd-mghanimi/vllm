# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 MoE recording (VLLM_ROCM_K3_RECORD) on CPU: what a recording holds."""

import enum
import os
import sys
import types

import pytest
import torch

from vllm.models.kimi_k3.amd import moe_record as mr


def _decode(blob):
    return blob["bytes"].view(getattr(torch, blob["dtype"])).reshape(blob["shape"])


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.uint8])
def test_tensor_blob_round_trip(dtype):
    t = (torch.randn(6, 10) * 50).to(dtype)[:, 2:]
    t.is_shuffled = True
    blob = mr.tensor_blob(t)
    assert blob["is_shuffled"] is True and blob["contiguous"] is False
    assert torch.equal(_decode(blob), t)


def test_recorder_writes_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr(mr, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setenv("VLLM_ROCM_K3_RECORD_LAYERS", "7")
    monkeypatch.setenv("VLLM_ROCM_K3_RECORD_EVERY", "2")
    monkeypatch.setenv("VLLM_ROCM_K3_RECORD_MAX", "65")
    rec = mr.MoERecorder(str(tmp_path))
    assert not rec.wants(8, 1) and not rec.wants(7, 33)
    taken = 0
    for i in range(200):
        if rec.wants(7, 4):
            rec.add(7, {"m": 4, "x": torch.full((4, 2), float(i))})
            taken += 1
    assert taken == 65
    d = tmp_path / "rank0" / "layer07"
    first = torch.load(d / "calls_00000.pt", weights_only=False)
    last = torch.load(d / "calls_00001.pt", weights_only=False)
    assert len(first) == 64 and len(last) == 1
    assert first[1]["x"][0, 0].item() == 2.0  # every 2nd call


class _Act(enum.Enum):
    SILU = 1


def test_spy_records_refs_and_values(monkeypatch):
    fake = types.ModuleType("fake_aiter")

    def topk(x, w, act=_Act.SILU, k=2):
        return x @ w, torch.zeros(1)

    def moe(y, w, scale):
        return y * scale

    fake.topk, fake.moe = topk, moe
    monkeypatch.setitem(sys.modules, "fake_aiter", fake)
    monkeypatch.setattr(
        mr.AiterSpy, "TARGETS", (("fake_aiter", "topk"), ("fake_aiter", "moe"))
    )
    x, w = torch.randn(2, 3), torch.randn(3, 3)
    with mr.AiterSpy({"in:x": x, "w:w": w}) as spy:
        y, _ = fake.topk(x, w, k=4)
        fake.moe(y, torch.ones(3), 0.5)
    assert fake.topk is topk
    c0, c1 = spy.calls
    assert [a[:2] for a in c0["args"]] == [("ref", "in:x"), ("ref", "w:w")]
    assert c0["kwargs"]["k"] == ("value", 4)
    assert c0["out"] == ["op0:out0", "op0:out1"]
    # an earlier call's output is a reference; a new tensor is saved by value
    assert c1["args"][0][:2] == ("ref", "op0:out0")
    assert c1["args"][1][0] == "tensor" and torch.equal(
        _decode(c1["args"][1][2]), torch.ones(3)
    )
    assert c1["args"][2] == ("value", 0.5)


def test_layer_index():
    assert mr.layer_index(types.SimpleNamespace(layer_name="model.layers.30.mlp")) == 30
    assert mr.layer_index(types.SimpleNamespace(layer_name="lm_head")) is None


def test_off_by_default():
    assert ("VLLM_ROCM_K3_RECORD" in os.environ) == (mr.RECORDER is not None)
