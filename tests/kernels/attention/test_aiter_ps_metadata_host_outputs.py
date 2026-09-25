# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The ROCm AITER MLA FP8 prefill builder plans into pinned host buffers and
copies them to the device on the current stream. That relies on
aiter.get_ps_metadata_v1 accepting host output tensors and producing the same
plan it would write into device tensors."""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCm only", allow_module_level=True)
aiter = pytest.importorskip("aiter")
if not hasattr(aiter, "get_ps_metadata_v1"):
    pytest.skip("aiter without get_ps_metadata_v1", allow_module_level=True)

TILE_Q = 256  # _FP8_PREFILL_TILE_Q
NUM_HEAD_K = 16
MAX_REQS = 256
MAX_TOKENS = 16384
OUTPUT_NAMES = (
    "work_indptr",
    "work_info",
    "reduce_indptr",
    "reduce_final_map",
    "reduce_partial_map",
)


def _alloc(device: str) -> list[torch.Tensor]:
    info = aiter.get_ps_metadata_info_v1(
        batch_size=MAX_REQS,
        num_head_k=NUM_HEAD_K,
        max_qlen=MAX_TOKENS,
        qlen_granularity=TILE_Q,
    )
    out = []
    for size, dtype in info:
        shape = size if isinstance(size, tuple) else (size,)
        pin = device == "cpu"
        out.append(torch.zeros(*shape, dtype=dtype, device=device, pin_memory=pin))
    return out


def _plan(bufs: list[torch.Tensor], seq_lens: list[int]) -> None:
    lens = torch.tensor(seq_lens, dtype=torch.int32)
    qo = torch.zeros(len(seq_lens) + 1, dtype=torch.int32)
    qo[1:] = lens.cumsum(0)
    aiter.get_ps_metadata_v1(
        qo,
        qo.clone(),
        lens,
        1,
        NUM_HEAD_K,
        *bufs,
        qhead_granularity=1,
        qlen_granularity=TILE_Q,
        kvlen_granularity=128,
        block_size=1,
        is_causal=True,
    )


@pytest.mark.parametrize(
    "seq_lens",
    [
        [8175],  # single unchunked prefill (the crash batch in the issue)
        [6992, 6896, 2495],  # several prefills with partial tiles
        [2] * MAX_REQS,  # cudagraph-capture style dummy batch
        [4096] * 4,
        [MAX_TOKENS],
    ],
)
def test_host_outputs_match_device_outputs(seq_lens):
    device_bufs = _alloc("cuda")
    host_bufs = _alloc("cpu")
    _plan(device_bufs, seq_lens)
    _plan(host_bufs, seq_lens)
    torch.cuda.synchronize()
    # Index 0 is work_metadata, which the host planner does not write.
    for name, dev, host in zip(OUTPUT_NAMES, device_bufs[1:], host_bufs[1:]):
        assert torch.equal(dev.cpu(), host), name
