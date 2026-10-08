# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 MoE layer recording for offline replay (debug).

Off unless VLLM_ROCM_K3_RECORD=<dir> is set. One serving run with it (eager
mode) writes, for each recorded layer on each recorded TP rank, under
<dir>/rank<r>/layer<NN>/:

    weights.pt           the layer's processed weights, once
    calls_<n>.pt         64 calls each: inputs and the stock path's outputs
    aiter_calls_m<M>.pt  the AITER calls the stock path made at M, with every
                         argument (AiterSpy), once per M

While recording, every MoE call takes the stock (multi-kernel) path. A
recording replays on one GPU with
benchmarks/kernels/benchmark_kimi_k3_moe_replay.py: the stock path exactly as
recorded, against the mono MoE launch, on real weights and inputs.
"""

import os
import time

import torch

from vllm.distributed import get_tensor_model_parallel_rank
from vllm.logger import init_logger

logger = init_logger(__name__)


def tensor_blob(t: torch.Tensor) -> dict:
    """A tensor as raw bytes plus dtype, so fp4/e8m0 weights survive torch.save."""
    is_shuffled = getattr(t, "is_shuffled", None)  # detach() drops attributes
    t = t.detach()
    return {
        "bytes": t.contiguous().reshape(-1).view(torch.uint8).cpu(),
        "dtype": str(t.dtype).removeprefix("torch."),
        "shape": tuple(t.shape),
        "contiguous": t.is_contiguous(),
        "is_shuffled": is_shuffled,
    }


class MoERecorder:
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
        self.layers = {
            int(v) for v in env("VLLM_ROCM_K3_RECORD_LAYERS", "1,30,60").split(",")
        }
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
                {
                    k: tensor_blob(v) if isinstance(v, torch.Tensor) else v
                    for k, v in weights.items()
                },
                path,
            )
            logger.info("K3 MoE record: weights of layer %d -> %s", layer, path)

    def add(self, layer: int, rec: dict) -> None:
        rec = {
            k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
            for k, v in rec.items()
        }
        buf = self.buf.setdefault(layer, [])
        buf.append(rec)
        self.saved[layer] = self.saved.get(layer, 0) + 1
        if len(buf) == self.CHUNK or self.saved[layer] == self.max_calls:
            n = (self.saved[layer] - 1) // self.CHUNK
            torch.save(buf, os.path.join(self.layer_dir(layer), f"calls_{n:05d}.pt"))
            buf.clear()


class AiterSpy:
    """Debug: the AITER calls the stock MoE path makes, with every argument, so an
    offline replay re-issues exactly them. Tensors that are a recorded weight, a
    forward input or a buffer of an earlier call are saved as references
    (("ref", name)), others by value; enums as ("enum", class, value)."""

    TARGETS = (("aiter", "biased_grouped_topk"), ("aiter.fused_moe", "fused_moe"))

    def __init__(self, named: dict):
        self.ptrs = {
            (t.data_ptr(), tuple(t.shape), t.dtype): n
            for n, t in named.items()
            if isinstance(t, torch.Tensor)
        }
        self.calls: list[dict] = []

    def _arg(self, v, op: int, slot: str):
        import enum

        if isinstance(v, torch.Tensor):
            key = (v.data_ptr(), tuple(v.shape), v.dtype)
            name = self.ptrs.get(key)
            if name is None:
                name = f"op{op}:{slot}"
                self.ptrs[key] = name
                return ("tensor", name, tensor_blob(v))
            return ("ref", name, {"is_shuffled": getattr(v, "is_shuffled", None)})
        if isinstance(v, enum.Enum):
            return ("enum", type(v).__name__, v.value)
        return ("value", v)

    def __enter__(self):
        import importlib

        self.saved = []
        for mod, attr in self.TARGETS:
            m = importlib.import_module(mod)
            orig = getattr(m, attr)
            self.saved.append((m, attr, orig))

            def spy(*args, _orig=orig, _name=f"{mod}.{attr}", **kwargs):
                op = len(self.calls)
                self.calls.append(
                    {
                        "fn": _name,
                        "args": [self._arg(v, op, str(i)) for i, v in enumerate(args)],
                        "kwargs": {k: self._arg(v, op, k) for k, v in kwargs.items()},
                    }
                )
                out = _orig(*args, **kwargs)
                outs = out if isinstance(out, tuple) else (out,)
                for i, o in enumerate(outs):
                    if isinstance(o, torch.Tensor):
                        self.ptrs.setdefault(
                            (o.data_ptr(), tuple(o.shape), o.dtype), f"op{op}:out{i}"
                        )
                self.calls[-1]["out"] = [
                    self.ptrs.get((o.data_ptr(), tuple(o.shape), o.dtype))
                    if isinstance(o, torch.Tensor)
                    else None
                    for o in outs
                ]
                return out

            setattr(m, attr, spy)
        return self

    def __exit__(self, *exc):
        for m, attr, orig in self.saved:
            setattr(m, attr, orig)


RECORDER = (
    MoERecorder(os.environ["VLLM_ROCM_K3_RECORD"])
    if os.environ.get("VLLM_ROCM_K3_RECORD")
    else None
)


def layer_index(runner) -> int | None:
    """The decoder layer number in the runner's layer name (model.layers.<n>.mlp...)."""
    parts = [
        p for p in str(getattr(runner, "layer_name", "")).split(".") if p.isdigit()
    ]
    return int(parts[0]) if parts else None


def record_weights(runner) -> dict:
    quant_config = runner._quant_method.moe_quant_config
    weights = dict(
        layer_name=str(runner.layer_name),
        w13=runner.routed_experts.w13_weight,
        w2=runner.routed_experts.w2_weight,
        w1_scale=quant_config.w1_scale,
        w2_scale=quant_config.w2_scale,
        bias=runner.router.e_score_correction_bias.data,
        topk=runner.router.top_k,
        situ_beta=runner.moe_config.activation_situ_beta,
        situ_linear_beta=runner.moe_config.activation_situ_linear_beta,
        mono_layer_ok=runner._mono_layer_ok,
    )
    if runner._shared_mlp_weights is not None:
        w_gu, w_dn, beta, linear_beta = runner._shared_mlp_weights
        weights.update(
            shared_w_gu=w_gu,
            shared_w_dn=w_dn,
            shared_beta=beta,
            shared_linear_beta=linear_beta,
        )
    return weights


def recorded_forward_impl(
    runner, forward, hidden_states, router_logits, shared_experts_input, input_ids
):
    """The stock path, with its inputs and outputs saved for offline replay."""
    layer = layer_index(runner)
    assert RECORDER is not None
    if not RECORDER.wants(layer, hidden_states.shape[0]):
        return forward(hidden_states, router_logits, shared_experts_input, input_ids)
    weights = record_weights(runner)
    RECORDER.save_weights(layer, weights)
    rec = dict(
        t=time.time(),
        m=hidden_states.shape[0],
        x=hidden_states.clone(),
        logits=router_logits.clone(),
        shared_x=None if shared_experts_input is None else shared_experts_input.clone(),
    )
    spy_path = os.path.join(
        RECORDER.layer_dir(layer), f"aiter_calls_m{rec['m']:02d}.pt"
    )
    if os.path.exists(spy_path):
        result = forward(hidden_states, router_logits, shared_experts_input, input_ids)
    else:
        named = {f"w:{k}": v for k, v in weights.items()}
        named.update({"in:x": hidden_states, "in:logits": router_logits})
        with AiterSpy(named) as spy:
            result = forward(
                hidden_states, router_logits, shared_experts_input, input_ids
            )
        torch.save(spy.calls, spy_path)
        logger.info(
            "K3 MoE record: %d AITER calls of layer %d (m=%d) -> %s",
            len(spy.calls),
            layer,
            rec["m"],
            spy_path,
        )
    shared_out, fused_out = result if isinstance(result, tuple) else (None, result)
    if isinstance(fused_out, torch.Tensor):
        rec.update(fused_out=fused_out, shared_out=shared_out)
        RECORDER.add(layer, rec)
    return result
