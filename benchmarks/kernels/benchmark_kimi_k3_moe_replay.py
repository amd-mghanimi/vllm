# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay recorded Kimi-K3 MoE layers: the multi-kernel path vs the mono MoE launch.

Input: a recording made with VLLM_ROCM_K3_RECORD=<dir> (see
vllm/models/kimi_k3/amd/moe_record.py), one rank's directory <dir>/rank<r>.
Each layer has its weights, recorded calls (inputs and the stock outputs) and,
per M, the exact AITER calls the stock path made. On one GPU, per layer and M,
one CUDA graph per configuration:

    multi||sh    the recorded AITER calls (biased_grouped_topk + fused_moe)
                 with the shared expert (KimiMLP) on an aux stream, forked
                 before and joined after, as vLLM serves it today
    sh;multi     the same kernels on one stream, shared expert first
    mono         the mono MoE launch: routing, routed and shared experts

Every configuration replays every sampled input of that M and is compared with
the outputs the stock path recorded in serving; the multi-kernel replay's own
distance to them is the run-to-run spread (fused_moe is not bit-reproducible).
Timing: median of --iters graph replays, warm (back to back, weights in cache).

    python3 benchmarks/kernels/benchmark_kimi_k3_moe_replay.py \\
        --dir <dir>/rank0 --layers 1,30,60 --ms 1,2,4,8,16 --out replay.json
"""

import argparse
import glob
import importlib
import json
import os
import statistics
from collections import defaultdict

import torch

CONFIGS = ("multi||sh", "sh;multi", "mono")


def load_tensor(blob: dict) -> torch.Tensor:
    t = blob["bytes"].view(getattr(torch, blob["dtype"])).reshape(blob["shape"])
    t = t.cuda()
    if blob.get("is_shuffled") is not None:
        t.is_shuffled = blob["is_shuffled"]
    return t


def load_weights(path: str) -> dict:
    w = torch.load(path, weights_only=False)
    return {
        k: load_tensor(v) if isinstance(v, dict) and "bytes" in v else v
        for k, v in w.items()
    }


def load_aiter_calls(layer_dir: str) -> dict[int, list]:
    return {
        int(os.path.basename(p)[len("aiter_calls_m") : -3]): torch.load(
            p, weights_only=False
        )
        for p in glob.glob(os.path.join(layer_dir, "aiter_calls_m*.pt"))
    }


def inputs_by_m(layer_dir: str, per_m: int) -> dict[int, list[dict]]:
    keys = ("logits", "x", "shared_x", "fused_out", "shared_out")
    by_m: dict[int, list[dict]] = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(layer_dir, "calls_*.pt"))):
        for rec in torch.load(f, weights_only=False):
            if len(by_m[rec["m"]]) < per_m:
                by_m[rec["m"]].append(
                    {k: rec[k].cuda() for k in keys if torch.is_tensor(rec.get(k))}
                )
    return dict(sorted(by_m.items()))


def _enum(cls_name: str, value):
    import aiter

    for mod in (aiter, importlib.import_module("aiter.ops.flydsl.moe_common")):
        cls = getattr(mod, cls_name, None)
        if cls is not None:
            return cls(value)
    raise KeyError(cls_name)


class MultiKernel:
    """The stock path's recorded AITER calls on fixed input buffers."""

    def __init__(self, calls: list, w: dict, rec0: dict):
        self.calls = calls
        self.env = {f"w:{k}": v for k, v in w.items()}
        for k in ("w:w13", "w:w2"):
            # fused_moe takes the FlyDSL a4w4 path only for preshuffled weights
            if getattr(self.env[k], "is_shuffled", None) is None:
                self.env[k].is_shuffled = True
        self.env["in:x"] = rec0["x"].clone()
        self.env["in:logits"] = rec0["logits"].clone()
        for c in calls:
            for v in [*c["args"], *c["kwargs"].values()]:
                if v[0] == "ref" and len(v) > 2 and v[2].get("is_shuffled") is not None:
                    self.env[v[1]].is_shuffled = v[2]["is_shuffled"]
                if v[0] == "tensor":
                    self.env[v[1]] = load_tensor(v[2])
        self.fns = []
        for c in calls:
            mod, attr = c["fn"].rsplit(".", 1)
            self.fns.append(getattr(importlib.import_module(mod), attr))
        self.out = None

    def _val(self, v):
        if v[0] in ("ref", "tensor"):
            return self.env[v[1]]
        if v[0] == "enum":
            return _enum(v[1], v[2])
        return v[1]

    def fill(self, rec: dict) -> None:
        self.env["in:x"].copy_(rec["x"])
        self.env["in:logits"].copy_(rec["logits"])

    def __call__(self) -> None:
        for c, fn in zip(self.calls, self.fns):
            args = [self._val(v) for v in c["args"]]
            out = fn(*args, **{k: self._val(v) for k, v in c["kwargs"].items()})
            outs = out if isinstance(out, tuple) else (out,)
            for name, o in zip(c.get("out", []), outs):
                if name is not None and torch.is_tensor(o):
                    self.env[name] = o
            self.out = outs[0]


class SharedExpert:
    """vLLM's KimiMLP on ROCm (unquantized bf16) on a fixed input buffer."""

    def __init__(self, w: dict, rec0: dict):
        self.w_gu, self.w_dn = w["shared_w_gu"], w["shared_w_dn"]
        self.beta = float(w["shared_beta"])
        lb = w["shared_linear_beta"]
        self.linear_beta = -1.0 if lb is None or lb <= 0 else float(lb)
        self.x = rec0["shared_x"].clone()
        self.out = None

    def fill(self, rec: dict) -> None:
        self.x.copy_(rec["shared_x"])

    def __call__(self) -> None:
        gemm = torch.ops.vllm.rocm_unquantized_gemm
        gu = gemm(self.x, self.w_gu, None)
        h = gu.new_empty(gu.shape[0], gu.shape[1] // 2)
        torch.ops._C.situ_and_mul(h, gu, self.beta, self.linear_beta)
        self.out = gemm(h, self.w_dn, None)


class Mono:
    """The mono MoE launch on fixed buffers."""

    def __init__(self, w: dict, rec0: dict):
        m = rec0["x"].shape[0]
        self.w = w
        self.lg, self.x = rec0["logits"].clone(), rec0["x"].clone()
        self.out = rec0["x"].new_empty(m, w["w2"].shape[1])
        self.tw = torch.empty(m, w["topk"], device="cuda")
        self.ti = torch.empty(m, w["topk"], dtype=torch.int32, device="cuda")
        self.sx = rec0["shared_x"].clone()
        self.sout = rec0["x"].new_empty(m, w["shared_w_dn"].shape[0])

    def fill(self, rec: dict) -> None:
        self.lg.copy_(rec["logits"])
        self.x.copy_(rec["x"])
        self.sx.copy_(rec["shared_x"])

    def __call__(self) -> None:
        from vllm.models.kimi_k3.amd.mono.runner import mono_moe

        w = self.w
        mono_moe(
            self.lg, w["bias"], self.x, w["w13"], w["w2"], w["w1_scale"],
            w["w2_scale"], self.sx, w["shared_w_gu"], w["shared_w_dn"],
            topk=w["topk"], situ_beta=w["situ_beta"],
            situ_linear_beta=w["situ_linear_beta"], shared_beta=w["shared_beta"],
            shared_linear_beta=w["shared_linear_beta"], out=self.out,
            shared_out=self.sout, topk_weights=self.tw, topk_ids=self.ti,
        )  # fmt: skip


class Layer:
    """One layer at one M: the three configurations on shared input buffers."""

    def __init__(self, w: dict, rec0: dict, calls: list):
        self.multi = MultiKernel(calls, w, rec0)
        self.mono = Mono(w, rec0)
        self.sh = SharedExpert(w, rec0)
        self.aux = torch.cuda.Stream()

    def fill(self, rec: dict) -> None:
        for p in (self.multi, self.mono, self.sh):
            p.fill(rec)

    def run(self, cfg: str) -> None:
        if cfg == "mono":
            self.mono()
        elif cfg == "sh;multi":
            self.sh()
            self.multi()
        else:
            main = torch.cuda.current_stream()
            self.aux.wait_stream(main)
            with torch.cuda.stream(self.aux):
                self.sh()
            self.multi()
            main.wait_stream(self.aux)

    def outputs(self, cfg: str):
        """(routed, shared) of the last run of cfg."""
        if cfg == "mono":
            return self.mono.out, self.mono.sout
        return self.multi.out, self.sh.out

    def capture(self, cfg: str) -> torch.cuda.CUDAGraph:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                self.run(cfg)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self.run(cfg)
        return g


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return (
        torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=1).min().item()
    )


def check(layer: Layer, cfg: str, g: torch.cuda.CUDAGraph, recs: list) -> dict:
    """Worst cosine against the recorded stock outputs over all inputs."""
    r = {"cos_min": 1.0, "sh_cos_min": None}
    for rec in recs:
        layer.fill(rec)
        g.replay()
        torch.cuda.synchronize()
        out, sout = layer.outputs(cfg)
        r["cos_min"] = min(r["cos_min"], _cos(out, rec["fused_out"]))
        if sout is not None and "shared_out" in rec:
            r["sh_cos_min"] = min(r["sh_cos_min"] or 1.0, _cos(sout, rec["shared_out"]))
    return r


def time_graph(layer: Layer, g: torch.cuda.CUDAGraph, recs: list, iters: int):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t = []
    for i in range(iters + 3):
        layer.fill(recs[i % len(recs)])
        s.record()
        g.replay()
        e.record()
        e.synchronize()
        if i >= 3:
            t.append(s.elapsed_time(e) * 1e3)
    return round(statistics.median(t), 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", required=True, help="<recording>/rank<r>")
    ap.add_argument("--layers", default="", help="e.g. 1,30,60 (default: all)")
    ap.add_argument("--ms", default="1,2,4,8,16")
    ap.add_argument("--per-m", type=int, default=14, help="inputs per M")
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    os.environ.setdefault("VLLM_ROCM_USE_AITER", "1")
    import vllm._custom_ops  # noqa: F401  (registers situ_and_mul)
    from vllm.model_executor.layers import utils as vutils

    vutils.warmup_rocm_skinny_gemm_workspaces(torch.device("cuda"))
    dirs = sorted(glob.glob(os.path.join(a.dir, "layer*")))
    if a.layers:
        keep = {f"layer{int(v):02d}" for v in a.layers.split(",")}
        dirs = [d for d in dirs if os.path.basename(d) in keep]
    ms = [int(v) for v in a.ms.split(",")]
    rows = []
    print(f"{'layer':>7} {'M':>3} " + " ".join(f"{c:>10}" for c in CONFIGS)
          + "  whole-layer uplift  worst cos")  # fmt: skip
    for d in dirs:
        w = load_weights(os.path.join(d, "weights.pt"))
        calls, by_m = load_aiter_calls(d), inputs_by_m(d, a.per_m)
        for m in ms:
            if m not in calls or m not in by_m or "shared_w_gu" not in w:
                continue
            recs = by_m[m]
            layer = Layer(w, recs[0], calls[m])
            row = {"layer": os.path.basename(d), "M": m, "inputs": len(recs)}
            for cfg in CONFIGS:
                g = layer.capture(cfg)
                row[f"{cfg}_us"] = time_graph(layer, g, recs, a.iters)
                row.update(
                    {f"{cfg}_{k}": v for k, v in check(layer, cfg, g, recs).items()}
                )
                del g
            row["uplift"] = round(row["mono_us"] / row["multi||sh_us"] - 1, 4)
            worst = min(row[f"{c}_cos_min"] for c in CONFIGS)
            print(f"{row['layer']:>7} {m:>3} "
                  + " ".join(f"{row[f'{c}_us']:>10.1f}" for c in CONFIGS)
                  + f"  {100 * row['uplift']:+17.1f}%  {worst:.6f}")  # fmt: skip
            rows.append(row)
            del layer
            torch.cuda.empty_cache()
        del w
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"gpu": torch.cuda.get_device_name(), "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
