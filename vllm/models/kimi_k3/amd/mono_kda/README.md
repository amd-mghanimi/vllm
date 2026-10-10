# Kimi-K3 mono KDA decode (ROCm CDNA4)

One persistent FlyDSL launch for a KDA layer's decode step, behind
`VLLM_ROCM_MONO_DECODE=1`.

## Scope

A step takes the launch when it is decode only, non-speculative, at most 16
rows (`plan.M_MAX`, one MFMA tile of tokens), and runs eagerly or as a FULL
CUDA graph; a layer takes it when the fused KDA decode kernel's weights are
staged (gfx950, width-major conv state) and in_proj / f_b_proj / o_proj are
bf16, unquantized and without bias. `../mono_kda_decode.py` holds the
conditions; everything else runs the layer as before. Both paths read and write
the same conv and recurrent states.

## Launch

The launch replaces, per KDA layer and step,

| vLLM | here |
|---|---|
| `in_proj_qkvgfab` (bf16 GEMM, M x 7168 x 6288 at TP8) | in_proj tiles (`stages/gemv.py`) |
| `f_b_proj` (bf16 GEMM, M x 128 x 1536) | decode task prologue (`stages/decode.py`) |
| `ops.fused_kda_decode` (conv update, gated delta rule, gated RMSNorm) | decode task (`stages/decode.py`) |
| `o_proj` before its all-reduce (bf16 GEMM, M x 1536 x 7168) | o_proj tiles (`stages/gemv.py`) |

o_proj's TP all-reduce runs after the launch, as before.

- `layer.py`: the kernel, its ticket loops and the launcher.
- `runner.py`: launcher cache, workspace, `supported` and `mono_kda`.
- `plan.py`: shapes, ticket order, control words, workspace layout.
- `sync.py`: tickets and counters between workgroups.
- `../mono_kda_decode.py`: vLLM's side, called first in
  `KimiK3DeltaAttention.forward`.

## Execution model and hand-offs

One workgroup (256 threads) a CU; workgroups loop on a global ticket counter:

1. in_proj tiles (16 columns, the four waves split K): f_a and beta first, then
   head by head q_h, k_h, v_h, g_h. A tile writes its bf16 columns through to
   memory and bumps the shared counter or its head's counter.
2. decode tasks (head h, row m), head-major: the state slice (64 KB), the W_fb
   rows, the conv weights and the conv taps load first; the task then waits for
   the shared and head-h counters, writes its gated-norm output row (bf16) and
   bumps the decode counter.
3. o_proj tiles: the weights load first; the tile waits for every decode task.

A ticket waits only on smaller tickets, so any grid size completes (the tests
run it on one workgroup). The workgroup that takes the last ticket returns the
counters to zero for the next launch.

## Numerics (the vLLM path each stage reproduces)

- GEMMs: fp32 accumulation, bf16 outputs (in_proj columns, g1, o_proj), as
  vLLM's unquantized GEMMs; the sum order differs.
- Decode: `fused_kda_decode_kernel_rocm.cu` step by step: conv on bf16 taps with
  fp32 weights and SiLU, gate `exp(-exp(A_log) softplus(g1 + dt_bias))` or the
  lower-bound form, L2 norm of q (scaled by 128^-0.5) and k, the rank-1 update
  folded into the output, gated RMSNorm with the fp32 weight. Reductions run as
  wave butterflies rather than the kernel's DPP row order.
- A row whose state slot is <= 0 (vLLM's padding) outputs zeros and leaves both
  states untouched.

## Validation

- `tests/models/kimi_k3/test_amd_mono_kda.py` (gfx950): output and both states
  against in_proj / f_b_proj / `ops.fused_kda_decode` / o_proj at TP8 shapes, M
  1..16, both gates, padded rows, three consecutive steps, grids of 1, 7, 64.
- `tests/models/kimi_k3/test_amd_mono_kda_gate.py` (CPU): which steps and
  layouts take the launch.
