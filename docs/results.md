# FORGE results

All measurements on **Qwen2.5-Coder-1.5B**, Apple M5 Max (128 GB), MPS backend, float32.
Calibration: 128 x 2048 tokens from wikitext2 (seed 0). Perplexity: full wikitext2 test
split, 146 non-overlapping 2048-token windows.

Reproduce with:

```bash
forge quantize --nsamples 128 --out docs/results_m2.md
```

---

## Headline

| Configuration | wikitext2 ppl | vs FP16 |
|---|---:|---:|
| FP16 (baseline) | **10.40** | 1.0x |
| Naive ternary, absmean scale | 506,714 | 48,700x |
| Naive ternary, optimal scale | 404,730 | 38,900x |
| + rotation (absmean) | 47,775 | 4,590x |
| **FORGE (rotation + GPTQ + sequential)** | **172.27** | **16.6x** |

FORGE improves on naive ternary by **2,350x**. It is **not yet a usable model**: 172 ppl
against FP16's 10.40 is a large gap, and no amount of framing changes that. See
[Where the remaining error is](#where-the-remaining-error-is).

## Cost

| | |
|---|---|
| Quantization time | **597 s** (10 min) for 1.5B, 128 sequences |
| Calibration capture alone | 195 s |
| Peak Hessian residency | **378 MB** (one block at a time) |
| Rotation fusion | 1.5 s |

Comfortably inside the 1-4 hour, single-consumer-GPU budget the project set out to hit.

## Per-tensor breakdown

`rel_error` is `||(W-Q)X|| / ||WX||` against calibration activations. `attenuation` is
`||QX|| / ||WX||` — 1.0 is neutral, below 1.0 means the layer is losing gain.
`H flatness` is `std(diag H)/mean(diag H)`, the outlier metric the rotation targets.

| tensor | rel_error | attenuation | sparsity | H flatness |
|---|---:|---:|---:|---:|
| `attn_k` | 0.1372 | 0.9903 | 0.4519 | 0.5073 |
| `attn_q` | 0.1638 | 0.9859 | 0.4529 | 0.5073 |
| `ffn_gate` | 0.1730 | 0.9837 | 0.4549 | 0.5971 |
| `attn_output` | 0.1915 | 0.9812 | 0.4533 | 0.7214 |
| `ffn_down` | 0.2257 | 0.9744 | 0.4663 | **10.5866** |
| `attn_v` | 0.2687 | 0.9626 | 0.4529 | 0.5073 |
| `ffn_up` | 0.2924 | 0.9552 | 0.4554 | 0.5971 |
| **mean** | **0.2075** | **0.9762** | 0.4553 | 2.0034 |

## Rotation is exactly a no-op

Verified on the real checkpoint (`forge verify-rotation`):

| | |
|---|---|
| max \|logit diff\| | 1.04e-3 |
| relative to \|logit\| | 3.15e-5 |
| wikitext2 ppl before / after | 10.3976 / 10.3976 |
| mean weight kurtosis | 6.50 → **4.39** (3.0 = Gaussian) |

## Attenuation, and what fixes it

Ternary reconstruction is an orthogonal projection onto the codebook, so `<W,Q> = ||Q||²`
and every layer comes out short. Measured:

| | attenuation |
|---|---:|
| RTN, per layer | 0.9000 |
| + per-channel least-squares rescale | 0.9012 (**alpha ~ 1.00 — a near no-op**) |
| GPTQ | 0.9486 |
| **GPTQ + sequential propagation** | **0.9762** |

Scaling a projection up only raises its MSE, so this cannot be fixed inside the layer.
Sequential propagation fixes it by letting each layer absorb its predecessors' shortfall.
`quant/rescale.py` is kept for the diagnostic, not the correction.

---

## Size and speed

Sizes from llama.cpp's own quantizer on the same checkpoint:

| build | size | vs F16 |
|---|---:|---:|
| F16 | 2.88 GiB | 1.0x |
| Q4_K_M | 934.7 MiB | 3.2x |
| **TQ2_0** | **505.3 MiB** | **5.8x** |

`llama-bench`, M5 Max, Metal backend, 5 repetitions:

| build | pp512 (t/s) | tg128 (t/s) |
|---|---:|---:|
| F16 | 10994 ± 355 | 132.7 ± 9.4 |
| Q4_K_M | 11357 ± 253 | 269.2 ± 7.4 |
| **TQ2_0** | **11963 ± 19** | **288.7 ± 9.5** |

**Honest reading of the speed numbers.** TQ2_0 decodes 2.2x faster than F16 but only
**7% faster than Q4_K_M**, despite being 1.85x smaller. At 0.5 GB on a machine with M5 Max
bandwidth, decode is not bandwidth-bound — it is latency- and kernel-bound, so the memory
saving does not convert into proportional throughput. The speed argument for ternary is
therefore about *memory-constrained* deployment (phones, base M-series, fitting a larger
model in the same RAM), not about raw throughput on a workstation. Expect a better ratio
at 7B, where the working set is large enough for bandwidth to dominate again.

Note these use llama.cpp's own amax quantizer, so their *quality* is not FORGE's; they are
here to measure the runtime, which is independent of how the scales were chosen.

---

## Where the remaining error is

`ffn_down`'s Hessian flatness is **10.59** against ~0.51 everywhere else — a 20x outlier.
That is the one layer whose input the fusable rotations cannot reach: it reads the SwiGLU
output, which is exactly where an online Hadamard (R4) would go. This is a measured,
localized argument for the deferred tier-2 work, not a speculative one.

Two candidate next steps, in order of expected value:

1. **Online R4 Hadamard before `down_proj`.** Directly targets the measured outlier. Costs
   a runtime graph change, so the output would no longer load on stock llama.cpp without a
   patch — the central trade-off of the project.
2. **Rank-r FP16 residual** (`W ~ s*T + AB^T`). Large accuracy recovery, but breaks the
   stock-GGUF property and moves storage to a sidecar format.

The 1.5B is also the hardest size to quantize — small models carry far less redundancy than
7B. Running the 7B before drawing conclusions about the ceiling is worthwhile.
