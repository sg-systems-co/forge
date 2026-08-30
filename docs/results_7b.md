# Qwen2.5-Coder-7B — the v1 target

Apple M5 Max, MPS, float32 pipeline. 128 calibration sequences x 2048 tokens (wikitext2,
seed 0). Cholesky in float32 on GPU.

## Verdict: **target hit, with per-tensor precision and no runtime change**

The shipping configuration ternarizes six of the seven transformer linears and keeps
`ffn_down` at Q6_K. That is ordinary per-tensor type mixing — stock `llama-quantize` takes
`--tensor-type ffn_down=q6_K` — so the checkpoint is a plain GGUF that unmodified
llama.cpp loads. **No online rotation, no custom graph, no fork.**

| build | size | wikitext2 ppl | vs FP16 | decode t/s |
|---|---:|---:|---:|---:|
| F16 | 15.24 GB | 6.2731 | 1.00x | 34.51 |
| Q4_K_M | 4.68 GB | 6.3727 | 1.02x | 101.43 |
| **FORGE mixed** | **3.51 GB** | **9.4150** | **1.50x** | **120.87** |

* **4.3x smaller than F16**, and **1.33x smaller than Q4_K_M**
* **3.5x faster decode than F16**, **1.19x faster than Q4_K_M**

## Two perplexity conventions, one ratio

`llama-perplexity` scores only the second half of each window (`const int first = n_ctx/2`
in `tools/perplexity/perplexity.cpp`), where context is richest. FORGE's PyTorch harness
scores every token, which is strictly harder. Absolute numbers therefore differ — but the
**degradation ratio agrees to three digits**, which is a useful cross-check between two
independent implementations:

| | FP16 | FORGE mixed | ratio |
|---|---:|---:|---:|
| llama.cpp (second half of window) | 6.2731 | 9.4150 | **1.50x** |
| FORGE PyTorch (all tokens) | 7.5922 | 11.4022 | **1.50x** |

Quote the ratio, not the absolute, when comparing across tools.

## The ablation

32 calibration sequences, 32 windows, PyTorch convention (so not comparable to the table
above in absolute terms; the ordering and step gains are what matter).

| config | ppl | x FP16 | step gain | rel_error | attenuation | `ffn_down`/other flatness |
|---|---:|---:|---:|---:|---:|---:|
| FP16 baseline | 7.52 | 1.00x | — | — | 1.0000 | — |
| naive ternary (absmean) | 24,632,412 | 3.3e6x | — | 0.5622 | 0.5713 | 5.29 / 6.57 |
| naive ternary (optimal scale) | 52,292.70 | 6955x | 471x | 0.4252 | 0.8008 | 5.29 / 6.57 |
| + rotation | 1,359.19 | 181x | 38.5x | 0.3849 | 0.8874 | 5.29 / 0.41 (13x) |
| + GPTQ solver | 19.39 | 2.58x | **70.1x** | 0.2122 | 0.9728 | 5.29 / 0.41 (13x) |
| + sequential (full FORGE) | 17.83 | 2.37x | 1.1x | 0.2001 | 0.9767 | 5.53 / 0.46 (12x) |

At full fidelity (128 sequences, full split) the all-ternary configuration reaches
**16.4415** — a 2.17x degradation, outside the 9–14 band. Excluding `ffn_down` brings that
to 11.4022 (1.50x), inside it.

## Scale absorbs quantization error

| | 1.5B | 7B |
|---|---:|---:|
| FP16 ppl | 10.40 | 7.59 |
| FORGE all-ternary ppl | 172.27 | 16.44 |
| **degradation** | **16.6x** | **2.17x** |

Same code, same hyperparameters. Parameter redundancy is worth ~7x in relative degradation.

## The `ffn_down` outlier

It dampens with scale but does not disappear, and remains the single worst layer:

| | 1.5B | 7B |
|---|---:|---:|
| `ffn_down` flatness | 10.59 | 5.42 |
| every other tensor | 0.51 | 0.44 |
| **outlier ratio** | **20.8x** | **12.3x** |

The unrotated ablation rows show `ffn_down` *below* average (5.29 vs 6.57). That is not a
contradiction: before rotation everything is badly conditioned and nothing stands out. The
rotation flattens every reachable tensor to ~0.41 and leaves `ffn_down` where it was. The
rotation **exposes** the outlier rather than causing it.

**Conditioning is not the whole story, which matters for the R4 decision.** `ffn_up` has
essentially the same reconstruction error as `ffn_down` (0.2622 vs 0.2646) at **12x lower
flatness**. An online Hadamard fixes conditioning, so it would help `ffn_down` and do
nothing for `ffn_up`. Per-tensor precision addresses both, and costs no graph change.

## Cost

| | |
|---|---|
| Quantization | 2220 s (37 min) all-ternary; 1958 s excluding `ffn_down` |
| Peak Hessian residency | 1744 MB |
| Peak process RSS | ~44 GB (7B in float32 + Hessians + activation buffers) |
| Export (save + convert + quantize) | 60 s |

Excluding `ffn_down` also removes the 18944^2 Cholesky, which was ~80% of solver time:
per-block quantization drops from 15.4 s to 3.1 s.

## Honest caveats

1. **This is no longer a 1.58-bit model.** `ffn_down` is 29% of the transformer-linear
   parameters, so at Q6_K the effective rate across those layers is ~2.8 bpw, not 2.06.
2. **Q4_K_M is nearly lossless (1.02x) while FORGE is 1.50x.** Against INT4 specifically,
   FORGE buys 1.33x size and 1.19x decode speed for a real quality cost. The case is
   memory-constrained deployment, not a free win over INT4.
3. `llama-cli` applies a chat template by default, and Qwen2.5-Coder-7B is a **base**
   model; that combination emits degenerate output regardless of quantization. Use
   `llama-simple` (or a completion endpoint) to sanity-check base-model checkpoints. The
   FORGE checkpoint generates correct, on-task Python in raw completion mode.

## Decode is bandwidth-bound at 7B, unlike 1.5B

| build | weights | tok/s | implied GB/s | speedup | if bandwidth-bound | efficiency |
|---|---:|---:|---:|---:|---:|---:|
| F16 | 15.24 GB | 34.51 | 526 | 1.00x | 1.00x | 100% |
| Q4_K_M | 4.68 GB | 101.43 | 475 | 2.94x | 3.26x | 90% |
| FORGE mixed | 3.51 GB | 120.87 | 424 | 3.50x | 4.34x | 81% |

Implied bandwidth is far more uniform than at 1.5B (438 / 286 / 174, efficiency 100 / 65 /
40%). This confirms the prediction in `results.md`: at 1.5B the working set was too small
for bandwidth to dominate, and at 7B it is. The ternary format recovers 81% of its ideal
speedup here against 40% at 1.5B.
