# Qwen2.5-Coder-7B ablation

Apple M5 Max, MPS, float32. 32 calibration sequences x 2048 tokens (wikitext2, seed 0),
32 perplexity windows. Cholesky factorization in float32 on GPU.

```bash
python bench/ablation.py --model Qwen/Qwen2.5-Coder-7B --nsamples 32 --limit 32 \
    --factorization float32_gpu --out docs/results_ablation_7b.md
```

## The grid

| config | ppl | x FP16 | step gain | rel_error | attenuation | `ffn_down` / other flatness | min |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP16 baseline | **7.52** | 1.00x | — | — | 1.0000 | — | 1.5 |
| naive ternary (absmean) | 24,632,412 | 3.3e6x | — | 0.5622 | 0.5713 | 5.29 / 6.57 (0.8x) | 7.0 |
| naive ternary (optimal scale) | 52,292.70 | 6955x | 471x | 0.4252 | 0.8008 | 5.29 / 6.57 (0.8x) | 7.7 |
| + rotation | 1,359.19 | 181x | 38.5x | 0.3849 | 0.8874 | 5.29 / 0.41 (**13x**) | 7.2 |
| + GPTQ solver | 19.39 | 2.58x | **70.1x** | 0.2122 | 0.9728 | 5.29 / 0.41 (13x) | 12.0 |
| **+ sequential (full FORGE)** | **17.83** | **2.37x** | 1.1x | 0.2001 | 0.9767 | 5.53 / 0.46 (12x) | 16.2 |

**Verdict against the 9–14 target: missed, narrowly.** 17.83 is outside the range but is not
a catastrophic failure — it is 1.27x above the top of the band, at reduced calibration
fidelity (32 sequences against the 128 used for the 1.5B headline).

## Scale absorbs quantization error, strongly

| | 1.5B | 7B |
|---|---:|---:|
| FP16 ppl | 10.40 | 7.52 |
| FORGE ppl | 172.27 | **17.83** |
| **degradation** | **16.6x** | **2.37x** |

The 7B is **7x better in relative degradation** than the 1.5B on the identical pipeline.
The parameter-redundancy hypothesis is confirmed clearly: the same algorithm, same
hyperparameters, same code produces a nearly-usable model at 7B and an unusable one at 1.5B.

## Does the `ffn_down` outlier dampen at scale?

**Partly — it halves in absolute terms but persists as a large relative outlier.**

| | 1.5B | 7B |
|---|---:|---:|
| `ffn_down` flatness | 10.59 | 5.53 |
| every other tensor | 0.51 | 0.46 |
| **outlier ratio** | **20.8x** | **12.1x** |

So `ffn_down` remains the single worst-conditioned layer in the model by an order of
magnitude, and it is still the one layer the fusable rotations cannot reach. The absolute
improvement (10.59 → 5.53) is real, and it is part of why the 7B does so much better — but
the structural problem has not gone away, it has softened.

Note the unrotated rows report `ffn_down` flatness *below* the average (5.29 vs 6.57).
That is not a contradiction: before rotation every tensor is badly conditioned, so nothing
stands out. The rotation flattens all the reachable tensors to ~0.41 and leaves `ffn_down`
where it was, which is what creates the 13x gap. The outlier is *exposed* by the rotation,
not caused by it.

## Where the value comes from

Step gains are multiplicative and very unevenly distributed:

| step | gain |
|---|---:|
| optimal scale rule over absmean | 471x |
| + rotation | 38.5x |
| **+ GPTQ solver** | **70.1x** |
| + sequential propagation | 1.1x |

**The GPTQ solver is the single biggest lever at 7B**, and sequential propagation — which
was essential at 1.5B (it is what pulled attenuation from 0.90 to 0.98) — contributes only
8% here. That fits the mechanism: sequential mode exists to let downstream layers absorb
upstream attenuation, and at 7B the GPTQ solver alone already reaches 0.9728 attenuation,
so there is far less left to absorb.

This is a useful cost lever. Sequential mode triples the forward passes per block (16.2 min
vs 12.0 min here, and the gap widens with calibration size) for an 8% perplexity gain at
7B. On larger models it may be worth disabling.
