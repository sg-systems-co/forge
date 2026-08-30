# FORGE results

Training-free ternary PTQ. All numbers reproducible from this repository; see
[Reproducing](#reproducing).

---

## v1 shipping configuration — Qwen2.5-Coder-7B

Six of the seven transformer linears at `TQ2_0`; `ffn_down` retained at `Q6_K`. This is
ordinary per-tensor type mixing that stock `llama-quantize` performs, so the checkpoint is
a plain GGUF that **unmodified llama.cpp loads**. No online rotation, no custom graph, no
fork.

| build | size | wikitext2 ppl | vs FP16 | decode t/s | vs FP16 |
|---|---:|---:|---:|---:|---:|
| F16 | 15.24 GB | 6.2731 | 1.00x | 34.51 | 1.00x |
| Q4_K_M | 4.68 GB | 6.3727 | 1.02x | 101.43 | 2.94x |
| **FORGE mixed** | **3.51 GB** | **9.4150** | **1.50x** | **120.87** | **3.50x** |

* **4.3x smaller than F16**, **1.33x smaller than Q4_K_M**
* **3.5x faster decode than F16**, **1.19x faster than Q4_K_M**

Measured with `llama-perplexity -c 2048 --chunks 12` and `llama-bench -n 128 -r 5` on the
actual GGUF files, Apple M5 Max, Metal backend.

### ⚠ Read the caveats before quoting any of this

Four, in descending order of how much they should change your plans. The fourth is the one
that matters most and is **not** visible in the table above.

---

## Caveat 1 — this is no longer a 1.58-bit model

`ffn_down` is **29% of the transformer-linear parameters** (1.901 B of 6.525 B at 7B).
Retaining it at Q6_K puts the effective rate across those layers at **~2.8 bpw**, not the
2.0625 bpw of pure `TQ2_0`.

| `ffn_down` type | total size | vs FP16 | vs Q4_K_M | effective bpw |
|---|---:|---:|---:|---:|
| TQ2_0 (pure ternary) | 2.58 GB | 5.9x | 1.82x smaller | 2.06 |
| **Q6_K (shipped)** | **3.51 GB** | **4.3x** | **1.33x smaller** | **~2.8** |
| Q4_K | 3.16 GB | 4.8x | 1.49x smaller | ~2.8 |

Describing this as a "1.58-bit" or "ternary" checkpoint would be misleading. It is a
**mixed 2.06/6.56-bit checkpoint** whose weight-dominant layers are ternary.

Pure ternary at 7B reaches 16.4415 ppl (2.17x FP16) — outside the 9–14 target band.
Excluding `ffn_down` is what brought it inside.

## Caveat 2 — 1.50x perplexity degradation, against a near-lossless INT4

Q4_K_M costs **1.02x** perplexity. FORGE costs **1.50x**. Against INT4 specifically, FORGE
buys 1.33x size and 1.19x decode speed **for a real and much larger quality loss**.

The honest case for FORGE is *memory-constrained deployment* — fitting a model in RAM that
otherwise would not fit at all. It is not a free win over INT4, and on a machine where
Q4_K_M already fits, Q4_K_M is the better checkpoint.

## Caveat 3 — base model, no chat template

Qwen2.5-Coder-7B is a **base** model. `llama-cli` and `llama-completion` apply a chat
template by default, and a base model fed a chat template emits degenerate output
**regardless of quantization** — this briefly looked like an export bug during development.

```bash
llama-simple -m out/qwen7b-forge-mixed.gguf -n 100 "def fibonacci(n):"   # correct
llama-cli    -m out/qwen7b-forge-mixed.gguf -p "def fibonacci(n):"       # degenerate
```

Use a raw-completion path for base weights.

## Caveat 4 — **perplexity does not predict code quality here**

> **The mixed checkpoint reaches 9.4150 ppl (1.50x FP16) and still produces zero
> syntactically valid Python across 8 standard prompts.**

| | parses as Python | degenerate repetition |
|---|---:|---:|
| FP16 | **6 / 8** | 0 / 8 |
| **FORGE mixed** | **0 / 8** | **3 / 8** |

Identical prompts, identical decoding path, identical seed. FP16's two non-parsing samples
are truncated at the token budget, not broken; FORGE's failures are structural — dropped
indentation, unclosed parentheses, repetition loops. Sampling with a repetition penalty
does not rescue them. Full side-by-side transcripts: [SAMPLE_OUTPUTS.md](SAMPLE_OUTPUTS.md).

**Consequence: perplexity was the wrong acceptance metric for a code model.** A 9–14 ppl
band was met while the actual downstream capability was not. Any future gate should be
task-based (HumanEval pass@1, or at minimum syntactic validity) rather than perplexity
alone. This was flagged as a risk in the original project plan; it is now measured.

---

## How we got here — the ablation

Qwen2.5-Coder-7B, 32 calibration sequences, 32 windows, FORGE PyTorch harness.

| config | ppl | x FP16 | step gain | rel_error | attenuation | `ffn_down`/other flatness |
|---|---:|---:|---:|---:|---:|---:|
| FP16 baseline | 7.52 | 1.00x | — | — | 1.0000 | — |
| naive ternary (absmean) | 24,632,412 | 3.3e6x | — | 0.5622 | 0.5713 | 5.29 / 6.57 |
| naive ternary (optimal scale) | 52,292.70 | 6955x | 471x | 0.4252 | 0.8008 | 5.29 / 6.57 |
| + rotation | 1,359.19 | 181x | 38.5x | 0.3849 | 0.8874 | 5.29 / 0.41 (13x) |
| + GPTQ solver | 19.39 | 2.58x | **70.1x** | 0.2122 | 0.9728 | 5.29 / 0.41 (13x) |
| + sequential (full FORGE) | 17.83 | 2.37x | 1.1x | 0.2001 | 0.9767 | 5.53 / 0.46 (12x) |

The GPTQ solver is the dominant contributor at 7B. Sequential propagation adds only 1.1x
here against being essential at 1.5B, because GPTQ alone already reaches 0.973 attenuation
and there is little left for downstream layers to absorb — it costs 3x the forward passes
for an 8% gain, so it is a cost lever on larger models.

## Scale absorbs quantization error

| | 1.5B | 7B |
|---|---:|---:|
| FP16 ppl | 10.40 | 7.59 |
| FORGE pure-ternary ppl | 172.27 | 16.44 |
| **degradation** | **16.6x** | **2.17x** |

Identical code and hyperparameters. Parameter redundancy is worth ~7x in relative
degradation — the single largest effect measured in this project.

## The `ffn_down` outlier

| | 1.5B | 7B |
|---|---:|---:|
| `ffn_down` Hessian flatness | 10.59 | 5.42 |
| every other tensor | 0.51 | 0.44 |
| **outlier ratio** | **20.8x** | **12.3x** |

It dampens with scale but remains the worst-conditioned layer by an order of magnitude, and
is the one layer the fusable rotations cannot reach (it reads the SwiGLU output).

Unrotated ablation rows show `ffn_down` *below* average (5.29 vs 6.57). Not a
contradiction: before rotation everything is badly conditioned and nothing stands out. The
rotation flattens every reachable tensor to ~0.41 and leaves `ffn_down` where it was — it
**exposes** the outlier rather than causing it.

**Conditioning is not the whole story.** `ffn_up` has essentially the same reconstruction
error as `ffn_down` (0.2622 vs 0.2646) at **12x lower flatness**. An online Hadamard fixes
conditioning, so it would help `ffn_down` and do nothing for `ffn_up`. That caps the
expected payoff from a custom-graph branch.

## Two perplexity conventions, one ratio

`llama-perplexity` scores only the second half of each window (`const int first = n_ctx/2`,
`tools/perplexity/perplexity.cpp`). FORGE's PyTorch harness scores every token, which is
strictly harder. Absolutes differ; the **ratio agrees to three digits** — a useful
cross-check between independent implementations.

| | FP16 | FORGE mixed | ratio |
|---|---:|---:|---:|
| llama.cpp (second half of window) | 6.2731 | 9.4150 | **1.50x** |
| FORGE PyTorch (all tokens) | 7.5922 | 11.4022 | **1.50x** |

Quote the ratio across tools, never the absolute.

## Decode is bandwidth-bound at 7B, and was not at 1.5B

If decode were bandwidth-bound, implied `GB/s = size x tok/s` would be constant.

**7B:**

| build | weights | tok/s | implied GB/s | speedup | ideal | efficiency |
|---|---:|---:|---:|---:|---:|---:|
| F16 | 15.24 GB | 34.51 | 526 | 1.00x | 1.00x | 100% |
| Q4_K_M | 4.68 GB | 101.43 | 475 | 2.94x | 3.26x | 90% |
| FORGE mixed | 3.51 GB | 120.87 | 424 | 3.50x | 4.34x | 81% |

**1.5B:**

| build | weights | tok/s | implied GB/s | speedup | ideal | efficiency |
|---|---:|---:|---:|---:|---:|---:|
| F16 | 3.09 GB | 141.6 | 438 | 1.00x | 1.00x | 100% |
| Q4_K_M | 0.98 GB | 291.6 | 286 | 2.06x | 3.16x | 65% |
| TQ2_0 | 0.53 GB | 327.7 | 174 | 2.31x | 5.84x | 40% |

At 1.5B the working set is too small for bandwidth to dominate and a fixed per-token cost
takes over. At 7B the format recovers **81%** of its ideal speedup against 40% at 1.5B.

## Cost

| | 1.5B | 7B |
|---|---|---|
| Quantization | ~10 min | 37 min (pure), 33 min (excl. `ffn_down`) |
| Peak Hessian residency | 378 MB | 1744 MB |
| Peak process RSS | ~8 GB | ~44 GB (float32 pipeline) |
| Export (save + convert + quantize) | — | 60 s |

Excluding `ffn_down` also removes the 18944² Cholesky, ~80% of solver time: per-block
solve drops from 15.4 s to 3.1 s. All well inside the 1–4 hour single-consumer-GPU budget.

## Measurement conditions

Throughput is load-sensitive and an early benchmark set here was contaminated by an
unrelated process, reading 6–14% low **while still showing plausible ~3% error bars** —
steady contention produces tight bars just as an idle machine does. Tight error bars are
evidence of stable conditions, not of an idle one. All throughput figures above are the
mean of three independent runs on a confirmed-idle machine.

Long-running wall-clock (quantization time) was *not* meaningfully affected; it is
dominated by sustained work on the same device rather than by scheduling latency.

**Accuracy is load-independent** and this is verified, not assumed: `10.3999` and
`404729.5582` reproduce to every digit across runs under different load, and per-block
`rel_error` is bit-identical between runs.

## Reproducing

```bash
uv venv --python 3.12 && uv pip install -e ".[dev]"
git submodule update --init
cmake -S llamacpp/llama.cpp -B llamacpp/llama.cpp/build -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SHARED_LIBS=ON -DGGML_METAL=ON
cmake --build llamacpp/llama.cpp/build -j

# the shipping checkpoint
python -m forge.cli quantize --model Qwen/Qwen2.5-Coder-7B \
    --nsamples 128 --factorization float32_gpu \
    --exclude ffn_down --exclude-type q6_K \
    --export out/qwen7b-forge-mixed.gguf

# the ablation grid
python bench/ablation.py --model Qwen/Qwen2.5-Coder-7B --nsamples 32 --limit 32
```

Every checkpoint ships a `.forge.json` sidecar with the full reproducibility
specification. Two runs with the same seed produce byte-identical output.
