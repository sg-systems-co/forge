# FORGE

Training-free ternary (1.58-bit) post-training quantization for open-weight LLMs.

FORGE converts FP16/BF16 checkpoints into ternary {-1, 0, 1} weights using calibration-based
PTQ — no continued pretraining. It runs on a single consumer GPU or an Apple Silicon laptop
in 1–4 hours for a 7B model.

Output is a **stock GGUF** built on llama.cpp's existing `TQ2_0` block type, so converted
checkpoints load on unmodified llama.cpp. No custom kernel, no forked runtime, no new type.

---

## Results at a glance

| model | payload | size | ppl vs FP16 | downstream | verdict |
|---|---|---:|---:|---|---|
| Qwen2.5-Coder-7B | code | 3.51 GB (4.3x) | 1.50x | **0/8 valid Python** | ✗ unusable |
| Falcon-H1-7B-Instruct v1 | NLP | 3.25 GB (4.7x) | 1.63x | 9/12 factual recall | ✗ hallucinates |
| **Falcon-H1-7B-Instruct v2** | NLP | **3.48 GB (4.4x)** | **1.39x** | **12/12 factual recall** | ✓ **ships** |

The engine works. **Whether the result is usable depends entirely on the payload**, and
perplexity does not tell you which case you are in — see [The two big lessons](#the-two-big-lessons).

## Quick start

```bash
uv venv --python 3.12 && uv pip install -e ".[dev]"
git submodule update --init
cmake -S llamacpp/llama.cpp -B llamacpp/llama.cpp/build -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SHARED_LIBS=ON -DGGML_METAL=ON
cmake --build llamacpp/llama.cpp/build -j

# the shipping NLP checkpoint
python -m forge.cli quantize --model tiiuae/Falcon-H1-7B-Instruct \
    --nsamples 128 --factorization float32_gpu \
    --exclude ffn_down ssm_out --exclude-type q6_K \
    --export out/falcon-h1-7b-forge-v2.gguf

# check it actually knows things
PYTHONPATH=. python bench/factual_probe.py out/falcon-h1-7b-forge-v2.gguf
```

## How it works

Three stages, each independently ablatable via `bench/ablation.py`.

**1. Rotate.** Fuse randomized Hadamard rotations into the weights so outlier channels stop
forcing a large block scale. Every rotation FORGE uses is **fusable offline** — a global
residual-stream rotation R1 and a per-head value rotation R3 — so there is zero runtime
overhead and no graph change. QuaRot's online Hadamards exist to tame *activation* outliers;
a weight-only method does not need them. Verified as a mathematical no-op: max |logit diff|
1.04e-3 on Qwen, 2.9e-06 on Falcon-H1.

Non-power-of-two widths are handled with Paley I/II constructions plus a Kronecker search
(1536 = 128x12, 3584 = 128x28, 3072 = 256x12, 18944 = 128x148).

**2. Solve.** GPTQ recursion over a ternary codebook against a calibration Hessian, with the
per-256-block scale refit on already-compensated weights. The ternary scale itself is
**exactly optimal** in O(n log n) — sort |w|, prefix-sum, `argmax prefix[k]²/k` — verified
against exhaustive 3ⁿ enumeration.

**3. Propagate.** Solve each layer against the activations the *quantized* model actually
produces. Ternary reconstruction is an orthogonal projection, so every layer attenuates to
~0.90 and that compounds over depth; it cannot be fixed inside a layer (the least-squares
per-channel gain comes out at 1.00) but downstream layers absorb it. Measured: 0.90 → 0.976.

Full derivations in [docs/algorithm.md](docs/algorithm.md); the storage layout is in
[docs/format.md](docs/format.md).

## The two big lessons

**Perplexity does not predict downstream capability.** Qwen2.5-Coder hit its 9–14 perplexity
band at 1.50x FP16 and produced **zero** syntactically valid Python across 8 prompts. A
perplexity gate passed a model that could not do its job.

**Neither does coherence.** After pivoting to NLP, an 8/8 "coherent English" audit passed a
model that answered *"a carrot cake is made of carrot cake mix, carrot cake mix, carrot cake
mix"* and named **Sydney** the capital of Australia. The audit used generative prompts
(summarize, draft, explain) where fluency is most of the task, and never probed world
knowledge.

Both times the gate measured the wrong thing, and the second failure mode is the worse one:
*fluent and confidently false*, with nothing in the output to signal it.

`bench/factual_probe.py` exists because of this. It asks 12 questions with keyword-checkable
answers, and it is what separated v1 (9/12) from v2 (12/12) when perplexity moved only
10%. **Gate on the task, not on a corpus average.**

## What survives ternary, and what does not

Consistently across every architecture tested, the layers that break are the **writers whose
input no fusable rotation can reach**:

| architecture | unreachable writer | Hessian flatness vs peers |
|---|---|---|
| Qwen2.5 (transformer) | `ffn_down` (post-SwiGLU) | 10.59 vs 0.51 — **20x** |
| Falcon-H1 (hybrid) | `ffn_down`, `ssm_out` (post-SSD) | 6.78 vs 0.33 — **20x** |

The fix is per-tensor precision, not a custom graph: keep those at Q6_K via stock
`llama-quantize --tensor-type`. On Falcon-H1 that costs +0.23 GB and buys back **full
factual recall**. This keeps zero-friction distribution — the alternative (an online R4
Hadamard) would require a forked runtime.

Note the honest consequence: these are **not 1.58-bit checkpoints**. With `ffn_down` and
`ssm_out` retained, the effective rate is ~2.8 bpw.

## Architectures

| family | adapter | status |
|---|---|---|
| Qwen2 / Qwen3 / Llama-3 | `forge/models/qwen2.py` | tested at 1.5B and 7B |
| Falcon-H1 (Mamba-2 SSD + attention) | `forge/models/falcon_h1.py` | **shipping target** |
| Mamba-1 / Falcon-Mamba | `forge/models/mamba.py` | adapter verified, not benchmarked |

Adding an architecture means writing one adapter — the solver and packing are untouched.
Zamba2 is **not** supportable: llama.cpp has no `zamba2` architecture, so it would quantize
in PyTorch and then be unexportable.

## Layout

```
forge/
├── forge/
│   ├── models/      architecture adapters (qwen2, falcon_h1, mamba) + registry
│   ├── rotate/      Hadamard construction, offline rotation fusion
│   ├── quant/       ternary codebook, GPTQ solver, rescale, layer-by-layer driver
│   ├── calib/       calibration data, streaming Hessians, teacher/student capture
│   ├── pack/        TQ2_0 bit-packing, GGUF export
│   └── eval/        perplexity, reporting
├── bench/           ablation grid, factual-recall probe
├── tests/           282 tests
├── docs/            algorithm, format, and every measured result
└── artifacts/       reproducibility sidecars for published checkpoints
```

## Testing

```bash
pytest -q -m "not slow"     # 282 tests
pytest -q                   # adds the end-to-end GGUF export
```

The load-bearing tests:

* **`test_rotation_invariance.py`** — fusion must be a mathematical no-op. Catches every
  transpose, sign, GQA head-grouping and gain-folding bug, before quantization noise can
  hide it. Includes a negative control proving a wrong head grouping *would* be caught.
* **`test_pack_bitexact.py`** — FORGE's packer is byte-identical to ggml's on 34 cases
  including adversarial half-way rounding, where `lroundf` disagrees with numpy. This is
  what keeps stock llama.cpp loading the output.
* **`test_ternary_solver.py`** — the O(n log n) solver matches exhaustive 3ⁿ enumeration.
* **`test_e2e_tiny.py`** — full pipeline on a tiny model. Caught a real bug: sequential and
  non-sequential returned bit-identical results because the runner advanced both activation
  buffers through the post-quantization output.

## Reference results

| document | contents |
|---|---|
| [docs/results.md](docs/results.md) | Qwen2.5-Coder-7B: full comparison, caveats, bandwidth analysis |
| [docs/results_7b.md](docs/results_7b.md) | Qwen 7B ablation grid |
| [docs/results_falcon_h1.md](docs/results_falcon_h1.md) | Falcon-H1 v1 |
| [docs/results_falcon_h1_v2.md](docs/results_falcon_h1_v2.md) | **Falcon-H1 v2 — the shipping artifact** |
| [docs/SAMPLE_OUTPUTS.md](docs/SAMPLE_OUTPUTS.md) | Qwen code generations vs FP16 control |
| [docs/SAMPLE_OUTPUTS_NLP.md](docs/SAMPLE_OUTPUTS_NLP.md) | Falcon-H1 NLP generations vs FP16 control |

Every checkpoint ships a `.forge.json` sidecar recording the source model, calibration spec,
every solver knob, and the pinned llama.cpp revision. Two runs with the same seed produce
byte-identical output.

## Caveats worth reading before quoting anything

1. **Not 1.58-bit.** Retaining `ffn_down` and `ssm_out` puts the effective rate at ~2.8 bpw.
2. **Q4_K_M is near-lossless (1.02x) while FORGE is 1.39–1.50x.** Against INT4 specifically,
   FORGE buys ~1.3x size and ~1.2x decode speed for a real quality cost. The case is
   *memory-constrained deployment*, not a free win.
3. **Perplexity conventions differ.** `llama-perplexity` scores only the second half of each
   window (`first = n_ctx/2`); FORGE's harness scores every token. Compare *ratios* across
   tools, never absolutes — they agreed to three digits (1.50x) on Qwen.
4. **Base models need raw completion.** `llama-cli` applies a chat template, which makes a
   base model emit garbage regardless of quantization. Use `llama-simple` for base weights;
   instruct models are fine with the template.
5. **Throughput is load-sensitive.** An early benchmark here read 6–14% low while still
   showing plausible ~3% error bars — steady contention produces tight bars just like an idle
   machine. Always take the mean of independent runs on a confirmed-idle machine.

## License

MIT
