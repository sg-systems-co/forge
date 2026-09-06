# Falcon-H1-7B-Instruct — the NLP payload

Target for the NLP pivot and the HELIX SSM_SCAN handoff. Produced by the **unmodified
v1.0.0-mixed-tq2 solver** with a new architecture adapter; `quant/` and `pack/` are
byte-identical to the frozen release.

## Artifact

```
/Users/sebastiangrebe/Documents/Git/forge/out/falcon-h1-7b-forge-mixed.gguf
```

| | |
|---|---|
| Size | **3.25 GB** (from 15.18 GB F16) |
| llama.cpp arch | `falcon-h1` (Mamba-2 SSD + attention, parallel hybrid) |
| Loads on | stock llama.cpp, unmodified |
| Sidecar | `out/falcon-h1-7b-forge-mixed.forge.json` |

## Results

| build | size | wikitext2 ppl | vs F16 | decode t/s | vs F16 |
|---|---:|---:|---:|---:|---:|
| F16 | 15.18 GB | 6.5875 | 1.00x | 29.88 | 1.00x |
| **FORGE mixed** | **3.25 GB** | **10.7590** | **1.63x** | **78.68** | **2.63x** |

**4.7x smaller, 2.6x faster to decode, 1.63x perplexity cost.**

Perplexity via `llama-perplexity -c 2048 --chunks 12` (scores the second half of each
window). FORGE's own all-token harness reported 13.3857 during the run; see the convention
note in `results.md` — compare ratios across tools, never absolutes.

## The NLP pivot worked

| | coherent output | mean distinct-3 | duplicated sentences |
|---|---:|---:|---:|
| Qwen2.5-Coder-7B (code) | **0 / 8** valid Python | — | 3/8 degenerate |
| Falcon-H1-7B F16 (NLP) | 8 / 8 | 0.940 | 0.00 |
| **Falcon-H1-7B FORGE (NLP)** | **8 / 8** | 0.661 | 0.11 |

Semantic content survives ternary rounding where structural syntax does not. The measurable
regression is **repetition, not incoherence** — a graceful failure mode — and most of it is
a greedy-decoding artifact that largely disappears in instruct mode with a repetition
penalty. Transcripts: [SAMPLE_OUTPUTS_NLP.md](SAMPLE_OUTPUTS_NLP.md).

## Per-tensor breakdown

| tensor | rel_error | attenuation | sparsity | H flatness |
|---|---:|---:|---:|---:|
| `attn_k` | 0.1694 | 0.9846 | 0.4551 | 0.3313 |
| `attn_q` | 0.2148 | 0.9756 | 0.4561 | 0.3313 |
| `ffn_gate` | 0.2257 | 0.9720 | 0.4558 | 0.3397 |
| `ssm_in` | 0.2442 | 0.9681 | 0.4562 | 0.3313 |
| `attn_output` | 0.2737 | 0.9587 | 0.4467 | 0.7237 |
| `attn_v` | 0.2988 | 0.9521 | 0.4557 | 0.3313 |
| **`ssm_out`** | 0.3243 | 0.9459 | 0.4604 | **6.7764** |
| `ffn_up` | 0.3711 | 0.9262 | 0.4569 | 0.3397 |
| `ffn_down` | — kept at Q6_K — | | | |
| **mean** | **0.2653** | **0.9604** | 0.4554 | 1.1881 |

### The structural prediction held

Before the run, the adapter documented: *"`ssm_out` reads the post-SSD activation, which no
fusable rotation can reach — the position `ffn_down` occupies in a transformer. Expect it to
be the outlier."*

`ssm_out` flatness is **6.78 against ~0.33 for every other reachable tensor — a 20x
outlier**, the same signature `ffn_down` showed on Qwen (12–20x). The rotation flattens
everything it can reach and leaves the post-SSD writer behind.

**This is the obvious next lever:** `ssm_out` is only 9.44M params per layer (6% of the
block) against `ffn_down`'s 37.75M. Keeping it at Q6_K too would cost roughly 0.3 GB and
should recover a meaningful part of the 1.63x — much cheaper than it was on Qwen.

## What HELIX needs to know

The quantization touches the projections **around** the scan, never the scan itself. Every
tensor `SSM_SCAN` consumes is untouched F32:

| tensor | type | note |
|---|---|---|
| `ssm_a` (A_log) | **F32** | exponentiated; drives recurrence stability |
| `ssm_d` (D) | **F32** | skip connection |
| `ssm_dt.bias` | **F32** | goes through softplus |
| `ssm_conv1d.{weight,bias}` | **F32** | llama.cpp refuses to quantize these anyway |
| `ssm_norm.weight` | **F32** | in-mixer gated RMSNorm |
| `ssm_in.weight` | TQ2_0 | projection *into* the scan |
| `ssm_out.weight` | TQ2_0 | projection *out of* the scan |

So the SSD kernel sees exactly the numerics it was built for. The win is that the
surrounding projections are 2.06 bpw, which is where the memory goes.

Confirmed Mamba-2, not Mamba-1: `falcon-h1` reaches `ggml_ssm_scan` via
`mamba-base.cpp:262` (the SSD path carrying `n_group`/`head_dim`/`n_head` views), not the
Mamba-1 site at `:123`. `mamba_n_groups = 1`.

## Cost

| | |
|---|---|
| Quantization | 3688 s (61 min), 44 blocks |
| Peak Hessian residency | 878 MB |
| Export | 44 s |
| Calibration | **32** sequences x 2048 tokens (not 128 — see caveat) |

## Caveats

1. **Calibration was 32 sequences, not the 128 used for Qwen.** The Mamba-2 scan has no
   fast kernel on MPS, costing ~83 s/block in sequential mode; 128 would have taken ~6.6
   hours. This is the most likely reason the ratio is 1.63x here against 1.50x on Qwen, and
   it is the first thing to try before blaming the architecture.
2. **Not a 1.58-bit model.** `ffn_down` at Q6_K plus F32 SSM internals put the effective
   rate above 2.06 bpw.
3. **Repetition is elevated** under greedy decoding (distinct-3 0.66 vs 0.94). Use
   sampling with a repetition penalty; this is an instruct model, so the chat template is
   correct here (unlike the Qwen base model).

## Reproducing

```bash
python -m forge.cli quantize --model tiiuae/Falcon-H1-7B-Instruct \
    --nsamples 32 --limit 32 --factorization float32_gpu \
    --exclude ffn_down --exclude-type q6_K \
    --export out/falcon-h1-7b-forge-mixed.gguf
```
