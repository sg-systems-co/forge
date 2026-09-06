# Falcon-H1-7B-Instruct v2 — the shipping NLP artifact

Fixes the factual degradation in v1 by preserving `ssm_out`, the tensor whose Hessian
flatness outlier the earlier run had already flagged.

## Artifact

```
/Users/sebastiangrebe/Documents/Git/forge/out/falcon-h1-7b-forge-v2.gguf
```

| | |
|---|---|
| Size | **3.48 GB** (from 15.18 GB F16) |
| llama.cpp arch | `falcon-h1` (Mamba-2 SSD + attention hybrid) |
| Loads on | stock llama.cpp, unmodified |
| Sidecar | `out/falcon-h1-7b-forge-v2.forge.json` |

## v1 vs v2

| build | size | wikitext2 ppl | vs F16 | **factual recall** | decode t/s |
|---|---:|---:|---:|---:|---:|
| F16 | 15.18 GB | 6.5875 | 1.00x | **12/12** | 29.1 |
| v1 (`ssm_out` ternary, 32 calib) | 3.25 GB | 10.7590 | 1.63x | **9/12** | ~63 |
| **v2 (`ssm_out` Q6_K, 128 calib)** | **3.48 GB** | **9.1633** | **1.39x** | **12/12** | ~63 |

**+0.23 GB buys back full factual recall and 15% of the perplexity gap.** Still 4.4x
smaller than F16 and ~2.2x faster to decode.

### What the +0.23 GB fixed

| probe | v1 | v2 |
|---|---|---|
| main ingredients in a carrot cake | `"carrot cake mix, carrot cake mix, …"` | `"Carrots, sugar, flour, eggs, butter, cinnamon…"` |
| capital of Australia | **`"Sydney"`** | `"Canberra"` |
| boiling point of water | **`"99°C"`** | `"100°C"` |

Only the first was a visible repetition loop. The other two were **fluent and confidently
wrong**, delivered with supporting explanations — the failure mode nothing in the output
signals, and the one a coherence metric scores as a pass.

## Tensor map

| tensor | type | note |
|---|---|---|
| `attn_{q,k,v,output}`, `ffn_{gate,up}`, `ssm_in` | `TQ2_0` | ternary, 2.06 bpw |
| **`ffn_down`** | `Q6_K` | post-SwiGLU writer, unreachable by fusable rotation |
| **`ssm_out`** | `Q6_K` | post-SSD writer, same position — **new in v2** |
| `ssm_a`, `ssm_d`, `ssm_dt.bias`, `ssm_conv1d`, `ssm_norm` | `F32` | SSM_SCAN inputs, untouched |
| `token_embd` / `output` | `Q4_K` / `Q6_K` | llama-quantize defaults |

Mean Hessian flatness across the *quantized* set drops **1.1881 → 0.3818** once `ssm_out`
is removed from it — the 6.78 outlier is simply no longer being crushed.

## Honest caveat: the experiment is confounded

v2 changed **two** things at once — preserving `ssm_out` *and* raising calibration from 32
to 128 sequences. The results cannot attribute the recovery between them. The flatness
evidence (6.78 against ~0.33) makes `ssm_out` the likelier cause, and the v1 failures were
knowledge-shaped rather than fluency-shaped, but this is inference and not measurement.

A third run — `ssm_out` preserved at 32 sequences — would separate them, at ~1 hour. Worth
doing before generalizing the rule to other architectures; not worth blocking this handoff.

## What HELIX needs to know

Unchanged from v1, and the important property still holds: **quantization touches the
projections around the scan, never the scan.** Every tensor `SSM_SCAN` consumes — `ssm_a`,
`ssm_d`, `ssm_dt.bias`, `ssm_conv1d`, `ssm_norm` — is untouched F32, so the SSD kernel sees
exactly the numerics it was built for.

In v2 the projection *out of* the scan (`ssm_out`) is Q6_K rather than ternary; `ssm_in`
remains TQ2_0. Confirmed Mamba-2, not Mamba-1: `falcon-h1` reaches `ggml_ssm_scan` through
`mamba-base.cpp:262` (the SSD path with `n_group`/`head_dim`/`n_head` views), not the
Mamba-1 site at `:123`. `mamba_n_groups = 1`.

## Cost

| | |
|---|---|
| Quantization | 15,459 s (4h 18m), 44 blocks at ~334 s |
| Peak Hessian residency | 878 MB |
| Export | 51 s |
| Calibration | 128 sequences x 2048 tokens |

The Mamba-2 scan has no fast MPS kernel, and sequential mode runs three forward passes per
block, which is essentially the entire wall-clock. Disabling sequential propagation would
cut this to roughly 1.4 h; it contributed only 1.1x on Qwen-7B.

## Measurement conditions

Decode throughput was re-measured after the machine settled: the first reading (78.68 t/s)
was taken while still warm from the 4-hour run and read high with wide error bars. The
figures above are the mean of two independent runs agreeing to ~0.2%. Perplexity and
factual recall are deterministic and load-independent.

## Reproducing

```bash
python -m forge.cli quantize --model tiiuae/Falcon-H1-7B-Instruct \
    --nsamples 128 --limit 32 --factorization float32_gpu \
    --exclude ffn_down ssm_out --exclude-type q6_K \
    --export out/falcon-h1-7b-forge-v2.gguf

PYTHONPATH=. python bench/factual_probe.py out/falcon-h1-7b-forge-v2.gguf
```
