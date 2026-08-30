# FORGE algorithm

Three stages, applied in this order: **rotate**, **solve**, **propagate**. Each is
independently ablatable (`bench/ablation.py`) so its contribution is measured, not assumed.

---

## 1. Rotation

### Why

Ternary rounding has three levels to spend, so a single outlier channel forces a large
block scale and everything else collapses to zero. Rotating the weights into a basis where
they are closer to Gaussian spreads that mass out. Measured on Qwen2.5-Coder-1.5B, mean
weight kurtosis drops **6.50 → 4.39** (3.0 is Gaussian).

### Which rotations are fusable, and why that matters

QuaRot and SpinQuant use *online* Hadamard transforms, which force a runtime graph change.
Those exist mainly to tame **activation** outliers. FORGE v1 is weight-only — activations
stay in the runtime's native precision — so the rotation only has to Gaussianize the
weights, and the two rotations that do that are both fusable offline:

| | what it rotates | fusable | in v1 |
|---|---|---|---|
| **R1** | residual stream (`hidden_size`) | yes | yes |
| **R3** | per-head value subspace (`head_dim`) | yes | yes |
| R4 | `down_proj` input, after SwiGLU | no — needs an online Hadamard | deferred |
| R2 | q/k after RoPE | no — does not commute with RoPE | deferred |

**Consequence: v1 needs no runtime change at all**, and the exported GGUF loads on stock
llama.cpp.

### R1 — rotating the residual basis

RMSNorm commutes with an orthogonal `R` because `||R.T x|| == ||x||`. The learned gain `g`
does not, so it is folded into the norm's consumers first (`W <- W @ diag(g)`, scaling
*columns*, i.e. input channels) and the norm is set to all-ones. Then with `x~ = R.T x`:

| tensor | role | transform |
|---|---|---|
| `token_embd` (rows are token vectors) | writer | `E <- E @ R` |
| `q,k,v,gate,up` | reader | `W <- W @ R` |
| `o_proj`, `down_proj` | writer | `W <- R.T @ W` |
| `lm_head` | reader | `W <- W @ R` |

Biases need no transform: an input rotation does not touch them, and both writers are
bias-free in Qwen2 and Llama-3 (pinned by `tests/test_graph.py`).

`R = H_d diag(s) / sqrt(d)` with `s` a fixed-seed sign vector. The random signs are what
make this *incoherence processing* rather than a fixed basis change.

**Non-power-of-two dimensions.** Sylvester alone does not cover real model widths, so
`forge/rotate/hadamard.py` adds Paley I (`p = 3 mod 4`, order `p+1`) and Paley II
(`p = 1 mod 4`, order `2(p+1)`) over prime fields, then searches for `n = 2^a * m`:

| dimension | factorization | construction |
|---|---|---|
| 1536 (1.5B hidden) | 128 x 12 | Sylvester (x) Paley I, p=11 |
| 3584 (7B hidden) | 128 x 28 | Sylvester (x) Paley II, p=13 |
| 8960 (1.5B ffn) | 64 x 140 | Sylvester (x) Paley I, p=139 |
| 18944 (7B ffn) | 128 x 148 | Sylvester (x) Paley II, p=73 |

Orders with no construction fall back to a random orthogonal matrix from QR.

### R3 — the per-head value rotation

Attention output is linear in V, so a `head_dim` Hadamard applied after `v_proj` cancels
exactly when its inverse is applied before `o_proj`:

```
W_v <- blockdiag(H, n_kv_heads)     @ W_v      (and v_proj's bias, which is rotated too)
W_o <- W_o @ blockdiag(H, n_heads).T
```

Under GQA the two block-diagonals have **different repeat counts**, because KV heads are
repeated to `n_heads` before `o_proj`. That asymmetry is the most bug-prone line in
`fuse.py`, and `test_rotation_invariance.py` covers every grouping plus a negative control
that confirms a wrong repeat count would actually be caught.

### Weight tying

Folding the final norm's gain into `lm_head` makes it differ from `token_embd`, so a tied
model must be untied first. Qwen2.5 ties at 0.5B/1.5B and unties at 7B — which is the main
reason the 1.5B's compression ratio (4.3x) is materially worse than the 7B's (5.9x).

### Verification

Fusion is a mathematical no-op, so it is directly testable. On the real 1.5B checkpoint:
max `|logit diff|` **1.04e-3** (3.2e-5 relative), and wikitext2 perplexity identical to four
decimal places (10.3976 both before and after).

---

## 2. Ternary codebook

For a 256-weight block, minimize `||w - s*t||^2` over `s > 0`, `t in {-1,0,1}^256`.

Fix the support `S`; on it the optimal sign is `sign(w_i)`, so the objective is
`||w||^2 - 2s sum_S |w_i| + s^2 |S|`. Minimizing over `s`:

```
s*(S)  = (sum_{i in S} |w_i|) / |S|
gain(S) = (sum_{i in S} |w_i|)^2 / |S|
```

The gain depends on `S` only through its size and the magnitude mass it collects, so the
best support of a given size takes the largest magnitudes: **the optimum is a prefix of
`|w|` sorted descending**. Sort, prefix-sum, `argmax_k prefix[k]^2 / k`. Exactly optimal in
`O(n log n)` — verified against exhaustive `3^n` enumeration in `test_ternary_solver.py`.

This strictly beats BitNet's `s = mean(|w|)`: relative error 0.44 vs 0.53 on Gaussian
weights, and the resulting ~45% zero fraction matches the known TWN threshold of ~0.6 sigma.

---

## 3. Layer-wise solve (GPTQ recursion)

```
minimize_Q  ||(W - Q) X||_F^2  =  tr( (W-Q) H (W-Q)^T ),   H = 2/N sum x x^T
```

Quantize one input column at a time and push its rounding error onto the columns not yet
quantized, weighted by the inverse Hessian:

```
q_j       = clamp(round(w_j / s_g), -1, 1) * s_g
e_j       = (w_j - q_j) / Hinv[j, j]
W[:, j:] -= e_j (x) Hinv[j, j:]
```

Two FORGE-specific choices:

* **The block scale is refit every 256 columns, on the already-compensated weights.**
  TQ2_0 stores one fp16 scale per 256 weights — finer-grained than BitNet's per-tensor
  scale — so this costs nothing and is what makes that per-block scale earn its keep.
* **Act-order is deliberately omitted.** Permuting columns by `diag(H)` would scramble
  which weights share a block scale, and the permutation cannot be folded away for
  `q/k/v/gate/up` (their input is the shared residual stream with many consumers). The
  Hadamard rotation already flattens `diag(H)`, so there is little left to buy —
  measurable via `hessian_flatness`.

Damping is `lambda = 0.01 * mean(diag H)`, retried at 10x steps if the Cholesky fails. The
factorization runs on CPU in float64 (MPS has no float64 Cholesky, and this is the one step
where conditioning genuinely matters).

---

## 4. Cross-layer propagation

### The attenuation problem

Ternary reconstruction is an **orthogonal projection** onto the codebook: at the optimum
`<W, Q> = ||Q||^2`, so `||Q|| < ||W||`. Measured, each layer attenuates its output to about
**0.90** of the original. Over 28 blocks that compounds, and it is a large part of why naive
ternary destroys a model even though each individual layer looks only ~40% off.

### Why it cannot be fixed inside the layer

Scaling a projection back up only increases its MSE. The least-squares per-channel gain

```
alpha_r = (W H Q^T)_rr / (Q H Q^T)_rr
```

comes out at **~1.00** in practice, so `quant/rescale.py` is close to a no-op on its own.
This is a measured negative result, kept because it is the reason the next section exists.

### What does fix it

Solve each layer against the activations the **quantized** model actually produces. The
driver keeps two buffers — the teacher's activations `X_fp` and the student's `X_q` — and
hands the solver `X_q`. A layer fed already-attenuated inputs produces a larger
least-squares solution, absorbing its predecessors' shortfall instead of stacking on it.

Measured effect: mean attenuation recovers **0.90 → 0.978**.

The pass ordering in `calib/capture.py` is what makes this safe to do in place:

1. accumulate `H` by replaying the student inputs through the still-original weights;
2. advance the teacher buffer through the still-original weights — this must happen
   *before* the consumer mutates anything, because the originals are unrecoverable after;
3. hand control to the solver, which quantizes in place;
4. advance the student buffer through the now-quantized weights.

---

## Remaining error, localized

Hessian flatness `std(diag H) / mean(diag H)` after rotation:

| tensor | flatness |
|---|---|
| `attn_q`, `attn_k`, `attn_v`, `attn_output` | ~0.48 |
| `ffn_gate`, `ffn_up` | ~0.56 |
| **`ffn_down`** | **10.9** |

`down_proj` is the one layer whose input the fusable rotations cannot reach — it reads the
SwiGLU output, which is exactly where R4's online Hadamard would go. This is a concrete,
measured argument for the deferred tier-2 work rather than a speculative one.
