# FORGE storage format

FORGE emits **stock GGUF**. It does not define a new ggml type. Converted checkpoints load
on unmodified llama.cpp; the accompanying kernels only make them faster.

## Tensor type map

| Tensor pattern | ggml type | bits/weight |
|---|---|---|
| `blk.*.attn_{q,k,v,output}.weight` | `TQ2_0` (35) | 2.0625 |
| `blk.*.ffn_{gate,up,down}.weight` | `TQ2_0` (35) | 2.0625 |
| `token_embd.weight`, `output.weight` | `Q6_K` (14) | 6.5625 |
| `*_norm.weight`, `blk.*.attn_{q,k,v}.bias` | `F32` (0) | 32 |

Every ternary tensor's **input dimension must be a multiple of 256**. FORGE refuses to
convert a model that violates this rather than padding, because padding would silently
change the layer's arithmetic. Verified for the target models:

| Model | hidden | ffn | kv out | all divisible by 256 |
|---|---|---|---|---|
| Qwen2.5-Coder-1.5B | 1536 | 8960 | 256 | yes |
| Qwen2.5-Coder-7B | 3584 | 18944 | 512 | yes |

## `block_tq2_0`

```c
#define QK_K 256
typedef struct {
    uint8_t qs[QK_K/4];   // 64 bytes, 2 bits per weight
    ggml_half d;          // fp16 block scale
} block_tq2_0;            // 66 bytes per 256 weights
```

Values are stored **biased**: `t in {-1,0,1}` is written as `q = t + 1 in {0,1,2}`.
Decoding is `(q - 1) * d`. The code `3` is unreachable and FORGE never emits it.

### Bit layout

The 64 bytes are **two independent groups of 32 bytes**, each encoding 128 consecutive
weights. This is read off `quantize_row_tq2_0_ref` in `ggml/src/ggml-quants.c`:

```
element e  ->  g = e // 128,  r = e % 128,  n = r // 32,  m = r % 32
               byte index = 32*g + m,   bit shift = 2*n
```

So byte `32g + m` packs elements `{128g+m, 128g+m+32, 128g+m+64, 128g+m+96}` at shifts
`0, 2, 4, 6`.

```
        byte 32g+m:   [ b7 b6 | b5 b4 | b3 b2 | b1 b0 ]
                        n=3     n=2     n=1     n=0
        element:      +96     +64     +32     +0      (all relative to 128g+m)
```

A SIMD kernel therefore recovers four contiguous 32-lane vectors with four shift/mask
pairs and **no cross-lane shuffles**, which is the property the NEON and Metal kernels
depend on.

### Scales

ggml's own `quantize_row_tq2_0_ref` sets `d = max|x|` over the block. FORGE does **not**:
it supplies the closed-form optimal scale from `forge/quant/ternary.py`, which is strictly
better. The two quantizers therefore agree on *layout* but not on *values*.
`tests/test_pack_bitexact.py` pins these two properties separately:

* FORGE's amax reimplementation is byte-identical to ggml's (proves the layout);
* `dequantize(pack(t, s)) == s * t` exactly (proves the contract stock llama.cpp relies on).

The scale is rounded to fp16 before packing, so the round trip is exact.

## FORGE metadata keys

Written into the GGUF KV store for reproducibility:

| Key | Meaning |
|---|---|
| `forge.version` | converter version |
| `forge.rotation.kind` | `randomized_hadamard` \| `random_orthogonal` \| `none` |
| `forge.rotation.seed` | seed for the sign vector and rotation |
| `forge.calib.dataset` | `wikitext2` \| `c4` |
| `forge.calib.nsamples` | calibration sequence count |
| `forge.calib.seqlen` | calibration sequence length |
| `forge.solver` | `rtn` \| `gptq_ternary` |
