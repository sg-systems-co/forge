# FORGE

Training-free ternary (1.58-bit) post-training quantization for open-weight LLMs.

FORGE converts FP16/BF16 checkpoints (Qwen2.5-Coder, Llama-3) into ternary {-1, 0, 1}
weights using calibration-based PTQ — no continued pretraining. The pipeline runs on a
single consumer GPU or an Apple Silicon laptop in 1-4 hours for a 3B-7B model.

Output is a **stock GGUF** using llama.cpp's existing `TQ2_0` block type, so converted
checkpoints load on unmodified llama.cpp. A companion Metal/ARM NEON GEMV kernel turns
the ~5.8x memory reduction into a real decode speedup.

## How it works

1. **Rotate** — fuse randomized Hadamard rotations into the weights to suppress outlier
   channels before rounding. All rotations FORGE uses are fusable offline, so there is
   zero runtime overhead and no graph change.
2. **Solve** — quantize layer by layer against a calibration Hessian, propagating each
   column's rounding error onto the columns not yet quantized (GPTQ recursion, ternary
   codebook, closed-form optimal per-block scale).
3. **Propagate** — feed each layer the *quantized* model's activations so error is
   absorbed rather than compounded across depth.
4. **Pack** — serialize to `TQ2_0` blocks (256 weights, 4 per byte, 2.0625 bpw).

## Status

Early development. See `docs/` for the format and algorithm specifications.

## License

MIT
