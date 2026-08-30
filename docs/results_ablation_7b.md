# Milestone 2b -- ablation

Model: `Qwen/Qwen2.5-Coder-7B`, 32 calibration sequences, 32 perplexity windows.

| config                        | ppl           | rel_error | attenuation | ffn_down_flatness | other_flatness | minutes |
|-------------------------------|---------------|-----------|-------------|-------------------|----------------|---------|
| FP16 baseline                 | 7.5190        | 0.0000    | 1.0000      | nan               |                | 1.5000  |
| naive ternary (absmean)       | 24632412.3590 | 0.5622    | 0.5713      | 5.2890            | 6.5720         | 7.0000  |
| naive ternary (optimal scale) | 52292.6970    | 0.4252    | 0.8008      | 5.2890            | 6.5720         | 7.7000  |
| + rotation                    | 1359.1860     | 0.3849    | 0.8874      | 5.2890            | 0.4050         | 7.2000  |
| + GPTQ solver                 | 19.3940       | 0.2122    | 0.9728      | 5.2890            | 0.4050         | 12.0000 |
| + sequential (full FORGE)     | 17.8340       | 0.2001    | 0.9767      | 5.5250            | 0.4580         | 16.2000 |
