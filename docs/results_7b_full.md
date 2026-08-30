# FORGE quantization

| tensor      | rel_error | attenuation | sparsity | H flatness |
|-------------|-----------|-------------|----------|------------|
| attn_k      | 0.1348    | 0.9906      | 0.4548   | 0.3765     |
| attn_output | 0.2145    | 0.9754      | 0.4506   | 0.7298     |
| attn_q      | 0.1702    | 0.9841      | 0.4555   | 0.3765     |
| attn_v      | 0.2451    | 0.9678      | 0.4556   | 0.3765     |
| ffn_down    | 0.2646    | 0.9626      | 0.4824   | 5.4194     |
| ffn_gate    | 0.1429    | 0.9872      | 0.4525   | 0.4358     |
| ffn_up      | 0.2622    | 0.9611      | 0.4567   | 0.4358     |

Model: `Qwen/Qwen2.5-Coder-7B`  
Config: solver=gptq, scale=optimal, rotate=True, sequential=True, rescale=True  
Calibration: 128 x 2048 from wikitext2  
**wikitext2 ppl = 16.4415**  
Time: 2219.6s, peak Hessian 1744 MB

