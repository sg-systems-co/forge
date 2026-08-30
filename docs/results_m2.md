# FORGE quantization

| tensor      | rel_error | attenuation | sparsity | H flatness |
|-------------|-----------|-------------|----------|------------|
| attn_k      | 0.1372    | 0.9903      | 0.4519   | 0.5073     |
| attn_output | 0.1915    | 0.9812      | 0.4533   | 0.7214     |
| attn_q      | 0.1638    | 0.9859      | 0.4529   | 0.5073     |
| attn_v      | 0.2687    | 0.9626      | 0.4529   | 0.5073     |
| ffn_down    | 0.2257    | 0.9744      | 0.4663   | 10.5866    |
| ffn_gate    | 0.1730    | 0.9837      | 0.4549   | 0.5971     |
| ffn_up      | 0.2924    | 0.9552      | 0.4554   | 0.5971     |

Model: `Qwen/Qwen2.5-Coder-1.5B`  
Config: solver=gptq, scale=optimal, rotate=True, sequential=True, rescale=True  
Calibration: 128 x 2048 from wikitext2  
**wikitext2 ppl = 172.2660**  
Time: 596.7s, peak Hessian 378 MB

