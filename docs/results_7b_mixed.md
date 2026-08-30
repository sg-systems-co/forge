# FORGE quantization

| tensor      | rel_error | attenuation | sparsity | H flatness |
|-------------|-----------|-------------|----------|------------|
| attn_k      | 0.1393    | 0.9900      | 0.4550   | 0.3536     |
| attn_output | 0.2211    | 0.9739      | 0.4506   | 0.7260     |
| attn_q      | 0.1760    | 0.9830      | 0.4556   | 0.3536     |
| attn_v      | 0.2505    | 0.9665      | 0.4557   | 0.3536     |
| ffn_gate    | 0.1479    | 0.9864      | 0.4526   | 0.4123     |
| ffn_up      | 0.2701    | 0.9589      | 0.4567   | 0.4123     |

Model: `Qwen/Qwen2.5-Coder-7B`  
Config: solver=gptq, scale=optimal, rotate=True, sequential=True, rescale=True  
Calibration: 128 x 2048 from wikitext2  
**wikitext2 ppl = 11.4022**  
Time: 1958.0s, peak Hessian 1744 MB

