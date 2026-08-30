# Milestone 1 -- baseline ternary reconstruction error

| tensor      | absmean (H-weighted) | optimal (H-weighted) | absmean (weight-space) | optimal (weight-space) |
|-------------|----------------------|----------------------|------------------------|------------------------|
| attn_k      | 0.6016               | 0.4657               | 0.5412                 | 0.4594                 |
| attn_output | 0.5654               | 0.4762               | 0.5322                 | 0.4493                 |
| attn_q      | 0.5537               | 0.4414               | 0.5344                 | 0.4509                 |
| attn_v      | 0.5785               | 0.4791               | 0.5545                 | 0.4682                 |
| ffn_down    | 0.5947               | 0.4526               | 0.5293                 | 0.4500                 |
| ffn_gate    | 0.4955               | 0.3513               | 0.5248                 | 0.4451                 |
| ffn_up      | 0.5331               | 0.4564               | 0.5247                 | 0.4444                 |

Model: `Qwen/Qwen2.5-Coder-1.5B`  
Calibration: 128 x 2048 tokens from wikitext2 (seed 0)  
Metric: `||(W-Q)X|| / ||WX||`, X from calibration activations  
Peak Hessian residency: 378 MB  
Capture time: 194.7s

