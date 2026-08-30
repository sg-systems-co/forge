"""Hessian-guided layer-wise reconstruction, ternary codebook.

Objective, per linear layer:

    argmin_Q ||(W - Q) X||_F^2  =  argmin_Q tr( (W-Q) H (W-Q)^T ),   H = 2/N sum x x^T

Solved with the GPTQ recursion: quantize one input column at a time, and push the rounding
error onto the columns not yet quantized, weighted by the inverse Hessian. A column that
the calibration data says matters gets compensated for by its correlated neighbours.

    q_j       = clamp(round(w_j / s_g), -1, 1) * s_g
    e_j       = (w_j - q_j) / Hinv[j, j]
    W[:, j:] -= e_j (x) Hinv[j, j:]

Two FORGE-specific details:

* The block scale s_g is refit at the start of every 256-column group, on the *already
  error-compensated* weights rather than on the original tensor. TQ2_0 stores one fp16
  scale per 256 weights, so this costs nothing and is what makes that per-block scale
  earn its keep.
* Act-order (processing columns by descending diag(H)) is deliberately omitted. It would
  scramble which weights share a block scale, and the permutation cannot be folded away
  for q/k/v/gate/up because their input is the shared residual stream with many consumers.
  The Hadamard rotation already flattens diag(H), so there is little left to gain --
  measurable via `forge.quant.gptq.hessian_flatness`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from forge.calib.hessian import inverse_cholesky
from forge.pack.tq2 import QK_K
from forge.quant.ternary import ternary_absmean, ternary_scale

SCALE_RULES = {"optimal": ternary_scale, "absmean": lambda w, block: w.abs()
               .reshape(w.shape[0], -1, block).mean(-1)}


@dataclass
class LayerResult:
    """One quantized layer: the ternary codes, the per-block scales, and diagnostics."""

    codes: torch.Tensor  # int8 (out, in), values in {-1, 0, 1}
    scales: torch.Tensor  # fp32 (out, in // 256)
    relative_error: float  # ||(W-Q)X|| / ||WX||
    sparsity: float  # fraction of zero codes

    def dequantize(self) -> torch.Tensor:
        out, n = self.codes.shape
        return self.codes.float() * self.scales.repeat_interleave(QK_K, dim=1)


def hessian_flatness(h: torch.Tensor) -> float:
    """std(diag H) / mean(diag H). Rotation should drive this toward 0.

    This is the quantity act-order exploits; if it is already small, act-order has
    nothing left to buy.
    """
    d = torch.diag(h).float()
    return float(d.std() / d.mean().clamp_min(1e-30))


@torch.no_grad()
def gptq_quantize_layer(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    block: int = QK_K,
    damping: float = 0.01,
    scale_rule: str = "optimal",
) -> LayerResult:
    """Quantize one linear layer's weights against its calibration Hessian.

    weight: (out, in) -- the layer's weight matrix, quantized along the input dimension.
    hessian: (in, in) -- from calib.hessian.HessianAccumulator.finalize().
    """
    out_features, n = weight.shape
    if n % block != 0:
        raise ValueError(f"input dimension {n} is not a multiple of block {block}")
    if hessian.shape != (n, n):
        raise ValueError(f"hessian is {tuple(hessian.shape)}, expected {(n, n)}")

    device = weight.device
    w_orig = weight.float()

    # Factorize on CPU in float64: MPS lacks a float64 Cholesky, and this is the one step
    # where conditioning actually matters.
    hinv = inverse_cholesky(hessian.cpu().double(), damping).to(device, torch.float32)

    residual = w_orig.clone()
    codes = torch.zeros((out_features, n), dtype=torch.int8, device=device)
    scales = torch.zeros((out_features, n // block), dtype=torch.float32, device=device)

    rule = SCALE_RULES[scale_rule]

    for b0 in range(0, n, block):
        b1 = b0 + block
        w_blk = residual[:, b0:b1].clone()
        err_blk = torch.zeros_like(w_blk)
        hinv_blk = hinv[b0:b1, b0:b1]

        # Scale fixed for the whole 256-column group, fit on the compensated weights.
        s = rule(w_blk, block).reshape(out_features)
        safe = torch.where(s > 0, s, torch.ones_like(s))
        scales[:, b0 // block] = s

        for i in range(block):
            col = w_blk[:, i]
            t = torch.clamp(torch.round(col / safe), -1.0, 1.0)
            t = torch.where(s > 0, t, torch.zeros_like(t))
            q = t * s

            codes[:, b0 + i] = t.to(torch.int8)

            e = (col - q) / hinv_blk[i, i]
            err_blk[:, i] = e
            # Compensate the columns still to come *inside* this block.
            if i + 1 < block:
                w_blk[:, i + 1 :] -= e[:, None] * hinv_blk[i, i + 1 :][None, :]

        # One rank-`block` update pushes this block's accumulated error onto every column
        # after it -- the same result as updating column by column, far fewer kernels.
        if b1 < n:
            residual[:, b1:] -= err_blk @ hinv[b0:b1, b1:]

    recon = codes.float() * scales.repeat_interleave(block, dim=1)
    err = w_orig - recon
    h32 = hessian.float()
    num = torch.sum((err @ h32) * err)
    den = torch.sum((w_orig @ h32) * w_orig)
    rel = float(torch.sqrt(torch.clamp(num, min=0) / torch.clamp(den, min=1e-30)))

    return LayerResult(
        codes=codes,
        scales=scales,
        relative_error=rel,
        sparsity=float((codes == 0).float().mean()),
    )


@torch.no_grad()
def rtn_quantize_layer(
    weight: torch.Tensor, hessian: torch.Tensor, block: int = QK_K, scale_rule: str = "optimal"
) -> LayerResult:
    """Round-to-nearest with no error compensation -- the ablation baseline."""
    from forge.quant.ternary import ternary_quantize

    w = weight.float()
    if scale_rule == "optimal":
        codes, scales = ternary_quantize(w, block)
    else:
        codes, scales = ternary_absmean(w, block)

    recon = codes.float() * scales.repeat_interleave(block, dim=1)
    err = w - recon
    h32 = hessian.float()
    num = torch.sum((err @ h32) * err)
    den = torch.sum((w @ h32) * w)
    return LayerResult(
        codes=codes,
        scales=scales,
        relative_error=float(torch.sqrt(torch.clamp(num, min=0) / torch.clamp(den, min=1e-30))),
        sparsity=float((codes == 0).float().mean()),
    )
