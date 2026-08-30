"""Closed-form per-output-channel rescaling.

Ternary quantization systematically *attenuates* a layer. The MSE-optimal reconstruction
is an orthogonal projection of W onto the ternary codebook, so <W, Q> == ||Q||^2 and
therefore ||Q|| < ||W||: every layer's output comes out slightly short. Across 28 blocks
that shrinkage compounds, and it is a large part of why naive ternary destroys a model
even though each individual layer looks only ~40% off.

Weight-space rescaling cannot fix this -- exactly because Q is already the projection,
the weight-space least-squares gain is identically 1. The fix has to be measured in
*activation* space, where the calibration Hessian carries the information:

    minimize_alpha ||W X - diag(alpha) Q X||_F^2

Row r separates, giving a closed form with no gradients and no training:

    alpha_r = (W H Q^T)_rr / (Q H Q^T)_rr

alpha then folds straight into the per-block scales, so it stays exactly representable in
TQ2_0 and costs nothing at inference time.
"""

from __future__ import annotations

import torch

from forge.pack.tq2 import QK_K


@torch.no_grad()
def channel_rescale(
    weight: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    hessian: torch.Tensor,
    block: int = QK_K,
    ridge: float = 1e-6,
    clamp: tuple[float, float] = (0.5, 2.0),
) -> torch.Tensor:
    """Return rescaled per-block scales. Shapes match `scales`.

    `clamp` bounds the correction: a row whose quantized output is nearly orthogonal to
    the original has a meaningless ratio, and letting it through would amplify noise.
    """
    w = weight.float()
    q = (codes.float() * scales.repeat_interleave(block, dim=1)).float()
    h = hessian.float()

    qh = q @ h
    num = torch.sum(qh * w, dim=1)  # diag(Q H W^T)
    den = torch.sum(qh * q, dim=1)  # diag(Q H Q^T)

    alpha = num / (den + ridge * den.abs().mean().clamp_min(1e-30))
    alpha = torch.where(den > 0, alpha, torch.ones_like(alpha))
    alpha = torch.clamp(alpha, clamp[0], clamp[1])

    return scales * alpha[:, None]


@torch.no_grad()
def attenuation(weight: torch.Tensor, recon: torch.Tensor, hessian: torch.Tensor) -> float:
    """||QX|| / ||WX||. Below 1 means the layer is losing gain; 1.0 is neutral."""
    w, q, h = weight.float(), recon.float(), hessian.float()
    num = torch.sum((q @ h) * q)
    den = torch.sum((w @ h) * w)
    return float(torch.sqrt(torch.clamp(num, min=0) / torch.clamp(den, min=1e-30)))
