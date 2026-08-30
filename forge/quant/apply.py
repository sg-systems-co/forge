"""Apply a ternary codebook to live nn.Linear weights (simulated quantization).

Used for baseline measurement and for PyTorch-side perplexity checks. The packing path
(forge/pack) is what produces real checkpoints; this module deliberately keeps the weights
in float so the rest of the model runs unchanged.
"""

from __future__ import annotations

import torch

from forge.quant.ternary import dequantize, ternary_absmean, ternary_quantize

METHODS = {"absmean": ternary_absmean, "optimal": ternary_quantize}


@torch.no_grad()
def quantize_weight(w: torch.Tensor, method: str = "optimal") -> torch.Tensor:
    """Return the ternary reconstruction s * t of w, same shape and dtype."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {sorted(METHODS)}")
    t, scale = METHODS[method](w.float())
    return dequantize(t, scale).to(w.dtype)


@torch.no_grad()
def quantize_model_rtn(model, graph, method: str = "optimal", progress=None) -> int:
    """Round-to-nearest ternary over every quantizable linear. Returns params touched.

    This is the Milestone 1 baseline: no rotation, no Hessian, no error propagation. It
    exists to establish how far the naive approach is from usable.
    """
    touched = 0
    for spec in graph.all_linears:
        module = model.get_submodule(spec.name)
        module.weight.data = quantize_weight(module.weight.data, method)
        touched += spec.numel
        if progress is not None:
            progress.update(1)
    return touched


@torch.no_grad()
def hessian_relative_error(w: torch.Tensor, q: torch.Tensor, h: torch.Tensor) -> float:
    """||(W - Q) X||_F / ||W X||_F, computed from H = 2/N sum x x^T without touching X.

    Since ||A X||_F^2 = tr(A (X X^T) A^T) and H is X X^T up to a constant factor, the
    constant cancels in the ratio.
    """
    w32, q32, h32 = w.float(), q.float(), h.float()
    err = w32 - q32
    num = torch.sum((err @ h32) * err)
    den = torch.sum((w32 @ h32) * w32)
    return float(torch.sqrt(torch.clamp(num, min=0) / torch.clamp(den, min=1e-30)))


@torch.no_grad()
def weight_relative_error(w: torch.Tensor, q: torch.Tensor) -> float:
    """||W - Q||_F / ||W||_F -- the data-free counterpart, for comparison."""
    w32 = w.float()
    return float(torch.linalg.norm(w32 - q.float()) / torch.linalg.norm(w32))
