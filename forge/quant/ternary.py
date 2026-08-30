"""Ternary codebook: scale assignment for w ~= s * t, t in {-1, 0, 1}.

The core routine solves

    argmin_{s > 0, t in {-1,0,1}^n}  ||w - s * t||^2

*exactly*, in O(n log n), rather than falling back to BitNet's absmean heuristic.

Derivation. Fix the support S = {i : t_i != 0}; on S the optimal sign is t_i = sign(w_i),
so the objective becomes ||w||^2 - 2 s * sum_{i in S} |w_i| + s^2 |S|. Minimizing over s,

    s*(S) = (sum_{i in S} |w_i|) / |S|                                        (1)
    ||w||^2 - objective = (sum_{i in S} |w_i|)^2 / |S|                        (2)

(2) depends on S only through its size and the sum of magnitudes it collects, so for a
given size the best S takes the largest magnitudes. The optimal support is therefore a
prefix of |w| sorted descending, and we only have to sweep n prefix sums.

This is the classic TWN result; the useful part is that it is exact and vectorizes over
every block of the weight matrix at once.
"""

from __future__ import annotations

import torch

TERNARY_BLOCK = 256  # QK_K: one TQ2_0 block, and one scale group


def _as_blocks(w: torch.Tensor, block: int) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Reshape (..., n) -> (num_blocks, block), returning the original shape."""
    shape = tuple(w.shape)
    if shape[-1] % block != 0:
        raise ValueError(
            f"last dimension {shape[-1]} is not divisible by block size {block}; "
            "FORGE refuses to pad silently"
        )
    return w.reshape(-1, block), shape


def ternary_scale(w: torch.Tensor, block: int = TERNARY_BLOCK) -> torch.Tensor:
    """Optimal per-block scale s*, shape (..., n // block).

    Used by the GPTQ solver, which needs the scale up front (fixed for a whole block)
    and then rounds columns against it one at a time.
    """
    blocks, shape = _as_blocks(w, block)
    mag = blocks.abs().to(torch.float32)

    ordered, _ = torch.sort(mag, dim=-1, descending=True)
    prefix = torch.cumsum(ordered, dim=-1)
    sizes = torch.arange(1, block + 1, device=w.device, dtype=torch.float32)

    # Equation (2): pick the support size maximizing prefix^2 / k.
    score = prefix * prefix / sizes
    best = torch.argmax(score, dim=-1)

    # Equation (1) at the winning support size.
    chosen_sum = torch.gather(prefix, 1, best[:, None]).squeeze(1)
    chosen_size = best.to(torch.float32) + 1
    scale = chosen_sum / chosen_size

    # An all-zero block has no meaningful scale; keep it finite and non-negative.
    scale = torch.where(torch.isfinite(scale), scale, torch.zeros_like(scale))
    return scale.reshape(*shape[:-1], shape[-1] // block)


def ternary_quantize(
    w: torch.Tensor, block: int = TERNARY_BLOCK
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact optimal ternary quantization. Returns (t, scale).

    t has the same shape as w with values in {-1, 0, 1}; scale has shape
    (..., n // block). The support is taken directly from the optimal prefix rather than
    by thresholding w / s, which makes this exactly optimal instead of approximately so.
    """
    blocks, shape = _as_blocks(w, block)
    mag = blocks.abs().to(torch.float32)

    ordered, order = torch.sort(mag, dim=-1, descending=True)
    prefix = torch.cumsum(ordered, dim=-1)
    sizes = torch.arange(1, block + 1, device=w.device, dtype=torch.float32)

    score = prefix * prefix / sizes
    best = torch.argmax(score, dim=-1)

    chosen_sum = torch.gather(prefix, 1, best[:, None]).squeeze(1)
    scale = chosen_sum / (best.to(torch.float32) + 1)
    scale = torch.where(torch.isfinite(scale), scale, torch.zeros_like(scale))

    # Support = the first (best + 1) entries of the descending order. Building the mask in
    # sorted space and scattering back avoids any tie-breaking ambiguity that a magnitude
    # threshold would introduce when several weights share a magnitude.
    ranks = torch.arange(block, device=w.device)[None, :]
    keep_sorted = ranks <= best[:, None]
    keep = torch.zeros_like(keep_sorted)
    keep.scatter_(1, order, keep_sorted)

    t = torch.sign(blocks) * keep
    return t.reshape(shape).to(torch.int8), scale.reshape(*shape[:-1], shape[-1] // block)


def ternary_absmean(
    w: torch.Tensor, block: int = TERNARY_BLOCK
) -> tuple[torch.Tensor, torch.Tensor]:
    """BitNet b1.58's absmean rule -- the round-to-nearest baseline Milestone 1 measures.

    scale = mean(|w|) over the block, then t = clamp(round(w / scale), -1, 1).
    """
    blocks, shape = _as_blocks(w, block)
    scale = blocks.abs().to(torch.float32).mean(dim=-1)
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    t = torch.clamp(torch.round(blocks.to(torch.float32) / safe[:, None]), -1, 1)
    t = torch.where(scale[:, None] > 0, t, torch.zeros_like(t))
    return t.reshape(shape).to(torch.int8), scale.reshape(*shape[:-1], shape[-1] // block)


def ternary_round_fixed_scale(w: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Round against a scale that is already fixed: t = clamp(round(w / s), -1, 1).

    This is the rule the GPTQ inner loop must use, because it quantizes one column at a
    time and cannot see the rest of the block. Broadcasting: w is (..., n), scale is
    (...,) or (..., 1).
    """
    s = scale if scale.dim() == w.dim() else scale.unsqueeze(-1)
    safe = torch.where(s > 0, s, torch.ones_like(s))
    t = torch.clamp(torch.round(w.to(torch.float32) / safe), -1, 1)
    return torch.where(s > 0, t, torch.zeros_like(t))


def dequantize(t: torch.Tensor, scale: torch.Tensor, block: int = TERNARY_BLOCK) -> torch.Tensor:
    """Reconstruct s * t from a ternary tensor and its per-block scales."""
    shape = tuple(t.shape)
    expanded = scale.reshape(*shape[:-1], shape[-1] // block, 1).expand(
        *shape[:-1], shape[-1] // block, block
    )
    return t.to(torch.float32) * expanded.reshape(shape)


def quantization_error(w: torch.Tensor, t: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Relative Frobenius error ||w - s*t|| / ||w||, the M1 per-layer reporting metric."""
    recon = dequantize(t, scale)
    return torch.linalg.vector_norm(w.to(torch.float32) - recon) / torch.linalg.vector_norm(
        w.to(torch.float32)
    )
