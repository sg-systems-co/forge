"""Second-moment (Hessian) accumulation for layer-wise reconstruction.

For the layer objective ||(W - Q) X||_F^2 the relevant curvature is

    H = 2 / N * sum_n x_n x_n^T                                              (fp32)

accumulated over calibration tokens. We keep the running sum rather than a running mean so
the accumulation is order-independent and exactly reproducible; the 2/N normalization is
applied once at the end. Damping is added at solve time, not accumulation time, so the same
captured H can be reused across damping ablations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class HessianAccumulator:
    """Streaming accumulator for one linear layer's input second moment."""

    in_features: int
    device: torch.device
    dtype: torch.dtype = torch.float32
    ntokens: int = 0
    _sum: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self):
        self._sum = torch.zeros(
            (self.in_features, self.in_features), device=self.device, dtype=self.dtype
        )

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        """x: (..., in_features). Flattened over every leading dimension."""
        flat = x.reshape(-1, x.shape[-1]).to(self.dtype)
        if flat.shape[-1] != self.in_features:
            raise ValueError(f"expected {self.in_features} features, got {flat.shape[-1]}")
        self._sum.addmm_(flat.T, flat)
        self.ntokens += flat.shape[0]

    @property
    def nbytes(self) -> int:
        return self._sum.element_size() * self._sum.numel()

    def finalize(self) -> torch.Tensor:
        """Return H = 2/N sum x x^T. Dead input channels are made solvable."""
        if self.ntokens == 0:
            raise RuntimeError("no tokens accumulated")
        h = self._sum * (2.0 / self.ntokens)

        # A channel that was exactly zero across the whole calibration set leaves a zero
        # row and column. Cholesky would fail; the honest fix is to mark the channel as
        # carrying no information (unit curvature, zero cross-terms) so the solver treats
        # its weights as free rather than dividing by zero.
        dead = torch.diag(h) == 0
        if dead.any():
            h[dead, :] = 0
            h[:, dead] = 0
            h[dead, dead] = 1.0
        return h

    def release(self) -> None:
        self._sum = torch.empty(0, device=self.device, dtype=self.dtype)


def damp(h: torch.Tensor, fraction: float = 0.01) -> torch.Tensor:
    """Add lambda * I with lambda = fraction * mean(diag(H)).

    Damping is what makes the Cholesky in the GPTQ recursion succeed on a Hessian that is
    rank-deficient (it always is: we have far fewer calibration tokens than the square of
    the feature count for the wide FFN layers).
    """
    lam = fraction * torch.mean(torch.diag(h))
    return h + torch.eye(h.shape[0], device=h.device, dtype=h.dtype) * lam


# Where and at what precision to factorize the Hessian. float64 on CPU is the safe
# reference; float32 on the GPU is ~5x faster at 7B scale (18944^2 takes 6.9 s instead of
# 34.8 s) and agrees to within the solver's own noise -- see tests/test_gptq.py.
FACTORIZATIONS = {
    "float64_cpu": (torch.float64, "cpu"),
    "float32_cpu": (torch.float32, "cpu"),
    "float32_gpu": (torch.float32, None),  # None = keep the Hessian where it already is
}


def inverse_cholesky(
    h: torch.Tensor, damping: float = 0.01, factorization: str = "float64_cpu"
) -> torch.Tensor:
    """Upper-triangular Cholesky factor of H^-1, as the GPTQ recursion consumes it.

    Retries with progressively heavier damping instead of failing outright -- a layer whose
    calibration Hessian is badly conditioned should still be quantized, just more
    conservatively. If it still fails at the requested precision, it falls back to float64
    on CPU before giving up, because a slow answer beats no answer.
    """
    if factorization not in FACTORIZATIONS:
        raise ValueError(
            f"unknown factorization {factorization!r}; expected one of {sorted(FACTORIZATIONS)}"
        )
    dtype, device = FACTORIZATIONS[factorization]

    # Move device FIRST, then cast. A fused h.to("cpu", torch.float64) on an MPS tensor
    # attempts the float64 cast while still on MPS, which has no float64 -- and rather
    # than raising it yields garbage that only surfaces later as a bogus
    # "not positive-definite" Cholesky failure. Pinned by test_gptq.py.
    work = h.to(device) if device is not None else h
    work = work.to(dtype)

    fraction = damping
    for _ in range(6):
        try:
            damped = damp(work, fraction)
            lower = torch.linalg.cholesky(damped)
            inv = torch.cholesky_inverse(lower)
            return torch.linalg.cholesky(inv, upper=True)
        except Exception:  # noqa: BLE001 - torch raises several types here
            fraction *= 10

    if factorization != "float64_cpu":
        return inverse_cholesky(h, damping, "float64_cpu")
    raise RuntimeError(f"Hessian not factorizable even at damping {fraction}")
