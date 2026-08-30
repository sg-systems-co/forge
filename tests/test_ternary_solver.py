"""Test 3 (first half) from the plan: the ternary codebook is exactly optimal."""

import itertools

import pytest
import torch

from forge.quant.ternary import (
    dequantize,
    quantization_error,
    ternary_absmean,
    ternary_quantize,
    ternary_round_fixed_scale,
    ternary_scale,
)

torch.manual_seed(0)


def brute_force_best_error(w: torch.Tensor) -> float:
    """Exhaustively minimize ||w - s*t||^2 over every t in {-1,0,1}^n and the optimal s.

    For a fixed t the optimal scale is the least-squares solution s = <w,t> / <t,t>.
    Only tractable for tiny n, which is exactly what we want for a ground-truth oracle.
    """
    n = w.numel()
    best = float(torch.sum(w * w))  # t = 0 everywhere
    for t in itertools.product((-1.0, 0.0, 1.0), repeat=n):
        tt = torch.tensor(t, dtype=torch.float64)
        denom = float(tt @ tt)
        if denom == 0:
            continue
        s = float(w @ tt) / denom
        if s <= 0:
            continue
        err = float(torch.sum((w - s * tt) ** 2))
        best = min(best, err)
    return best


@pytest.mark.parametrize("n", [4, 6, 8])
@pytest.mark.parametrize("trial", range(12))
def test_matches_brute_force(n, trial):
    """The O(n log n) prefix sweep must match exhaustive 3^n enumeration."""
    w = torch.randn(1, n, dtype=torch.float64)
    t, s = ternary_quantize(w, block=n)
    got = float(torch.sum((w - dequantize(t, s, block=n).double()) ** 2))
    want = brute_force_best_error(w[0])
    assert got <= want + 1e-9, f"solver found {got}, brute force found {want}"
    assert got >= want - 1e-9, f"solver claims {got} but brute force floor is {want}"


@pytest.mark.parametrize("trial", range(20))
def test_beats_absmean(trial):
    """The closed form must never be worse than BitNet's absmean heuristic."""
    w = torch.randn(8, 256)
    _, opt = ternary_quantize(w)
    t_opt, s_opt = ternary_quantize(w)
    t_am, s_am = ternary_absmean(w)
    assert quantization_error(w, t_opt, s_opt) <= quantization_error(w, t_am, s_am) + 1e-6


def test_output_domain_and_shapes():
    w = torch.randn(7, 1024)
    t, s = ternary_quantize(w)
    assert t.dtype == torch.int8
    assert t.shape == w.shape
    assert s.shape == (7, 4)
    assert torch.all((t >= -1) & (t <= 1))
    # Sign must always follow the original weight where the support is non-zero.
    nz = t != 0
    assert torch.all(torch.sign(w[nz]) == t[nz].float())


def test_scale_is_positive_and_matches_quantize():
    w = torch.randn(3, 512)
    _, s_from_quant = ternary_quantize(w)
    s_direct = ternary_scale(w)
    torch.testing.assert_close(s_direct, s_from_quant)
    assert torch.all(s_direct > 0)


def test_all_zero_block_is_handled():
    """A dead block must produce zeros and a finite scale, not NaN."""
    w = torch.zeros(2, 256)
    t, s = ternary_quantize(w)
    assert torch.all(t == 0)
    assert torch.all(torch.isfinite(s))
    t2, s2 = ternary_absmean(w)
    assert torch.all(t2 == 0) and torch.all(torch.isfinite(s2))


def test_gaussian_sparsity_matches_twn_theory():
    """For Gaussian weights the optimal ternary support keeps ~55% of entries.

    The known TWN result is a threshold near 0.6 sigma; this pins the solver against
    theory so a future 'optimization' that quietly changes the support gets caught.
    """
    w = torch.randn(64, 256)
    t, _ = ternary_quantize(w)
    frac_nonzero = float((t != 0).float().mean())
    assert 0.50 < frac_nonzero < 0.60, frac_nonzero


def test_scale_invariance():
    """Scaling w by c must scale s by c and leave the ternary pattern unchanged."""
    w = torch.randn(4, 256)
    t1, s1 = ternary_quantize(w)
    t2, s2 = ternary_quantize(w * 3.5)
    torch.testing.assert_close(t1, t2)
    torch.testing.assert_close(s1 * 3.5, s2, rtol=1e-5, atol=1e-6)


def test_fixed_scale_rounding_rule():
    """The GPTQ inner-loop rounding rule thresholds at s/2 and clamps to +/-1."""
    w = torch.tensor([[0.0, 0.4, 0.6, 5.0, -0.6, -5.0]])
    s = torch.tensor([1.0])
    t = ternary_round_fixed_scale(w, s)
    torch.testing.assert_close(t, torch.tensor([[0.0, 0.0, 1.0, 1.0, -1.0, -1.0]]))


def test_rejects_unaligned_last_dim():
    """Row lengths that are not a multiple of the block must fail loudly, not pad."""
    with pytest.raises(ValueError, match="not divisible"):
        ternary_quantize(torch.randn(2, 300))
