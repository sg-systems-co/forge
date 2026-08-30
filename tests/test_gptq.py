"""Tests 3 (second half) and 4 from the plan: the Hessian-guided solver.

The properties that must hold no matter how the solver is tuned:

  * with H = I the recursion has nothing to propagate and must reduce to plain RTN;
  * with a real correlated H it must never do worse than RTN;
  * block boundaries and shapes must be handled exactly, never silently padded.
"""

import numpy as np
import pytest
import torch

from forge.calib.hessian import HessianAccumulator, damp, inverse_cholesky
from forge.pack.tq2 import QK_K
from forge.quant.gptq import (
    gptq_quantize_layer,
    hessian_flatness,
    rtn_quantize_layer,
)
from forge.quant.rescale import attenuation, channel_rescale

torch.manual_seed(0)


def correlated_hessian(n, ntokens=4096, strength=0.05, seed=0):
    """A Hessian from genuinely correlated activations, the regime GPTQ is built for."""
    g = torch.Generator().manual_seed(seed)
    mix = torch.randn(n, n, generator=g) * strength
    x = torch.randn(ntokens, n, generator=g) @ mix
    return 2.0 / ntokens * (x.T @ x)


def hessian_error(w, recon, h):
    err = (w - recon).float()
    num = torch.sum((err @ h.float()) * err)
    den = torch.sum((w.float() @ h.float()) * w.float())
    return float(torch.sqrt(num.clamp_min(0) / den.clamp_min(1e-30)))


# ------------------------------------------------------------------ correctness


def test_reduces_to_rtn_when_hessian_is_identity():
    """With H = I there is no correlation to exploit; GPTQ must match RTN exactly.

    Both use the same fixed-scale rounding rule, so the codes must be identical --
    not merely similar.
    """
    w = torch.randn(64, QK_K * 2)
    h = torch.eye(QK_K * 2)

    got = gptq_quantize_layer(w, h)

    # RTN under the *same* rounding rule: scale from the block, then round against it.
    from forge.quant.ternary import ternary_round_fixed_scale, ternary_scale

    blocks = w.reshape(64, -1, QK_K)
    s = ternary_scale(w)
    expected = torch.cat(
        [ternary_round_fixed_scale(blocks[:, i], s[:, i]) for i in range(blocks.shape[1])],
        dim=1,
    )
    torch.testing.assert_close(got.codes.float(), expected)


@pytest.mark.parametrize("seed", range(6))
def test_never_worse_than_rtn(seed):
    """On correlated data the compensated solution must beat plain rounding."""
    n = QK_K * 3
    w = torch.randn(96, n, generator=torch.Generator().manual_seed(seed))
    h = correlated_hessian(n, seed=seed)

    rtn = rtn_quantize_layer(w, h)
    gptq = gptq_quantize_layer(w, h)
    assert gptq.relative_error < rtn.relative_error, (gptq.relative_error, rtn.relative_error)


def test_reported_error_matches_recomputation():
    """The solver's own diagnostic must agree with an independent computation."""
    n = QK_K * 2
    w = torch.randn(48, n)
    h = correlated_hessian(n)
    r = gptq_quantize_layer(w, h)
    assert abs(r.relative_error - hessian_error(w, r.dequantize(), h)) < 1e-5


def test_output_domain_and_shapes():
    n = QK_K * 4
    r = gptq_quantize_layer(torch.randn(32, n), correlated_hessian(n))
    assert r.codes.dtype == torch.int8
    assert r.codes.shape == (32, n)
    assert r.scales.shape == (32, n // QK_K)
    assert torch.all((r.codes >= -1) & (r.codes <= 1))
    assert torch.all(r.scales >= 0)
    assert 0.0 < r.sparsity < 1.0


def test_dequantize_is_codes_times_block_scale():
    n = QK_K * 2
    r = gptq_quantize_layer(torch.randn(16, n), correlated_hessian(n))
    expected = r.codes.float() * r.scales.repeat_interleave(QK_K, dim=1)
    torch.testing.assert_close(r.dequantize(), expected)


@pytest.mark.parametrize("scale_rule", ["optimal", "absmean"])
def test_both_scale_rules_run(scale_rule):
    n = QK_K * 2
    r = gptq_quantize_layer(torch.randn(24, n), correlated_hessian(n), scale_rule=scale_rule)
    assert np.isfinite(r.relative_error)


# ------------------------------------------------------------------ shape guards


def test_rejects_unaligned_input_dimension():
    with pytest.raises(ValueError, match="not a multiple"):
        gptq_quantize_layer(torch.randn(8, 300), torch.eye(300))


def test_rejects_mismatched_hessian():
    with pytest.raises(ValueError, match="hessian is"):
        gptq_quantize_layer(torch.randn(8, QK_K), torch.eye(QK_K * 2))


def test_handles_dead_input_channels():
    """A channel that never fired leaves a zero row/col in H; the solve must survive."""
    n = QK_K
    acc = HessianAccumulator(n, torch.device("cpu"))
    x = torch.randn(512, n)
    x[:, 3] = 0.0
    x[:, 17] = 0.0
    acc.update(x)
    h = acc.finalize()

    assert h[3, 3] == 1.0 and h[3, 0] == 0.0  # marked as carrying no information
    r = gptq_quantize_layer(torch.randn(16, n), h)
    assert np.isfinite(r.relative_error)


def test_singular_hessian_is_damped_not_crashed():
    """A rank-deficient Hessian must be rescued by damping, not raise."""
    n = QK_K
    x = torch.randn(4, n)  # rank 4 << 256
    h = 2.0 / 4 * (x.T @ x)
    r = gptq_quantize_layer(torch.randn(8, n), h)
    assert np.isfinite(r.relative_error)


# ------------------------------------------------------------------ diagnostics


def test_hessian_flatness_detects_outlier_channels():
    n = QK_K
    flat = torch.eye(n)
    assert hessian_flatness(flat) < 1e-6

    spiky = torch.eye(n)
    spiky[0, 0] = 500.0
    assert hessian_flatness(spiky) > 1.0


def test_attenuation_is_below_one_for_ternary():
    """Ternary reconstruction is an orthogonal projection, so it loses gain.

    This pins the empirical fact that motivates sequential propagation: each layer comes
    out roughly 10% short, and that shortfall has to be absorbed downstream.
    """
    n = QK_K * 2
    w = torch.randn(64, n)
    h = correlated_hessian(n)
    r = rtn_quantize_layer(w, h)
    att = attenuation(w, r.dequantize(), h)
    assert 0.8 < att < 0.98, att


def test_channel_rescale_preserves_shape_and_stays_bounded():
    n = QK_K * 2
    w = torch.randn(32, n)
    h = correlated_hessian(n)
    r = gptq_quantize_layer(w, h)
    new_scales = channel_rescale(w, r.codes, r.scales, h)

    assert new_scales.shape == r.scales.shape
    ratio = new_scales / r.scales.clamp_min(1e-12)
    assert torch.all(ratio >= 0.5 - 1e-6) and torch.all(ratio <= 2.0 + 1e-6)
    # Folding alpha into the block scales must keep the layer representable in TQ2_0:
    # the codes are untouched, only the fp16 scales move.
    assert torch.all((r.codes >= -1) & (r.codes <= 1))


def test_rescale_never_increases_hessian_error_much():
    """alpha is a least-squares fit, so it cannot make the H-weighted error worse."""
    n = QK_K * 2
    w = torch.randn(48, n)
    h = correlated_hessian(n)
    r = gptq_quantize_layer(w, h)
    before = hessian_error(w, r.dequantize(), h)
    new_scales = channel_rescale(w, r.codes, r.scales, h)
    after = hessian_error(w, r.codes.float() * new_scales.repeat_interleave(QK_K, dim=1), h)
    assert after <= before + 1e-4


# ------------------------------------------------------------------ hessian utils


def test_damping_scales_with_diagonal():
    h = torch.eye(8) * 4.0
    assert torch.allclose(torch.diag(damp(h, 0.01)), torch.full((8,), 4.04))


def test_inverse_cholesky_reconstructs_inverse():
    n = 32
    h = correlated_hessian(n, ntokens=256)
    c = inverse_cholesky(h.double(), 0.01)
    assert torch.all(torch.triu(c) == c)
    torch.testing.assert_close(c.T @ c, torch.linalg.inv(damp(h.double(), 0.01)), atol=1e-8,
                               rtol=1e-6)
