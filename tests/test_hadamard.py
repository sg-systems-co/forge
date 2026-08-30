"""Test 1 from the plan: Hadamard construction correctness."""

import numpy as np
import pytest

from forge.rotate.hadamard import (
    HadamardNotFound,
    build_rotation,
    factorize,
    hadamard_matrix,
    paley1,
    paley2,
    random_hadamard_matrix,
    sylvester,
)

# Dimensions the target models actually need.
MODEL_DIMS = [
    128,    # head_dim (R3), Qwen2.5 and Llama3
    512,    # 7B kv projection output
    1536,   # Qwen2.5-Coder-1.5B hidden  = 128 * 12
    3584,   # Qwen2.5-Coder-7B   hidden  = 128 * 28
    8960,   # 1.5B intermediate          = 64  * 140
    18944,  # 7B   intermediate          = 128 * 148
]


# Above this order the full n^3 gram check stops being worth the wall-clock; we switch
# to a random row sample, which still catches any real construction bug immediately.
FULL_GRAM_LIMIT = 2048
SAMPLE_ROWS = 512


def assert_hadamard(h, n, seed=0):
    """Assert h is a Hadamard matrix of order n: +/-1 entries and H @ H.T == n * I."""
    assert h.shape == (n, n), f"expected {n}x{n}, got {h.shape}"
    assert set(np.unique(h)).issubset({-1, 1}), "entries must be +/-1"

    hf = h.astype(np.float64)  # float64 goes through BLAS; int64 matmul does not
    if n <= FULL_GRAM_LIMIT:
        np.testing.assert_allclose(hf @ hf.T, n * np.eye(n), atol=1e-6)
        return

    rows = np.random.default_rng(seed).choice(n, size=SAMPLE_ROWS, replace=False)
    sub = hf[rows]
    np.testing.assert_allclose(sub @ sub.T, n * np.eye(SAMPLE_ROWS), atol=1e-6)


@pytest.mark.parametrize("k", range(0, 10))
def test_sylvester_is_hadamard(k):
    n = 2**k
    assert_hadamard(sylvester(n), n)


@pytest.mark.parametrize("p", [3, 7, 11, 19, 23, 31, 43, 47, 59, 139])
def test_paley1_is_hadamard(p):
    assert_hadamard(paley1(p), p + 1)


@pytest.mark.parametrize("p", [5, 13, 17, 29, 37, 41, 73])
def test_paley2_is_hadamard(p):
    assert_hadamard(paley2(p), 2 * (p + 1))


def test_paley_rejects_wrong_residue():
    with pytest.raises(HadamardNotFound):
        paley1(13)  # 13 = 1 (mod 4)
    with pytest.raises(HadamardNotFound):
        paley2(11)  # 11 = 3 (mod 4)
    with pytest.raises(HadamardNotFound):
        paley1(9)  # not prime


@pytest.mark.parametrize("n", MODEL_DIMS)
def test_model_dims_construct(n):
    """Every dimension in the target models must have a Hadamard matrix."""
    assert_hadamard(hadamard_matrix(n), n)


@pytest.mark.parametrize(
    "n,expected",
    [
        (128, (128, 1)),
        (1536, (128, 12)),
        (3584, (128, 28)),
        (8960, (64, 140)),
        (18944, (128, 148)),
    ],
)
def test_factorize(n, expected):
    assert factorize(n) == expected
    pow2, base = expected
    assert pow2 * base == n


@pytest.mark.parametrize("n", [128, 1536, 3584])
def test_random_hadamard_is_orthogonal(n):
    r = random_hadamard_matrix(n, seed=0)
    np.testing.assert_allclose(r @ r.T, np.eye(n), atol=1e-9)
    np.testing.assert_allclose(r.T @ r, np.eye(n), atol=1e-9)


def test_random_hadamard_is_seed_deterministic():
    a = random_hadamard_matrix(256, seed=1234)
    b = random_hadamard_matrix(256, seed=1234)
    c = random_hadamard_matrix(256, seed=1235)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


@pytest.mark.parametrize("kind", ["randomized_hadamard", "hadamard", "random_orthogonal"])
def test_build_rotation_orthogonal(kind):
    r = build_rotation(1536, seed=7, kind=kind)
    np.testing.assert_allclose(r @ r.T, np.eye(1536), atol=1e-9)


def test_build_rotation_falls_back_for_impossible_order():
    """Order 6 has no Hadamard matrix; we must still get an orthogonal rotation."""
    with pytest.raises(HadamardNotFound):
        hadamard_matrix(6)
    r = build_rotation(6, seed=0)
    np.testing.assert_allclose(r @ r.T, np.eye(6), atol=1e-12)
