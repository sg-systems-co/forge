"""Hadamard matrix construction and application.

FORGE fuses its rotations offline, so construction speed does not matter much, but
*existence* does: Qwen/Llama hidden sizes are rarely powers of two (Qwen2.5-Coder-1.5B
is 1536, the 7B is 3584), so Sylvester alone is not enough. We combine Sylvester with
Paley I/II over prime fields and a Kronecker search:

    n = 2^a * m,  where m is an order we can build directly.

Covered orders include 12 (Paley I, p=11), 28 (Paley II, p=13), 140 (Paley I, p=139)
and 148 (Paley II, p=73), which is what the target models actually need:

    1536  = 128 * 12     3584  = 128 * 28
    8960  = 64  * 140    18944 = 128 * 148
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np


class HadamardNotFound(ValueError):
    """No known construction for a Hadamard matrix of the requested order."""


def _is_pow2(n: int) -> bool:
    return n >= 1 and (n & (n - 1)) == 0


def _is_prime(p: int) -> bool:
    if p < 2:
        return False
    if p % 2 == 0:
        return p == 2
    i = 3
    while i * i <= p:
        if p % i == 0:
            return False
        i += 2
    return True


def sylvester(n: int) -> np.ndarray:
    """Sylvester Hadamard matrix of order n, n a power of two. Entries in {-1, +1}."""
    if not _is_pow2(n):
        raise HadamardNotFound(f"sylvester requires a power of two, got {n}")
    h = np.ones((1, 1), dtype=np.int8)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]]).astype(np.int8)
    return h


def _jacobsthal(p: int) -> np.ndarray:
    """Jacobsthal matrix Q[i, j] = chi(j - i) over GF(p), chi the quadratic character."""
    residues = np.zeros(p, dtype=np.int8)
    residues[1:] = -1
    for x in range(1, p):
        residues[(x * x) % p] = 1
    idx = (np.arange(p)[None, :] - np.arange(p)[:, None]) % p
    return residues[idx]


def paley1(p: int) -> np.ndarray:
    """Paley construction I: Hadamard matrix of order p + 1 for prime p = 3 (mod 4)."""
    if not _is_prime(p) or p % 4 != 3:
        raise HadamardNotFound(f"paley1 requires a prime p = 3 (mod 4), got {p}")
    n = p + 1
    s = np.zeros((n, n), dtype=np.int8)
    s[0, 1:] = 1
    s[1:, 0] = -1
    s[1:, 1:] = _jacobsthal(p)
    return (np.eye(n, dtype=np.int8) + s).astype(np.int8)


def paley2(p: int) -> np.ndarray:
    """Paley construction II: Hadamard matrix of order 2(p + 1) for prime p = 1 (mod 4)."""
    if not _is_prime(p) or p % 4 != 1:
        raise HadamardNotFound(f"paley2 requires a prime p = 1 (mod 4), got {p}")
    m = p + 1
    s = np.zeros((m, m), dtype=np.int8)
    s[0, 1:] = 1
    s[1:, 0] = 1
    s[1:, 1:] = _jacobsthal(p)  # symmetric when p = 1 (mod 4)

    a = np.array([[1, 1], [1, -1]], dtype=np.int8)
    b = np.array([[1, -1], [-1, -1]], dtype=np.int8)
    return (np.kron(s, a) + np.kron(np.eye(m, dtype=np.int8), b)).astype(np.int8)


def _direct_orders(limit: int = 512) -> dict[int, tuple[str, int]]:
    """Orders we can build without a Kronecker product -> (construction, parameter)."""
    table: dict[int, tuple[str, int]] = {1: ("trivial", 0), 2: ("sylvester", 2)}
    for p in range(3, limit):
        if not _is_prime(p):
            continue
        if p % 4 == 3 and p + 1 <= limit:
            table.setdefault(p + 1, ("paley1", p))
        if p % 4 == 1 and 2 * (p + 1) <= limit:
            table.setdefault(2 * (p + 1), ("paley2", p))
    for k in range(1, 12):
        table[2**k] = ("sylvester", 2**k)
    return table


def _build_direct(n: int) -> np.ndarray:
    if n == 1:
        return np.ones((1, 1), dtype=np.int8)
    kind, param = _direct_orders()[n]
    if kind == "sylvester":
        return sylvester(param)
    if kind == "paley1":
        return paley1(param)
    if kind == "paley2":
        return paley2(param)
    raise HadamardNotFound(f"unknown construction {kind!r}")


def factorize(n: int) -> tuple[int, int]:
    """Split n into (pow2_part, base_order) with n == pow2_part * base_order.

    Prefers the smallest non-trivial base order, i.e. the largest Sylvester factor,
    because Sylvester blocks admit the cheapest structured apply.
    """
    if n < 1:
        raise HadamardNotFound(f"order must be positive, got {n}")
    if _is_pow2(n):
        return n, 1
    table = _direct_orders()
    for base in sorted(table):
        if base == 1 or n % base != 0:
            continue
        if _is_pow2(n // base):
            return n // base, base
    raise HadamardNotFound(
        f"no Hadamard factorization n = 2^a * m for n={n} with m in the known-order table"
    )


@lru_cache(maxsize=16)
def hadamard_matrix(n: int) -> np.ndarray:
    """Unnormalized Hadamard matrix of order n, entries in {-1, +1}, H @ H.T == n * I."""
    pow2, base = factorize(n)
    if base == 1:
        return sylvester(pow2)
    h_base = _build_direct(base)
    if pow2 == 1:
        return h_base
    return np.kron(sylvester(pow2), h_base).astype(np.int8)


def random_hadamard_matrix(n: int, seed: int, dtype=np.float64) -> np.ndarray:
    """Orthogonal randomized Hadamard R = (1/sqrt(n)) * H @ diag(s), s uniform in {-1,+1}.

    The random sign flip is what makes the rotation *incoherence-processing* rather than a
    fixed basis change: it makes the probability of an unlucky alignment between the
    weight matrix and the transform vanishingly small (Halko et al., and the same argument
    QuaRot uses). The seed is recorded in the GGUF so a checkpoint is reproducible.
    """
    h = hadamard_matrix(n).astype(dtype)
    signs = np.random.default_rng(seed).integers(0, 2, size=n).astype(dtype) * 2 - 1
    return (h * signs[None, :]) / np.sqrt(n)


def random_orthogonal_matrix(n: int, seed: int, dtype=np.float64) -> np.ndarray:
    """Fallback rotation via QR of a Gaussian, for orders with no Hadamard construction."""
    g = np.random.default_rng(seed).standard_normal((n, n)).astype(dtype)
    q, r = np.linalg.qr(g)
    return q * np.sign(np.diag(r))[None, :]


def build_rotation(n: int, seed: int, kind: str = "randomized_hadamard") -> np.ndarray:
    """Build the orthogonal rotation FORGE fuses into the weights."""
    if kind == "randomized_hadamard":
        try:
            return random_hadamard_matrix(n, seed)
        except HadamardNotFound:
            return random_orthogonal_matrix(n, seed)
    if kind == "hadamard":
        return hadamard_matrix(n).astype(np.float64) / np.sqrt(n)
    if kind == "random_orthogonal":
        return random_orthogonal_matrix(n, seed)
    raise ValueError(f"unknown rotation kind {kind!r}")
