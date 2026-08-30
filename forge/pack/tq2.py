"""TQ2_0 block serialization.

Mirrors llama.cpp's `block_tq2_0` exactly, so FORGE checkpoints load on stock llama.cpp:

    typedef struct {
        uint8_t qs[QK_K/4];   // 64 bytes, 2 bits per weight
        ggml_half d;          // fp16 block scale
    } block_tq2_0;            // 66 bytes per 256 weights = 2.0625 bpw

Layout (read off `quantize_row_tq2_0_ref` in ggml/src/ggml-quants.c, not guessed). The 64
bytes are TWO independent groups of 32 bytes; each group encodes 128 consecutive weights:

    element e  ->  group g = e // 128,  r = e % 128,  n = r // 32,  m = r % 32
                   byte index = 32*g + m,  bit shift = 2*n

so byte (32g + m) packs elements {128g + m, 128g + m + 32, 128g + m + 64, 128g + m + 96}
at shifts 0, 2, 4, 6. A SIMD kernel therefore recovers four contiguous 32-lane vectors
with four shift/mask pairs and no cross-lane shuffles.

Values are stored biased: t in {-1, 0, 1} is written as q = t + 1 in {0, 1, 2}. The code
3 is unreachable and FORGE never emits it. Dequantization is `(q - 1) * d`.

Note on scales: ggml's own `quantize_row_tq2_0_ref` sets d = max|x| over the block, which
is a deliberately cheap choice. FORGE supplies its own (optimal, see quant/ternary.py)
scale, so the two quantizers agree on *layout* but not on *values* -- see
tests/test_pack_bitexact.py, which pins both properties separately.
"""

from __future__ import annotations

import numpy as np

QK_K = 256
TQ2_0_QS_BYTES = QK_K // 4  # 64
TQ2_0_BLOCK_BYTES = TQ2_0_QS_BYTES + 2  # + fp16 scale
BITS_PER_WEIGHT = TQ2_0_BLOCK_BYTES * 8 / QK_K  # 2.0625

_SHIFT_WEIGHTS = np.array([1, 4, 16, 64], dtype=np.uint16)  # 1 << (2*n)


def _to_group_view(x: np.ndarray) -> np.ndarray:
    """(nblocks, 256) -> (nblocks, 2, 4, 32) indexed [block, group, shift, byte]."""
    return x.reshape(-1, 2, 4, 32)


def pack_tq2_0(t: np.ndarray, scale: np.ndarray) -> bytes:
    """Pack ternary values and per-block scales into TQ2_0 blocks.

    t: integer array (..., n) with values in {-1, 0, 1}; n must be a multiple of 256.
    scale: float array (..., n // 256), one scale per block.
    Returns the raw little-endian byte string, blocks in row-major order.
    """
    t = np.asarray(t)
    scale = np.asarray(scale, dtype=np.float32)

    if t.shape[-1] % QK_K != 0:
        raise ValueError(f"row length {t.shape[-1]} is not a multiple of {QK_K}")
    nblocks = t.size // QK_K
    if scale.size != nblocks:
        raise ValueError(f"expected {nblocks} scales, got {scale.size}")
    if t.min() < -1 or t.max() > 1:
        raise ValueError("ternary values must lie in {-1, 0, 1}")

    q = (t.reshape(-1, QK_K).astype(np.int16) + 1).astype(np.uint16)  # {0,1,2}

    # [block, group, shift, byte] -> weighted sum over the shift axis.
    grouped = _to_group_view(q)
    packed = (grouped * _SHIFT_WEIGHTS[None, None, :, None]).sum(axis=2, dtype=np.uint16)
    qs = packed.reshape(nblocks, TQ2_0_QS_BYTES).astype(np.uint8)

    d = scale.reshape(nblocks).astype(np.float16).view(np.uint8).reshape(nblocks, 2)

    out = np.empty((nblocks, TQ2_0_BLOCK_BYTES), dtype=np.uint8)
    out[:, :TQ2_0_QS_BYTES] = qs
    out[:, TQ2_0_QS_BYTES:] = d
    return out.tobytes()


def unpack_tq2_0(buf: bytes | np.ndarray, nweights: int) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of pack_tq2_0. Returns (t, scale) with t in {-1, 0, 1}."""
    if nweights % QK_K != 0:
        raise ValueError(f"nweights {nweights} is not a multiple of {QK_K}")
    nblocks = nweights // QK_K
    raw = np.frombuffer(buf, dtype=np.uint8)
    if raw.size != nblocks * TQ2_0_BLOCK_BYTES:
        raise ValueError(
            f"buffer is {raw.size} bytes, expected {nblocks * TQ2_0_BLOCK_BYTES} "
            f"for {nblocks} blocks"
        )
    raw = raw.reshape(nblocks, TQ2_0_BLOCK_BYTES)

    qs = raw[:, :TQ2_0_QS_BYTES].reshape(nblocks, 2, 1, 32)
    shifts = np.array([0, 2, 4, 6], dtype=np.uint8)[None, None, :, None]
    q = (qs >> shifts) & 3  # (nblocks, 2, 4, 32)

    t = q.reshape(nblocks, QK_K).astype(np.int8) - 1
    scale = raw[:, TQ2_0_QS_BYTES:].copy().view(np.float16).reshape(nblocks).astype(np.float32)
    return t, scale


def dequantize_tq2_0(buf: bytes | np.ndarray, nweights: int) -> np.ndarray:
    """Decode a TQ2_0 buffer to float32, matching ggml's `dequantize_row_tq2_0`."""
    t, scale = unpack_tq2_0(buf, nweights)
    return (t.astype(np.float32) * scale[:, None]).reshape(nweights)


def quantize_amax_reference(x: np.ndarray) -> bytes:
    """Reimplementation of ggml's own `quantize_row_tq2_0_ref` (d = max|x|).

    FORGE does not use this to build checkpoints -- its scale is strictly worse than the
    optimal one in quant/ternary.py -- but it lets us assert byte-for-byte agreement with
    upstream and so prove our layout is right.
    """
    x = np.asarray(x, dtype=np.float32).reshape(-1, QK_K)
    amax = np.abs(x).max(axis=1)
    d = amax.astype(np.float16).astype(np.float32)
    inv = np.where(amax != 0, 1.0 / np.where(amax != 0, amax, 1.0), 0.0)

    # ggml rounds with lroundf (half away from zero), which is NOT numpy's banker rounding.
    scaled = x * inv[:, None]
    t = np.sign(scaled) * np.floor(np.abs(scaled) + 0.5)
    return pack_tq2_0(t.astype(np.int8), d)


def packed_nbytes(nweights: int) -> int:
    """Size on disk of a TQ2_0 tensor with `nweights` elements."""
    if nweights % QK_K != 0:
        raise ValueError(f"nweights {nweights} is not a multiple of {QK_K}")
    return nweights // QK_K * TQ2_0_BLOCK_BYTES
