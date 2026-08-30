"""Test 5 from the plan: FORGE's TQ2_0 packer agrees with ggml byte for byte.

Two separate properties, deliberately pinned by separate tests:

1. *Layout*: given the same (ternary values, scale) choice ggml itself would make, FORGE
   must emit identical bytes. This proves we understand block_tq2_0's bit layout.
2. *Contract*: whatever scale FORGE chooses -- and its optimal scale is deliberately NOT
   ggml's amax -- packing then dequantizing must reproduce s * t exactly. This is what
   actually keeps stock llama.cpp loading FORGE checkpoints correctly.
"""

import ctypes

import numpy as np
import pytest

from forge.pack.tq2 import (
    QK_K,
    TQ2_0_BLOCK_BYTES,
    dequantize_tq2_0,
    pack_tq2_0,
    packed_nbytes,
    quantize_amax_reference,
    unpack_tq2_0,
)

GGML_TYPE_TQ2_0 = 35


def ggml_quantize(ggml, x: np.ndarray) -> bytes:
    """Quantize rows of x to TQ2_0 using the real ggml implementation."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    nrows, n_per_row = x.shape
    dst = ctypes.create_string_buffer(packed_nbytes(x.size))
    written = ggml.ggml_quantize_chunk(
        GGML_TYPE_TQ2_0,
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(dst, ctypes.c_void_p),
        0,
        nrows,
        n_per_row,
        None,
    )
    assert written == packed_nbytes(x.size), (written, packed_nbytes(x.size))
    return dst.raw[: packed_nbytes(x.size)]


# --------------------------------------------------------------------------- layout


@pytest.mark.needs_ggml
@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("shape", [(1, 256), (4, 256), (3, 1536), (2, 8960)])
def test_byte_identical_to_ggml(ggml, seed, shape):
    """FORGE's amax reimplementation must be byte-identical to ggml's quantizer."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(shape).astype(np.float32)
    assert quantize_amax_reference(x) == ggml_quantize(ggml, x)


@pytest.mark.needs_ggml
def test_byte_identical_on_adversarial_inputs(ggml):
    """Exact ties, zeros, and dead blocks are where a rounding-rule mismatch shows up."""
    cases = [
        np.zeros((1, 256), np.float32),
        np.full((1, 256), 1.0, np.float32),
        np.full((1, 256), -1.0, np.float32),
        # Exactly 0.5 after scaling: lroundf rounds half away from zero, numpy rounds
        # half to even. quantize_amax_reference must follow ggml, not numpy.
        np.tile(np.array([1.0, 0.5, -0.5, 0.0], np.float32), 64).reshape(1, 256),
        np.tile(np.array([2.0, 1.0, -1.0, 1e-8], np.float32), 64).reshape(1, 256),
    ]
    for x in cases:
        assert quantize_amax_reference(x) == ggml_quantize(ggml, x), x[0, :4]


@pytest.mark.needs_ggml
def test_ggml_bytes_decode_with_forge(ggml):
    """FORGE's decoder must correctly read buffers produced by ggml itself."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal((4, 512)).astype(np.float32)
    buf = ggml_quantize(ggml, x)
    t, scale = unpack_tq2_0(buf, x.size)

    amax = np.abs(x.reshape(-1, QK_K)).max(axis=1).astype(np.float16).astype(np.float32)
    np.testing.assert_array_equal(scale, amax)

    expected = np.sign(x.reshape(-1, QK_K)) * np.floor(
        np.abs(x.reshape(-1, QK_K)) / amax[:, None] + 0.5
    )
    np.testing.assert_array_equal(t, expected.astype(np.int8))


# --------------------------------------------------------------------------- contract


@pytest.mark.parametrize("seed", range(10))
def test_roundtrip_is_exact(seed):
    """pack -> unpack must be lossless for the ternary values and fp16-exact for scales."""
    rng = np.random.default_rng(seed)
    t = rng.integers(-1, 2, size=(5, 1024)).astype(np.int8)
    scale = (rng.random((5, 4)).astype(np.float32) + 0.25).astype(np.float16).astype(np.float32)

    buf = pack_tq2_0(t, scale)
    t2, s2 = unpack_tq2_0(buf, t.size)

    np.testing.assert_array_equal(t2.reshape(t.shape), t)
    np.testing.assert_array_equal(s2.reshape(scale.shape), scale)


def test_dequantize_matches_s_times_t():
    """The decode contract: dequant(pack(t, s)) == s * t, exactly, in fp32."""
    rng = np.random.default_rng(7)
    t = rng.integers(-1, 2, size=(3, 512)).astype(np.int8)
    scale = rng.random((3, 2)).astype(np.float16).astype(np.float32)

    got = dequantize_tq2_0(pack_tq2_0(t, scale), t.size).reshape(3, 512)
    want = t.astype(np.float32) * np.repeat(scale, QK_K, axis=1)
    np.testing.assert_array_equal(got, want)


# --------------------------------------------------------------------------- structure


def test_size_and_bitrate():
    assert packed_nbytes(256) == TQ2_0_BLOCK_BYTES == 66
    assert packed_nbytes(8960) == 8960 // 256 * 66
    assert packed_nbytes(256) * 8 / 256 == 2.0625


def test_code_three_is_never_emitted():
    """q == 3 dequantizes to 2*d and must be unreachable from FORGE's packer."""
    rng = np.random.default_rng(1)
    t = rng.integers(-1, 2, size=(8, 256)).astype(np.int8)
    buf = np.frombuffer(pack_tq2_0(t, np.ones(8, np.float32)), np.uint8)
    qs = buf.reshape(8, TQ2_0_BLOCK_BYTES)[:, :64]
    for shift in (0, 2, 4, 6):
        assert np.all(((qs >> shift) & 3) != 3)


def test_layout_places_elements_where_documented():
    """Byte 32g+m holds elements 128g+m+32n at shift 2n -- a direct probe of the layout."""
    for elem in (0, 1, 31, 32, 96, 127, 128, 159, 255):
        t = np.zeros((1, 256), np.int8)
        t[0, elem] = 1  # encodes as q = 2 -> bit pattern 0b10
        buf = np.frombuffer(pack_tq2_0(t, np.ones(1, np.float32)), np.uint8)

        g, r = divmod(elem, 128)
        n, m = divmod(r, 32)
        byte_index, shift = 32 * g + m, 2 * n

        # Every other element is 0 -> q = 1, so the "background" byte value is 0b01010101.
        assert (buf[byte_index] >> shift) & 3 == 2
        assert buf[byte_index] == 0x55 + (1 << shift)


def test_rejects_bad_shapes_and_values():
    with pytest.raises(ValueError, match="not a multiple"):
        pack_tq2_0(np.zeros((1, 300), np.int8), np.zeros(1, np.float32))
    with pytest.raises(ValueError, match="expected"):
        pack_tq2_0(np.zeros((1, 512), np.int8), np.zeros(1, np.float32))
    with pytest.raises(ValueError, match="ternary values"):
        pack_tq2_0(np.full((1, 256), 2, np.int8), np.ones(1, np.float32))
    with pytest.raises(ValueError, match="buffer is"):
        unpack_tq2_0(b"\x00" * 10, 256)


# ------------------------------------------------------- lossless re-quantization
#
# FORGE exports by writing its ternary *reconstruction* (s * t, in float) to a normal
# safetensors checkpoint, converting that to GGUF with llama.cpp's own converter, and
# then running llama-quantize to TQ2_0. That reuses all of upstream's metadata handling
# instead of hand-rolling a GGUF writer -- but it is only correct if ggml's amax
# quantizer is *lossless* on weights that are already exactly ternary.
#
# It is, and this is why: for a block whose values are s * t with t in {-1,0,1} and at
# least one non-zero, amax == s exactly, so lround(w / amax) == t exactly. These tests
# pin that property, including the edge cases where it could fail.


def _reconstruct(t, scale):
    return (t.astype(np.float32) * np.repeat(scale, QK_K, axis=1)).astype(np.float32)


@pytest.mark.parametrize("seed", range(10))
def test_amax_requantization_is_lossless_on_ternary_weights(seed):
    """The property the whole export path rests on."""
    rng = np.random.default_rng(seed)
    t = rng.integers(-1, 2, size=(6, 1024)).astype(np.int8)
    # Guarantee each block has at least one non-zero, so amax == scale.
    t[:, ::QK_K] = 1
    scale = (rng.random((6, 4)).astype(np.float32) + 0.1).astype(np.float16).astype(np.float32)

    buf = quantize_amax_reference(_reconstruct(t, scale))
    t2, s2 = unpack_tq2_0(buf, t.size)

    np.testing.assert_array_equal(t2.reshape(t.shape), t)
    np.testing.assert_array_equal(s2.reshape(scale.shape), scale)


@pytest.mark.needs_ggml
@pytest.mark.parametrize("seed", range(5))
def test_ggml_requantization_is_lossless_on_ternary_weights(ggml, seed):
    """Same property, but asserted against the real ggml quantizer rather than ours."""
    rng = np.random.default_rng(seed)
    t = rng.integers(-1, 2, size=(4, 768)).astype(np.int8)
    t[:, ::QK_K] = -1
    scale = (rng.random((4, 3)).astype(np.float32) + 0.1).astype(np.float16).astype(np.float32)

    buf = ggml_quantize(ggml, _reconstruct(t, scale))
    t2, s2 = unpack_tq2_0(buf, t.size)

    np.testing.assert_array_equal(t2.reshape(t.shape), t)
    np.testing.assert_array_equal(s2.reshape(scale.shape), scale)


def test_all_zero_block_survives_requantization():
    """A dead block has amax == 0, so ggml writes d = 0 and all-zero codes.

    The reconstruction is 0 either way, so this is lossless in *value* even though the
    scale is not preserved. Worth pinning explicitly: it is the one case where the
    round trip does not reproduce the scale.
    """
    t = np.zeros((1, QK_K), np.int8)
    scale = np.array([[0.75]], np.float32)
    buf = quantize_amax_reference(_reconstruct(t, scale))
    t2, s2 = unpack_tq2_0(buf, QK_K)
    np.testing.assert_array_equal(t2, t)
    assert s2[0] == 0.0
    np.testing.assert_array_equal(
        t2.astype(np.float32) * s2[0], np.zeros((1, QK_K), np.float32)
    )


def test_scale_must_be_fp16_representable():
    """FORGE rounds scales to fp16 before packing; if it did not, the export round trip
    would silently change them. This documents why that rounding is mandatory."""
    t = np.ones((1, QK_K), np.int8)
    scale_f32 = np.array([[0.1234567]], np.float32)
    buf = quantize_amax_reference(_reconstruct(t, scale_f32))
    _, s2 = unpack_tq2_0(buf, QK_K)
    assert s2[0] == np.float16(0.1234567).astype(np.float32)
    assert s2[0] != scale_f32[0, 0]  # the fp32 value did not survive, as expected
