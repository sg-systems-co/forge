import ctypes
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
GGML_CANDIDATES = [
    REPO / "llamacpp/llama.cpp/build/bin/libggml-base.dylib",
    REPO / "llamacpp/llama.cpp/build/bin/libggml.dylib",
    REPO / "llamacpp/llama.cpp/build/ggml/src/libggml-base.dylib",
]

GGML_TYPE_TQ2_0 = 35


@pytest.fixture(scope="session")
def ggml():
    """Load the real libggml so we can diff FORGE's packer against upstream's.

    Skips (rather than fails) when llama.cpp has not been built, so the pure-Python
    suite stays runnable on a fresh checkout.
    """
    lib_path = next((p for p in GGML_CANDIDATES if p.exists()), None)
    if lib_path is None:
        pytest.skip("libggml not built; run cmake --build llamacpp/llama.cpp/build")

    lib = ctypes.CDLL(str(lib_path))
    lib.ggml_quantize_chunk.restype = ctypes.c_size_t
    lib.ggml_quantize_chunk.argtypes = [
        ctypes.c_int,  # enum ggml_type
        ctypes.POINTER(ctypes.c_float),  # src
        ctypes.c_void_p,  # dst
        ctypes.c_int64,  # start
        ctypes.c_int64,  # nrows
        ctypes.c_int64,  # n_per_row
        ctypes.POINTER(ctypes.c_float),  # imatrix
    ]
    if hasattr(lib, "ggml_quantize_init"):
        lib.ggml_quantize_init.argtypes = [ctypes.c_int]
        lib.ggml_quantize_init(GGML_TYPE_TQ2_0)
    return lib
