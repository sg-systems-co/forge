"""Export-path guards.

The full export (save -> convert_hf_to_gguf -> llama-quantize) is exercised end to end by
tests/test_e2e_tiny.py; these cover the parts that must hold without invoking the toolchain.
"""

import pytest

from forge.config import ForgeConfig
from forge.pack.gguf_writer import FORGE_VERSION, ExportPaths, forge_metadata

# Every knob that changes the produced bytes must appear in the sidecar, or a checkpoint
# is not reproducible from it.
REQUIRED_KEYS = {
    "forge.version", "forge.source_model", "forge.llamacpp_revision", "forge.compute_dtype",
    "forge.calib.dataset", "forge.calib.split", "forge.calib.nsamples",
    "forge.calib.seqlen", "forge.calib.seed",
    "forge.solver.method", "forge.solver.scale_rule", "forge.solver.damping",
    "forge.solver.sequential", "forge.solver.rescale", "forge.solver.factorization",
    "forge.solver.block_size", "forge.solver.excluded_tensors",
    "forge.rotation.enabled", "forge.rotation.kind", "forge.rotation.seed",
    "forge.eval.convention",
}


def test_metadata_records_everything_needed_to_reproduce():
    cfg = ForgeConfig()
    cfg.calib.nsamples = 256
    cfg.rotation.seed = 99
    meta = forge_metadata(cfg)

    missing = REQUIRED_KEYS - set(meta)
    assert not missing, f"sidecar is missing {sorted(missing)}"

    assert meta["forge.version"] == FORGE_VERSION
    assert meta["forge.calib.nsamples"] == "256"
    assert meta["forge.rotation.seed"] == "99"
    assert meta["forge.rotation.kind"] == "randomized_hadamard"
    assert meta["forge.solver"] == "gptq_ternary_sequential"
    assert all(isinstance(v, str) for v in meta.values()), "GGUF KV values must be strings"


def test_metadata_records_the_pinned_llamacpp_revision():
    """The packing layout and the quantizer both come from llama.cpp, so the checkpoint is
    only reproducible if the revision that wrote it is recorded."""
    rev = forge_metadata(ForgeConfig())["forge.llamacpp_revision"]
    assert rev != "unknown" and len(rev) == 12, rev


def test_metadata_records_excluded_tensors_and_eval_convention():
    cfg = ForgeConfig()
    assert forge_metadata(cfg)["forge.solver.excluded_tensors"] == "none"
    cfg.solver.exclude = ("ffn_down",)
    assert forge_metadata(cfg)["forge.solver.excluded_tensors"] == "ffn_down"
    # Perplexity is meaningless without saying which convention produced it.
    assert forge_metadata(cfg)["forge.eval.convention"] == "all_tokens"


def test_metadata_records_rotation_disabled():
    cfg = ForgeConfig()
    cfg.rotation.enabled = False
    assert forge_metadata(cfg)["forge.rotation.kind"] == "none"


def test_metadata_records_non_sequential_solver():
    cfg = ForgeConfig()
    cfg.solver.sequential = False
    assert forge_metadata(cfg)["forge.solver"] == "gptq_ternary"


def test_export_paths_point_at_the_submodule():
    paths = ExportPaths.default()
    assert paths.converter.name == "convert_hf_to_gguf.py"
    assert paths.quantize_bin.name == "llama-quantize"
    assert paths.repo.name == "llama.cpp"


def test_missing_toolchain_gives_an_actionable_error(tmp_path):
    paths = ExportPaths(repo=tmp_path)
    with pytest.raises(FileNotFoundError, match="submodule update"):
        paths.check()
