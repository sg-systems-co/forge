"""Export a FORGE-quantized model to a stock GGUF.

FORGE does not hand-roll a GGUF writer. It reuses llama.cpp's own converter and quantizer,
which is possible because of one property proved in tests/test_pack_bitexact.py:

    ggml's amax quantizer is LOSSLESS on weights that are already exactly ternary.

For a block whose values are s * t with t in {-1,0,1} and at least one non-zero entry,
`amax == s` exactly, so `lroundf(w / amax) == t` exactly. So the pipeline can be:

    quantized model (weights = s * t, still float)
      -> save_pretrained            (a normal safetensors checkpoint)
      -> convert_hf_to_gguf.py      (all the tokenizer/rope/metadata handling, for free)
      -> llama-quantize TQ2_0       (reproduces FORGE's codes and scales bit for bit)

and the result is a stock GGUF carrying FORGE's optimal scales rather than amax scales,
even though upstream's amax quantizer is what wrote it. Hand-writing the container would
have meant reimplementing tokenizer export and chat templates for every architecture, and
getting one detail wrong produces a file that loads but generates garbage.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from forge.pack.tq2 import QK_K

FORGE_VERSION = "1.0.0"


@dataclass
class ExportPaths:
    """Where the llama.cpp tooling lives. Defaults assume the pinned submodule."""

    repo: Path

    @classmethod
    def default(cls) -> ExportPaths:
        return cls(repo=Path(__file__).resolve().parents[2] / "llamacpp" / "llama.cpp")

    @property
    def converter(self) -> Path:
        return self.repo / "convert_hf_to_gguf.py"

    @property
    def quantize_bin(self) -> Path:
        return self.repo / "build" / "bin" / "llama-quantize"

    @property
    def new_metadata(self) -> Path:
        return self.repo / "gguf-py" / "gguf" / "scripts" / "gguf_new_metadata.py"

    def check(self) -> None:
        if not self.converter.exists():
            raise FileNotFoundError(
                f"{self.converter} not found; run `git submodule update --init`"
            )
        if not self.quantize_bin.exists():
            raise FileNotFoundError(
                f"{self.quantize_bin} not found; build llama.cpp first:\n"
                f"  cmake -S {self.repo} -B {self.repo}/build -DCMAKE_BUILD_TYPE=Release\n"
                f"  cmake --build {self.repo}/build -j"
            )


def _run(cmd: list[str], desc: str) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or proc.stdout).splitlines()[-20:])
        raise RuntimeError(f"{desc} failed (exit {proc.returncode}):\n{tail}")


def _llamacpp_revision() -> str:
    """Pinned llama.cpp SHA -- the packing layout and quantizer come from it."""
    try:
        pin = Path(__file__).resolve().parents[2] / "llamacpp" / "PINNED_SHA"
        return pin.read_text().strip()[:12] if pin.exists() else "unknown"
    except OSError:
        return "unknown"


def forge_metadata(cfg, rotation_plan=None) -> dict[str, str]:
    """The full reproducibility specification for a FORGE checkpoint.

    Everything needed to regenerate the identical file: the source model, the calibration
    corpus and how it was sampled, every solver knob, the rotation seed, and the pinned
    llama.cpp revision whose quantizer wrote the container. Two runs with the same values
    produce byte-identical output.
    """
    meta = {
        # provenance
        "forge.version": FORGE_VERSION,
        "forge.source_model": str(cfg.model),
        "forge.llamacpp_revision": _llamacpp_revision(),
        "forge.compute_dtype": str(cfg.dtype),
        # calibration
        "forge.calib.dataset": cfg.calib.dataset,
        "forge.calib.split": "train",
        "forge.calib.nsamples": str(cfg.calib.nsamples),
        "forge.calib.seqlen": str(cfg.calib.seqlen),
        "forge.calib.seed": str(cfg.calib.seed),
        # solver
        "forge.solver": (
            f"{cfg.solver.method}_ternary"
            f"{'_sequential' if cfg.solver.sequential else ''}"
        ),
        "forge.solver.method": cfg.solver.method,
        "forge.solver.scale_rule": cfg.solver.scale_rule,
        "forge.solver.damping": str(cfg.solver.damping),
        "forge.solver.sequential": str(bool(cfg.solver.sequential)).lower(),
        "forge.solver.rescale": str(bool(cfg.solver.rescale)).lower(),
        "forge.solver.factorization": cfg.solver.factorization,
        "forge.solver.block_size": str(QK_K),
        "forge.solver.excluded_tensors": ",".join(cfg.solver.exclude) or "none",
        # rotation
        "forge.rotation.enabled": str(bool(cfg.rotation.enabled)).lower(),
        "forge.rotation.kind": cfg.rotation.kind if cfg.rotation.enabled else "none",
        "forge.rotation.seed": str(cfg.rotation.seed),
        "forge.rotation.head_dim_rotation": str(bool(cfg.rotation.rotate_head_dim)).lower(),
        # evaluation convention, so a reported ppl is never ambiguous
        "forge.eval.convention": "all_tokens",
        "forge.eval.note": (
            "llama-perplexity scores only the second half of each window "
            "(first = n_ctx/2); compare ratios, not absolutes"
        ),
    }
    if rotation_plan is not None:
        meta["forge.rotation.hidden_size"] = str(rotation_plan.hidden_size)
        meta["forge.rotation.head_dim"] = str(rotation_plan.head_dim or 0)
    return meta


def export_gguf(
    model,
    tokenizer,
    out_path: str | Path,
    cfg,
    rotation_plan=None,
    paths: ExportPaths | None = None,
    workdir: str | Path | None = None,
    keep_intermediate: bool = False,
    tensor_types: dict[str, str] | None = None,
) -> Path:
    """Write a stock TQ2_0 GGUF for a model whose weights are already exactly ternary.

    The model must be the *reconstruction* (s * t in float), which is what
    forge.quant.sequential.quantize_model leaves behind.

    `tensor_types` maps a GGUF tensor stem to a ggml type for tensors that were left out
    of ternarization (see SolverConfig.exclude), e.g. {"ffn_down": "q6_K"}. These become
    `--tensor-type` overrides, which stock llama-quantize supports -- so a mixed-precision
    FORGE checkpoint is still an ordinary GGUF that unmodified llama.cpp loads.
    """
    paths = paths or ExportPaths.default()
    paths.check()
    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tmp = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="forge-export-"))
    tmp.mkdir(parents=True, exist_ok=True)
    hf_dir = tmp / "hf"
    f16_path = tmp / "model-f16.gguf"

    try:
        # float32 keeps the exact s*t values; f16 would perturb them before the converter
        # ever sees them, and the losslessness argument depends on them being exact.
        model.to("cpu").float().save_pretrained(hf_dir, safe_serialization=True)
        tokenizer.save_pretrained(hf_dir)

        _run(
            [sys.executable, str(paths.converter), str(hf_dir),
             "--outfile", str(f16_path), "--outtype", "f16"],
            "convert_hf_to_gguf",
        )
        quantize_cmd = [str(paths.quantize_bin)]
        for stem, ggml_type in (tensor_types or {}).items():
            quantize_cmd += ["--tensor-type", f"{stem}={ggml_type}"]
        quantize_cmd += [str(f16_path), str(out_path), "TQ2_0", "8"]
        _run(quantize_cmd, "llama-quantize")

        # Upstream's gguf_new_metadata.py only exposes a fixed set of general.* flags, so
        # FORGE's custom keys go in a sidecar next to the GGUF rather than being forced
        # into fields that mean something else. The GGUF itself stays exactly what stock
        # llama.cpp expects.
        meta = forge_metadata(cfg, rotation_plan)
        if tensor_types:
            meta["forge.tensor_types"] = ",".join(
                f"{k}={v}" for k, v in sorted(tensor_types.items())
            )
        out_path.with_suffix(".forge.json").write_text(json.dumps(meta, indent=2))

        return out_path
    finally:
        if not keep_intermediate:
            shutil.rmtree(tmp, ignore_errors=True)
