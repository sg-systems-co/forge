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

FORGE_VERSION = "0.1.0"


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


def forge_metadata(cfg, rotation_plan=None) -> dict[str, str]:
    """The KV entries that make a FORGE checkpoint reproducible."""
    meta = {
        "forge.version": FORGE_VERSION,
        "forge.solver": (
            f"{cfg.solver.method}_ternary"
            f"{'_sequential' if cfg.solver.sequential else ''}"
        ),
        "forge.scale_rule": cfg.solver.scale_rule,
        "forge.damping": str(cfg.solver.damping),
        "forge.calib.dataset": cfg.calib.dataset,
        "forge.calib.nsamples": str(cfg.calib.nsamples),
        "forge.calib.seqlen": str(cfg.calib.seqlen),
        "forge.calib.seed": str(cfg.calib.seed),
        "forge.rotation.kind": cfg.rotation.kind if cfg.rotation.enabled else "none",
        "forge.rotation.seed": str(cfg.rotation.seed),
    }
    if rotation_plan is not None:
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
) -> Path:
    """Write a stock TQ2_0 GGUF for a model whose weights are already exactly ternary.

    The model must be the *reconstruction* (s * t in float), which is what
    forge.quant.sequential.quantize_model leaves behind.
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
        _run(
            [str(paths.quantize_bin), str(f16_path), str(out_path), "TQ2_0", "8"],
            "llama-quantize",
        )

        # Upstream's gguf_new_metadata.py only exposes a fixed set of general.* flags, so
        # FORGE's custom keys go in a sidecar next to the GGUF rather than being forced
        # into fields that mean something else. The GGUF itself stays exactly what stock
        # llama.cpp expects.
        meta = forge_metadata(cfg, rotation_plan)
        out_path.with_suffix(".forge.json").write_text(json.dumps(meta, indent=2))

        return out_path
    finally:
        if not keep_intermediate:
            shutil.rmtree(tmp, ignore_errors=True)
