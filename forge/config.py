"""Run configuration. Every knob the milestones ablate lives here."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


@dataclass
class CalibConfig:
    dataset: str = "wikitext2"
    nsamples: int = 128
    seqlen: int = 2048
    seed: int = 0


@dataclass
class RotationConfig:
    enabled: bool = True
    kind: str = "randomized_hadamard"
    seed: int = 0
    rotate_head_dim: bool = True  # R3, the fusable per-head value rotation


@dataclass
class SolverConfig:
    method: str = "gptq"  # "rtn" | "gptq"
    scale_rule: str = "optimal"  # "optimal" | "absmean"
    damping: float = 0.01
    sequential: bool = True
    rescale: bool = True


@dataclass
class ForgeConfig:
    model: str = "Qwen/Qwen2.5-Coder-1.5B"
    device: str = "mps"
    dtype: str = "bfloat16"
    calib: CalibConfig = field(default_factory=CalibConfig)
    rotation: RotationConfig = field(default_factory=RotationConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)

    @classmethod
    def load(cls, path: str | Path) -> ForgeConfig:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        return cls(
            **{
                **{k: v for k, v in raw.items() if k not in ("calib", "rotation", "solver")},
                "calib": CalibConfig(**raw.get("calib", {})),
                "rotation": RotationConfig(**raw.get("rotation", {})),
                "solver": SolverConfig(**raw.get("solver", {})),
            }
        )

    def to_dict(self) -> dict:
        return asdict(self)
