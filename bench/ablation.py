"""Milestone 2b ablation grid.

Each component of the pipeline is switched off in turn so its marginal contribution is
measurable rather than asserted. Runs are deliberately small (32 calibration sequences,
16 perplexity windows) so the whole grid finishes in roughly an hour; the headline number
comes from a single full-size run.

    python bench/ablation.py --nsamples 32 --limit 16 --out docs/results_ablation.md
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from forge.calib.datasets import calibration_batch, evaluation_batch
from forge.cli import load_model
from forge.config import ForgeConfig
from forge.eval.perplexity import perplexity
from forge.eval.report import markdown_table
from forge.quant.sequential import quantize_model
from forge.rotate.fuse import fuse_rotations

# (label, rotate, method, scale_rule, sequential, rescale)
GRID = [
    ("RTN absmean, no rotation",        False, "rtn",  "absmean", False, False),
    ("RTN optimal, no rotation",        False, "rtn",  "optimal", False, False),
    ("+ rotation",                      True,  "rtn",  "optimal", False, False),
    ("+ GPTQ solver",                   True,  "gptq", "optimal", False, False),
    ("+ sequential propagation",        True,  "gptq", "optimal", True,  False),
    ("+ channel rescale (full FORGE)",  True,  "gptq", "optimal", True,  True),
    ("full FORGE, no rotation",         False, "gptq", "optimal", True,  True),
    ("full FORGE, absmean scale",       True,  "gptq", "absmean", True,  True),
]


def run_one(spec, args) -> dict:
    label, rotate, method, scale_rule, sequential, rescale = spec
    cfg = ForgeConfig(model=args.model, dtype="float32")
    cfg.calib.nsamples = args.nsamples
    cfg.rotation.enabled = rotate
    cfg.solver.method = method
    cfg.solver.scale_rule = scale_rule
    cfg.solver.sequential = sequential
    cfg.solver.rescale = rescale

    device = torch.device(cfg.device)
    model, tok, graph = load_model(cfg)
    if rotate:
        fuse_rotations(model, graph, seed=cfg.rotation.seed, kind=cfg.rotation.kind,
                       dtype=torch.float32)

    ids = calibration_batch(tok, cfg.calib.dataset, cfg.calib.nsamples, cfg.calib.seqlen,
                            cfg.calib.seed)
    t0 = time.time()
    report = quantize_model(model, graph, ids, cfg, device, verbose=False)

    windows = evaluation_batch(tok, cfg.calib.seqlen)[: args.limit]
    ppl = perplexity(model, windows, device, progress=False)

    del model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return {
        "config": label,
        "ppl": round(ppl, 3),
        "rel_error": round(report.mean("rel_error"), 4),
        "attenuation": round(report.mean("attenuation"), 4),
        "minutes": round((time.time() - t0) / 60, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B")
    ap.add_argument("--nsamples", type=int, default=32)
    ap.add_argument("--limit", type=int, default=16)
    ap.add_argument("--out", default="docs/results_ablation.md")
    args = ap.parse_args()

    rows = []
    for spec in GRID:
        print(f"=== {spec[0]} ===", flush=True)
        try:
            row = run_one(spec, args)
        except Exception as exc:  # keep the grid going; a failed cell is data too
            row = {"config": spec[0], "ppl": float("nan"), "error": str(exc)[:80]}
        rows.append(row)
        print(f"    {row}", flush=True)
        Path(args.out).with_suffix(".json").write_text(json.dumps(rows, indent=2))

    table = markdown_table(rows, ["config", "ppl", "rel_error", "attenuation", "minutes"])
    Path(args.out).write_text(
        f"# Milestone 2b -- ablation\n\n"
        f"Model: `{args.model}`, {args.nsamples} calibration sequences, "
        f"{args.limit} perplexity windows.\n\n{table}\n"
    )
    print("\n" + table)


if __name__ == "__main__":
    main()
