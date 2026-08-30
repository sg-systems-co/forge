"""FORGE command line interface."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from forge.calib.capture import LayerwiseRunner
from forge.calib.datasets import calibration_batch, evaluation_batch
from forge.config import ForgeConfig
from forge.eval.perplexity import perplexity
from forge.eval.report import markdown_table, write_report
from forge.models.registry import build_graph
from forge.quant.apply import (
    hessian_relative_error,
    quantize_model_rtn,
    quantize_weight,
    weight_relative_error,
)

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def load_model(cfg: ForgeConfig):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.model)
    model = AutoModelForCausalLM.from_pretrained(cfg.model, dtype=DTYPES[cfg.dtype])
    model.eval()
    model.to(torch.device(cfg.device))
    return model, tok, build_graph(model.config)


def _tensor_kind(gguf_name: str) -> str:
    return gguf_name.split(".", 2)[-1].rsplit(".", 1)[0]


def cmd_baseline(args) -> None:
    """Milestone 1: capture calibration Hessians and measure naive ternary error."""
    cfg = ForgeConfig.load(args.config) if args.config else ForgeConfig(model=args.model)
    if args.nsamples:
        cfg.calib.nsamples = args.nsamples
    device = torch.device(cfg.device)

    t0 = time.time()
    model, tok, graph = load_model(cfg)
    print(graph.summary(), flush=True)
    print(f"\nloaded in {time.time()-t0:.1f}s; capturing {cfg.calib.nsamples} sequences", flush=True)

    ids = calibration_batch(tok, cfg.calib.dataset, cfg.calib.nsamples, cfg.calib.seqlen,
                            cfg.calib.seed)
    runner = LayerwiseRunner(model, graph, device)

    rows, peak_hessian = [], 0
    t_cap = time.time()
    for block_pass in runner.run(ids, sequential=False):
        peak_hessian = max(peak_hessian, block_pass.hessian_bytes())
        for spec in block_pass.block.linears:
            w = model.get_submodule(spec.name).weight.data
            h = block_pass.hessians[spec.name].finalize()
            row = {"layer": spec.layer_index, "tensor": _tensor_kind(spec.gguf_name)}
            for method in ("absmean", "optimal"):
                q = quantize_weight(w, method)
                row[f"{method}_H"] = hessian_relative_error(w, q, h)
                row[f"{method}_W"] = weight_relative_error(w, q)
            rows.append(row)
            del h
        print(f"  block {block_pass.index:2d}/{graph.num_layers}  "
              f"[{time.time()-t_cap:6.1f}s]", flush=True)

    kinds = sorted({r["tensor"] for r in rows})
    summary = []
    for kind in kinds:
        sel = [r for r in rows if r["tensor"] == kind]
        summary.append({
            "tensor": kind,
            "absmean (H-weighted)": sum(r["absmean_H"] for r in sel) / len(sel),
            "optimal (H-weighted)": sum(r["optimal_H"] for r in sel) / len(sel),
            "absmean (weight-space)": sum(r["absmean_W"] for r in sel) / len(sel),
            "optimal (weight-space)": sum(r["optimal_W"] for r in sel) / len(sel),
        })
    print("\n" + markdown_table(summary))
    print(f"\npeak Hessian residency: {peak_hessian/1e6:.0f} MB")
    print(f"total capture time: {time.time()-t_cap:.1f}s")

    if args.out:
        write_report(
            args.out,
            "Milestone 1 -- baseline ternary reconstruction error",
            summary,
            notes=(
                f"Model: `{cfg.model}`  \n"
                f"Calibration: {cfg.calib.nsamples} x {cfg.calib.seqlen} tokens "
                f"from {cfg.calib.dataset} (seed {cfg.calib.seed})  \n"
                f"Metric: `||(W-Q)X|| / ||WX||`, X from calibration activations  \n"
                f"Peak Hessian residency: {peak_hessian/1e6:.0f} MB  \n"
                f"Capture time: {time.time()-t_cap:.1f}s\n"
            ),
        )
        Path(args.out).with_name("m1_per_layer.json").write_text(json.dumps(rows, indent=2))
        print(f"wrote {args.out}")


def cmd_ppl(args) -> None:
    """End-to-end perplexity, optionally after naive ternary round-to-nearest."""
    cfg = ForgeConfig.load(args.config) if args.config else ForgeConfig(model=args.model)
    device = torch.device(cfg.device)
    model, tok, graph = load_model(cfg)

    if args.rtn:
        n = quantize_model_rtn(model, graph, method=args.method)
        print(f"applied {args.method} ternary RTN to {n/1e9:.3f}B params", flush=True)

    windows = evaluation_batch(tok, cfg.calib.seqlen)
    if args.limit:
        windows = windows[: args.limit]
    t0 = time.time()
    ppl = perplexity(model, windows, device)
    label = f"ternary-rtn-{args.method}" if args.rtn else "fp16"
    print(f"{label}: wikitext2 ppl = {ppl:.4f}  "
          f"({windows.shape[0]} x {windows.shape[1]} tokens, {time.time()-t0:.1f}s)")


def main() -> None:
    parser = argparse.ArgumentParser(prog="forge", description=__doc__)
    parser.add_argument("--config", help="YAML config path")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("baseline", help="Milestone 1: calibration + baseline ternary error")
    p.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B")
    p.add_argument("--nsamples", type=int)
    p.add_argument("--out", help="write a markdown report here")
    p.set_defaults(func=cmd_baseline)

    p = sub.add_parser("ppl", help="wikitext2 perplexity")
    p.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B")
    p.add_argument("--rtn", action="store_true", help="apply naive ternary first")
    p.add_argument("--method", default="optimal", choices=["optimal", "absmean"])
    p.add_argument("--limit", type=int, help="only evaluate the first N windows")
    p.set_defaults(func=cmd_ppl)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
