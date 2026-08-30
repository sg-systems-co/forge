"""Layer-by-layer quantization driver with cross-layer error propagation.

Each block is solved against the activations the *quantized* model actually produces, not
the ones the original model would have. That single change is what stops error from
compounding across depth:

  * Ternary reconstruction is an orthogonal projection, so every layer attenuates its
    output by roughly ||Q||/||W|| ~ 0.9. That shortfall cannot be fixed inside the layer
    (scaling a projection up only raises its MSE -- measured, and it is why
    quant/rescale.py's alpha comes out at ~1.0).
  * It can be fixed *downstream*. When layer i is solved against the already-attenuated
    X_q, the least-squares solution naturally comes out larger, absorbing its
    predecessors' shortfall instead of stacking on top of it.

So the driver keeps two buffers -- the teacher's activations and the student's -- and
hands the solver the student's. See calib/capture.py for the pass ordering that makes
this safe to do in place.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from forge.calib.capture import LayerwiseRunner
from forge.config import ForgeConfig
from forge.quant.gptq import gptq_quantize_layer, hessian_flatness, rtn_quantize_layer
from forge.quant.rescale import attenuation, channel_rescale


@dataclass
class LayerRecord:
    layer: int
    tensor: str
    rel_error: float
    attenuation: float
    sparsity: float
    flatness: float


@dataclass
class QuantizationReport:
    records: list[LayerRecord] = field(default_factory=list)
    seconds: float = 0.0
    peak_hessian_mb: float = 0.0

    def mean(self, field_name: str) -> float:
        vals = [getattr(r, field_name) for r in self.records]
        return sum(vals) / len(vals) if vals else float("nan")

    def by_tensor(self) -> list[dict]:
        kinds = sorted({r.tensor for r in self.records})
        rows = []
        for kind in kinds:
            sel = [r for r in self.records if r.tensor == kind]
            rows.append({
                "tensor": kind,
                "rel_error": sum(r.rel_error for r in sel) / len(sel),
                "attenuation": sum(r.attenuation for r in sel) / len(sel),
                "sparsity": sum(r.sparsity for r in sel) / len(sel),
                "H flatness": sum(r.flatness for r in sel) / len(sel),
            })
        return rows


def _tensor_kind(gguf_name: str) -> str:
    return gguf_name.split(".", 2)[-1].rsplit(".", 1)[0]


@torch.no_grad()
def quantize_model(
    model,
    graph,
    input_ids: torch.Tensor,
    cfg: ForgeConfig,
    device: torch.device,
    verbose: bool = True,
) -> QuantizationReport:
    """Quantize every linear in the model in place, returning per-layer diagnostics.

    Weights are replaced by their ternary *reconstruction* (still float), so the model
    stays runnable and can be evaluated directly. Packing to TQ2_0 is a separate step.
    """
    solver = cfg.solver
    report = QuantizationReport()
    runner = LayerwiseRunner(model, graph, device)
    t_start = time.time()

    for block_pass in runner.run(input_ids, sequential=solver.sequential):
        report.peak_hessian_mb = max(report.peak_hessian_mb, block_pass.hessian_bytes() / 1e6)
        t_block = time.time()

        for spec in block_pass.block.linears:
            if _tensor_kind(spec.gguf_name) in solver.exclude:
                continue
            module = model.get_submodule(spec.name)
            weight = module.weight.data.float()
            hessian = block_pass.hessians[spec.name].finalize()

            if solver.method == "gptq":
                result = gptq_quantize_layer(
                    weight, hessian, damping=solver.damping, scale_rule=solver.scale_rule,
                    factorization=solver.factorization,
                )
            elif solver.method == "rtn":
                result = rtn_quantize_layer(weight, hessian, scale_rule=solver.scale_rule)
            else:
                raise ValueError(f"unknown solver method {solver.method!r}")

            if solver.rescale:
                result.scales = channel_rescale(weight, result.codes, result.scales, hessian)

            recon = result.dequantize()
            module.weight.data = recon.to(module.weight.dtype)

            report.records.append(
                LayerRecord(
                    layer=spec.layer_index,
                    tensor=_tensor_kind(spec.gguf_name),
                    rel_error=result.relative_error,
                    attenuation=attenuation(weight, recon, hessian),
                    sparsity=result.sparsity,
                    flatness=hessian_flatness(hessian),
                )
            )
            del hessian, weight, recon

        if verbose:
            per_block = len(block_pass.block.linears) - len(solver.exclude)
            recent = report.records[-per_block:] if per_block else []
            mean_err = (
                sum(r.rel_error for r in recent) / len(recent) if recent else float("nan")
            )
            print(
                f"  block {block_pass.index:2d}/{graph.num_layers}  "
                f"rel_err={mean_err:.4f}  "
                f"[{time.time()-t_block:5.1f}s, {time.time()-t_start:6.1f}s total]",
                flush=True,
            )

    report.seconds = time.time() - t_start
    return report
