"""Layer-by-layer calibration driver.

The whole pipeline is streaming: exactly one transformer block is active at a time, so
peak memory is set by one block's Hessians plus the cached hidden states, not by the model
size. That is what keeps a 7B conversion inside a consumer GPU's budget.

The driver also carries the two activation buffers the sequential mode needs:

    X_fp  -- hidden states from the *unmodified* model  (the teacher)
    X_q   -- hidden states from the *quantized-so-far* model (the student)

Layer i's Hessian is built from X_q, so each layer sees the error its predecessors already
made and absorbs it, rather than each layer being solved against a clean input it will
never actually receive at inference time.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import nn

from forge.calib.hessian import HessianAccumulator
from forge.models.graph import BlockGraph


class _StopForward(Exception):
    """Control-flow sentinel: we only want the inputs to layer 0, not a full forward."""


# past_key_values would accumulate across replayed sequences; the rest are per-call flags
# that must not leak between blocks.
_DROP_KWARGS = (
    "past_key_values", "past_key_value", "use_cache", "cache_position",
    "cache_params",  # Mamba: a mutable state object that would leak between replays
)


def _sanitize(kwargs: dict) -> dict:
    return {k: v for k, v in kwargs.items() if k not in _DROP_KWARGS}


def get_layers(model, layers_path: str):
    """Resolve the ModuleList of transformer/SSM blocks by dotted path."""
    obj = model
    for part in layers_path.split("."):
        obj = getattr(obj, part)
    return obj


@torch.no_grad()
def capture_block_inputs(
    model,
    input_ids: torch.Tensor,
    device: torch.device,
    store_dtype=torch.float16,
    layers_path: str = "model.layers",
) -> tuple[torch.Tensor, dict]:
    """Run embeddings and rotary, and catch what layer 0 would have received.

    Returns (hidden_states, shared_kwargs). Sequences are pushed through one at a time so
    the captured position_embeddings match the batch size used during replay.
    """
    layers = get_layers(model, layers_path)
    captured: list[torch.Tensor] = []
    shared: dict = {}

    class Catcher(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, hidden_states, **kwargs):
            captured.append(hidden_states.detach().to("cpu", store_dtype))
            shared.update(_sanitize(kwargs))
            raise _StopForward

    layers[0] = Catcher(layers[0])
    try:
        for i in range(input_ids.shape[0]):
            try:
                model(input_ids[i : i + 1].to(device), use_cache=False)
            except _StopForward:
                pass
    finally:
        layers[0] = layers[0].inner

    if not captured:
        raise RuntimeError("failed to capture layer-0 inputs")
    return torch.cat(captured, dim=0), shared


@contextmanager
def accumulate_hessians(
    layer: nn.Module, block: BlockGraph, device: torch.device
) -> Iterator[dict[str, HessianAccumulator]]:
    """Hook every quantizable linear in `layer` and accumulate its input second moment."""
    accs: dict[str, HessianAccumulator] = {}
    handles = []

    for spec in block.linears:
        local = spec.name.split(".", 3)[-1]  # "self_attn.q_proj" within the block
        module = layer.get_submodule(local)
        accs[spec.name] = HessianAccumulator(spec.in_features, device)

        def hook(_mod, args, _kwargs=None, _name=spec.name):
            accs[_name].update(args[0].detach())

        handles.append(module.register_forward_pre_hook(hook))

    try:
        yield accs
    finally:
        for h in handles:
            h.remove()


@dataclass
class BlockPass:
    """One block's worth of calibration state handed to the solver."""

    index: int
    layer: nn.Module
    block: BlockGraph
    hessians: dict[str, HessianAccumulator]
    inputs_fp: torch.Tensor
    inputs_q: torch.Tensor

    def hessian_bytes(self) -> int:
        return sum(a.nbytes for a in self.hessians.values())


class LayerwiseRunner:
    """Walks the model one block at a time, maintaining the teacher/student buffers."""

    def __init__(self, model, graph, device: torch.device, store_dtype=torch.float16):
        self.model = model
        self.graph = graph
        self.device = device
        self.store_dtype = store_dtype
        self._kwargs: dict = {}

    @torch.no_grad()
    def _forward_block(self, layer: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
        """Replay a block over cached hidden states, one sequence at a time."""
        outs = []
        for i in range(hidden.shape[0]):
            x = hidden[i : i + 1].to(self.device, self.model.dtype)
            y = layer(x, **self._kwargs)
            if isinstance(y, tuple):
                y = y[0]
            outs.append(y.detach().to("cpu", self.store_dtype))
        return torch.cat(outs, dim=0)

    @torch.no_grad()
    def run(self, input_ids: torch.Tensor, sequential: bool = True) -> Iterator[BlockPass]:
        """Yield one BlockPass per transformer block, in order.

        The consumer may quantize the block's weights in place before control returns. The
        pass ordering below is what makes that safe:

          1. accumulate H by replaying the *student* inputs through the still-original
             weights (the GPTQ convention: curvature from the inputs the layer will really
             see, weights not yet touched);
          2. advance the *teacher* buffer through the still-original weights -- this must
             happen before the consumer mutates anything, because the original weights are
             unrecoverable afterwards;
          3. hand control to the consumer, which quantizes in place;
          4. advance the *student* buffer through the now-quantized weights, so the next
             block inherits the error this one just made.
        """
        hidden, self._kwargs = capture_block_inputs(
            self.model, input_ids, self.device, self.store_dtype, self.graph.layers_path
        )
        layers = get_layers(self.model, self.graph.layers_path)
        inputs_fp = hidden
        inputs_q = hidden.clone() if sequential else hidden

        for index, block in enumerate(self.graph.blocks):
            layer = layers[index]

            with accumulate_hessians(layer, block, self.device) as accs:
                hessian_pass = self._forward_block(layer, inputs_q)

            # Step 2: teacher output, while the weights are still pristine. This is the
            # only chance to see the unquantized output, and non-sequential mode is
            # *defined* by feeding it forward.
            #
            # In non-sequential mode the two buffers are the same tensor, so the pass
            # above already computed exactly this -- reuse it rather than paying for an
            # identical second forward over the whole calibration set.
            outputs_fp = (
                self._forward_block(layer, inputs_fp) if sequential else hessian_pass
            )
            del hessian_pass

            yield BlockPass(
                index=index,
                layer=layer,
                block=block,
                hessians=accs,
                inputs_fp=inputs_fp,
                inputs_q=inputs_q,
            )

            for acc in accs.values():
                acc.release()

            if sequential:
                # Step 4: student output, through whatever the consumer left behind, so
                # the next block inherits the error this one just made.
                inputs_q = self._forward_block(layer, inputs_q)
                inputs_fp = outputs_fp
            else:
                # Every block sees clean inputs; error is never propagated. Advancing
                # through the *quantized* layer here would silently make this identical
                # to sequential mode, which is what tests/test_e2e_tiny.py caught.
                inputs_fp = inputs_q = outputs_fp
