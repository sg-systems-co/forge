"""Fuse orthogonal rotations into the weights.

Everything here happens offline and changes no computation the runtime performs: the
rotated model is mathematically identical to the original, which is exactly what
tests/test_rotation_invariance.py asserts. The point is that after rotation the *weights*
are far closer to Gaussian, so ternary rounding destroys much less.

Why every rotation FORGE uses is fusable
----------------------------------------
QuaRot and SpinQuant need online Hadamard transforms at inference time, which would force
a runtime graph change. Those online rotations exist mainly to tame *activation* outliers.
FORGE v1 is weight-only -- activations stay in the runtime's native precision -- so the
rotation's only job is to Gaussianize the weights, and the two rotations that do that are
both fusable:

  R1  a global rotation of the residual stream (hidden_size)
  R3  a per-head rotation between v_proj and o_proj (head_dim)

The two non-fusable ones (an online Hadamard before down_proj, and a post-RoPE q/k
rotation) are deliberately out of scope for v1.

R1: rotating the residual basis
-------------------------------
For orthogonal R, ||R.T x|| == ||x||, so a *plain* RMSNorm commutes with it. Qwen/Llama
RMSNorm carries a learned gain g, which does not, so the gain is folded into its consumers
first (W <- W @ diag(g)) and the norm is set to all-ones. Then with x~ = R.T x:

    token_embd (writer)   E  <- E @ R          rows are token vectors
    q,k,v,gate,up (reader) W <- W @ R
    o_proj, down_proj (writer) W <- R.T @ W
    lm_head (reader)      W  <- W @ R

Biases are untouched: an input rotation does not affect them, and the two writers are
bias-free in both architectures (asserted in tests/test_graph.py).

R3: the per-head value rotation
-------------------------------
Attention output is linear in V, so a head_dim x head_dim Hadamard applied to v_proj's
output cancels exactly when the inverse is applied to o_proj's input:

    W_v <- blockdiag(H, n_kv_heads) @ W_v       (and the same for v_proj's bias)
    W_o <- W_o @ blockdiag(H, n_heads).T

Under GQA the KV heads are repeated to n_heads before o_proj, so the two block-diagonals
have *different* repeat counts. Getting that wrong is the single most likely bug in this
file, which is why test_rotation_invariance covers GQA explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from forge.models.graph import ModelGraph, Role
from forge.rotate.hadamard import build_rotation, hadamard_matrix


@dataclass
class RotationPlan:
    """The rotations applied to a model, kept so they can be recorded in the GGUF."""

    kind: str
    seed: int
    hidden_size: int
    head_dim: int | None
    r1: torch.Tensor
    r3: torch.Tensor | None

    def metadata(self) -> dict[str, object]:
        return {
            "forge.rotation.kind": self.kind,
            "forge.rotation.seed": self.seed,
            "forge.rotation.head_dim": self.head_dim if self.r3 is not None else 0,
        }


def _block_diag(h: torch.Tensor, repeats: int) -> torch.Tensor:
    """blockdiag(h, repeats) without materializing the zeros first."""
    n = h.shape[0]
    out = torch.zeros(n * repeats, n * repeats, dtype=h.dtype, device=h.device)
    for i in range(repeats):
        out[i * n : (i + 1) * n, i * n : (i + 1) * n] = h
    return out


@torch.no_grad()
def untie_embeddings(model) -> bool:
    """Give lm_head its own storage. Returns True if a tie was actually broken.

    Required whenever the final norm's gain is folded into lm_head: after folding,
    lm_head is no longer equal to the embedding matrix, so they cannot share storage.
    Qwen2.5 ties at 0.5B/1.5B and unties at 7B, so this fires only on the small models --
    and it is why the 1.5B's compression ratio is materially worse than the 7B's.
    """
    embed = model.get_input_embeddings()
    head = model.get_output_embeddings()
    if head is None or head.weight is not embed.weight:
        return False
    head.weight = nn.Parameter(embed.weight.detach().clone())
    if hasattr(model.config, "tie_word_embeddings"):
        model.config.tie_word_embeddings = False
    if hasattr(model, "_tied_weights_keys"):
        model._tied_weights_keys = []
    return True


@torch.no_grad()
def fold_norm_gain(model, norm_name: str, consumer_names: tuple[str, ...]) -> None:
    """Fold an RMSNorm's learned gain into its consumers, leaving the norm at all-ones.

    W @ diag(g) scales the *columns* of W, i.e. the input channels -- which is what the
    gain multiplies. Scaling rows instead would be silently wrong for any non-square layer
    and merely wrong for square ones, so this is worth stating.
    """
    norm = model.get_submodule(norm_name)
    gain = norm.weight.data.detach().float().clone()

    for name in consumer_names:
        linear = model.get_submodule(name)
        w = linear.weight.data
        linear.weight.data = (w.float() * gain[None, :]).to(w.dtype)

    norm.weight.data = torch.ones_like(norm.weight.data)


@torch.no_grad()
def fold_all_norms(model, graph: ModelGraph) -> None:
    """Fold every RMSNorm gain in the model, including the final pre-lm_head norm."""
    for block in graph.blocks:
        for norm in block.norms:
            fold_norm_gain(model, norm.name, norm.consumers)
    fold_norm_gain(model, graph.final_norm_name, (graph.lm_head_name,))


@torch.no_grad()
def apply_r1(model, graph: ModelGraph, r1: torch.Tensor) -> None:
    """Rotate the residual stream basis. Assumes norm gains are already folded."""
    dev, dt = r1.device, r1.dtype

    embed = model.get_submodule(graph.embed_name)
    embed.weight.data = (embed.weight.data.to(dev, dt) @ r1).to(embed.weight.dtype)

    for spec in graph.all_linears:
        linear = model.get_submodule(spec.name)
        w = linear.weight.data
        if spec.role is Role.READER:
            linear.weight.data = (w.to(dev, dt) @ r1).to(w.dtype)
        else:
            linear.weight.data = (r1.T @ w.to(dev, dt)).to(w.dtype)

    head = model.get_submodule(graph.lm_head_name)
    head.weight.data = (head.weight.data.to(dev, dt) @ r1).to(head.weight.dtype)


@torch.no_grad()
def apply_r3(model, graph: ModelGraph, h: torch.Tensor) -> None:
    """Rotate each attention head's value subspace; cancels exactly inside o_proj."""
    dev, dt = h.device, h.dtype
    bv = _block_diag(h, graph.num_key_value_heads)  # v_proj has n_kv heads
    bo = _block_diag(h, graph.num_attention_heads)  # o_proj sees n_heads after repeat_kv

    for block in graph.blocks:
        prefix = f"model.layers.{block.layer_index}"
        v = model.get_submodule(f"{prefix}.self_attn.v_proj")
        o = model.get_submodule(f"{prefix}.self_attn.o_proj")

        v.weight.data = (bv @ v.weight.data.to(dev, dt)).to(v.weight.dtype)
        if v.bias is not None:
            # v_proj's output is rotated, so its bias must be rotated with it.
            v.bias.data = (bv @ v.bias.data.to(dev, dt)).to(v.bias.dtype)

        o.weight.data = (o.weight.data.to(dev, dt) @ bo.T).to(o.weight.dtype)


@torch.no_grad()
def fuse_rotations(
    model,
    graph: ModelGraph,
    seed: int = 0,
    kind: str = "randomized_hadamard",
    rotate_head_dim: bool = True,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> RotationPlan:
    """Fold norm gains and fuse R1 (and optionally R3) into the weights, in place.

    Order matters: untie before folding (folding makes lm_head differ from the embedding),
    and fold before rotating (the gain does not commute with the rotation).
    """
    device = device or next(model.parameters()).device

    untie_embeddings(model)
    fold_all_norms(model, graph)

    r1_np = build_rotation(graph.hidden_size, seed=seed, kind=kind)
    r1 = torch.from_numpy(r1_np).to(device, dtype)
    apply_r1(model, graph, r1)

    r3 = None
    if rotate_head_dim:
        h_np = hadamard_matrix(graph.head_dim).astype(np.float64) / np.sqrt(graph.head_dim)
        r3 = torch.from_numpy(h_np).to(device, dtype)
        apply_r3(model, graph, r3)

    return RotationPlan(
        kind=kind,
        seed=seed,
        hidden_size=graph.hidden_size,
        head_dim=graph.head_dim,
        r1=r1,
        r3=r3,
    )
