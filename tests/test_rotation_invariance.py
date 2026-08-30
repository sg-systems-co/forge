"""Test 2 from the plan -- the highest-value test in the suite.

Rotation and norm folding are supposed to change *nothing* about what the model computes.
So: fuse the rotations with no quantization at all, and assert the logits are unchanged.

Every transpose, sign, GQA head-grouping, gain-folding and weight-tying bug in
forge/rotate/fuse.py shows up here, and shows up *before* quantization noise can hide it.
The per-block variants localize a failure to a specific fusion step.
"""

import numpy as np
import pytest
import torch

from forge.models.registry import build_graph
from forge.rotate.fuse import (
    apply_r1,
    apply_r3,
    fold_all_norms,
    fuse_rotations,
    untie_embeddings,
)
from forge.rotate.hadamard import build_rotation, hadamard_matrix

TOL = 1e-3


def tiny_model(tie: bool = False, n_kv_heads: int = 2, seed: int = 0):
    """A small Qwen2 with GQA, float32, and deliberately non-trivial RMSNorm gains."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(seed)
    cfg = Qwen2Config(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=n_kv_heads,
        head_dim=64,
        vocab_size=512,
        max_position_embeddings=128,
        tie_word_embeddings=tie,
        attention_bias=True,
    )
    model = Qwen2ForCausalLM(cfg).to(torch.float32).eval()

    # RMSNorm initializes to all-ones, which would make gain folding a silent no-op and
    # hide any bug in it. Force real gains.
    for name, param in model.named_parameters():
        if "norm" in name:
            param.data = torch.rand_like(param.data) * 1.5 + 0.25
        elif "bias" in name:
            param.data = torch.randn_like(param.data) * 0.1
    return model


@torch.no_grad()
def logits_of(model, ids):
    return model(ids, use_cache=False).logits.float().clone()


def sample_ids(model, n=2, length=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, model.config.vocab_size, (n, length), generator=g)


def max_abs_diff(a, b):
    return float((a - b).abs().max())


# ------------------------------------------------------------------ whole-model


@pytest.mark.parametrize("tie", [False, True])
@pytest.mark.parametrize("n_kv_heads", [1, 2, 4])
def test_full_fusion_preserves_logits(tie, n_kv_heads):
    """The headline invariant, across weight tying and every GQA grouping."""
    model = tiny_model(tie=tie, n_kv_heads=n_kv_heads)
    graph = build_graph(model.config)
    ids = sample_ids(model)

    before = logits_of(model, ids)
    fuse_rotations(model, graph, seed=1234, dtype=torch.float64)
    after = logits_of(model, ids)

    assert max_abs_diff(before, after) < TOL, max_abs_diff(before, after)


@pytest.mark.parametrize("kind", ["randomized_hadamard", "hadamard", "random_orthogonal"])
def test_invariance_for_every_rotation_kind(kind):
    model = tiny_model()
    graph = build_graph(model.config)
    ids = sample_ids(model)

    before = logits_of(model, ids)
    fuse_rotations(model, graph, seed=7, kind=kind, dtype=torch.float64)
    assert max_abs_diff(before, logits_of(model, ids)) < TOL


# ------------------------------------------------------------------ per-step


def test_norm_folding_alone_is_invariant():
    """Isolates gain folding: no rotation involved."""
    model = tiny_model()
    graph = build_graph(model.config)
    ids = sample_ids(model)

    before = logits_of(model, ids)
    untie_embeddings(model)
    fold_all_norms(model, graph)
    after = logits_of(model, ids)

    assert max_abs_diff(before, after) < TOL
    # Every norm must now be exactly all-ones, or the gain was folded twice / not at all.
    for name, param in model.named_parameters():
        if "norm" in name:
            assert torch.allclose(param.data, torch.ones_like(param.data)), name


def test_r1_alone_is_invariant_after_folding():
    """Isolates the residual-stream rotation."""
    model = tiny_model()
    graph = build_graph(model.config)
    ids = sample_ids(model)

    before = logits_of(model, ids)
    untie_embeddings(model)
    fold_all_norms(model, graph)
    r1 = torch.from_numpy(build_rotation(graph.hidden_size, seed=3)).to(torch.float64)
    apply_r1(model, graph, r1)

    assert max_abs_diff(before, logits_of(model, ids)) < TOL


@pytest.mark.parametrize("n_kv_heads", [1, 2, 4])
def test_r3_alone_is_invariant(n_kv_heads):
    """Isolates the per-head value rotation, which is where GQA grouping can go wrong.

    R3 needs no norm folding: it lives entirely between v_proj and o_proj.
    """
    model = tiny_model(n_kv_heads=n_kv_heads)
    graph = build_graph(model.config)
    ids = sample_ids(model)

    before = logits_of(model, ids)
    h = torch.from_numpy(hadamard_matrix(graph.head_dim) / np.sqrt(graph.head_dim))
    apply_r3(model, graph, h.to(torch.float64))

    assert max_abs_diff(before, logits_of(model, ids)) < TOL


def test_r3_wrong_head_grouping_is_actually_detected():
    """Guard on the guard: if R3 used the wrong repeat count, the test above must fail.

    Without this, a GQA bug that happened to be invariant would leave the suite green for
    the wrong reason.
    """
    model = tiny_model(n_kv_heads=2)
    graph = build_graph(model.config)
    ids = sample_ids(model)
    before = logits_of(model, ids)

    # Deliberately rotate o_proj as if every head were its own KV head.
    from forge.rotate.fuse import _block_diag

    h = torch.from_numpy(hadamard_matrix(graph.head_dim) / np.sqrt(graph.head_dim)).double()
    bad = _block_diag(h, graph.num_key_value_heads)
    for block in graph.blocks:
        prefix = f"model.layers.{block.layer_index}"
        v = model.get_submodule(f"{prefix}.self_attn.v_proj")
        o = model.get_submodule(f"{prefix}.self_attn.o_proj")
        v.weight.data = (bad @ v.weight.data.double()).float()
        v.bias.data = (bad @ v.bias.data.double()).float()
        wrong = _block_diag(h, graph.num_key_value_heads)  # should be num_attention_heads
        o.weight.data = (o.weight.data.double() @ wrong.T.repeat(2, 2) / 2).float()

    assert max_abs_diff(before, logits_of(model, ids)) > TOL


# ------------------------------------------------------------------ tying


def test_untie_gives_lm_head_independent_storage():
    model = tiny_model(tie=True)
    embed = model.get_input_embeddings()
    head = model.get_output_embeddings()
    assert head.weight is embed.weight

    assert untie_embeddings(model) is True
    assert head.weight is not embed.weight
    torch.testing.assert_close(head.weight.data, embed.weight.data)
    assert model.config.tie_word_embeddings is False

    assert untie_embeddings(model) is False  # idempotent


def test_untied_model_is_left_alone():
    model = tiny_model(tie=False)
    assert untie_embeddings(model) is False


def test_tied_model_diverges_after_folding():
    """After folding the final norm gain, lm_head must no longer equal the embedding."""
    model = tiny_model(tie=True)
    graph = build_graph(model.config)
    untie_embeddings(model)
    fold_all_norms(model, graph)
    embed = model.get_input_embeddings().weight.data
    head = model.get_output_embeddings().weight.data
    assert not torch.allclose(embed, head)


# ------------------------------------------------------------------ the actual point


def test_rotation_reduces_weight_kurtosis():
    """Why we rotate at all: it Gaussianizes the weights so ternary rounding hurts less.

    Kurtosis is the outlier metric -- a Gaussian sits at 3.0.
    """
    model = tiny_model()
    graph = build_graph(model.config)

    def mean_kurtosis():
        vals = []
        for spec in graph.all_linears:
            w = model.get_submodule(spec.name).weight.data.float().flatten()
            centered = w - w.mean()
            vals.append(float((centered**4).mean() / (centered**2).mean() ** 2))
        return sum(vals) / len(vals)

    # Give the weights genuine outliers, the way a trained model has them.
    torch.manual_seed(0)
    for spec in graph.all_linears:
        w = model.get_submodule(spec.name).weight.data
        spike = torch.rand_like(w) < 0.001
        w[spike] *= 25.0

    before = mean_kurtosis()
    fuse_rotations(model, graph, seed=0, dtype=torch.float64)
    after = mean_kurtosis()

    assert after < before, f"kurtosis rose: {before:.2f} -> {after:.2f}"
    assert after < 5.0, f"still heavy-tailed after rotation: {after:.2f}"
