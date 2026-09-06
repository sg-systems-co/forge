"""Falcon-H1 adapter: parallel Mamba-2 (SSD) + attention hybrid."""

from types import SimpleNamespace

import pytest

from forge.models.falcon_h1 import mamba_dims
from forge.models.graph import Role
from forge.models.registry import build_graph, supported_architectures

FALCON_H1_7B = dict(
    architectures=["FalconH1ForCausalLM"], model_type="falcon_h1",
    hidden_size=3072, intermediate_size=12288, num_hidden_layers=44,
    num_attention_heads=12, num_key_value_heads=2, head_dim=128,
    mamba_d_ssm=3072, mamba_d_state=256, mamba_n_groups=1, mamba_n_heads=24,
    mamba_d_head=128, mamba_d_conv=4, vocab_size=130049,
    tie_word_embeddings=False, attention_bias=False, mlp_bias=False, projectors_bias=False,
)


def cfg(**overrides):
    return SimpleNamespace(**{**FALCON_H1_7B, **overrides})


def test_registered():
    assert "FalconH1ForCausalLM" in supported_architectures()


def test_nine_projections_per_block():
    """Attention (4) + SSD (2) + MLP (3). conv1d, A_log, D, dt_bias and ssm_norm are
    deliberately absent -- they drive the recurrence and are excluded."""
    block = build_graph(cfg()).blocks[0]
    assert len(block.linears) == 9
    assert {s.gguf_name for s in block.linears} == {
        "blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_v.weight",
        "blk.0.attn_output.weight", "blk.0.ssm_in.weight", "blk.0.ssm_out.weight",
        "blk.0.ffn_gate.weight", "blk.0.ffn_up.weight", "blk.0.ffn_down.weight",
    }


def test_three_writers_per_block():
    """Attention, SSD and MLP each write into the residual stream."""
    block = build_graph(cfg()).blocks[0]
    assert {s.gguf_name.split(".")[2] for s in block.writers} == {
        "attn_output", "ssm_out", "ffn_down"
    }


def test_attention_and_ssm_share_the_first_norm():
    """The parallel hybrid's defining property: both branches read the same normalized
    residual, so the gain folds into q/k/v AND mamba.in_proj."""
    norms = build_graph(cfg()).blocks[0].norms
    assert norms[0].name.endswith("input_layernorm")
    assert set(norms[0].consumers) == {
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.v_proj",
        "model.layers.0.mamba.in_proj",
    }
    assert norms[1].name.endswith("pre_ff_layernorm")
    assert set(norms[1].consumers) == {
        "model.layers.0.feed_forward.gate_proj",
        "model.layers.0.feed_forward.up_proj",
    }


def test_ssd_projection_widths():
    """in_proj emits [gate | x | B | C | dt] concatenated."""
    d = mamba_dims(cfg())
    assert d["d_ssm"] == 3072
    assert d["conv_dim"] == 3072 + 2 * 1 * 256
    assert d["projection_size"] == 3072 + d["conv_dim"] + 24
    block = build_graph(cfg()).blocks[0]
    assert block.by_name("model.layers.0.mamba.in_proj").out_features == d["projection_size"]
    assert block.by_name("model.layers.0.mamba.out_proj").in_features == d["d_ssm"]


def test_narrow_attention_is_allowed():
    """Falcon-H1's attention spans 12*128 = 1536, only half of hidden 3072. That is
    legitimate and must not trip the shared validator (it does hold for Qwen, which
    asserts it in its own adapter)."""
    g = build_graph(cfg())
    assert g.num_attention_heads * g.head_dim != g.hidden_size
    g.validate()


def test_r3_dimensions_are_guarded():
    """The validator catches a graph whose head geometry does not match its tensors.

    An adapter derives v_proj/o_proj from n_heads and n_kv_heads, so it is always
    self-consistent by construction -- the guard exists for the case where that derivation
    is *wrong*, which is what this simulates. Without it, apply_r3 would silently build a
    block-diagonal of the wrong size and corrupt every attention layer.
    """
    from dataclasses import replace

    graph = build_graph(cfg())
    block = graph.blocks[0]
    bad = tuple(
        replace(s, out_features=s.out_features * 2) if s.gguf_name.endswith("attn_v.weight")
        else s
        for s in block.linears
    )
    graph.blocks[0] = replace(block, linears=bad)

    with pytest.raises(ValueError, match="R3 would be built at the wrong size"):
        graph.validate()


def test_all_inputs_block_aligned():
    assert all(s.block_aligned for s in build_graph(cfg()).all_linears)


def test_ffn_down_is_the_excludable_writer():
    """The v1 shipping config keeps ffn_down at Q6_K; the stem must match so --exclude
    ffn_down selects it."""
    stems = {s.gguf_name.split(".", 2)[-1].rsplit(".", 1)[0] for s in build_graph(cfg()).all_linears}
    assert "ffn_down" in stems and "ssm_out" in stems


def test_parameter_split():
    """MLP dominates, SSD is second; the excluded SSM internals are negligible."""
    block = build_graph(cfg()).blocks[0]
    by = {s.gguf_name.split(".", 2)[-1].rsplit(".", 1)[0]: s.numel for s in block.linears}
    total = sum(by.values())
    assert by["ffn_down"] / total > 0.2
    assert by["ssm_in"] > by["ssm_out"] > by["attn_q"]
