"""Mamba/SSM adapter: what must hold for the frozen pipeline to apply unchanged."""

from types import SimpleNamespace

import pytest

from forge.models.graph import Role
from forge.models.registry import build_graph, supported_architectures

FALCON_MAMBA_7B = dict(
    architectures=["FalconMambaForCausalLM"], model_type="falcon_mamba",
    hidden_size=4096, intermediate_size=8192, num_hidden_layers=64,
    vocab_size=65024, tie_word_embeddings=False, use_bias=False, expand=2,
)


def cfg(**overrides):
    return SimpleNamespace(**{**FALCON_MAMBA_7B, **overrides})


def test_registered():
    assert "FalconMambaForCausalLM" in supported_architectures()
    assert "MambaForCausalLM" in supported_architectures()


def test_only_the_residual_stream_projections_are_quantized():
    """in_proj and out_proj only. The SSM internals drive a recurrence through exp() and
    softplus, and are ~4% of a block's parameters -- nothing to win, much to break."""
    block = build_graph(cfg()).blocks[0]
    assert len(block.linears) == 2
    assert {s.name.split(".")[-1] for s in block.linears} == {"in_proj", "out_proj"}
    assert {s.gguf_name for s in block.linears} == {
        "blk.0.ssm_in.weight", "blk.0.ssm_out.weight"
    }


def test_reader_writer_roles():
    block = build_graph(cfg()).blocks[0]
    assert block.by_name("backbone.layers.0.mixer.in_proj").role is Role.READER
    assert block.by_name("backbone.layers.0.mixer.out_proj").role is Role.WRITER


def test_projection_shapes():
    """in_proj emits the SSM input and its gate concatenated, hence 2 * d_inner."""
    block = build_graph(cfg()).blocks[0]
    assert block.by_name("backbone.layers.0.mixer.in_proj").out_features == 2 * 8192
    assert block.by_name("backbone.layers.0.mixer.in_proj").in_features == 4096
    assert block.by_name("backbone.layers.0.mixer.out_proj").out_features == 4096
    assert block.by_name("backbone.layers.0.mixer.out_proj").in_features == 8192


def test_norm_feeds_only_in_proj():
    """out_proj reads the post-SSM activation, not the normalized residual, so folding the
    norm gain into it would be wrong."""
    norms = build_graph(cfg()).blocks[0].norms
    assert len(norms) == 1
    assert norms[0].consumers == ("backbone.layers.0.mixer.in_proj",)


def test_module_paths_are_mamba_shaped():
    g = build_graph(cfg())
    assert g.layers_path == "backbone.layers"
    assert g.embed_name == "backbone.embeddings"
    assert g.final_norm_name == "backbone.norm_f"
    assert g.lm_head_name == "lm_head"


def test_no_attention():
    g = build_graph(cfg())
    assert g.has_attention is False
    assert g.num_attention_heads == 0 and g.head_dim == 0
    g.validate()  # must not raise despite head_dim == 0


def test_block_alignment_for_the_real_model():
    g = build_graph(cfg())
    assert all(s.block_aligned for s in g.all_linears), "TQ2_0 needs in_features % 256 == 0"


def test_rejects_unaligned_inner_dimension():
    with pytest.raises(ValueError, match="multiples of 256"):
        build_graph(cfg(intermediate_size=8100))


def test_parameter_accounting():
    """in_proj + out_proj should be ~96% of a block; the rest is deliberately excluded."""
    g = build_graph(cfg())
    per_block = sum(s.numel for s in g.blocks[0].linears)
    d_inner, hidden, dt_rank, d_state = 8192, 4096, 256, 16
    excluded = (
        (dt_rank + 2 * d_state) * d_inner  # x_proj
        + d_inner * dt_rank                # dt_proj
        + d_inner * d_state                # A_log
        + d_inner                          # D
        + d_inner * 4                      # conv1d
    )
    assert per_block / (per_block + excluded) > 0.95


def test_expand_fallback_when_intermediate_size_absent():
    c = cfg()
    del c.intermediate_size
    g = build_graph(c)
    assert g.intermediate_size == 2 * 4096
