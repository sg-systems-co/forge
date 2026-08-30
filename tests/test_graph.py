"""Model graph descriptors and the guardrails that stop unrepresentable conversions."""

from types import SimpleNamespace

import pytest

from forge.models.graph import Role
from forge.models.registry import build_graph, supported_architectures

QWEN_1_5B = dict(
    architectures=["Qwen2ForCausalLM"], hidden_size=1536, intermediate_size=8960,
    num_attention_heads=12, num_key_value_heads=2, num_hidden_layers=28,
    vocab_size=151936, tie_word_embeddings=True, head_dim=128,
)
QWEN_7B = dict(
    architectures=["Qwen2ForCausalLM"], hidden_size=3584, intermediate_size=18944,
    num_attention_heads=28, num_key_value_heads=4, num_hidden_layers=28,
    vocab_size=152064, tie_word_embeddings=False, head_dim=128,
)


def cfg(**overrides):
    return SimpleNamespace(**{**QWEN_1_5B, **overrides})


def test_builds_expected_linears():
    g = build_graph(cfg())
    assert g.num_layers == 28
    assert len(g.all_linears) == 28 * 7
    names = {s.gguf_name for s in g.blocks[0].linears}
    assert names == {
        "blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_v.weight",
        "blk.0.attn_output.weight", "blk.0.ffn_gate.weight", "blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight",
    }


def test_reader_writer_roles():
    """Only o_proj and down_proj write into the residual stream."""
    block = build_graph(cfg()).blocks[0]
    assert {s.name.split(".")[-1] for s in block.writers} == {"o_proj", "down_proj"}
    assert {s.name.split(".")[-1] for s in block.readers} == {
        "q_proj", "k_proj", "v_proj", "gate_proj", "up_proj",
    }


def test_gqa_shapes():
    """k/v project to n_kv_heads * head_dim, not hidden_size."""
    block = build_graph(cfg()).blocks[0]
    assert block.by_name("model.layers.0.self_attn.k_proj").out_features == 2 * 128
    assert block.by_name("model.layers.0.self_attn.q_proj").out_features == 12 * 128
    assert block.by_name("model.layers.0.mlp.down_proj").in_features == 8960


def test_writers_have_no_bias():
    """Residual-stream writers must be bias-free or rotation fusion would need to move
    the bias too. This asserts the assumption the fusion code relies on."""
    for spec in build_graph(cfg()).all_linears:
        if spec.role is Role.WRITER:
            assert not spec.has_bias, spec.name


def test_qwen_has_qkv_bias_llama_does_not():
    qwen = {s.name.split(".")[-1] for s in build_graph(cfg()).all_linears if s.has_bias}
    assert qwen == {"q_proj", "k_proj", "v_proj"}
    llama_cfg = cfg(architectures=["LlamaForCausalLM"], attention_bias=False)
    assert not any(s.has_bias for s in build_graph(llama_cfg).all_linears)


def test_all_layers_block_aligned():
    for c in (cfg(), cfg(**QWEN_7B)):
        assert all(s.block_aligned for s in build_graph(c).all_linears)


def test_rejects_unaligned_dimensions():
    """A model whose FFN width is not a multiple of 256 must be refused, not padded."""
    with pytest.raises(ValueError, match="not\n?\\s*multiples of 256|multiples of 256"):
        build_graph(cfg(intermediate_size=8900))


def test_rejects_inconsistent_head_dim():
    with pytest.raises(ValueError, match="head_dim"):
        build_graph(cfg(head_dim=256))  # 256-aligned, but 12 * 256 != 1536


def test_rejects_unknown_architecture():
    with pytest.raises(ValueError, match="no FORGE adapter"):
        build_graph(cfg(architectures=["MambaForCausalLM"]))
    assert "Qwen2ForCausalLM" in supported_architectures()


@pytest.mark.parametrize("c,expect_ternary", [(QWEN_1_5B, 1.31e9), (QWEN_7B, 6.53e9)])
def test_ternary_param_counts(c, expect_ternary):
    g = build_graph(SimpleNamespace(**c))
    assert abs(g.ternary_params - expect_ternary) / expect_ternary < 0.01
