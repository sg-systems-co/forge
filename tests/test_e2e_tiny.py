"""Test 7 from the plan: the whole pipeline, end to end, on a model small enough for CI.

This is the test that catches integration breakage -- a change that keeps every unit test
green but makes rotate -> solve -> pack -> export stop composing. It runs on a randomly
initialized tiny Qwen2, so it needs no downloads and no network.

The export leg is marked `slow` because it shells out to convert_hf_to_gguf.py and
llama-quantize.
"""

import numpy as np
import pytest
import torch

from forge.calib.capture import LayerwiseRunner
from forge.config import ForgeConfig
from forge.models.registry import build_graph
from forge.pack.tq2 import QK_K, dequantize_tq2_0
from forge.quant.sequential import quantize_model
from forge.rotate.fuse import fuse_rotations

DEVICE = torch.device("cpu")


def tiny_model(seed=0, tie=False, vocab_size=512):
    """A 2-layer Qwen2 with GQA. `vocab_size` defaults to something tiny for speed; the
    export test must pass the real tokenizer's size, because convert_hf_to_gguf.py asserts
    max(tokenizer.vocab.values()) < config.vocab_size."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(seed)
    cfg = Qwen2Config(
        hidden_size=256, intermediate_size=512, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=64,
        vocab_size=vocab_size, max_position_embeddings=128,
        tie_word_embeddings=tie, attention_bias=True,
    )
    model = Qwen2ForCausalLM(cfg).to(torch.float32).eval()
    for name, param in model.named_parameters():
        if "norm" in name:
            param.data = torch.rand_like(param.data) * 1.5 + 0.25
    return model


def tiny_config(**overrides):
    cfg = ForgeConfig(device="cpu", dtype="float32")
    cfg.calib.nsamples, cfg.calib.seqlen = 4, 64
    for key, value in overrides.items():
        head, _, tail = key.partition(".")
        setattr(getattr(cfg, head) if tail else cfg, tail or head, value)
    return cfg


def ids_for(model, n=4, length=64):
    g = torch.Generator().manual_seed(1)
    return torch.randint(0, model.config.vocab_size, (n, length), generator=g)


# ------------------------------------------------------------------ pipeline


def test_full_pipeline_runs_and_produces_ternary_weights():
    model = tiny_model()
    graph = build_graph(model.config)
    cfg = tiny_config()

    fuse_rotations(model, graph, seed=0, dtype=torch.float64)
    report = quantize_model(model, graph, ids_for(model), cfg, DEVICE, verbose=False)

    assert len(report.records) == graph.num_layers * 7
    assert all(np.isfinite(r.rel_error) for r in report.records)

    # Every quantized weight must now be exactly representable as s * t per 256 block.
    for spec in graph.all_linears:
        w = model.get_submodule(spec.name).weight.data
        blocks = w.reshape(-1, QK_K)
        amax = blocks.abs().max(dim=1, keepdim=True).values
        safe = torch.where(amax > 0, amax, torch.ones_like(amax))
        codes = blocks / safe
        assert torch.allclose(codes, codes.round(), atol=1e-5), spec.name
        assert codes.abs().max() <= 1.0 + 1e-5, spec.name


def test_pipeline_beats_rtn_on_reconstruction_error():
    """The solver must actually help end to end, not just on synthetic Hessians."""
    ids = ids_for(tiny_model())

    def mean_error(**solver):
        model = tiny_model()
        graph = build_graph(model.config)
        cfg = tiny_config()
        for k, v in solver.items():
            setattr(cfg.solver, k, v)
        if solver.get("rotate", True):
            fuse_rotations(model, graph, seed=0, dtype=torch.float64)
        return quantize_model(model, graph, ids, cfg, DEVICE, verbose=False).mean("rel_error")

    rtn = mean_error(method="rtn", sequential=False, rescale=False)
    gptq = mean_error(method="gptq", sequential=False, rescale=False)
    assert gptq < rtn, (gptq, rtn)


def test_sequential_improves_attenuation():
    """Sequential propagation exists to pull layer gain back toward 1.0."""
    ids = ids_for(tiny_model())

    def mean_attenuation(sequential):
        model = tiny_model()
        graph = build_graph(model.config)
        cfg = tiny_config()
        cfg.solver.sequential = sequential
        cfg.solver.rescale = False
        fuse_rotations(model, graph, seed=0, dtype=torch.float64)
        return quantize_model(model, graph, ids, cfg, DEVICE, verbose=False).mean("attenuation")

    assert mean_attenuation(True) > mean_attenuation(False)


def test_quantized_model_still_generates():
    """A model that quantizes cleanly but cannot run forward is not done."""
    model = tiny_model()
    graph = build_graph(model.config)
    fuse_rotations(model, graph, seed=0, dtype=torch.float64)
    quantize_model(model, graph, ids_for(model), tiny_config(), DEVICE, verbose=False)

    with torch.no_grad():
        out = model(ids_for(model, n=1, length=16))
    assert out.logits.shape == (1, 16, model.config.vocab_size)
    assert torch.isfinite(out.logits).all()


# ------------------------------------------------------------------ pack round trip


def test_quantized_weights_survive_the_packing_round_trip():
    """The bridge between the solver and the exporter: reconstructed weights must pack
    to TQ2_0 and come back bit-identical."""
    from forge.pack.tq2 import quantize_amax_reference

    model = tiny_model()
    graph = build_graph(model.config)
    fuse_rotations(model, graph, seed=0, dtype=torch.float64)
    quantize_model(model, graph, ids_for(model), tiny_config(), DEVICE, verbose=False)

    for spec in graph.all_linears:
        w = model.get_submodule(spec.name).weight.data.float().numpy()
        buf = quantize_amax_reference(w)
        back = dequantize_tq2_0(buf, w.size).reshape(w.shape)
        np.testing.assert_allclose(back, w, rtol=1e-3, atol=1e-6, err_msg=spec.name)


def test_capture_runner_advances_both_buffers():
    """Teacher and student buffers must diverge once the consumer quantizes."""
    model = tiny_model()
    graph = build_graph(model.config)
    runner = LayerwiseRunner(model, graph, DEVICE, store_dtype=torch.float32)

    seen = []
    for block_pass in runner.run(ids_for(model), sequential=True):
        seen.append((block_pass.inputs_fp.clone(), block_pass.inputs_q.clone()))
        # Perturb the block so the student diverges from the teacher.
        for spec in block_pass.block.linears:
            model.get_submodule(spec.name).weight.data *= 0.5

    assert len(seen) == graph.num_layers
    torch.testing.assert_close(seen[0][0], seen[0][1])  # identical at the first block
    assert not torch.allclose(seen[1][0], seen[1][1])  # diverged by the second


# ------------------------------------------------------------------ export


@pytest.mark.slow
def test_export_produces_a_loadable_gguf(tmp_path):
    """The end of the line: a real GGUF that llama.cpp can read back."""
    from transformers import AutoTokenizer

    from forge.pack.gguf_writer import ExportPaths, export_gguf

    paths = ExportPaths.default()
    try:
        paths.check()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B")
    model = tiny_model(vocab_size=len(tok))
    graph = build_graph(model.config)
    cfg = tiny_config()
    fuse_rotations(model, graph, seed=0, dtype=torch.float64)
    quantize_model(model, graph, ids_for(model), cfg, DEVICE, verbose=False)

    out = export_gguf(model, tok, tmp_path / "tiny.gguf", cfg)

    assert out.exists() and out.stat().st_size > 0
    assert out.with_suffix(".forge.json").exists()
    assert out.read_bytes()[:4] == b"GGUF"


def test_exclude_leaves_named_tensors_untouched():
    """`--exclude ffn_down` must leave those weights bit-identical and quantize the rest.

    This is the stock-GGUF-compatible lever for the worst-conditioned layer: it costs
    memory (ffn_down is 29% of the 7B's ternary parameters) but needs no graph change,
    unlike an online rotation.
    """
    model = tiny_model()
    graph = build_graph(model.config)
    cfg = tiny_config()
    cfg.solver.exclude = ("ffn_down",)

    fuse_rotations(model, graph, seed=0, dtype=torch.float64)
    before = {
        s.name: model.get_submodule(s.name).weight.data.clone() for s in graph.all_linears
    }
    report = quantize_model(model, graph, ids_for(model), cfg, DEVICE, verbose=False)

    assert not any(r.tensor == "ffn_down" for r in report.records)
    assert len(report.records) == graph.num_layers * 6

    for spec in graph.all_linears:
        after = model.get_submodule(spec.name).weight.data
        if spec.gguf_name.split(".", 2)[-1].rsplit(".", 1)[0] == "ffn_down":
            torch.testing.assert_close(after, before[spec.name])  # untouched
        else:
            assert not torch.allclose(after, before[spec.name]), spec.name  # quantized


def test_exclude_nothing_is_the_default():
    model = tiny_model()
    graph = build_graph(model.config)
    report = quantize_model(model, graph, ids_for(model), tiny_config(), DEVICE, verbose=False)
    assert len(report.records) == graph.num_layers * 7
